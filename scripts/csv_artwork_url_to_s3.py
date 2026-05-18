#!/usr/bin/env python3
"""
Read a CSV with MRELG_ID and ARTWORK_URL, download each image, upload to S3 as
``album_art/{MRELG_ID}.jpg`` (default bucket ``parquetgarage``).

Skips rows with missing ID/URL. Optionally skips keys that already exist (default).

  python3 scripts/csv_artwork_url_to_s3.py \\
    --csv /Users/ronannayak/Downloads/2025_revenue_with_url.csv

Requires: pandas, boto3 (see requirements.txt). HTTP via stdlib urllib.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CSV = "/Users/ronannayak/Downloads/2025_revenue_with_url.csv"
DEFAULT_BUCKET = "parquetgarage"
DEFAULT_PREFIX = "album_art/"


def _norm_cols(df: pd.DataFrame) -> dict[str, str]:
    """Map UPPER_SNAKE keys to original column names."""
    return {str(c).strip().upper().replace(" ", "_"): c for c in df.columns}


def _http_get_bytes(url: str, *, timeout: float = 60.0) -> tuple[bytes, Optional[str]]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "tide-api-csv-artwork-sync/1.0",
            "Accept": "image/*,*/*;q=0.8",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        ctype = resp.headers.get("Content-Type")
        return body, (ctype.split(";")[0].strip() if ctype else None)


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Upload ARTWORK_URL images to S3 as MRELG_ID.jpg")
    p.add_argument("--csv", type=Path, default=Path(DEFAULT_CSV), help="Input CSV path")
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket")
    p.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Key prefix (trailing slash recommended), e.g. album_art/",
    )
    p.add_argument("--dry-run", action="store_true", help="Log actions only; no HTTP/S3 writes")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="Process at most N rows")
    p.add_argument("--sleep", type=float, default=0.2, help="Seconds after each successful upload")
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Upload even if s3://bucket/prefixMRELG_ID.jpg already exists",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    prefix = args.prefix.strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    skip_existing = not args.no_skip_existing

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as e:
        raise SystemExit("boto3 is required. pip install boto3") from e

    s3 = boto3.client("s3")

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("CSV is empty")
        return 0

    norm = _norm_cols(df)
    mcol = norm.get("MRELG_ID") or norm.get("MRELG")
    ucol = norm.get("ARTWORK_URL") or norm.get("ARTWORK") or norm.get("IMAGE_URL")
    if not mcol or not ucol:
        raise SystemExit(
            f"Need MRELG_ID (or MRELG) and ARTWORK_URL columns. Got: {list(df.columns)}"
        )

    work = df[[mcol, ucol]].copy()
    work["_m"] = work[mcol].astype(str).str.strip()
    work["_u"] = work[ucol].astype(str).str.strip()
    work = work[(work["_m"] != "") & (work["_u"] != "")]
    work = work.drop_duplicates(subset=["_m"], keep="first")

    processed = uploaded = skipped_existing = dry_run = failed = 0
    for _, row in work.iterrows():
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1
        mid = row["_m"]
        url = row["_u"]
        key = f"{prefix}{mid}.jpg"

        if skip_existing:
            try:
                s3.head_object(Bucket=args.bucket, Key=key)
                logger.info("SKIP existing s3://%s/%s", args.bucket, key)
                skipped_existing += 1
                continue
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                    logger.warning("head_object %s: %s — trying upload anyway", key, e)

        if args.dry_run:
            logger.info("DRY-RUN would fetch %s → s3://%s/%s", url, args.bucket, key)
            dry_run += 1
            continue

        try:
            raw, ctype = _http_get_bytes(url)
        except urllib.error.HTTPError as e:
            logger.warning("HTTP %s for %s (%s)", e.code, mid, url)
            failed += 1
            continue
        except Exception as e:
            logger.warning("download failed %s: %s", mid, e)
            failed += 1
            continue

        if not raw:
            logger.warning("empty body %s", mid)
            failed += 1
            continue

        content_type = "image/jpeg"
        if ctype and ctype.lower().startswith("image/"):
            content_type = ctype.lower()

        s3.put_object(
            Bucket=args.bucket,
            Key=key,
            Body=raw,
            ContentType=content_type,
        )
        logger.info("UPLOAD s3://%s/%s (%d bytes) ← %s", args.bucket, key, len(raw), url)
        uploaded += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    logger.info(
        "done processed=%d uploaded=%d skipped_existing=%d dry_run=%d failed=%d",
        processed,
        uploaded,
        skipped_existing,
        dry_run,
        failed,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
