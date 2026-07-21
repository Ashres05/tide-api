#!/usr/bin/env python3
"""
Resolve main-artist IDs and profile-image URLs for streaming-roster MRELGs,
write a CSV, then upload images to:

    s3://parquetgarage/artist_art/{LUMINATE_ARTIST_ID}.jpeg

The uploader skips existing S3 keys, so repeated cron runs are inexpensive.

Usage:
  python3 scripts/fetch_roster_artist_artwork.py
  python3 scripts/fetch_roster_artist_artwork.py --csv-only
  python3 scripts/fetch_roster_artist_artwork.py --dry-run
  python3 scripts/fetch_roster_artist_artwork.py --skip-snowflake
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from snowflake_conn import get_snowflake_connection, load_sql  # noqa: E402

logger = logging.getLogger(__name__)

DB_PATH = REPO_ROOT / "marketshare_data.db"
ARTIST_ART_QUERY_FILE = "query_roster_artist_artwork_urls.sql"
OUTPUT_CSV = REPO_ROOT / "model" / "data" / "roster_artist_artwork_urls.csv"
UPLOADER_SCRIPT = REPO_ROOT / "scripts" / "csv_artwork_url_to_s3.py"

SNOWFLAKE_BATCH_SIZE = 500
S3_BUCKET = "parquetgarage"
S3_PREFIX = "artist_art/"


def get_roster_mrelg_ids(db_path: Path = DB_PATH) -> list[str]:
    """Read all MRELG IDs from the local streaming roster."""
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT MRELG_ID FROM STREAMING_ROSTER_2026").fetchall()
    ids = [str(row[0]).strip() for row in rows if row[0] and str(row[0]).strip()]
    logger.info("roster: %d mrelg_ids from %s", len(ids), db_path.name)
    return ids


def query_artist_artwork_urls(mrelg_ids: list[str]) -> pd.DataFrame:
    """Resolve main artists from MRELG.ARTISTS and fetch profile-image URLs."""
    sql_template = load_sql(ARTIST_ART_QUERY_FILE)
    frames: list[pd.DataFrame] = []

    with get_snowflake_connection() as sf:
        for i in range(0, len(mrelg_ids), SNOWFLAKE_BATCH_SIZE):
            batch = mrelg_ids[i : i + SNOWFLAKE_BATCH_SIZE]
            # MRELG IDs are internal identifiers, but quote defensively.
            id_list = ", ".join(f"'{mid.replace(chr(39), chr(39) * 2)}'" for mid in batch)
            sql = sql_template.replace("{MRELG_ID_LIST}", id_list)
            logger.info(
                "snowflake: querying batch %d-%d of %d",
                i + 1,
                min(i + SNOWFLAKE_BATCH_SIZE, len(mrelg_ids)),
                len(mrelg_ids),
            )
            df = sf.query(sql)
            if not df.empty:
                frames.append(df)

    columns = ["LUMINATE_ARTIST_ID", "ARTIST_NAME", "PROFILE_IMAGE"]
    if not frames:
        return pd.DataFrame(columns=columns)

    result = pd.concat(frames, ignore_index=True)
    result.columns = [str(column).strip().upper() for column in result.columns]
    result = result.dropna(subset=["LUMINATE_ARTIST_ID", "PROFILE_IMAGE"])
    result["LUMINATE_ARTIST_ID"] = result["LUMINATE_ARTIST_ID"].astype(str).str.strip()
    result["PROFILE_IMAGE"] = result["PROFILE_IMAGE"].astype(str).str.strip()
    result = result[
        (result["LUMINATE_ARTIST_ID"] != "") & (result["PROFILE_IMAGE"] != "")
    ]
    result = result.drop_duplicates(subset=["LUMINATE_ARTIST_ID"], keep="first")
    result = result.sort_values(["ARTIST_NAME", "LUMINATE_ARTIST_ID"])
    logger.info("snowflake: %d artists with profile-image URLs", len(result))
    return result[columns]


def write_csv(df: pd.DataFrame, path: Path = OUTPUT_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("wrote %d rows -> %s", len(df), path)
    return path


def run_uploader(csv_path: Path, *, dry_run: bool = False, verbose: bool = False) -> int:
    cmd = [
        sys.executable,
        str(UPLOADER_SCRIPT),
        "--csv",
        str(csv_path),
        "--bucket",
        S3_BUCKET,
        "--prefix",
        S3_PREFIX,
        "--id-column",
        "LUMINATE_ARTIST_ID",
        "--url-column",
        "PROFILE_IMAGE",
        "--extension",
        "jpeg",
    ]
    if dry_run:
        cmd.append("--dry-run")
    if verbose:
        cmd.append("--verbose")
    logger.info("running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch roster artist profile images and upload them to S3"
    )
    parser.add_argument("--csv-only", action="store_true", help="Write CSV only")
    parser.add_argument(
        "--skip-snowflake",
        action="store_true",
        help="Reuse the existing CSV and run only the uploader",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not download/upload")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.skip_snowflake:
        if not args.output_csv.is_file():
            logger.error("--skip-snowflake but CSV not found: %s", args.output_csv)
            return 1
        logger.info("skipping Snowflake, reusing %s", args.output_csv)
    else:
        mrelg_ids = get_roster_mrelg_ids(args.db)
        if not mrelg_ids:
            logger.warning("roster is empty, nothing to do")
            return 0
        df = query_artist_artwork_urls(mrelg_ids)
        if df.empty:
            logger.warning("no artist profile-image URLs returned from Snowflake")
            return 0
        write_csv(df, args.output_csv)

    if args.csv_only:
        logger.info("--csv-only: stopping before S3 upload")
        return 0

    return run_uploader(args.output_csv, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
