#!/usr/bin/env python3
"""
Pull artwork URLs from Snowflake for streaming roster mrelg_ids, write a CSV,
then invoke csv_artwork_url_to_s3.py to download images into S3.

Usage (standalone):
  python3 scripts/fetch_roster_artwork.py                    # full pipeline
  python3 scripts/fetch_roster_artwork.py --csv-only         # just write CSV, skip S3 upload
  python3 scripts/fetch_roster_artwork.py --dry-run          # pass --dry-run to uploader
  python3 scripts/fetch_roster_artwork.py --skip-snowflake   # reuse existing CSV, just run uploader

Designed to run as a daily/weekly cron. The S3 uploader skips keys that already
exist, so re-runs are cheap.
"""

from __future__ import annotations

import argparse
import logging
import os
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
ARTWORK_QUERY_FILE = "query_roster_artwork_urls.sql"
OUTPUT_CSV = REPO_ROOT / "model" / "data" / "roster_artwork_urls.csv"
UPLOADER_SCRIPT = REPO_ROOT / "scripts" / "csv_artwork_url_to_s3.py"

SNOWFLAKE_BATCH_SIZE = 500


def get_roster_mrelg_ids(db_path: Path = DB_PATH) -> list[str]:
    """Read all MRELG_IDs from the local STREAMING_ROSTER_2026 table."""
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT MRELG_ID FROM STREAMING_ROSTER_2026").fetchall()
    finally:
        conn.close()
    ids = [r[0] for r in rows if r[0]]
    logger.info("roster: %d mrelg_ids from %s", len(ids), db_path.name)
    return ids


def query_artwork_urls(mrelg_ids: list[str]) -> pd.DataFrame:
    """Query Snowflake for artwork URLs, batching to avoid query-length limits."""
    sql_template = load_sql(ARTWORK_QUERY_FILE)
    frames: list[pd.DataFrame] = []

    with get_snowflake_connection() as sf:
        for i in range(0, len(mrelg_ids), SNOWFLAKE_BATCH_SIZE):
            batch = mrelg_ids[i : i + SNOWFLAKE_BATCH_SIZE]
            id_list = ", ".join(f"'{mid}'" for mid in batch)
            sql = sql_template.replace("{MRELG_ID_LIST}", id_list)
            logger.info(
                "snowflake: querying batch %d–%d of %d",
                i + 1,
                min(i + SNOWFLAKE_BATCH_SIZE, len(mrelg_ids)),
                len(mrelg_ids),
            )
            df = sf.query(sql)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["MRELG_ID", "ARTWORK_URL"])

    result = pd.concat(frames, ignore_index=True)
    result.columns = [str(c).strip().upper() for c in result.columns]
    result = result.dropna(subset=["ARTWORK_URL"])
    result = result[result["ARTWORK_URL"].str.strip() != ""]
    result = result.drop_duplicates(subset=["MRELG_ID"], keep="first")
    logger.info("snowflake: %d mrelg_ids with artwork URLs", len(result))
    return result


def write_csv(df: pd.DataFrame, path: Path = OUTPUT_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("wrote %d rows → %s", len(df), path)
    return path


def run_uploader(csv_path: Path, *, dry_run: bool = False, verbose: bool = False) -> int:
    cmd = [
        sys.executable,
        str(UPLOADER_SCRIPT),
        "--csv", str(csv_path),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if verbose:
        cmd.append("--verbose")
    logger.info("running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch roster artwork URLs and upload to S3")
    p.add_argument("--csv-only", action="store_true", help="Write CSV only, skip S3 upload")
    p.add_argument("--skip-snowflake", action="store_true", help="Reuse existing CSV, just run uploader")
    p.add_argument("--dry-run", action="store_true", help="Pass --dry-run to csv_artwork_url_to_s3.py")
    p.add_argument("--db", type=Path, default=DB_PATH, help="Path to marketshare_data.db")
    p.add_argument("--output-csv", type=Path, default=OUTPUT_CSV, help="Output CSV path")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

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
        df = query_artwork_urls(mrelg_ids)
        if df.empty:
            logger.warning("no artwork URLs returned from Snowflake")
            return 0
        write_csv(df, args.output_csv)

    if args.csv_only:
        logger.info("--csv-only: stopping before S3 upload")
        return 0

    return run_uploader(args.output_csv, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
