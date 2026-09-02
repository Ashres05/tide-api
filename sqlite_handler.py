from snowflake_conn import get_snowflake_connection, load_sql
from search_text import normalize_search_text
import contextlib
import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta
from pathlib import Path
import logging

# TODO: Add a cron job to run this script every week.

# Database name
DATABASE_NAME = str(Path(__file__).resolve().parent / 'marketshare_data.db')
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def sqlite_connect(database: str | None = None, **kwargs):
    """
    Open SQLite and always close the connection.

    CPython 3.14 ``with sqlite3.connect()`` only commit/rollback on exit; it
    does not close. Weekly roster prewarm opened one connection per title and
    leaked FDs until boot.json hit EMFILE (Aug 17, soft ulimit 1024).
    """
    conn = sqlite3.connect(database or DATABASE_NAME, **kwargs)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        # After a checkpoint, keep the WAL file from retaining multi-GB of
        # already-checkpointed frames when other connections are still open.
        conn.execute("PRAGMA journal_size_limit=67108864")
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def wal_checkpoint(mode: str = "TRUNCATE") -> tuple:
    """
    Merge WAL frames into the main DB and optionally truncate the WAL file.

    Call after batch writes (prewarm, backfill) and before boot.json export so
    a leaked connection cannot pin multi-GB of already-checkpointed frames.
    Returns the SQLite triple (busy, log, checkpointed).
    """
    allowed = {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}
    mode_u = (mode or "TRUNCATE").upper()
    if mode_u not in allowed:
        mode_u = "TRUNCATE"
    with sqlite_connect() as conn:
        row = conn.execute(f"PRAGMA wal_checkpoint({mode_u})").fetchone()
    logger.info("wal_checkpoint(%s)=%s", mode_u, row)
    return tuple(row) if row else (0, 0, 0)


# Create table queries
CREATE_EXPECTED_RELEASES_TABLE = 'create_expected_releases_table.sql'
CREATE_MARKETSHARE_RELEASE_METRICS_TABLE = 'create_marketshare_release_metrics.sql'
CREATE_WEEKLY_MARKETSHARE_TABLE = 'create_weekly_marketshare_table.sql'
CREATE_YTD_MARKETSHARE_TABLE = 'create_ytd_marketshare_table.sql'
CREATE_MARKETSHARE_SEARCH_SUMMARY_TABLE = 'create_marketshare_search_summary_table.sql'
CREATE_MARKETSHARE_SEARCH_SUMMARY_SINGLES_TABLE = (
    'create_marketshare_search_summary_table_singles.sql'
)
CREATE_MARKETSHARE_SEARCH_ARTISTS_TABLE = 'create_marketshare_search_artists_table.sql'
CREATE_MARKETSHARE_SEARCH_DISTRIBUTORS_TABLE = (
    'create_marketshare_search_distributors_table.sql'
)
CREATE_DAILY_GLOBAL_STREAMS_TABLE = 'create_daily_global_streams_table.sql'
CREATE_WEEKLY_GLOBAL_STREAMS_TABLE = 'create_weekly_global_streams_table.sql'
CREATE_MARKETSHARE_REVENUE_2025_TABLE = 'create_marketshare_revenue_2025_table.sql'
CREATE_STREAMING_ROSTER_2026_TABLE = 'create_streaming_roster_2026_table.sql'

# Select queries
WEEKLY_MARKETSHARE_QUERY = 'query_weekly_marketshare_query.sql'
YTD_MARKETSHARE_QUERY = 'query_ytd_marketshare_query.sql'
MARKETSHARE_RELEASE_METRICS_QUERY = 'query_marketshare_release_metrics.sql'
EXPECTED_RELEASES_QUERY = 'release_get_all.sql'
MARKETSHARE_SEARCH_SUMMARY_QUERY = 'query_marketshare_search_summary.sql'
MARKETSHARE_SEARCH_SUMMARY_SINGLES_QUERY = 'query_marketshare_search_summary_singles.sql'
MARKETSHARE_SEARCH_ARTISTS_ROLLUP_QUERY = 'query_marketshare_search_artists_rollup.sql'
MARKETSHARE_SEARCH_DISTRIBUTORS_ROLLUP_QUERY = (
    'query_marketshare_search_distributors_rollup.sql'
)
DAILY_GLOBAL_STREAMING_SF_QUERY = 'query_daily_global_streaming.sql'
DAILY_GLOBAL_STREAMS_SQLITE_QUERY = 'query_daily_global_streams_sqlite.sql'
WEEKLY_GLOBAL_STREAMING_SF_QUERY = 'query_release_global_streaming.sql'
WEEKLY_GLOBAL_STREAMS_SQLITE_QUERY = 'query_weekly_global_streams_sqlite.sql'
MARKETSHARE_REVENUE_2025_BY_MRELG_QUERY = 'query_marketshare_revenue_2025_by_mrelg.sql'

# Insert queries
INSERT_WEEKLY_MARKETSHARE = 'insert_weekly_marketshare.sql'
INSERT_YTD_MARKETSHARE = 'insert_ytd_marketshare.sql'
INSERT_MARKETSHARE_RELEASE_METRICS = 'insert_marketshare_release_metrics.sql'
INSERT_MARKETSHARE_SEARCH_SUMMARY = 'insert_marketshare_search_summary.sql'
INSERT_MARKETSHARE_SEARCH_SUMMARY_SINGLES = 'insert_marketshare_search_summary_singles.sql'
INSERT_MARKETSHARE_SEARCH_ARTISTS = 'insert_marketshare_search_artists.sql'
INSERT_MARKETSHARE_SEARCH_DISTRIBUTORS = 'insert_marketshare_search_distributors.sql'
INSERT_DAILY_GLOBAL_STREAMS = 'insert_daily_global_streams.sql'
INSERT_WEEKLY_GLOBAL_STREAMS = 'insert_weekly_global_streams.sql'
INSERT_MARKETSHARE_REVENUE_2025 = 'insert_marketshare_revenue_2025.sql'

# Delete queries
DELETE_MARKETSHARE_SEARCH_SUMMARY = 'delete_marketshare_search_summary.sql'
DELETE_MARKETSHARE_SEARCH_SUMMARY_SINGLES = 'delete_marketshare_search_summary_singles.sql'
DELETE_MARKETSHARE_SEARCH_ARTISTS = 'delete_marketshare_search_artists.sql'
DELETE_MARKETSHARE_SEARCH_DISTRIBUTORS = 'delete_marketshare_search_distributors.sql'
DELETE_MARKETSHARE_REVENUE_2025 = 'delete_marketshare_revenue_2025.sql'

def ensure_streaming_roster_2026_table(conn: sqlite3.Connection) -> None:
    """Create STREAMING_ROSTER_2026 if missing (streaming revenue board roster)."""
    conn.execute(load_sql(CREATE_STREAMING_ROSTER_2026_TABLE))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(STREAMING_ROSTER_2026)")
    existing = {row[1] for row in cur.fetchall()}
    for col in ("PARENT_GROUP", "LUMINATE_ARTIST_ID"):
        if col not in existing:
            try:
                cur.execute(f"ALTER TABLE STREAMING_ROSTER_2026 ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
    conn.commit()


def ensure_expected_releases_fw_columns(conn: sqlite3.Connection) -> None:
    """
    Add first-week component AE columns if missing (older DBs pre-date UI breakdown).
    Safe to call on every connection; no-op when columns already exist.

    Uses try/ignore duplicate so concurrent callers or a schema already updated by
    CREATE TABLE do not raise sqlite3.OperationalError.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='EXPECTED_RELEASES'"
    )
    if cur.fetchone() is None:
        return
    for col, ddl in (
        ("FW_STREAMS", "REAL NOT NULL DEFAULT 0"),
        ("FW_SONGS", "REAL NOT NULL DEFAULT 0"),
        ("FW_SALES", "REAL NOT NULL DEFAULT 0"),
        ("PRODUCT_TYPE", "TEXT"),
    ):
        cur.execute("PRAGMA table_info(EXPECTED_RELEASES)")
        existing = {row[1] for row in cur.fetchall()}
        if col in existing:
            continue
        try:
            cur.execute(f"ALTER TABLE EXPECTED_RELEASES ADD COLUMN {col} {ddl}")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise

    # CLUSTER=0 was historically the API default while the field was unused; NULL means
    # "auto" (computed mixture). One-time bump of user_version avoids re-running.
    cur.execute("PRAGMA user_version")
    _uv_row = cur.fetchone()
    user_ver = int(_uv_row[0]) if _uv_row and _uv_row[0] is not None else 0
    if user_ver < 2:
        try:
            cur.execute(
                "UPDATE EXPECTED_RELEASES SET CLUSTER = NULL WHERE CLUSTER = 0"
            )
        except sqlite3.OperationalError:
            pass
        cur.execute("PRAGMA user_version = 2")


# Street-week first-week AE stored on EXPECTED_RELEASES so UI/boot do not
# recompute on every load. If the chart week containing RELEASE_DATE is a
# stub (<= this fraction of the next week), use week 2 instead.
FW_STUB_MAX_FRAC_OF_W2 = 0.10


def _parse_iso_date(value) -> date | None:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def resolve_street_week_fw(
    release_date,
    weeks: list[tuple],
    *,
    stub_frac: float = FW_STUB_MAX_FRAC_OF_W2,
) -> dict | None:
    """
    Pick first-week AE for an expected release.

    ``weeks`` is chronological ``(week_ending_date, ae, streams, sales, songs)``.
    Uses the Luminate chart week that contains ``release_date`` (week_end-6
    through week_end). If that week's AE is <= ``stub_frac`` of the following
    week's AE, the following week is used instead (pre-street leakage).
    """
    rel = _parse_iso_date(release_date)
    if rel is None or not weeks:
        return None

    parsed: list[tuple] = []
    for row in weeks:
        we = _parse_iso_date(row[0])
        if we is None:
            continue
        ae = float(row[1] or 0)
        streams = float(row[2] or 0) if len(row) > 2 else 0.0
        sales = float(row[3] or 0) if len(row) > 3 else 0.0
        songs = float(row[4] or 0) if len(row) > 4 else 0.0
        parsed.append((we, ae, streams, sales, songs))
    if not parsed:
        return None
    parsed.sort(key=lambda r: r[0])

    w1_idx = None
    for i, (we, *_rest) in enumerate(parsed):
        week_start = we - timedelta(days=6)
        if week_start <= rel <= we:
            w1_idx = i
            break
    if w1_idx is None:
        for i, (we, *_rest) in enumerate(parsed):
            if we >= rel:
                w1_idx = i
                break
    if w1_idx is None:
        return None

    chosen = w1_idx
    if w1_idx + 1 < len(parsed):
        w1_ae = parsed[w1_idx][1]
        w2_ae = parsed[w1_idx + 1][1]
        days_apart = (parsed[w1_idx + 1][0] - parsed[w1_idx][0]).days
        # Only treat the next metrics row as "week 2" when it is the next
        # chart week (7 days). Gaps are missing data, not a stub-vs-street pair.
        if days_apart == 7 and w2_ae > 0 and w1_ae <= stub_frac * w2_ae:
            chosen = w1_idx + 1

    we, ae, streams, sales, songs = parsed[chosen]
    return {
        "week_ending_date": we.strftime("%Y-%m-%d"),
        "fw_vol": ae,
        "fw_streams": streams,
        "fw_sales": sales,
        "fw_songs": songs,
        "skipped_stub_week": chosen != w1_idx,
    }


def persist_expected_release_fw_vols(release_id: int | None = None) -> dict:
    """
    Write street-week fw_vol (+ component FW_* columns) onto EXPECTED_RELEASES.

    Call after MARKETSHARE_RELEASE_METRICS is refreshed. Skips rows with no
    usable metrics. Returns counts for logging.
    """
    stats = {"updated": 0, "skipped_no_metrics": 0, "skipped_stub": 0, "errors": 0}
    with sqlite_connect() as conn:
        ensure_expected_releases_fw_columns(conn)
        cur = conn.cursor()
        if release_id is not None:
            rel_rows = cur.execute(
                "SELECT RELEASE_ID, RELEASE_DATE FROM EXPECTED_RELEASES WHERE RELEASE_ID = ?",
                (int(release_id),),
            ).fetchall()
        else:
            rel_rows = cur.execute(
                "SELECT RELEASE_ID, RELEASE_DATE FROM EXPECTED_RELEASES"
            ).fetchall()

        for rid, release_date in rel_rows:
            weeks = cur.execute(
                "SELECT WEEK_ENDING_DATE, ALBUM_EQUIVALENT, STREAMING_EQUIVALENT, "
                "PRODUCT_SALES, SONG_SALE_EQUIVALENT "
                "FROM MARKETSHARE_RELEASE_METRICS WHERE RELEASE_ID = ? "
                "ORDER BY date(WEEK_ENDING_DATE) ASC",
                (int(rid),),
            ).fetchall()
            resolved = resolve_street_week_fw(release_date, weeks)
            if resolved is None:
                stats["skipped_no_metrics"] += 1
                continue
            try:
                cur.execute(
                    "UPDATE EXPECTED_RELEASES SET "
                    "EXPECTED_ALBUM_EQUIVALENT = ?, FW_STREAMS = ?, "
                    "FW_SALES = ?, FW_SONGS = ? WHERE RELEASE_ID = ?",
                    (
                        resolved["fw_vol"],
                        resolved["fw_streams"],
                        resolved["fw_sales"],
                        resolved["fw_songs"],
                        int(rid),
                    ),
                )
                stats["updated"] += 1
                if resolved["skipped_stub_week"]:
                    stats["skipped_stub"] += 1
            except sqlite3.Error:
                stats["errors"] += 1
                logger.exception("persist_expected_release_fw_vols: failed id=%s", rid)
        conn.commit()
    logger.info("persist_expected_release_fw_vols: %s", stats)
    return stats


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def _snowflake_str(value: str) -> str:
    """Module-level Snowflake string-literal escaper."""
    return "'" + str(value).replace("'", "''") + "'"


def _min_week_anchor(cursor: sqlite3.Cursor, table: str, lookback_days: int, fallback: str = "2018-01-01") -> str:
    """
    Incremental lower-bound date for Snowflake pulls:
    max existing WEEK_ENDING_DATE in SQLite minus lookback_days.
    """
    try:
        cursor.execute(f"SELECT MAX(WEEK_ENDING_DATE) FROM {table}")
        row = cursor.fetchone()
        max_week = row[0] if row else None
    except sqlite3.Error:
        max_week = None
    if not max_week:
        return fallback
    max_dt = pd.to_datetime(max_week, errors="coerce")
    if pd.isna(max_dt):
        return fallback
    return (max_dt - pd.Timedelta(days=max(0, int(lookback_days)))).strftime("%Y-%m-%d")


def ensure_marketshare_search_summary_columns(conn: sqlite3.Connection) -> None:
    """
    Online migration for the search-summary table. Adds the persisted
    normalized columns (and supporting indexes) to databases that pre-date
    the search-latency optimization so production rollouts don't require a
    full rebuild before the new prefilter path can run.

    Safe to call on every connection; ALTER/CREATE statements are idempotent.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='MARKETSHARE_SEARCH_SUMMARY'"
    )
    if cur.fetchone() is None:
        return

    cur.execute("PRAGMA table_info(MARKETSHARE_SEARCH_SUMMARY)")
    existing = {row[1] for row in cur.fetchall()}
    for col in ("ARTIST_SEARCH", "TITLE_SEARCH", "LUMINATE_ARTIST_ID", "RELEASE_TYPE"):
        if col in existing:
            continue
        try:
            cur.execute(
                f"ALTER TABLE MARKETSHARE_SEARCH_SUMMARY ADD COLUMN {col} TEXT"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise

    cur.execute(
        "CREATE INDEX IF NOT EXISTS IDX_MARKETSHARE_SEARCH_SUMMARY_STREAMS "
        "ON MARKETSHARE_SEARCH_SUMMARY (DAILY_GLOBAL_STREAMS DESC)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS IDX_MARKETSHARE_SEARCH_SUMMARY_ARTIST_ID "
        "ON MARKETSHARE_SEARCH_SUMMARY (LUMINATE_ARTIST_ID)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS IDX_MARKETSHARE_SEARCH_SUMMARY_LABEL_DATE "
        "ON MARKETSHARE_SEARCH_SUMMARY (LABEL_NAME, RELEASE_DATE DESC, MRELG_ID)"
    )


def ensure_marketshare_search_summary_singles_columns(conn: sqlite3.Connection) -> None:
    """
    Online migration for the singles search-summary table. Same persisted
    normalized columns and streams index as the album table.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='MARKETSHARE_SEARCH_SUMMARY_SINGLES'"
    )
    if cur.fetchone() is None:
        return

    cur.execute("PRAGMA table_info(MARKETSHARE_SEARCH_SUMMARY_SINGLES)")
    existing = {row[1] for row in cur.fetchall()}
    for col in ("ARTIST_SEARCH", "TITLE_SEARCH", "LUMINATE_ARTIST_ID", "RELEASE_TYPE"):
        if col in existing:
            continue
        try:
            cur.execute(
                f"ALTER TABLE MARKETSHARE_SEARCH_SUMMARY_SINGLES ADD COLUMN {col} TEXT"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise

    cur.execute(
        "CREATE INDEX IF NOT EXISTS IDX_MARKETSHARE_SEARCH_SUMMARY_SINGLES_STREAMS "
        "ON MARKETSHARE_SEARCH_SUMMARY_SINGLES (DAILY_GLOBAL_STREAMS DESC)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS IDX_MARKETSHARE_SEARCH_SUMMARY_SINGLES_ARTIST_ID "
        "ON MARKETSHARE_SEARCH_SUMMARY_SINGLES (LUMINATE_ARTIST_ID)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS IDX_MARKETSHARE_SEARCH_SUMMARY_SINGLES_LABEL_DATE "
        "ON MARKETSHARE_SEARCH_SUMMARY_SINGLES (LABEL_NAME, RELEASE_DATE DESC, MRELG_ID)"
    )


def ensure_marketshare_search_artists_table(conn: sqlite3.Connection) -> None:
    """
    Create MARKETSHARE_SEARCH_ARTISTS and its typeahead indexes.

    Safe to call on every connection; CREATE/ALTER/INDEX are idempotent.
    """
    conn.execute(load_sql(CREATE_MARKETSHARE_SEARCH_ARTISTS_TABLE))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(MARKETSHARE_SEARCH_ARTISTS)")
    existing = {row[1] for row in cur.fetchall()}
    for col, ddl in (
        ("ARTIST_SEARCH", "TEXT"),
        ("DAILY_GLOBAL_STREAMS", "INTEGER"),
        ("RELEASE_COUNT", "INTEGER"),
        ("HAS_ARTWORK", "INTEGER"),
    ):
        if col in existing:
            continue
        try:
            cur.execute(
                f"ALTER TABLE MARKETSHARE_SEARCH_ARTISTS ADD COLUMN {col} {ddl}"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
    cur.execute(
        "CREATE INDEX IF NOT EXISTS IDX_MARKETSHARE_SEARCH_ARTISTS_SEARCH "
        "ON MARKETSHARE_SEARCH_ARTISTS (ARTIST_SEARCH)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS IDX_MARKETSHARE_SEARCH_ARTISTS_STREAMS "
        "ON MARKETSHARE_SEARCH_ARTISTS (DAILY_GLOBAL_STREAMS DESC)"
    )


def ensure_marketshare_search_distributors_table(conn: sqlite3.Connection) -> None:
    """Create MARKETSHARE_SEARCH_DISTRIBUTORS and typeahead indexes."""
    conn.execute(load_sql(CREATE_MARKETSHARE_SEARCH_DISTRIBUTORS_TABLE))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(MARKETSHARE_SEARCH_DISTRIBUTORS)")
    existing = {row[1] for row in cur.fetchall()}
    for col, ddl in (
        ("LABEL_SEARCH", "TEXT"),
        ("DAILY_GLOBAL_STREAMS", "INTEGER"),
        ("RELEASE_COUNT", "INTEGER"),
    ):
        if col in existing:
            continue
        try:
            cur.execute(
                f"ALTER TABLE MARKETSHARE_SEARCH_DISTRIBUTORS ADD COLUMN {col} {ddl}"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
    cur.execute(
        "CREATE INDEX IF NOT EXISTS IDX_MARKETSHARE_SEARCH_DISTRIBUTORS_SEARCH "
        "ON MARKETSHARE_SEARCH_DISTRIBUTORS (LABEL_SEARCH)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS IDX_MARKETSHARE_SEARCH_DISTRIBUTORS_STREAMS "
        "ON MARKETSHARE_SEARCH_DISTRIBUTORS (DAILY_GLOBAL_STREAMS DESC)"
    )


def refresh_marketshare_search_distributors(*, persist: bool = True) -> int:
    """
    Rebuild MARKETSHARE_SEARCH_DISTRIBUTORS from local search snapshots.
    One row per LABEL_NAME. Does not query Snowflake.
    """
    logger.info(
        "sqlite_handler: refreshing MARKETSHARE_SEARCH_DISTRIBUTORS (db=%s)",
        DATABASE_NAME,
    )
    with sqlite_connect() as conn:
        ensure_marketshare_search_summary_columns(conn)
        ensure_marketshare_search_summary_singles_columns(conn)
        ensure_marketshare_search_distributors_table(conn)
        cur = conn.cursor()
        try:
            cur.execute(load_sql(MARKETSHARE_SEARCH_DISTRIBUTORS_ROLLUP_QUERY))
            rollup = cur.fetchall()
        except sqlite3.Error:
            logger.exception(
                "sqlite_handler: distributor rollup failed; leaving existing index"
            )
            raise

        rows = []
        for label, streams, release_count in rollup:
            name = str(label).strip() if label is not None else ""
            if not name:
                continue
            try:
                stream_val = int(streams or 0)
            except (TypeError, ValueError):
                stream_val = 0
            try:
                n_releases = int(release_count or 0)
            except (TypeError, ValueError):
                n_releases = 0
            rows.append(
                (
                    name,
                    normalize_search_text(name),
                    stream_val,
                    n_releases,
                )
            )

        cur.execute(load_sql(DELETE_MARKETSHARE_SEARCH_DISTRIBUTORS))
        if rows:
            cur.executemany(load_sql(INSERT_MARKETSHARE_SEARCH_DISTRIBUTORS), rows)
        conn.commit()

    logger.info(
        "sqlite_handler: MARKETSHARE_SEARCH_DISTRIBUTORS rebuilt (rows=%d)",
        len(rows),
    )
    if persist and rows:
        _persist_marketshare_db_to_s3()
    return len(rows)


def refresh_marketshare_search_artists() -> int:
    """
    Rebuild MARKETSHARE_SEARCH_ARTISTS from the local album + singles
    search snapshots. One row per LUMINATE_ARTIST_ID; display name is the
    most-streamed ARTIST string for that id. Does not query Snowflake or
    scan releases at request time.

    Rows with a missing artist id are skipped. Returns the number of
    artist rows written. HAS_ARTWORK is set from the S3 artist_art/ index
    when that listing succeeds; otherwise all rows stay 0.
    """
    logger.info("sqlite_handler: refreshing MARKETSHARE_SEARCH_ARTISTS (db=%s)", DATABASE_NAME)

    with sqlite_connect() as conn:
        ensure_marketshare_search_summary_columns(conn)
        ensure_marketshare_search_summary_singles_columns(conn)
        ensure_marketshare_search_artists_table(conn)
        cur = conn.cursor()

        album_ready = _search_summary_has_artist_id(cur, "MARKETSHARE_SEARCH_SUMMARY")
        singles_ready = _search_summary_has_artist_id(
            cur, "MARKETSHARE_SEARCH_SUMMARY_SINGLES"
        )
        if not album_ready and not singles_ready:
            cur.execute(load_sql(DELETE_MARKETSHARE_SEARCH_ARTISTS))
            conn.commit()
            logger.info(
                "sqlite_handler: MARKETSHARE_SEARCH_ARTISTS empty "
                "(search snapshots have no LUMINATE_ARTIST_ID yet)"
            )
            return 0

        try:
            cur.execute(load_sql(MARKETSHARE_SEARCH_ARTISTS_ROLLUP_QUERY))
            rollup = cur.fetchall()
        except sqlite3.Error as e:
            logger.exception(
                "sqlite_handler: artist rollup failed; leaving existing index: %s", e
            )
            raise

        if not rollup:
            cur.execute(load_sql(DELETE_MARKETSHARE_SEARCH_ARTISTS))
            conn.commit()
            logger.info(
                "sqlite_handler: MARKETSHARE_SEARCH_ARTISTS rebuilt (rows=0)"
            )
            return 0

        artwork_ids: set[str] = set()
        try:
            from artist_art import known_artist_ids

            artwork_ids = {aid.upper() for aid in known_artist_ids()}
        except Exception:
            logger.warning(
                "sqlite_handler: artist_art index unavailable; HAS_ARTWORK=0",
                exc_info=True,
            )

        rows = []
        for artist_id, artist, streams, release_count in rollup:
            aid = str(artist_id).strip() if artist_id is not None else ""
            if not aid:
                continue
            name = artist if artist is not None else ""
            try:
                stream_val = int(streams or 0)
            except (TypeError, ValueError):
                stream_val = 0
            try:
                n_releases = int(release_count or 0)
            except (TypeError, ValueError):
                n_releases = 0
            rows.append(
                (
                    aid,
                    name,
                    normalize_search_text(name),
                    stream_val,
                    n_releases,
                    1 if aid.upper() in artwork_ids else 0,
                )
            )

        cur.execute(load_sql(DELETE_MARKETSHARE_SEARCH_ARTISTS))
        if rows:
            cur.executemany(load_sql(INSERT_MARKETSHARE_SEARCH_ARTISTS), rows)
        conn.commit()

    logger.info("sqlite_handler: MARKETSHARE_SEARCH_ARTISTS rebuilt (rows=%d)", len(rows))
    if rows:
        _persist_marketshare_db_to_s3()
    return len(rows)


def fill_missing_search_snapshot_artist_ids(*, batch_size: int = 8000) -> dict:
    """
    Resolve LUMINATE_ARTIST_ID for existing album + singles search-snapshot
    rows from Snowflake (first Main Artist), then UPDATE in place.

    Use this when the snapshot tables are populated but artist IDs were
    wiped (e.g. an S3 db pull of an older file) and a full snapshot rebuild
    is not practical. Does not insert or delete releases.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    with sqlite_connect() as conn:
        ensure_marketshare_search_summary_columns(conn)
        ensure_marketshare_search_summary_singles_columns(conn)
        conn.commit()
        cur = conn.cursor()
        missing: list[str] = []
        for table in (
            "MARKETSHARE_SEARCH_SUMMARY",
            "MARKETSHARE_SEARCH_SUMMARY_SINGLES",
        ):
            cur.execute(
                f"""
                SELECT MRELG_ID FROM {table}
                WHERE MRELG_ID IS NOT NULL
                  AND TRIM(MRELG_ID) != ''
                  AND (
                    LUMINATE_ARTIST_ID IS NULL
                    OR TRIM(LUMINATE_ARTIST_ID) = ''
                  )
                """
            )
            missing.extend(
                str(row[0]).strip() for row in cur.fetchall() if row[0]
            )
    missing = list(dict.fromkeys(missing))
    stats = {
        "missing_before": len(missing),
        "resolved": 0,
        "album_updated": 0,
        "singles_updated": 0,
        "batches": 0,
        "errors": [],
    }
    if not missing:
        logger.info("sqlite_handler: search snapshot artist ids already filled")
        return stats

    mapping: dict[str, str] = {}
    missing_set = set(missing)
    logger.info(
        "sqlite_handler: filling search snapshot artist ids (missing=%d)",
        len(missing),
    )
    # Read-only: first Main Artist per MRELG for Album/EP/Single. Filter to
    # local snapshot IDs in Python so we do not CREATE TEMP TABLE.
    join_sql = """
        SELECT
            m.mrelg_id,
            f.value:ARTIST_ID::STRING AS luminate_artist_id
        FROM LUMINATE_PROD.EXTRACT_S.VW_MUSICAL_RELEASE_GROUP_DS m,
             LATERAL FLATTEN(input => m.ARTISTS) f
        WHERE m.release_type IN ('Album', 'EP', 'Single')
          AND LOWER(COALESCE(f.value:ROLE::STRING, '')) = 'main artist'
          AND f.value:ARTIST_ID IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY m.mrelg_id
            ORDER BY f.index
        ) = 1
    """
    try:
        with get_snowflake_connection() as sf:
            logger.info("sqlite_handler: scanning Snowflake Main Artist ids")
            n_rows = 0
            for df in sf.iter_query(join_sql, chunksize=25000):
                if df is None or df.empty:
                    continue
                n_rows += len(df)
                stats["batches"] += 1
                norm = {str(c).strip().lower(): c for c in df.columns}
                mid_col = norm.get("mrelg_id")
                aid_col = norm.get("luminate_artist_id")
                if not mid_col or not aid_col:
                    stats["errors"].append(
                        {"error": f"unexpected columns {list(df.columns)}"}
                    )
                    break
                for mid, aid in zip(df[mid_col], df[aid_col]):
                    mid_s = str(mid or "").strip()
                    if mid_s not in missing_set:
                        continue
                    aid_s = str(aid or "").strip()
                    if aid_s and aid_s.lower() not in ("none", "nan"):
                        mapping[mid_s] = aid_s
                if stats["batches"] == 1 or stats["batches"] % 40 == 0:
                    logger.info(
                        "sqlite_handler: artist-id scan rows=%d matched=%d/%d",
                        n_rows,
                        len(mapping),
                        len(missing),
                    )
                if len(mapping) >= len(missing_set):
                    logger.info(
                        "sqlite_handler: artist-id scan complete early "
                        "(matched=%d rows_seen=%d)",
                        len(mapping),
                        n_rows,
                    )
                    break
    except Exception as e:
        logger.exception("sqlite_handler: search snapshot artist-id Snowflake failed")
        stats["errors"].append({"stage": "snowflake", "error": str(e)})
        return stats

    stats["resolved"] = len(mapping)
    if not mapping:
        logger.info(
            "sqlite_handler: %d missing snapshot ids, 0 resolved from Snowflake",
            len(missing),
        )
        return stats

    updates = [(aid, mid) for mid, aid in mapping.items()]
    try:
        with sqlite_connect() as conn:
            ensure_marketshare_search_summary_columns(conn)
            ensure_marketshare_search_summary_singles_columns(conn)
            cur = conn.cursor()
            cur.execute("PRAGMA synchronous = NORMAL")
            cur.executemany(
                """
                UPDATE MARKETSHARE_SEARCH_SUMMARY
                SET LUMINATE_ARTIST_ID = ?
                WHERE MRELG_ID = ?
                  AND (LUMINATE_ARTIST_ID IS NULL OR TRIM(LUMINATE_ARTIST_ID) = '')
                """,
                updates,
            )
            stats["album_updated"] = cur.rowcount if cur.rowcount >= 0 else 0
            cur.executemany(
                """
                UPDATE MARKETSHARE_SEARCH_SUMMARY_SINGLES
                SET LUMINATE_ARTIST_ID = ?
                WHERE MRELG_ID = ?
                  AND (LUMINATE_ARTIST_ID IS NULL OR TRIM(LUMINATE_ARTIST_ID) = '')
                """,
                updates,
            )
            stats["singles_updated"] = cur.rowcount if cur.rowcount >= 0 else 0
            conn.commit()
    except sqlite3.Error as e:
        stats["errors"].append({"stage": "sqlite", "error": str(e)})
        return stats

    logger.info(
        "sqlite_handler: search snapshot artist ids filled "
        "(missing_before=%d resolved=%d album_updated=%d singles_updated=%d)",
        stats["missing_before"],
        stats["resolved"],
        stats["album_updated"],
        stats["singles_updated"],
    )
    if stats["album_updated"] or stats["singles_updated"]:
        _persist_marketshare_db_to_s3()
    return stats


def refresh_marketshare_search_snapshots() -> dict:
    """
    Rebuild album + singles search snapshots from Snowflake, then the artist
    typeahead index once both tables are in place.
    """
    albums = refresh_marketshare_search_summary(rebuild_artists=False)
    singles = refresh_marketshare_search_summary_singles(rebuild_artists=False)
    _refresh_search_artists_best_effort()
    _refresh_search_distributors_best_effort()
    return {"albums": albums, "singles": singles}


def _search_summary_has_artist_id(cur: sqlite3.Cursor, table: str) -> bool:
    """True when ``table`` exists and has a LUMINATE_ARTIST_ID column."""
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    )
    if cur.fetchone() is None:
        return False
    cur.execute(f"PRAGMA table_info({table})")
    return "LUMINATE_ARTIST_ID" in {row[1] for row in cur.fetchall()}


_SEARCH_SNAPSHOT_CHUNKSIZE = 25000
_SEARCH_SNAPSHOT_SECONDARY_INDEXES = {
    "MARKETSHARE_SEARCH_SUMMARY": (
        "IDX_MARKETSHARE_SEARCH_SUMMARY_STREAMS",
        "IDX_MARKETSHARE_SEARCH_SUMMARY_ARTIST_ID",
    ),
    "MARKETSHARE_SEARCH_SUMMARY_SINGLES": (
        "IDX_MARKETSHARE_SEARCH_SUMMARY_SINGLES_STREAMS",
        "IDX_MARKETSHARE_SEARCH_SUMMARY_SINGLES_ARTIST_ID",
    ),
}
_SEARCH_SNAPSHOT_TEXT_COLS = (
    "ARTIST_SEARCH",
    "TITLE_SEARCH",
    "LUMINATE_ARTIST_ID",
    "RELEASE_TYPE",
)


def _ensure_search_snapshot_text_columns(conn: sqlite3.Connection, table: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    )
    if cur.fetchone() is None:
        return
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}
    for col in _SEARCH_SNAPSHOT_TEXT_COLS:
        if col in existing:
            continue
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")


def _prepare_search_snapshot_rows(
    df: pd.DataFrame,
    *,
    column_map: dict,
    target_cols: list,
    default_release_type: str,
) -> list:
    if df is None or df.empty:
        return []
    df = df.rename(columns=str.upper)
    for src in column_map:
        if src not in df.columns:
            df[src] = None
    df = df.rename(columns=column_map)

    if "RELEASE_DATE" in df.columns:
        df["RELEASE_DATE"] = df["RELEASE_DATE"].astype(str)

    if "DAILY_GLOBAL_STREAMS" in df.columns:
        streams = pd.to_numeric(df["DAILY_GLOBAL_STREAMS"], errors="coerce")
        streams = streams.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
        df["DAILY_GLOBAL_STREAMS"] = streams.astype("int64")

    for col in (
        "MRELG_ID",
        "TITLE",
        "ARTIST",
        "LUMINATE_ARTIST_ID",
        "RELEASE_TYPE",
        "PARENT_NAME",
        "LABEL_NAME",
        "GENRE",
    ):
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), None)

    if "LUMINATE_ARTIST_ID" in df.columns:
        df["LUMINATE_ARTIST_ID"] = (
            df["LUMINATE_ARTIST_ID"].astype(str).str.strip().replace(
                {"": None, "nan": None, "None": None}
            )
        )

    if "RELEASE_TYPE" in df.columns:
        df["RELEASE_TYPE"] = (
            df["RELEASE_TYPE"].astype(str).str.strip().replace(
                {"": None, "nan": None, "None": None}
            )
        )
        df["RELEASE_TYPE"] = df["RELEASE_TYPE"].fillna(default_release_type)

    df = df.loc[df["MRELG_ID"].astype(str).str.strip() != ""].copy()
    df["ARTIST_SEARCH"] = df["ARTIST"].map(normalize_search_text)
    df["TITLE_SEARCH"] = df["TITLE"].map(normalize_search_text)
    return list(df[target_cols].itertuples(index=False, name=None))


def _drop_search_snapshot_secondary_indexes(
    conn: sqlite3.Connection, table_name: str
) -> None:
    for index_name in _SEARCH_SNAPSHOT_SECONDARY_INDEXES.get(table_name, ()):
        conn.execute(f"DROP INDEX IF EXISTS {index_name}")


def _search_snapshot_use_stage_swap() -> bool:
    return os.environ.get("TIDE_SEARCH_SNAPSHOT_STAGE_SWAP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _refresh_search_snapshot_table(
    *,
    query_file: str,
    create_file: str,
    insert_file: str,
    table_name: str,
    column_map: dict,
    target_cols: list,
    default_release_type: str,
    ensure_live_fn,
) -> int:
    """
    Stream a Snowflake search snapshot into SQLite in chunks (avoids OOM on
    the widened Album/EP extract).

    Default path upserts into the live table so peak disk is live growing in
    place. A leftover `{table}_STAGE` is dropped first (DROP does not shrink
    the file until VACUUM). A killed load leaves a mix of old and new rows.

    Set TIDE_SEARCH_SNAPSHOT_STAGE_SWAP=1 to rebuild via a staging table and
    atomic rename (kill-safe, needs ~2x table free space). Use that once to
    convert an existing rowid table to WITHOUT ROWID.
    """
    stage_name = f"{table_name}_STAGE"
    use_stage = _search_snapshot_use_stage_swap()
    create_live = load_sql(create_file)
    insert_sql = load_sql(insert_file)
    if use_stage:
        insert_sql = insert_sql.replace(
            f"INSERT INTO {table_name}",
            f"INSERT INTO {stage_name}",
            1,
        )

    total_rows = 0
    with get_snowflake_connection() as sf, sqlite_connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(create_live)
        conn.execute(f"DROP TABLE IF EXISTS {stage_name}")
        if use_stage:
            ensure_live_fn(conn)
            create_stage = create_live.replace(
                f"CREATE TABLE IF NOT EXISTS {table_name}",
                f"CREATE TABLE IF NOT EXISTS {stage_name}",
                1,
            )
            conn.execute(create_stage)
            _ensure_search_snapshot_text_columns(conn, stage_name)
        else:
            _ensure_search_snapshot_text_columns(conn, table_name)
            _drop_search_snapshot_secondary_indexes(conn, table_name)
        conn.commit()

        for i, chunk in enumerate(
            sf.iter_query(load_sql(query_file), chunksize=_SEARCH_SNAPSHOT_CHUNKSIZE),
            start=1,
        ):
            rows = _prepare_search_snapshot_rows(
                chunk,
                column_map=column_map,
                target_cols=target_cols,
                default_release_type=default_release_type,
            )
            if rows:
                conn.executemany(insert_sql, rows)
                total_rows += len(rows)
            if i == 1 or i % 20 == 0:
                logger.info(
                    "sqlite_handler: %s %s chunk=%d rows_so_far=%d",
                    table_name,
                    "stage" if use_stage else "upsert",
                    i,
                    total_rows,
                )
            conn.commit()

        if use_stage:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"ALTER TABLE {stage_name} RENAME TO {table_name}")
        ensure_live_fn(conn)
        conn.execute(f"DROP TABLE IF EXISTS {stage_name}")
        conn.commit()

    logger.info(
        "sqlite_handler: %s rebuilt (rows=%d, stage_swap=%s)",
        table_name,
        total_rows,
        use_stage,
    )
    return total_rows


def refresh_marketshare_search_summary(*, rebuild_artists: bool = True) -> int:
    """
    Rebuild MARKETSHARE_SEARCH_SUMMARY from Snowflake in streamed chunks.
    Upserts into the live table by default (see
    _refresh_search_snapshot_table). Returns the number of rows written.
    """
    logger.info("sqlite_handler: refreshing MARKETSHARE_SEARCH_SUMMARY (db=%s)", DATABASE_NAME)
    column_map = {
        "MRELG_ID": "MRELG_ID",
        "TITLE": "TITLE",
        "ARTIST": "ARTIST",
        "LUMINATE_ARTIST_ID": "LUMINATE_ARTIST_ID",
        "RELEASE_TYPE": "RELEASE_TYPE",
        "LABEL": "LABEL_NAME",
        "RELEASE_DATE": "RELEASE_DATE",
        "GENRE": "GENRE",
        "DAILY_STREAMS": "DAILY_GLOBAL_STREAMS",
    }
    target_cols = [
        "MRELG_ID",
        "TITLE",
        "ARTIST",
        "LUMINATE_ARTIST_ID",
        "RELEASE_TYPE",
        "LABEL_NAME",
        "RELEASE_DATE",
        "GENRE",
        "DAILY_GLOBAL_STREAMS",
        "ARTIST_SEARCH",
        "TITLE_SEARCH",
    ]
    n = _refresh_search_snapshot_table(
        query_file=MARKETSHARE_SEARCH_SUMMARY_QUERY,
        create_file=CREATE_MARKETSHARE_SEARCH_SUMMARY_TABLE,
        insert_file=INSERT_MARKETSHARE_SEARCH_SUMMARY,
        table_name="MARKETSHARE_SEARCH_SUMMARY",
        column_map=column_map,
        target_cols=target_cols,
        default_release_type="Album",
        ensure_live_fn=ensure_marketshare_search_summary_columns,
    )
    if rebuild_artists:
        _refresh_search_artists_best_effort()
        _refresh_search_distributors_best_effort()
    return n


def refresh_marketshare_search_summary_singles(*, rebuild_artists: bool = True) -> int:
    """
    Rebuild MARKETSHARE_SEARCH_SUMMARY_SINGLES from Snowflake in streamed
    chunks. Same live upsert as the album snapshot.
    """
    logger.info(
        "sqlite_handler: refreshing MARKETSHARE_SEARCH_SUMMARY_SINGLES (db=%s)",
        DATABASE_NAME,
    )
    column_map = {
        "MRELG_ID": "MRELG_ID",
        "TITLE": "TITLE",
        "ARTIST": "ARTIST",
        "LUMINATE_ARTIST_ID": "LUMINATE_ARTIST_ID",
        "RELEASE_TYPE": "RELEASE_TYPE",
        "LABEL": "LABEL_NAME",
        "RELEASE_DATE": "RELEASE_DATE",
        "GENRE": "GENRE",
        "DAILY_STREAMS": "DAILY_GLOBAL_STREAMS",
    }
    target_cols = [
        "MRELG_ID",
        "TITLE",
        "ARTIST",
        "LUMINATE_ARTIST_ID",
        "RELEASE_TYPE",
        "LABEL_NAME",
        "RELEASE_DATE",
        "GENRE",
        "DAILY_GLOBAL_STREAMS",
        "ARTIST_SEARCH",
        "TITLE_SEARCH",
    ]
    n = _refresh_search_snapshot_table(
        query_file=MARKETSHARE_SEARCH_SUMMARY_SINGLES_QUERY,
        create_file=CREATE_MARKETSHARE_SEARCH_SUMMARY_SINGLES_TABLE,
        insert_file=INSERT_MARKETSHARE_SEARCH_SUMMARY_SINGLES,
        table_name="MARKETSHARE_SEARCH_SUMMARY_SINGLES",
        column_map=column_map,
        target_cols=target_cols,
        default_release_type="Single",
        ensure_live_fn=ensure_marketshare_search_summary_singles_columns,
    )
    if rebuild_artists:
        _refresh_search_artists_best_effort()
        _refresh_search_distributors_best_effort()
    return n


def _refresh_search_artists_best_effort() -> None:
    """Rebuild the artist typeahead index; snapshot refresh still succeeds if this fails."""
    try:
        refresh_marketshare_search_artists()
    except Exception:
        logger.exception("sqlite_handler: MARKETSHARE_SEARCH_ARTISTS rebuild failed")


def _refresh_search_distributors_best_effort() -> None:
    """Rebuild the distributor typeahead index; snapshot refresh still succeeds if this fails."""
    try:
        refresh_marketshare_search_distributors()
    except Exception:
        logger.exception(
            "sqlite_handler: MARKETSHARE_SEARCH_DISTRIBUTORS rebuild failed"
        )


def _persist_marketshare_db_to_s3() -> None:
    """Push marketshare_data.db so artist search survives API restart / S3 pull."""
    try:
        from api.s3_pull import sync_db_to_s3

        sync_db_to_s3()
    except Exception:
        logger.exception(
            "sqlite_handler: failed to persist marketshare_data.db to S3"
        )


def ensure_marketshare_search_artists_index() -> int:
    """
    If MARKETSHARE_SEARCH_ARTISTS is empty but the snapshots already have
    LUMINATE_ARTIST_ID, rebuild the typeahead locally (no Snowflake).
    """
    with sqlite_connect() as conn:
        ensure_marketshare_search_summary_columns(conn)
        ensure_marketshare_search_summary_singles_columns(conn)
        ensure_marketshare_search_artists_table(conn)
        conn.commit()
        cur = conn.cursor()
        artist_n = cur.execute(
            "SELECT COUNT(*) FROM MARKETSHARE_SEARCH_ARTISTS"
        ).fetchone()[0]
        if artist_n:
            return int(artist_n)
        has_ids = False
        for table in (
            "MARKETSHARE_SEARCH_SUMMARY",
            "MARKETSHARE_SEARCH_SUMMARY_SINGLES",
        ):
            cur.execute(
                f"""
                SELECT 1 FROM {table}
                WHERE LUMINATE_ARTIST_ID IS NOT NULL
                  AND TRIM(LUMINATE_ARTIST_ID) != ''
                LIMIT 1
                """
            )
            if cur.fetchone() is not None:
                has_ids = True
                break
    if not has_ids:
        return 0
    logger.info(
        "sqlite_handler: MARKETSHARE_SEARCH_ARTISTS empty with snapshot ids; rebuilding"
    )
    return refresh_marketshare_search_artists()


def ensure_marketshare_search_distributors_index() -> int:
    """
    If MARKETSHARE_SEARCH_DISTRIBUTORS is empty but snapshots have LABEL_NAME,
    rebuild the typeahead locally (no Snowflake).
    """
    with sqlite_connect() as conn:
        ensure_marketshare_search_summary_columns(conn)
        ensure_marketshare_search_summary_singles_columns(conn)
        ensure_marketshare_search_distributors_table(conn)
        conn.commit()
        cur = conn.cursor()
        n = cur.execute(
            "SELECT COUNT(*) FROM MARKETSHARE_SEARCH_DISTRIBUTORS"
        ).fetchone()[0]
        if n:
            return int(n)
        has_labels = False
        for table in (
            "MARKETSHARE_SEARCH_SUMMARY",
            "MARKETSHARE_SEARCH_SUMMARY_SINGLES",
        ):
            cur.execute(
                f"""
                SELECT 1 FROM {table}
                WHERE LABEL_NAME IS NOT NULL AND TRIM(LABEL_NAME) != ''
                LIMIT 1
                """
            )
            if cur.fetchone() is not None:
                has_labels = True
                break
    if not has_labels:
        return 0
    logger.info(
        "sqlite_handler: MARKETSHARE_SEARCH_DISTRIBUTORS empty; rebuilding"
    )
    return refresh_marketshare_search_distributors()


def recompute_ytd_share_from_current_data(data_path: Path) -> pd.DataFrame:
    """
    Match current_data_test.py logic exactly:
      1) Read current_data_all.csv (7-label panel)
      2) Use ALBUM_EQUIVALENT / ALBUM_EQUIVALENT_SHARE to estimate weekly total market
      3) Compute YTD by label as cumulative(label weekly volume) / cumulative(total market)
    Returns percent-point shares (e.g. 8.85).
    """
    from model.marketshare_labels import TARGET_LABELS

    current = pd.read_csv(data_path)
    current.columns = [str(c).strip().lstrip("\ufeff") for c in current.columns]
    year_col = next((c for c in ("YEAR", "Year", "year") if c in current.columns), None)
    if year_col is None:
        return pd.DataFrame(columns=["YEAR", "WEEK_ENDING_DATE", "LABEL_NAME", "ALBUM_EQUIVALENT_SHARE"])

    current["YEAR"] = pd.to_numeric(current[year_col], errors="coerce")
    current["ALBUM_EQUIVALENT"] = pd.to_numeric(current["ALBUM_EQUIVALENT"], errors="coerce")
    current["ALBUM_EQUIVALENT_SHARE"] = pd.to_numeric(current["ALBUM_EQUIVALENT_SHARE"], errors="coerce")

    share_median = current["ALBUM_EQUIVALENT_SHARE"].dropna().median()
    if pd.notna(share_median) and share_median > 1:
        current["ALBUM_EQUIVALENT_SHARE"] = current["ALBUM_EQUIVALENT_SHARE"] / 100.0

    current = current[
        current["LABEL_NAME"].isin(TARGET_LABELS)
        & current["YEAR"].notna()
        & current["ALBUM_EQUIVALENT"].notna()
        & current["ALBUM_EQUIVALENT_SHARE"].notna()
        & (current["ALBUM_EQUIVALENT_SHARE"] > 0)
    ].copy()
    if current.empty:
        return pd.DataFrame(columns=["YEAR", "WEEK_ENDING_DATE", "LABEL_NAME", "ALBUM_EQUIVALENT_SHARE"])

    current["WEEK_ENDING_DATE"] = pd.to_datetime(current["WEEK_ENDING_DATE"], errors="coerce")
    current = current[current["WEEK_ENDING_DATE"].notna()].copy()
    current["__total_market_est"] = current["ALBUM_EQUIVALENT"] / current["ALBUM_EQUIVALENT_SHARE"]

    weekly_total = (
        current.groupby(["YEAR", "WEEK_ENDING_DATE"], as_index=False)["__total_market_est"]
        .median()
        .rename(columns={"__total_market_est": "__weekly_total_market"})
    )
    label_weekly = (
        current.groupby(["YEAR", "WEEK_ENDING_DATE", "LABEL_NAME"], as_index=False)["ALBUM_EQUIVALENT"]
        .sum()
        .rename(columns={"ALBUM_EQUIVALENT": "__label_weekly_volume"})
    )
    out = label_weekly.merge(weekly_total, on=["YEAR", "WEEK_ENDING_DATE"], how="inner")
    out = out.sort_values(["LABEL_NAME", "YEAR", "WEEK_ENDING_DATE"]).reset_index(drop=True)
    out["__cum_num"] = out.groupby(["LABEL_NAME", "YEAR"])["__label_weekly_volume"].cumsum()
    out["__cum_den"] = out.groupby(["LABEL_NAME", "YEAR"])["__weekly_total_market"].cumsum()
    out["ALBUM_EQUIVALENT_SHARE"] = (out["__cum_num"] / out["__cum_den"]) * 100.0
    # Benchmark parity: truncate to 2 decimals (do not round).
    out["ALBUM_EQUIVALENT_SHARE"] = np.trunc(out["ALBUM_EQUIVALENT_SHARE"] * 100) / 100
    out["YEAR"] = out["YEAR"].astype(int)
    out["WEEK_ENDING_DATE"] = out["WEEK_ENDING_DATE"].dt.strftime("%Y-%m-%d")
    return out[["YEAR", "WEEK_ENDING_DATE", "LABEL_NAME", "ALBUM_EQUIVALENT_SHARE"]]


def update_sqlite_main() -> None:
    """
    Loads the weekly and ytd marketshare data from Snowflake and saves it to SQLite database.
    The SQLite database is located in the data folder.
    Other SQLite database tables are currently not being updated by this script.
    """
    logger.info("sqlite_handler: starting refresh (db=%s)", DATABASE_NAME)
    # Connect to SQLite database
    sqlite_conn = sqlite3.connect(DATABASE_NAME)
    cursor = sqlite_conn.cursor()

    # Create tables
    logger.info("sqlite_handler: ensuring SQLite tables exist")
    cursor.execute(load_sql(CREATE_EXPECTED_RELEASES_TABLE))
    ensure_expected_releases_fw_columns(sqlite_conn)
    cursor.execute(load_sql(CREATE_WEEKLY_MARKETSHARE_TABLE))
    cursor.execute(load_sql(CREATE_YTD_MARKETSHARE_TABLE))
    cursor.execute(load_sql(CREATE_MARKETSHARE_RELEASE_METRICS_TABLE))

    full_refresh = os.environ.get("TIDE_SQLITE_FULL_REFRESH", "").strip().lower() in (
        "1", "true", "yes",
    )
    weekly_ytd_lookback_days = int(os.environ.get("TIDE_MARKETSHARE_LOOKBACK_DAYS", "120"))
    release_metrics_lookback_days = int(os.environ.get("TIDE_RELEASE_METRICS_LOOKBACK_DAYS", "560"))

    if full_refresh:
        min_week_marketshare = "2018-01-01"
        min_week_metrics = "2018-01-01"
        logger.info("sqlite_handler: full Snowflake refresh enabled (TIDE_SQLITE_FULL_REFRESH=1)")
    else:
        min_week_marketshare = _min_week_anchor(
            cursor, "MARKETSHARE_WEEKLY", weekly_ytd_lookback_days
        )
        min_week_metrics = _min_week_anchor(
            cursor, "MARKETSHARE_RELEASE_METRICS", release_metrics_lookback_days
        )
        logger.info(
            "sqlite_handler: incremental Snowflake pull (marketshare_min_week=%s, metrics_min_week=%s)",
            min_week_marketshare,
            min_week_metrics,
        )

    cursor.execute(load_sql(EXPECTED_RELEASES_QUERY))
    expected_releases_columns = [d[0] for d in cursor.description]
    expected_releases_rows = cursor.fetchall()
    expected_releases_df = pd.DataFrame(expected_releases_rows, columns=expected_releases_columns)
    logger.info("sqlite_handler: loaded %d expected releases from SQLite", len(expected_releases_df))

    def _snowflake_string_literal(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    from model.marketshare_labels import TARGET_LABELS

    target_labels_sql = ", ".join(
        "'" + lab.replace("'", "''") + "'" for lab in TARGET_LABELS
    )

    release_ids = (
        expected_releases_df['MRELG_ID']
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.strip() != '']
        .unique()
        .tolist()
    )
    # Get data from Snowflake
    logger.info(
        "sqlite_handler: querying Snowflake weekly/ytd marketshare tables for %d labels",
        len(TARGET_LABELS),
    )
    with get_snowflake_connection() as sf:
        weekly_marketshare_data = sf.query(
            load_sql(WEEKLY_MARKETSHARE_QUERY)
            .replace("{MIN_WEEK_END_DATE}", min_week_marketshare)
            .replace("{TARGET_LABELS}", target_labels_sql)
        )
        ytd_marketshare_data = sf.query(
            load_sql(YTD_MARKETSHARE_QUERY)
            .replace("{MIN_WEEK_END_DATE}", min_week_marketshare)
            .replace("{TARGET_LABELS}", target_labels_sql)
        )
        logger.info(
            "sqlite_handler: Snowflake rows weekly=%d ytd=%d",
            len(weekly_marketshare_data),
            len(ytd_marketshare_data),
        )

        if release_ids:
            # Query in batches to avoid one very heavy, long-running Snowflake statement.
            batch_size = 5
            batches = _chunked(release_ids, batch_size)
            metric_frames: list[pd.DataFrame] = []
            logger.info(
                "sqlite_handler: querying Snowflake release metrics for %d releases in %d batches (size=%d)",
                len(release_ids),
                len(batches),
                batch_size,
            )
            for idx, batch in enumerate(batches, start=1):
                release_ids_str = ','.join(_snowflake_string_literal(rid) for rid in batch)
                metrics_sql = load_sql(MARKETSHARE_RELEASE_METRICS_QUERY).replace(
                    '{RELEASE_IDS}', release_ids_str
                ).replace("{MIN_WEEK_END_DATE}", min_week_metrics)
                logger.info(
                    "sqlite_handler: release metrics batch %d/%d (%d releases)",
                    idx,
                    len(batches),
                    len(batch),
                )
                df_batch = sf.query(metrics_sql)
                if not df_batch.empty:
                    metric_frames.append(df_batch)
            if metric_frames:
                marketshare_release_metrics_data = pd.concat(metric_frames, ignore_index=True)
            else:
                marketshare_release_metrics_data = pd.DataFrame(
                    columns=[
                        'WEEK_ENDING_DATE', 'MRELG_ID', 'ALBUM_EQUIVALENT', 'PRODUCT_SALES',
                        'SONG_SALE_EQUIVALENT', 'STREAMING_EQUIVALENT',
                    ]
                )
            marketshare_release_metrics_data = marketshare_release_metrics_data.rename(
                columns=str.upper,
            )
            logger.info(
                "sqlite_handler: Snowflake rows release_metrics=%d",
                len(marketshare_release_metrics_data),
            )
            # After rename(columns=str.upper)
            if "WEEK_ENDING_DATE" in marketshare_release_metrics_data.columns:
                marketshare_release_metrics_data["WEEK_ENDING_DATE"] = marketshare_release_metrics_data["WEEK_ENDING_DATE"].astype(str)

            for col in ("ALBUM_EQUIVALENT", "PRODUCT_SALES", "SONG_SALE_EQUIVALENT", "STREAMING_EQUIVALENT"):
                if col in marketshare_release_metrics_data.columns:
                    s = pd.to_numeric(marketshare_release_metrics_data[col], errors="coerce")
                    s = s.replace([float("inf"), float("-inf")], pd.NA)
                    marketshare_release_metrics_data[col] = s.where(~s.isna(), None).astype(object)
        else:
            marketshare_release_metrics_data = pd.DataFrame(
                columns=[
                    'WEEK_ENDING_DATE', 'MRELG_ID', 'ALBUM_EQUIVALENT', 'PRODUCT_SALES',
                    'SONG_SALE_EQUIVALENT', 'STREAMING_EQUIVALENT',
                ]
            )

    # Normalize date columns so sqlite3 does not rely on deprecated
    # implicit date adapters in Python 3.12+.
    if "WEEK_ENDING_DATE" in weekly_marketshare_data.columns:
        weekly_marketshare_data["WEEK_ENDING_DATE"] = weekly_marketshare_data["WEEK_ENDING_DATE"].astype(str)
    if "WEEK_ENDING_DATE" in ytd_marketshare_data.columns:
        ytd_marketshare_data["WEEK_ENDING_DATE"] = ytd_marketshare_data["WEEK_ENDING_DATE"].astype(str)
    try:
        current_path = (
            Path(__file__).resolve().parent / "model" / "data" / "current_data_all.csv"
        )
        ytd_recalc = recompute_ytd_share_from_current_data(current_path)
        if not ytd_recalc.empty:
            ytd_marketshare_data = ytd_marketshare_data.merge(
                ytd_recalc,
                on=["YEAR", "WEEK_ENDING_DATE", "LABEL_NAME"],
                how="left",
                suffixes=("", "_RECALC"),
            )
            ytd_marketshare_data["ALBUM_EQUIVALENT_SHARE"] = ytd_marketshare_data[
                "ALBUM_EQUIVALENT_SHARE_RECALC"
            ].where(
                ytd_marketshare_data["ALBUM_EQUIVALENT_SHARE_RECALC"].notna(),
                ytd_marketshare_data["ALBUM_EQUIVALENT_SHARE"],
            )
            ytd_marketshare_data = ytd_marketshare_data.drop(columns=["ALBUM_EQUIVALENT_SHARE_RECALC"])
    except Exception:
        # If current_data_all.csv is unavailable, keep Snowflake-provided YTD share.
        pass

    # Update tables
    logger.info("sqlite_handler: writing weekly/ytd/metrics rows into SQLite")
    cursor.executemany(load_sql(INSERT_WEEKLY_MARKETSHARE), weekly_marketshare_data[[
        'WEEK_ENDING_DATE', 'COUNTRY_CODE', 'RELEASE_AGE', 'LABEL_NAME',
        'STREAMING_TOTAL', 'ALBUM_EQUIVALENT', 'PRODUCT_SALES', 'SONG_SALE_EQUIVALENT',
        'STREAMING_EQUIVALENT', 'ALBUM_EQUIVALENT_SHARE', 'PRODUCT_SALES_SHARE',
        'SONG_SALE_EQUIVALENT_SHARE', 'STREAMING_EQUIVALENT_SHARE'
    ]].itertuples(index=False, name=None))

    cursor.executemany(load_sql(INSERT_YTD_MARKETSHARE), ytd_marketshare_data[[
        'WEEK_ENDING_DATE', 'YEAR', 'WEEK_NUM', 'COUNTRY_CODE', 'RELEASE_AGE', 'LABEL_NAME',
        'STREAMING_TOTAL', 'ALBUM_EQUIVALENT', 'PRODUCT_SALES', 'SONG_SALE_EQUIVALENT',
        'STREAMING_EQUIVALENT', 'ALBUM_EQUIVALENT_SHARE', 'PRODUCT_SALES_SHARE',
        'SONG_SALE_EQUIVALENT_SHARE', 'STREAMING_EQUIVALENT_SHARE'
    ]].itertuples(index=False, name=None))

    marketshare_release_metrics_data = pd.merge(
        marketshare_release_metrics_data,
        expected_releases_df[['MRELG_ID', 'RELEASE_ID']],
        on='MRELG_ID',
        how='left',
    )

    metric_cols = [
        'RELEASE_ID', 'MRELG_ID', 'WEEK_ENDING_DATE', 'ALBUM_EQUIVALENT',
        'PRODUCT_SALES', 'SONG_SALE_EQUIVALENT', 'STREAMING_EQUIVALENT',
    ]
    
    marketshare_release_metrics_data = marketshare_release_metrics_data[metric_cols]
    
    cursor.executemany(load_sql(INSERT_MARKETSHARE_RELEASE_METRICS),
        marketshare_release_metrics_data.itertuples(index=False, name=None))

    # Commit and close connection
    sqlite_conn.commit()
    sqlite_conn.close()

    # Street-week fw_vol on EXPECTED_RELEASES (chart week of RELEASE_DATE,
    # skipping pre-street stubs). Stored so UI/boot do not recompute the 10%
    # rule on every load.
    try:
        persist_expected_release_fw_vols()
    except Exception:
        logger.exception("sqlite_handler: persist_expected_release_fw_vols failed")

    # Rebuild the search-summary table from Snowflake. Daily-stream snapshot
    # data is fully replaced rather than merged so search results never carry
    # stale popularity numbers between refreshes.
    try:
        refresh_marketshare_search_summary()
    except Exception as e:
        logger.exception("sqlite_handler: search summary refresh failed: %s", e)

    try:
        refresh_marketshare_search_summary_singles()
    except Exception as e:
        logger.exception("sqlite_handler: singles search summary refresh failed: %s", e)

    logger.info("sqlite_handler: refresh complete")


# ---------------------------------------------------------------------------
# Worldwide streams caches (Live Revenue / streaming roster)
#
# Per-MRELG SQLite caches filled from Snowflake. Incremental pulls only
# request dates/weeks after the latest cached point (with overlap). Batch
# prewarm walks STREAMING_ROSTER_2026 via model_handler.prewarm_streaming_roster_caches.
# ---------------------------------------------------------------------------

DAILY_STREAMS_STALE_DAYS = 2
WEEKLY_STREAMS_STALE_DAYS = 6
DAILY_STREAMS_INCREMENTAL_OVERLAP_DAYS = 3
WEEKLY_STREAMS_INCREMENTAL_OVERLAP_WEEKS = 2
STREAMING_ROSTER_TABLE = "STREAMING_ROSTER_2026"


def _ensure_daily_global_streams_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute(load_sql(CREATE_DAILY_GLOBAL_STREAMS_TABLE))


def _ensure_weekly_global_streams_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute(load_sql(CREATE_WEEKLY_GLOBAL_STREAMS_TABLE))


def _snowflake_date_literal(iso_date: str) -> str:
    return _snowflake_str(str(iso_date).split(" ")[0][:10])


def _incremental_daily_filter(max_report_date: str | None) -> str:
    if not max_report_date:
        return ""
    max_dt = pd.to_datetime(max_report_date, errors="coerce")
    if pd.isna(max_dt):
        return ""
    overlap = int(os.environ.get("TIDE_DAILY_STREAMS_OVERLAP_DAYS", str(DAILY_STREAMS_INCREMENTAL_OVERLAP_DAYS)))
    floor = (max_dt - pd.Timedelta(days=overlap)).strftime("%Y-%m-%d")
    return f"\n    AND da.datename > {_snowflake_date_literal(floor)}"


def _incremental_weekly_filter(max_week_ending_date: str | None) -> str:
    """
    Min week-ending floor for incremental Snowflake weekly-stream pulls.

    When the SQLite cache already has a max WEEK_ENDING_DATE for the MRELG,
    only request weeks on/after (max - overlap). This is what keeps boot
    prewarm / daily+weekly crons from re-pulling the full YTD window every run.
    Cold cache (max is None) returns "" so the SQL release_date / YTD floors apply.
    """
    if not max_week_ending_date:
        return ""
    max_dt = pd.to_datetime(max_week_ending_date, errors="coerce")
    if pd.isna(max_dt):
        return ""
    overlap = int(os.environ.get("TIDE_WEEKLY_STREAMS_OVERLAP_WEEKS", str(WEEKLY_STREAMS_INCREMENTAL_OVERLAP_WEEKS)))
    floor = (max_dt - pd.Timedelta(weeks=overlap)).strftime("%Y-%m-%d")
    # Inclusive >= so overlap weeks re-upsert cleanly (INSERT OR REPLACE / UNIQUE).
    return f"\n        AND da.week_end_date >= {_snowflake_date_literal(floor)}"


def _max_daily_streams_report_date(cursor: sqlite3.Cursor, mrelg_id: str) -> str | None:
    cursor.execute(
        "SELECT MAX(REPORT_DATE) FROM MARKETSHARE_DAILY_GLOBAL_STREAMS WHERE MRELG_ID = ?",
        (mrelg_id,),
    )
    row = cursor.fetchone()
    return (row[0] if row else None) or None


def _daily_streams_is_fresh(cursor: sqlite3.Cursor, mrelg_id: str) -> bool:
    """
    True iff cache has rows AND the latest report_date is within
    DAILY_STREAMS_STALE_DAYS of today. Snowflake itself excludes the most
    recent day (DATEADD(DAY, -1, CURRENT_DATE())) so we expect max_report_date
    to be roughly today-2.
    """
    _ensure_daily_global_streams_table(cursor)
    max_report = _max_daily_streams_report_date(cursor, mrelg_id)
    if not max_report:
        return False
    max_dt = pd.to_datetime(max_report, errors="coerce")
    if pd.isna(max_dt):
        return False
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    return (today - max_dt).days <= DAILY_STREAMS_STALE_DAYS


def _weekly_streams_is_fresh(cursor: sqlite3.Cursor, mrelg_id: str) -> bool:
    _ensure_weekly_global_streams_table(cursor)
    cursor.execute(
        "SELECT MAX(WEEK_ENDING_DATE) FROM MARKETSHARE_WEEKLY_GLOBAL_STREAMS WHERE MRELG_ID = ?",
        (mrelg_id,),
    )
    row = cursor.fetchone()
    max_week = (row[0] if row else None) or None
    if not max_week:
        return False
    max_dt = pd.to_datetime(max_week, errors="coerce")
    if pd.isna(max_dt):
        return False
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    return (today - max_dt).days <= WEEKLY_STREAMS_STALE_DAYS


def refresh_daily_global_streams_for_mrelg(
    mrelg_id: str,
    release_date: str,
    sf_conn=None,
    *,
    force_full: bool = False,
) -> int:
    """
    Pull daily worldwide stream counts for a single MRELG from Snowflake and
    upsert into MARKETSHARE_DAILY_GLOBAL_STREAMS.

    When the cache already has rows and force_full is False, only pulls dates
    after the latest cached report_date (minus overlap). Returns rows written.
    """
    if not mrelg_id:
        return 0
    max_cached = None
    if not force_full:
        with sqlite_connect() as conn:
            cur = conn.cursor()
            _ensure_daily_global_streams_table(cur)
            max_cached = _max_daily_streams_report_date(cur, mrelg_id)

    sql = (
        load_sql(DAILY_GLOBAL_STREAMING_SF_QUERY)
        .replace("{RELEASE_DATE}", _snowflake_str(release_date))
        .replace("{MRELG_ID}", _snowflake_str(mrelg_id))
        .replace("{MIN_REPORT_DATE_FILTER}", _incremental_daily_filter(max_cached))
    )

    import contextlib

    @contextlib.contextmanager
    def _maybe_conn():
        if sf_conn is not None:
            yield sf_conn
        else:
            with get_snowflake_connection() as fresh:
                yield fresh

    with _maybe_conn() as sf:
        df = sf.query(sql)

    if df is None or df.empty:
        return 0
    df = df.rename(columns=str.upper)
    if "REPORT_DATE" not in df.columns or "GLOBAL_STREAMS" not in df.columns:
        logger.warning(
            "refresh_daily_global_streams_for_mrelg: unexpected columns %s for mrelg=%s",
            list(df.columns),
            mrelg_id,
        )
        return 0
    df["REPORT_DATE"] = df["REPORT_DATE"].astype(str)
    df["GLOBAL_STREAMS"] = pd.to_numeric(df["GLOBAL_STREAMS"], errors="coerce")
    df = df.dropna(subset=["GLOBAL_STREAMS"])

    rows = [(mrelg_id, str(d), float(v)) for d, v in zip(df["REPORT_DATE"], df["GLOBAL_STREAMS"])]
    with sqlite_connect() as conn:
        cur = conn.cursor()
        _ensure_daily_global_streams_table(cur)
        cur.executemany(load_sql(INSERT_DAILY_GLOBAL_STREAMS), rows)
        conn.commit()
    logger.info(
        "refresh_daily_global_streams_for_mrelg: wrote %d rows for mrelg=%s (since %s)",
        len(rows),
        mrelg_id,
        release_date,
    )
    return len(rows)


def get_daily_global_streams_for_mrelg(
    mrelg_id: str,
    release_date: str | None = None,
    refresh_if_stale: bool = True,
) -> pd.DataFrame:
    """
    Read cached daily worldwide streams for a MRELG. When the cache is empty
    or older than DAILY_STREAMS_STALE_DAYS, refresh from Snowflake first
    (requires release_date so the Snowflake query can bound itself). Pass
    refresh_if_stale=False to always serve cached rows even if stale.
    """
    if not mrelg_id:
        return pd.DataFrame(columns=["REPORT_DATE", "GLOBAL_STREAMS"])

    with sqlite_connect() as conn:
        cur = conn.cursor()
        _ensure_daily_global_streams_table(cur)
        fresh = _daily_streams_is_fresh(cur, mrelg_id)

    if refresh_if_stale and not fresh and release_date:
        try:
            refresh_daily_global_streams_for_mrelg(mrelg_id, release_date)
        except Exception as e:
            logger.exception(
                "get_daily_global_streams_for_mrelg: refresh failed for mrelg=%s: %s",
                mrelg_id,
                e,
            )

    with sqlite_connect() as conn:
        df = pd.read_sql_query(
            load_sql(DAILY_GLOBAL_STREAMS_SQLITE_QUERY),
            conn,
            params=(mrelg_id,),
        )
    return df


def refresh_weekly_global_streams_for_mrelg(
    mrelg_id: str,
    release_date: str,
    sf_conn=None,
    *,
    force_full: bool = False,
) -> int:
    """
    Pull weekly worldwide streams (global_streaming input) into SQLite.

    Incremental when cache exists: only weeks after max(WEEK_ENDING_DATE) minus
    overlap. Cold cache or force_full runs the full window query.
    """
    if not mrelg_id:
        return 0

    max_cached = None
    if not force_full:
        with sqlite_connect() as conn:
            cur = conn.cursor()
            _ensure_weekly_global_streams_table(cur)
            cur.execute(
                "SELECT MAX(WEEK_ENDING_DATE) FROM MARKETSHARE_WEEKLY_GLOBAL_STREAMS WHERE MRELG_ID = ?",
                (mrelg_id,),
            )
            row = cur.fetchone()
            max_cached = (row[0] if row else None) or None

    sql = (
        load_sql(WEEKLY_GLOBAL_STREAMING_SF_QUERY)
        .replace("{MRELG_ID}", _snowflake_str(mrelg_id))
        .replace("{RELEASE_DATE}", _snowflake_str(release_date))
        .replace("{MIN_WEEK_ENDING_DATE_FILTER}", _incremental_weekly_filter(max_cached))
    )

    import contextlib

    @contextlib.contextmanager
    def _maybe_conn():
        if sf_conn is not None:
            yield sf_conn
        else:
            with get_snowflake_connection() as fresh:
                yield fresh

    with _maybe_conn() as sf:
        df = sf.query(sql)

    if df is None or df.empty:
        return 0
    df = df.rename(columns=str.lower)
    if "week_ending_date" not in df.columns or "global_streams" not in df.columns:
        logger.warning(
            "refresh_weekly_global_streams_for_mrelg: unexpected columns %s for mrelg=%s",
            list(df.columns),
            mrelg_id,
        )
        return 0

    rows = []
    for wk, val in zip(df["week_ending_date"], df["global_streams"]):
        wk_str = wk.strftime("%Y-%m-%d") if hasattr(wk, "strftime") else str(wk).split(" ")[0][:10]
        rows.append((mrelg_id, wk_str, float(val)))

    with sqlite_connect() as conn:
        cur = conn.cursor()
        _ensure_weekly_global_streams_table(cur)
        cur.executemany(load_sql(INSERT_WEEKLY_GLOBAL_STREAMS), rows)
        conn.commit()
    logger.info(
        "refresh_weekly_global_streams_for_mrelg: wrote %d rows for mrelg=%s (incremental=%s)",
        len(rows),
        mrelg_id,
        bool(max_cached),
    )
    return len(rows)


def get_weekly_global_streams_for_mrelg(
    mrelg_id: str,
    release_date: str | None = None,
    *,
    refresh_if_stale: bool = True,
    sf_conn=None,
) -> pd.DataFrame:
    """
    Read cached weekly worldwide streams for global_streaming forecasts.
    Refreshes from Snowflake when empty or stale (see WEEKLY_STREAMS_STALE_DAYS).
    """
    if not mrelg_id:
        return pd.DataFrame(columns=["week_ending_date", "global_streams"])

    with sqlite_connect() as conn:
        cur = conn.cursor()
        _ensure_weekly_global_streams_table(cur)
        fresh = _weekly_streams_is_fresh(cur, mrelg_id)

    if refresh_if_stale and not fresh and release_date:
        try:
            refresh_weekly_global_streams_for_mrelg(
                mrelg_id, release_date, sf_conn=sf_conn
            )
        except Exception as e:
            logger.exception(
                "get_weekly_global_streams_for_mrelg: refresh failed for mrelg=%s: %s",
                mrelg_id,
                e,
            )

    with sqlite_connect() as conn:
        df = pd.read_sql_query(
            load_sql(WEEKLY_GLOBAL_STREAMS_SQLITE_QUERY),
            conn,
            params=(mrelg_id,),
        )
    if df.empty:
        return df
    return df.rename(columns=str.lower)


# ---------------------------------------------------------------------------
# 2025 catalog revenue (Live Revenue board only)
#
# One-shot load from s3://parquetgarage/model/data/2025_revenue_catalog.csv
# into MARKETSHARE_REVENUE_2025. The CSV is the source of truth (~5M rows of
# Luminate catalog revenue), so the table is fully replaced on refresh.
# Intentionally NOT wired into update_sqlite_main — the weekly Snowflake cron
# does not need to re-pull a static S3 file.
# ---------------------------------------------------------------------------

REVENUE_2025_S3_KEY = "model/data/2025_revenue_catalog.csv"


def _open_revenue_2025_csv_from_s3():
    """
    Stream the 2025 revenue CSV from S3. Resolves bucket via the same env
    as api/s3_pull.py (TIDE_ARTIFACTS_S3_URI / TIDE_ARTIFACTS_S3_BUCKET /
    TIDE_S3_DEFAULT_BUCKET). Returns a file-like body suitable for
    pandas.read_csv. Caller is responsible for closing.
    """
    import boto3

    uri = os.environ.get("TIDE_ARTIFACTS_S3_URI", "").strip()
    bucket = os.environ.get("TIDE_ARTIFACTS_S3_BUCKET", "").strip()
    if uri:
        if not uri.lower().startswith("s3://"):
            raise ValueError(f"TIDE_ARTIFACTS_S3_URI must start with s3://, got {uri!r}")
        rest = uri[5:].split("/", 1)
        bucket = rest[0].strip()
    if not bucket:
        bucket = os.environ.get("TIDE_S3_DEFAULT_BUCKET", "parquetgarage").strip()
    if not bucket:
        raise RuntimeError(
            "No S3 bucket configured for revenue 2025 load (set TIDE_ARTIFACTS_S3_URI or TIDE_S3_DEFAULT_BUCKET)."
        )
    logger.info("revenue_2025: streaming s3://%s/%s", bucket, REVENUE_2025_S3_KEY)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=REVENUE_2025_S3_KEY)
    return obj["Body"]


def refresh_marketshare_revenue_2025() -> int:
    """
    Rebuild MARKETSHARE_REVENUE_2025 from the S3 CSV. Bulk-inserts in
    chunks so we never hold the full ~350MB frame in memory. The CSV is
    the source of truth; the table is fully cleared first.

    Returns the number of rows written.
    """
    logger.info("sqlite_handler: refreshing MARKETSHARE_REVENUE_2025 (db=%s)", DATABASE_NAME)

    body = _open_revenue_2025_csv_from_s3()

    # Stream-parse the CSV: only the two columns we need. usecols by name
    # tolerates the BOM-prefixed first header ('﻿MRELG_ID') because
    # pandas strips it during header parsing.
    chunk_iter = pd.read_csv(
        body,
        usecols=["MRELG_ID", "2025_revenue"],
        dtype={"MRELG_ID": "string", "2025_revenue": "float64"},
        chunksize=200_000,
        encoding="utf-8",
    )

    total_rows = 0
    insert_sql = load_sql(INSERT_MARKETSHARE_REVENUE_2025)

    with sqlite_connect() as conn:
        cursor = conn.cursor()
        cursor.execute(load_sql(CREATE_MARKETSHARE_REVENUE_2025_TABLE))
        cursor.execute(load_sql(DELETE_MARKETSHARE_REVENUE_2025))
        for chunk in chunk_iter:
            chunk = chunk.dropna(subset=["MRELG_ID"])
            chunk["MRELG_ID"] = chunk["MRELG_ID"].astype(str).str.strip()
            chunk = chunk.loc[chunk["MRELG_ID"] != ""]
            chunk["2025_revenue"] = pd.to_numeric(chunk["2025_revenue"], errors="coerce")
            chunk = chunk.replace([float("inf"), float("-inf")], pd.NA)
            rows = [
                (mid, None if pd.isna(rev) else float(rev))
                for mid, rev in zip(chunk["MRELG_ID"], chunk["2025_revenue"])
            ]
            if rows:
                cursor.executemany(insert_sql, rows)
                total_rows += len(rows)
                logger.info("revenue_2025: wrote chunk (%d rows, total=%d)", len(rows), total_rows)
        conn.commit()

    logger.info("sqlite_handler: MARKETSHARE_REVENUE_2025 rebuilt (rows=%d)", total_rows)
    return total_rows


def get_catalog_revenue_2025_for_mrelg(mrelg_id: str) -> float | None:
    """
    Single-row lookup for the Live Revenue board. Returns None when the
    MRELG isn't present in the CSV (frontend treats null as "not in file").
    """
    if not isinstance(mrelg_id, str) or not mrelg_id.strip():
        return None
    mrelg_id = mrelg_id.strip()
    try:
        with sqlite_connect() as conn:
            cur = conn.cursor()
            cur.execute(load_sql(MARKETSHARE_REVENUE_2025_BY_MRELG_QUERY), (mrelg_id,))
            row = cur.fetchone()
    except sqlite3.OperationalError as e:
        # Table missing on a fresh db — treat as "no data" rather than 500.
        if "no such table" in str(e).lower():
            logger.warning("get_catalog_revenue_2025_for_mrelg: table missing; returning None")
            return None
        raise
    if row is None or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def drop_table(table_name: str) -> None:
    """
    Drops a table from the SQLite database.
    """
    try:
        with sqlite_connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error dropping table: {e}")


if __name__ == "__main__":
    update_sqlite_main()
