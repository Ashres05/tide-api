-- SQLite's sqlite3.Cursor.execute() permits only one statement per call, so
-- this file must contain exactly one DDL. Supporting indexes (streams,
-- LUMINATE_ARTIST_ID) are created idempotently in
-- ensure_marketshare_search_summary_singles_columns() in sqlite_handler.py, which
-- runs immediately after this CREATE TABLE.
CREATE TABLE IF NOT EXISTS MARKETSHARE_SEARCH_SUMMARY_SINGLES (
    MRELG_ID TEXT PRIMARY KEY NOT NULL,
    TITLE TEXT,
    ARTIST TEXT,
    LUMINATE_ARTIST_ID TEXT,
    RELEASE_TYPE TEXT,
    LABEL_NAME TEXT,
    RELEASE_DATE DATE,
    GENRE TEXT,
    DAILY_GLOBAL_STREAMS INTEGER,
    -- Pre-normalized text columns populated at refresh time. Storing them
    -- on disk lets the search endpoint feed candidates directly to the
    -- Python fuzzy scorer without re-running unicode normalization on every
    -- request, and makes LIKE-based prefilters case- and accent-insensitive.
    ARTIST_SEARCH TEXT,
    TITLE_SEARCH TEXT
) WITHOUT ROWID;
