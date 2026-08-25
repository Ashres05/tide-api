-- One row per LUMINATE_ARTIST_ID, rolled up from MARKETSHARE_SEARCH_SUMMARY*
-- at snapshot refresh. Typeahead (GET /v1/revenue/search_artists) hits this
-- table only — never albums/singles/EPs. Supporting indexes are created in
-- ensure_marketshare_search_artists_table() in sqlite_handler.py.
CREATE TABLE IF NOT EXISTS MARKETSHARE_SEARCH_ARTISTS (
    LUMINATE_ARTIST_ID TEXT PRIMARY KEY,
    ARTIST TEXT,
    ARTIST_SEARCH TEXT,
    DAILY_GLOBAL_STREAMS INTEGER,
    RELEASE_COUNT INTEGER,
    HAS_ARTWORK INTEGER DEFAULT 0
);
