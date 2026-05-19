INSERT INTO MARKETSHARE_SEARCH_SUMMARY_SINGLES (
    MRELG_ID, TITLE, ARTIST, LABEL_NAME, RELEASE_DATE, GENRE, DAILY_GLOBAL_STREAMS,
    ARTIST_SEARCH, TITLE_SEARCH
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(MRELG_ID)
DO UPDATE SET
    TITLE                = excluded.TITLE,
    ARTIST               = excluded.ARTIST,
    LABEL_NAME           = excluded.LABEL_NAME,
    RELEASE_DATE         = excluded.RELEASE_DATE,
    GENRE                = excluded.GENRE,
    DAILY_GLOBAL_STREAMS = excluded.DAILY_GLOBAL_STREAMS,
    ARTIST_SEARCH        = excluded.ARTIST_SEARCH,
    TITLE_SEARCH         = excluded.TITLE_SEARCH
;
