-- Nightly project grain: one row per (MRELG_ID, Main Artist) from
-- query_artist_ppr_current.sql. Same artist on two titles is two rows;
-- PPR_VALUE is still that artist's latest rate.
CREATE TABLE IF NOT EXISTS ARTIST_PPR_BY_RELEASE (
    MRELG_ID TEXT NOT NULL,
    LUMINATE_ARTIST_ID TEXT NOT NULL,
    ARTIST_INDEX INTEGER,
    PPR_VALUE REAL,
    CALCULATION_DATE TEXT,
    PRIMARY KEY (MRELG_ID, LUMINATE_ARTIST_ID)
) WITHOUT ROWID;
