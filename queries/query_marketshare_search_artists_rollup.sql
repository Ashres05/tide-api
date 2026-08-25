-- Roll up album + singles search snapshots to one row per LUMINATE_ARTIST_ID.
-- Display name is the ARTIST string on that id's most-streamed release.
-- DAILY_GLOBAL_STREAMS is the sum of yesterday's worldwide OnDemand snapshot
-- across that artist's titles. Called from refresh_marketshare_search_artists().
WITH combined AS (
    SELECT
        TRIM(LUMINATE_ARTIST_ID) AS LUMINATE_ARTIST_ID,
        ARTIST,
        COALESCE(DAILY_GLOBAL_STREAMS, 0) AS DAILY_GLOBAL_STREAMS
    FROM MARKETSHARE_SEARCH_SUMMARY
    WHERE LUMINATE_ARTIST_ID IS NOT NULL
      AND TRIM(LUMINATE_ARTIST_ID) != ''
      AND LOWER(TRIM(LUMINATE_ARTIST_ID)) NOT IN ('none', 'nan')
    UNION ALL
    SELECT
        TRIM(LUMINATE_ARTIST_ID),
        ARTIST,
        COALESCE(DAILY_GLOBAL_STREAMS, 0)
    FROM MARKETSHARE_SEARCH_SUMMARY_SINGLES
    WHERE LUMINATE_ARTIST_ID IS NOT NULL
      AND TRIM(LUMINATE_ARTIST_ID) != ''
      AND LOWER(TRIM(LUMINATE_ARTIST_ID)) NOT IN ('none', 'nan')
),
ranked AS (
    SELECT
        LUMINATE_ARTIST_ID,
        ARTIST,
        DAILY_GLOBAL_STREAMS,
        ROW_NUMBER() OVER (
            PARTITION BY LUMINATE_ARTIST_ID
            ORDER BY DAILY_GLOBAL_STREAMS DESC, ARTIST COLLATE NOCASE
        ) AS rn
    FROM combined
),
agg AS (
    SELECT
        LUMINATE_ARTIST_ID,
        SUM(DAILY_GLOBAL_STREAMS) AS DAILY_GLOBAL_STREAMS,
        COUNT(*) AS RELEASE_COUNT
    FROM combined
    GROUP BY LUMINATE_ARTIST_ID
)
SELECT
    a.LUMINATE_ARTIST_ID,
    r.ARTIST,
    a.DAILY_GLOBAL_STREAMS,
    a.RELEASE_COUNT
FROM agg a
JOIN ranked r
  ON r.LUMINATE_ARTIST_ID = a.LUMINATE_ARTIST_ID
 AND r.rn = 1
;