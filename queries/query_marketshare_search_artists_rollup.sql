-- Roll up album + singles search snapshots to one row per LUMINATE_ARTIST_ID.
--
-- Display name is the canonical artist string for that id, NOT the billing
-- line on the single most-streamed release (that is how "Shakira & Burna Boy"
-- replaced Burna Boy). Fold case, pick the LOWER(ARTIST) with the most titles
-- (then the most day-2 streams), then pick a mixed-case spelling of that key.
-- DAILY_GLOBAL_STREAMS remains the sum across all of that id's snapshot rows.
-- Called from refresh_marketshare_search_artists().
WITH combined AS (
    SELECT
        TRIM(LUMINATE_ARTIST_ID) AS LUMINATE_ARTIST_ID,
        TRIM(ARTIST) AS ARTIST,
        COALESCE(DAILY_GLOBAL_STREAMS, 0) AS DAILY_GLOBAL_STREAMS
    FROM MARKETSHARE_SEARCH_SUMMARY
    WHERE LUMINATE_ARTIST_ID IS NOT NULL
      AND TRIM(LUMINATE_ARTIST_ID) != ''
      AND LOWER(TRIM(LUMINATE_ARTIST_ID)) NOT IN ('none', 'nan')
    UNION ALL
    SELECT
        TRIM(LUMINATE_ARTIST_ID),
        TRIM(ARTIST),
        COALESCE(DAILY_GLOBAL_STREAMS, 0)
    FROM MARKETSHARE_SEARCH_SUMMARY_SINGLES
    WHERE LUMINATE_ARTIST_ID IS NOT NULL
      AND TRIM(LUMINATE_ARTIST_ID) != ''
      AND LOWER(TRIM(LUMINATE_ARTIST_ID)) NOT IN ('none', 'nan')
),
spelling AS (
    SELECT
        LUMINATE_ARTIST_ID,
        LOWER(ARTIST) AS artist_key,
        ARTIST,
        SUM(DAILY_GLOBAL_STREAMS) AS spelling_streams,
        COUNT(*) AS spelling_count,
        CASE
            WHEN ARTIST != ''
             AND ARTIST = UPPER(ARTIST)
             AND ARTIST != LOWER(ARTIST)
            THEN 1
            ELSE 0
        END AS is_all_caps
    FROM combined
    WHERE ARTIST IS NOT NULL
      AND ARTIST != ''
    GROUP BY
        LUMINATE_ARTIST_ID,
        LOWER(ARTIST),
        ARTIST
),
name_totals AS (
    SELECT
        LUMINATE_ARTIST_ID,
        artist_key,
        SUM(spelling_streams) AS name_streams,
        SUM(spelling_count) AS name_count
    FROM spelling
    GROUP BY
        LUMINATE_ARTIST_ID,
        artist_key
),
best_key AS (
    SELECT
        LUMINATE_ARTIST_ID,
        artist_key
    FROM (
        SELECT
            LUMINATE_ARTIST_ID,
            artist_key,
            ROW_NUMBER() OVER (
                PARTITION BY LUMINATE_ARTIST_ID
                ORDER BY name_count DESC, name_streams DESC, artist_key
            ) AS rn
        FROM name_totals
    ) ranked_keys
    WHERE rn = 1
),
best_spelling AS (
    SELECT
        LUMINATE_ARTIST_ID,
        ARTIST
    FROM (
        SELECT
            s.LUMINATE_ARTIST_ID,
            s.ARTIST,
            ROW_NUMBER() OVER (
                PARTITION BY s.LUMINATE_ARTIST_ID
                ORDER BY s.is_all_caps ASC, s.spelling_streams DESC, s.spelling_count DESC, s.ARTIST
            ) AS rn
        FROM spelling s
        JOIN best_key k
          ON k.LUMINATE_ARTIST_ID = s.LUMINATE_ARTIST_ID
         AND k.artist_key = s.artist_key
    ) ranked_spell
    WHERE rn = 1
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
    b.ARTIST,
    a.DAILY_GLOBAL_STREAMS,
    a.RELEASE_COUNT
FROM agg a
JOIN best_spelling b
  ON b.LUMINATE_ARTIST_ID = a.LUMINATE_ARTIST_ID
;
