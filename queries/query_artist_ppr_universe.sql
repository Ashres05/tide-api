-- MRELG_IDs for nightly artist PPR (roster ∪ expected releases).
-- Snowflake expands each project to every Main Artist on VW_MUSICAL_RELEASE_GROUP_DS.ARTISTS.
-- SQLite. Caller quotes the ids into {MRELG_ID_LIST} for query_artist_ppr_current.sql.

SELECT MRELG_ID
FROM (
    SELECT TRIM(MRELG_ID) AS MRELG_ID
    FROM STREAMING_ROSTER_2026
    WHERE MRELG_ID IS NOT NULL
      AND TRIM(MRELG_ID) != ''
    UNION
    SELECT TRIM(MRELG_ID)
    FROM EXPECTED_RELEASES
    WHERE MRELG_ID IS NOT NULL
      AND TRIM(MRELG_ID) != ''
)
ORDER BY MRELG_ID
;
