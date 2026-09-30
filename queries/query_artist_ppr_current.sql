-- Current artist PPR for roster ∪ expected projects.
-- Snowflake. Nightly materialization.
--
-- {MRELG_ID_LIST} is replaced at runtime with a comma-separated quoted list
-- from query_artist_ppr_universe.sql (SQLite).
--
-- Grain: one row per (MRELG_ID, LUMINATE_ARTIST_ID). A collab with two Main
-- Artists is two rows (same project, each artist's latest PPR). Roster SQLite
-- only stores the first Main Artist; this query flattens ARTISTS in Snowflake.
--
-- ARTIST_PPR_DAILY_V2 is daily. Collapse to max CALCULATION_DATE per Sodatone
-- artist, then stamp that rate on every project row for that artist.
-- Join key is sodatone_artist_id (not sodatone_id).
--
-- artist_master can have more than one Sodatone row per Luminate id;
-- QUALIFY keeps one bridge row per (project, luminate artist).

WITH project_artists AS (
    SELECT
        TRIM(m.mrelg_id) AS mrelg_id,
        TRIM(f.value:ARTIST_ID::STRING) AS luminate_artist_id,
        f.index AS artist_index
    FROM LUMINATE_PROD.EXTRACT_S.VW_MUSICAL_RELEASE_GROUP_DS m,
         LATERAL FLATTEN(input => m.ARTISTS) f
    WHERE m.mrelg_id IN ({MRELG_ID_LIST})
      AND f.value:ARTIST_ID IS NOT NULL
      AND TRIM(f.value:ARTIST_ID::STRING) != ''
      AND LOWER(COALESCE(f.value:ROLE::STRING, '')) = 'main artist'
),
bridge AS (
    SELECT
        p.mrelg_id,
        p.luminate_artist_id,
        p.artist_index,
        am.sodatone_artist_id
    FROM project_artists p
    JOIN CURRENT_DEV.DATA.ARTIST_MASTER am
      ON TRIM(TO_VARCHAR(am.luminate_artist_id)) = p.luminate_artist_id
    WHERE am.sodatone_artist_id IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY p.mrelg_id, p.luminate_artist_id
        ORDER BY IFF(am.sodatone_artist_id IS NULL, 1, 0)
    ) = 1
),
latest_ppr AS (
    SELECT
        v.sodatone_artist_id,
        v.calculation_date,
        v.ppr_value
    FROM CURRENT_DEV.DATA.ARTIST_PPR_DAILY_V2 v
    JOIN bridge b
      ON b.sodatone_artist_id = v.sodatone_artist_id
    WHERE v.ppr_value IS NOT NULL
      AND v.calculation_date IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.sodatone_artist_id
        ORDER BY v.calculation_date DESC NULLS LAST
    ) = 1
)
SELECT
    b.mrelg_id AS MRELG_ID,
    b.luminate_artist_id AS LUMINATE_ARTIST_ID,
    b.sodatone_artist_id AS SODATONE_ARTIST_ID,
    b.artist_index AS ARTIST_INDEX,
    p.calculation_date AS CALCULATION_DATE,
    p.ppr_value AS PPR_VALUE
FROM bridge b
JOIN latest_ppr p
  ON p.sodatone_artist_id = b.sodatone_artist_id
ORDER BY
    b.mrelg_id,
    b.artist_index
;
