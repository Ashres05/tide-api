-- Live current PPR for one artist (search / artist page).
-- Snowflake. One row, not per release.
--
-- {LUMINATE_ARTIST_ID} is replaced at runtime with a quoted Luminate id
-- (same helper as {MRELG_ID}). Call from GET /v1/revenue/releases_by_artist
-- or a dedicated artist_ppr endpoint — not from typeahead.
--
-- Same collapse as nightly: max CALCULATION_DATE on ARTIST_PPR_DAILY_V2,
-- bridged via ARTIST_MASTER on sodatone_artist_id.

WITH bridge AS (
    SELECT
        TRIM(TO_VARCHAR(am.luminate_artist_id)) AS luminate_artist_id,
        am.sodatone_artist_id
    FROM CURRENT_DEV.DATA.ARTIST_MASTER am
    WHERE am.sodatone_artist_id IS NOT NULL
      AND TRIM(TO_VARCHAR(am.luminate_artist_id)) = {LUMINATE_ARTIST_ID}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY TRIM(TO_VARCHAR(am.luminate_artist_id))
        ORDER BY IFF(am.sodatone_artist_id IS NULL, 1, 0)
    ) = 1
)
SELECT
    b.luminate_artist_id AS LUMINATE_ARTIST_ID,
    v.sodatone_artist_id AS SODATONE_ARTIST_ID,
    v.calculation_date AS CALCULATION_DATE,
    v.ppr_value AS PPR_VALUE
FROM CURRENT_DEV.DATA.ARTIST_PPR_DAILY_V2 v
JOIN bridge b
  ON b.sodatone_artist_id = v.sodatone_artist_id
WHERE v.ppr_value IS NOT NULL
  AND v.calculation_date IS NOT NULL
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY b.luminate_artist_id
    ORDER BY v.calculation_date DESC NULLS LAST
) = 1
;
