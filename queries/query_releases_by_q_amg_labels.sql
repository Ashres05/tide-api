-- US non-compilation releases for AMG level-3 distributor labels.
-- {MIN_FIRST_SALE_DATE} is max(first_sale_date) in releases_by_q_amg_labels.csv
-- (or '2023-09-01' on cold start). Incremental pulls use strict ``>`` on that anchor.
WITH all_releases AS (
    SELECT DISTINCT
        mrel.mrelg_id,
        i.level_3_distributor
    FROM luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel
    JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp
        ON mp.mrel_id = mrel.mrel_id
    JOIN current_dev.data.marketshare_map_icpns i
        ON i.mp_id = mp.mp_id
    WHERE i.level_3_distributor IN (
        'Atlantic Records',
        '300 Entertainment',
        '10K Projects'
    )
      AND i.country_code = 'US'
)
SELECT
    m.mrelg_id AS MRELG_ID,
    m.display_artist AS DISPLAY_ARTIST,
    m.title AS TITLE,
    t.level_3_distributor AS LEVEL_3_DISTRIBUTOR,
    m.first_sale_date AS FIRST_SALE_DATE
FROM luminate_prod.extract_s.vw_musical_release_group_ds m
JOIN all_releases t
    ON t.mrelg_id = m.mrelg_id
WHERE m.compilation_type = 'Non Compilation'
  AND m.first_sale_date > '{MIN_FIRST_SALE_DATE}'
  AND m.first_sale_date <= CURRENT_DATE()
ORDER BY
    m.display_artist,
    m.title;
