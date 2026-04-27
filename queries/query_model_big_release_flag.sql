-- Phase 2: incremental refresh.
-- {MIN_WEEK_END_DATE} is the max WEEK_END_DATE currently in alist_75k.csv,
-- injected from Python ('2018-01-01' on cold start). Use an inclusive -1 day
-- guard so the most recently completed week is included promptly.
--
-- NOTE: target_albums keeps the rolling 18-month qualifier (`first_sale_date >=
-- -18mo`) unchanged — that's the "is this album less-than-18-months-old" test
-- the user called out as the exception to the incremental scoping rule. We
-- only tighten the *week* dimension on the performance/output side.
WITH target_albums AS (
    SELECT
        m.mrelg_id,
        m.first_sale_date
    FROM
        luminate_prod.extract_s.vw_musical_release_group_ds m
        JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel 
            ON mrel.mrelg_id = m.mrelg_id
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp 
            ON mp.mrel_id = mrel.mrel_id
        JOIN luminate_prod.extract_s.vw_musical_product_ds prod 
            ON prod.mp_id = mp.mp_id
    WHERE
        m.compilation_type = 'Non Compilation'
        AND LOWER(prod.display_artist) NOT LIKE '%various%' 
        -- rolling 18mo album-freshness qualifier (do NOT replace with MIN_WEEK_END_DATE)
        AND m.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY m.mrelg_id
        ORDER BY prod.mp_id
    ) = 1
),

debut_week_performance AS (
    SELECT
        da.week_end_date,
        a.mrelg_id,
        SUM(s.equivalent_quantity) AS equivalent_quantity 
    FROM
        luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
        JOIN target_albums a 
            ON a.mrelg_id = s.mrelg_id
        JOIN luminate_prod.extract_s.vw_date_ds da 
            ON da.datename = s.report_date
    WHERE
        s.country_code = 'US'
        -- Tightened from -18mo to MIN_WEEK_END_DATE. Partition-prune the fact
        -- table to only the weeks we don't already have on disk. Debut weeks
        -- that landed before MIN_WEEK_END_DATE are already persisted in the
        -- prior CSV row; the Python layer's dedupe keeps them.
        AND s.report_date >= '{MIN_WEEK_END_DATE}'
        AND da.week_end_date >= '{MIN_WEEK_END_DATE}'
        AND da.week_end_date <= DATEADD(DAY, -1, CURRENT_DATE())
        AND a.first_sale_date BETWEEN DATEADD(DAY, -7, da.week_end_date) AND da.week_end_date
    GROUP BY
        da.week_end_date,
        a.mrelg_id
),

mrelg_to_distributor AS (
    SELECT
        mrel.mrelg_id,
        i.level_2_distributor
    FROM
        luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp 
            ON mp.mrel_id = mrel.mrel_id
        JOIN CURRENT_DEV.DATA.MARKETSHARE_MAP_ICPNS i 
            ON i.mp_id = mp.mp_id
    WHERE
        i.country_code = 'US'
        AND EXISTS (
            SELECT 1 
            FROM target_albums a 
            WHERE a.mrelg_id = mrel.mrelg_id
        )
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY mrel.mrelg_id
        ORDER BY i.mp_id
    ) = 1
)

SELECT
    dwp.week_end_date,
    MAX(IFF(
        d.level_2_distributor IN ('Atlantic Music Group', 'Atlantic Records') 
        AND dwp.equivalent_quantity >= 75000, 
        1, 
        0
    )) AS Big_Release_Atlantic,
    MAX(IFF(
        d.level_2_distributor = 'Interscope/Geffen/A&M' 
        AND dwp.equivalent_quantity >= 75000, 
        1, 
        0
    )) AS Big_Release_Interscope
FROM
    debut_week_performance dwp
    LEFT JOIN mrelg_to_distributor d 
        ON dwp.mrelg_id = d.mrelg_id
GROUP BY
    dwp.week_end_date
ORDER BY
    dwp.week_end_date DESC;
