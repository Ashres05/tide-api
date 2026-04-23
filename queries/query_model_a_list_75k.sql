-- Phase 2: incremental refresh.
-- {MIN_WEEK_END_DATE} is the max WEEK_END_DATE currently in alist_75k.csv, injected
-- from Python. On a cold start the Python layer passes '2018-01-01' to reproduce
-- the original full-history behavior. The -2 day guard excludes the in-progress
-- week so partial data is never written to the CSV.
--
-- The a-list concept here is "any week in which an album moved >= 75k AE units".
-- That is per-week data — adding MIN_WEEK_END_DATE to the lower bound does not
-- change qualification logic; it only caps the *output* to weeks we do not
-- already have.
WITH major_releases AS (
    SELECT
        da.week_end_date,
        m.mrelg_id,
        SUM(s.equivalent_quantity) AS equivalent_quantity
    FROM
        luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
        JOIN luminate_prod.extract_s.vw_musical_release_group_ds m ON m.mrelg_id = s.mrelg_id
        JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
    WHERE
        s.country_code = 'US'
        AND m.compilation_type = 'Non Compilation'
        AND s.report_date >= '{MIN_WEEK_END_DATE}'
        AND DATEADD(DAY, -2, CURRENT_DATE()) > da.week_end_date
        AND s.report_date >= m.first_sale_date
    GROUP BY
        da.week_end_date,
        m.mrelg_id
    HAVING
        SUM(s.equivalent_quantity) >= 75000
),
mrelg_to_distributor AS (
    SELECT
        mrel.mrelg_id,
        i.level_2_distributor
    FROM
        luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp ON mp.mrel_id = mrel.mrel_id
        JOIN CURRENT_DEV.DATA.MARKETSHARE_MAP_ICPNS i ON i.mp_id = mp.mp_id
    WHERE
        i.country_code = 'US'
        AND EXISTS (
            SELECT
                1
            FROM
                major_releases mr
            WHERE
                mr.mrelg_id = mrel.mrelg_id
        ) QUALIFY ROW_NUMBER() OVER (
            PARTITION BY mrel.mrelg_id
            ORDER BY
                i.mp_id
        ) = 1
)
SELECT
    m.week_end_date,
    SUM(
        IFF(
            d.level_2_distributor IN ('Atlantic Music Group', 'Atlantic Records'),
            m.equivalent_quantity,
            0
        )
    ) AS amg_albums,
    SUM(
        IFF(
            d.level_2_distributor = 'Interscope/Geffen/A&M',
            m.equivalent_quantity,
            0
        )
    ) AS interscope_albums,
    SUM(m.equivalent_quantity) AS market_albums
FROM
    major_releases m
    LEFT JOIN mrelg_to_distributor d ON m.mrelg_id = d.mrelg_id
GROUP BY
    m.week_end_date
ORDER BY
    m.week_end_date DESC;
