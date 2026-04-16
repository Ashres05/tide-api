WITH current_date AS (
    SELECT
        DATEADD(MONTH, -18, CURRENT_DATE()) AS cur_date
),

major_releases AS (
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
        AND s.report_date BETWEEN m.first_sale_date
        AND (SELECT cur_date FROM current_date)
    GROUP BY
        da.week_end_date,
        m.mrelg_id -- isolate big albums
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
        JOIN (
            SELECT
                DISTINCT mrelg_id
            FROM
                major_releases
        ) mr ON mr.mrelg_id = mrel.mrelg_id
    WHERE
        i.country_code = 'US' QUALIFY ROW_NUMBER() OVER (
            PARTITION BY mrel.mrelg_id
            ORDER BY
                i.mp_id
        ) = 1
),
alist_weekly AS (
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
)
SELECT
    *
FROM
    alist_weekly
ORDER BY
    week_end_date DESC;
