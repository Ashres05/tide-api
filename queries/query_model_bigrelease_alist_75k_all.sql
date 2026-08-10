-- Incremental refresh for bigrelease_alist_75k_all.csv.
-- {MIN_WEEK_END_DATE} is substituted from Python as the max WEEK_END_DATE
-- currently in bigrelease_alist_75k_all.csv (or '2018-01-01' on a cold start).
-- Upper bound excludes the in-progress week.
-- IGA / CMG are level_3 distributors; other labels remain level_2.
WITH weekly_performance AS (
    SELECT
        da.week_end_date,
        m.mrelg_id,
        m.first_sale_date,
        SUM(s.equivalent_quantity) AS equivalent_quantity
    FROM
        luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
        JOIN luminate_prod.extract_s.vw_musical_release_group_ds m
            ON m.mrelg_id = s.mrelg_id
        JOIN luminate_prod.extract_s.vw_date_ds da
            ON da.datename = s.report_date
    WHERE
        s.country_code = 'US'
        AND m.compilation_type = 'Non Compilation'
        AND s.report_date >= '{MIN_WEEK_END_DATE}'
        AND da.week_end_date <= DATEADD(DAY, -1, CURRENT_DATE())
        AND s.report_date >= m.first_sale_date
    GROUP BY
        da.week_end_date,
        m.mrelg_id,
        m.first_sale_date
    HAVING
        SUM(s.equivalent_quantity) >= 75000
),

album_metadata AS (
    SELECT
        mrel.mrelg_id,
        i.level_2_distributor,
        i.level_3_distributor,
        prod.display_artist
    FROM
        luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp
            ON mp.mrel_id = mrel.mrel_id
        JOIN current_dev.data.marketshare_map_icpns i
            ON i.mp_id = mp.mp_id
        JOIN luminate_prod.extract_s.vw_musical_product_ds prod
            ON prod.mp_id = mp.mp_id
    WHERE
        i.country_code = 'US'
        AND EXISTS (
            SELECT 1
            FROM weekly_performance wp
            WHERE wp.mrelg_id = mrel.mrelg_id
        )
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY mrel.mrelg_id
        ORDER BY i.mp_id
    ) = 1
)

SELECT
    wp.week_end_date AS WEEK_END_DATE,

    SUM(IFF(am.level_2_distributor = 'Warner Records', wp.equivalent_quantity, 0)) AS WARNER_ALBUMS,
    SUM(IFF(am.level_2_distributor IN ('Atlantic Music Group', 'Atlantic Records'), wp.equivalent_quantity, 0)) AS AMG_ALBUMS,
    SUM(IFF(am.level_3_distributor = 'IGA', wp.equivalent_quantity, 0)) AS IGA_ALBUMS,
    SUM(IFF(am.level_3_distributor = 'CMG', wp.equivalent_quantity, 0)) AS CMG_ALBUMS,
    SUM(IFF(am.level_2_distributor = 'REPUBLIC Collective', wp.equivalent_quantity, 0)) AS REPUBLIC_ALBUMS,
    SUM(IFF(am.level_2_distributor = 'THE ORCHARD', wp.equivalent_quantity, 0)) AS ORCHARD_ALBUMS,
    SUM(IFF(am.level_2_distributor = 'RCA Records', wp.equivalent_quantity, 0)) AS RCA_ALBUMS,
    SUM(IFF(am.level_2_distributor = 'Columbia Records', wp.equivalent_quantity, 0)) AS COLUMBIA_ALBUMS,

    SUM(wp.equivalent_quantity) AS MARKET_ALBUMS,

    MAX(IFF(
        am.level_2_distributor = 'Warner Records'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_WARNER,

    MAX(IFF(
        am.level_2_distributor IN ('Atlantic Music Group', 'Atlantic Records')
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_ATLANTIC,

    MAX(IFF(
        am.level_3_distributor = 'IGA'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_IGA,

    MAX(IFF(
        am.level_3_distributor = 'CMG'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_CMG,

    MAX(IFF(
        am.level_2_distributor = 'REPUBLIC Collective'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_REPUBLIC,

    MAX(IFF(
        am.level_2_distributor = 'THE ORCHARD'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_ORCHARD,

    MAX(IFF(
        am.level_2_distributor = 'RCA Records'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_RCA,

    MAX(IFF(
        am.level_2_distributor = 'Columbia Records'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_COLUMBIA

FROM
    weekly_performance wp
    LEFT JOIN album_metadata am
        ON wp.mrelg_id = am.mrelg_id
GROUP BY
    wp.week_end_date
ORDER BY
    wp.week_end_date DESC;
