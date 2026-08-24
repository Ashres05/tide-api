-- Incremental refresh for bigrelease_alist_75k_all.csv.
-- {MIN_WEEK_END_DATE} is substituted from Python as the max WEEK_END_DATE
-- currently in bigrelease_alist_75k_all.csv (or '2018-01-01' on a cold start).
-- Upper bound excludes the in-progress week.
--
-- label_group must match query_streaming_roster_ytd.sql / query_release_backfill.sql:
--   level_3 IGA/CMG as-is; legacy level_2 Interscope/Geffen/A&M → IGA;
--   Atlantic Records → Atlantic Music Group; else level_2.
-- Do not alias Interscope-Capitol (IGA vs CMG).
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
        CASE
            WHEN i.level_3_distributor IN ('IGA', 'CMG') THEN i.level_3_distributor
            WHEN i.level_2_distributor = 'Interscope/Geffen/A&M' THEN 'IGA'
            WHEN i.level_2_distributor IN ('Atlantic Music Group', 'Atlantic Records') THEN 'Atlantic Music Group'
            ELSE i.level_2_distributor
        END AS label_group,
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
        AND i.is_current = TRUE
        AND EXISTS (
            SELECT 1
            FROM weekly_performance wp
            WHERE wp.mrelg_id = mrel.mrelg_id
        )
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY mrel.mrelg_id
        ORDER BY
            IFF(
                CASE
                    WHEN i.level_3_distributor IN ('IGA', 'CMG') THEN i.level_3_distributor
                    WHEN i.level_2_distributor = 'Interscope/Geffen/A&M' THEN 'IGA'
                    WHEN i.level_2_distributor IN ('Atlantic Music Group', 'Atlantic Records') THEN 'Atlantic Music Group'
                    ELSE i.level_2_distributor
                END IN (
                    'Warner Records',
                    'Atlantic Music Group',
                    'IGA',
                    'CMG',
                    'REPUBLIC Collective',
                    'THE ORCHARD',
                    'RCA Records',
                    'Columbia Records'
                ),
                0,
                1
            ),
            i.mp_id
    ) = 1
)

SELECT
    wp.week_end_date AS WEEK_END_DATE,

    SUM(IFF(am.label_group = 'Warner Records', wp.equivalent_quantity, 0)) AS WARNER_ALBUMS,
    SUM(IFF(am.label_group = 'Atlantic Music Group', wp.equivalent_quantity, 0)) AS AMG_ALBUMS,
    SUM(IFF(am.label_group = 'IGA', wp.equivalent_quantity, 0)) AS IGA_ALBUMS,
    SUM(IFF(am.label_group = 'CMG', wp.equivalent_quantity, 0)) AS CMG_ALBUMS,
    SUM(IFF(am.label_group = 'REPUBLIC Collective', wp.equivalent_quantity, 0)) AS REPUBLIC_ALBUMS,
    SUM(IFF(am.label_group = 'THE ORCHARD', wp.equivalent_quantity, 0)) AS ORCHARD_ALBUMS,
    SUM(IFF(am.label_group = 'RCA Records', wp.equivalent_quantity, 0)) AS RCA_ALBUMS,
    SUM(IFF(am.label_group = 'Columbia Records', wp.equivalent_quantity, 0)) AS COLUMBIA_ALBUMS,

    SUM(wp.equivalent_quantity) AS MARKET_ALBUMS,

    MAX(IFF(
        am.label_group = 'Warner Records'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_WARNER,

    MAX(IFF(
        am.label_group = 'Atlantic Music Group'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_ATLANTIC,

    MAX(IFF(
        am.label_group = 'IGA'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_IGA,

    MAX(IFF(
        am.label_group = 'CMG'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_CMG,

    MAX(IFF(
        am.label_group = 'REPUBLIC Collective'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_REPUBLIC,

    MAX(IFF(
        am.label_group = 'THE ORCHARD'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_ORCHARD,

    MAX(IFF(
        am.label_group = 'RCA Records'
        AND LOWER(am.display_artist) NOT LIKE '%various%'
        AND wp.first_sale_date >= DATEADD(MONTH, -18, CURRENT_DATE())
        AND wp.first_sale_date BETWEEN DATEADD(DAY, -7, wp.week_end_date) AND wp.week_end_date,
        1, 0
    )) AS BIG_RELEASE_RCA,

    MAX(IFF(
        am.label_group = 'Columbia Records'
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
