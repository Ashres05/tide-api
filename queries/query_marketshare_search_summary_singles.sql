WITH mrelg_labels AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY mrelg_id
            ORDER BY
                percent_owned DESC
        ) AS rn
    FROM
        (
            SELECT
                DISTINCT mrel.mrelg_id,
                i.level_2_distributor,
                i.percent_owned
            FROM
                current_dev.data.marketshare_map_icpns i
                JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp ON mp.mp_id = i.mp_id
                JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel ON mrel.mrel_id = mp.mrel_id
        ) QUALIFY rn = 1
),
mrelg_summary AS (
    SELECT
        s.mrelg_id,
        s.title,
        s.display_artist AS artist,
        l.level_2_distributor AS label,
        COALESCE(s.first_sale_date, s.release_date) AS release_date,
        GET(
            FILTER(genres, x -> x:CLIENT_DOMAIN = 'Billboard'),
            0
        ):MAIN_GENRE::STRING AS genre,
        COALESCE(SUM(ss.quantity), 0) AS daily_streams
    FROM
        luminate_prod.extract_s.vw_musical_release_group_ds s
        JOIN luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds ss ON ss.mrelg_id = s.mrelg_id
        AND ss.country_code = 'AA'
        AND ss.report_date = DATEADD(DAY, -2, CURRENT_DATE())
        AND ss.metric_category = 'Streams'
        AND ss.service_type = 'OnDemand'
        JOIN mrelg_labels l ON l.mrelg_id = s.mrelg_id
    WHERE
        s.release_type = 'Single'
    GROUP BY
        ALL
)
SELECT
    *
FROM
    mrelg_summary
