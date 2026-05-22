-- Incremental YTD fiscal proxy revenue by distributor label (weekly).
-- {MIN_WEEK_END_DATE} is the max week_end_date in ytd_fiscal_revenue_by_label.csv
-- (or '2018-01-01' on cold start). Upper bound excludes the in-progress week.
WITH target_albums AS (
    SELECT
        mrg.mrelg_id,
        i.level_1_distributor,
        i.level_2_distributor,
        i.level_3_distributor
    FROM luminate_prod.extract_s.vw_musical_release_group_ds mrg
    JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds ml ON ml.mrelg_id = mrg.mrelg_id
    JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp ON mp.mrel_id = ml.mrel_id
    JOIN current_dev.data.marketshare_map_icpns i ON i.mp_id = mp.mp_id
    WHERE mrg.release_type IN ('Album', 'EP', 'Single')
        AND i.level_2_distributor IS NOT NULL
        AND mrg.compilation_type != 'Compilation'
    GROUP BY ALL
),
weekly_streams AS (
    SELECT
        da.week_end_date,
        dq.level_1_distributor,
        dq.level_2_distributor,
        dq.level_3_distributor,
        COALESCE(
            SUM(
                IFF(
                    s.metric_category = 'Streams'
                    AND s.service_type = 'OnDemand',
                    s.quantity,
                    0
                )
            ),
            0
        ) AS worldwide_streams
    FROM target_albums dq
    JOIN luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s ON s.mrelg_id = dq.mrelg_id
    JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
    WHERE s.country_code = 'AA'
        AND da.week_end_date >= '{MIN_WEEK_END_DATE}'
        AND da.week_end_date <= DATEADD(DAY, -1, CURRENT_DATE())
    GROUP BY ALL
)
SELECT
    ws.week_end_date,
    ws.level_1_distributor,
    ws.level_2_distributor,
    ws.level_3_distributor,
    (ws.worldwide_streams * 0.004) AS proxy_revenue
FROM weekly_streams ws
ORDER BY
    ws.week_end_date,
    ws.level_1_distributor,
    ws.level_2_distributor,
    ws.level_3_distributor;
