WITH prelim_query AS (
    SELECT
        da.week_end_date AS week_ending_date,
        SUM(s.quantity) AS global_streams,
        ROW_NUMBER() OVER (
            ORDER BY week_end_date
        ) AS rn
    FROM luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
        JOIN luminate_prod.extract_s.vw_musical_release_group_ds m ON s.mrelg_id = m.mrelg_id
        JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
    WHERE
        s.country_code = 'AA'
        AND s.metric_category = 'Streams'
        AND s.service_type = 'OnDemand'
        AND s.mrelg_id = {MRELG_ID}
        AND da.week_end_date < DATEADD(DAY, -2, CURRENT_DATE())
    GROUP BY
        1
)

SELECT
    weekly.week_ending_date,
    weekly.global_streams
FROM prelim_query weekly
    JOIN prelim_query week_after ON week_after.rn = weekly.rn + 1
WHERE
    NOT (
        weekly.rn = 1
        AND week_after.global_streams / weekly.global_streams > 100 
        -- streams week 2 are 100x greater
    )
