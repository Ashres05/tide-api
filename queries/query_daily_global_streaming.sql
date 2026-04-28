SELECT
    da.datename AS report_date,
    SUM(s.quantity) AS global_streams
FROM luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
    JOIN luminate_prod.extract_s.vw_musical_release_group_ds m ON s.mrelg_id = m.mrelg_id
    JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
WHERE
    s.country_code = 'AA'
    AND s.metric_category = 'Streams'
    AND s.service_type = 'OnDemand'
    AND da.datename >= {RELEASE_DATE} 
    AND s.mrelg_id = {MRELG_ID}
    AND da.datename < DATEADD(DAY, -1, CURRENT_DATE())
GROUP BY 
    da.datename
ORDER BY 
    da.datename;
