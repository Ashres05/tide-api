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
    AND s.mrelg_id = {MRELG_ID}
    
    -- Floor 1: Prevent pulling zeroes before the album actually dropped
    AND da.datename >= {RELEASE_DATE} 
    
    -- Floor 2 (NEW): Drop ancient catalog history, keep only YTD + 12 weeks
    AND da.datename >= DATEADD(WEEK, -12, DATE_TRUNC('YEAR', CURRENT_DATE()))
    
    AND da.datename < DATEADD(DAY, -1, CURRENT_DATE())
    {MIN_REPORT_DATE_FILTER}
GROUP BY 
    da.datename
ORDER BY 
    da.datename;
