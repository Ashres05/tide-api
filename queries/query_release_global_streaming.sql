-- Single-pass weekly aggregation with an inline spike filter.
--
-- Replaces the previous CTE + self-join pattern. The original query
-- materialized weekly totals into ``prelim_query`` and self-joined it on
-- ``rn = rn + 1`` just to compare each week to the next, which forced
-- Snowflake to scan/aggregate the same rows twice and silently dropped the
-- final week (the last row had no week_after). LEAD() over the same window
-- gets the next-week comparison in one pass and keeps the tail row.
--
-- Filter semantics are preserved: drop the very first observed week iff the
-- following week's streams are >100x larger (treats the first week as a
-- pre-release leak / metadata error and prefers the second week as the
-- true cold-start anchor).
SELECT
    week_ending_date,
    global_streams
FROM (
    SELECT
        da.week_end_date AS week_ending_date,
        SUM(s.quantity)  AS global_streams,
        ROW_NUMBER() OVER (ORDER BY da.week_end_date) AS rn,
        LEAD(SUM(s.quantity)) OVER (ORDER BY da.week_end_date) AS next_global_streams
    FROM luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
        JOIN luminate_prod.extract_s.vw_musical_release_group_ds m ON s.mrelg_id = m.mrelg_id
        JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
    WHERE
        s.country_code      = 'AA'
        AND s.metric_category = 'Streams'
        AND s.service_type    = 'OnDemand'
        AND s.mrelg_id        = {MRELG_ID}
        
        -- Keeps the release date floor so we don't pull data from before the release existed
        AND da.week_end_date >= {RELEASE_DATE}
        
        -- NEW: Apply the YTD + 12-week buffer to drop heavy catalog history
        AND da.week_end_date >= DATEADD(WEEK, -12, DATE_TRUNC('YEAR', CURRENT_DATE()))
        
        AND da.week_end_date < DATEADD(DAY, -2, CURRENT_DATE())
    GROUP BY da.week_end_date
)
WHERE NOT (
    rn = 1
    AND COALESCE(next_global_streams / NULLIF(global_streams, 0), 0) > 100
)
ORDER BY week_ending_date
