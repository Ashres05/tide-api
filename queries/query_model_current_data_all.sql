-- Incremental refresh for current_data_all.csv (7-label marketshare panel).
-- {MIN_WEEK_END_DATE} is substituted from Python as the max WEEK_ENDING_DATE
-- currently in current_data_all.csv (or '2018-01-01' on a cold start).
-- Upper bound excludes the in-progress week.
SELECT *
FROM current_dev.data.marketshare_weekly w
WHERE w.release_age = 'Current'
    AND w.country_code = 'US'
    AND w.label_name IN (
        'Warner Records',
        'Atlantic Music Group',
        'IGA',
        'CMG',
        'REPUBLIC Collective',
        'THE ORCHARD',
        'RCA Records',
        'Columbia Records'
    )
    AND w.week_ending_date >= '{MIN_WEEK_END_DATE}'
    AND w.week_ending_date <= DATEADD(DAY, -1, CURRENT_DATE())
ORDER BY w.week_ending_date DESC;
