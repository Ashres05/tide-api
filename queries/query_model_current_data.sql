-- Phase 2: incremental refresh.
-- {MIN_WEEK_END_DATE} is substituted from Python as the max WEEK_END_DATE currently
-- in alist_75k.csv (or '2018-01-01' on a cold start). The -2 day guard excludes
-- the current, still-reporting week so we never persist partial data.
SELECT *
FROM current_dev.data.marketshare_weekly w
WHERE w.release_age = 'Current'
    AND w.country_code = 'US'
    AND w.label_name IN ('Atlantic Music Group', 'Interscope/Geffen/A&M')
    AND w.week_ending_date >= '{MIN_WEEK_END_DATE}'
    AND DATEADD(DAY, -2, CURRENT_DATE()) > w.week_ending_date
ORDER BY WEEK_ENDING_DATE DESC;
