-- Phase 2: incremental refresh.
-- {MIN_WEEK_END_DATE} is substituted from Python as the max WEEK_END_DATE currently
-- in alist_75k.csv (or '2018-01-01' on a cold start). Use an inclusive -1 day
-- guard so the most recently completed weekly row is picked up promptly.
SELECT *
FROM current_dev.data.marketshare_weekly w
WHERE w.release_age = 'Current'
    AND w.country_code = 'US'
    AND w.label_name IN ('Atlantic Music Group', 'Interscope/Geffen/A&M')
    AND w.week_ending_date >= '{MIN_WEEK_END_DATE}'
    AND w.week_ending_date <= DATEADD(DAY, -1, CURRENT_DATE())
ORDER BY WEEK_ENDING_DATE DESC;
