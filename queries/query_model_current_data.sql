SELECT *
FROM current_dev.data.marketshare_weekly w
WHERE w.release_age = 'Current'
    AND w.country_code = 'US'
    AND w.label_name IN ('Atlantic Music Group', 'Interscope/Geffen/A&M')
ORDER BY WEEK_ENDING_DATE DESC;
