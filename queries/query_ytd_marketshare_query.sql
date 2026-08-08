SELECT *
FROM current_dev.data.marketshare_ytd
WHERE
    label_name IN ({TARGET_LABELS})
    AND country_code = 'US'
    AND week_ending_date >= '{MIN_WEEK_END_DATE}'
    AND DATEADD(DAY, -2, CURRENT_DATE()) > week_ending_date
ORDER BY week_ending_date DESC;
