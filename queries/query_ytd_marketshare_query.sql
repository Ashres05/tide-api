SELECT *
FROM current_dev.data.marketshare_ytd
WHERE
    label_name IN ('Atlantic Music Group', 'Interscope/Geffen/A&M')
    AND country_code = 'US'
ORDER BY week_ending_date DESC;
