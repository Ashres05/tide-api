-- Current US weekly panel (all labels present in marketshare_weekly, not only eight).
SELECT
    w.week_ending_date AS WEEK_ENDING_DATE,
    w.label_name AS LABEL_NAME,
    w.album_equivalent AS ALBUM_EQUIVALENT,
    w.album_equivalent_share AS ALBUM_EQUIVALENT_SHARE
FROM current_dev.data.marketshare_weekly w
WHERE w.country_code = 'US'
  AND w.release_age = 'Current'
  AND w.week_ending_date >= '{MIN_WEEK_END_DATE}'
  AND w.week_ending_date <= DATEADD(DAY, -1, CURRENT_DATE())
ORDER BY w.week_ending_date, w.label_name
;