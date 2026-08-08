-- Restrict to the current Luminate chart-year. Luminate's chart year doesn't
-- align with the calendar year (bounded on chart Fridays), so filtering by
-- week_ending_date.startsWith('2026') in the frontend would mis-bucket weeks
-- straddling the boundary. The YEAR column is populated authoritatively from
-- current_data_all.csv via recompute_ytd_share_from_current_data() — which is
-- Luminate's own chart-year designation.
--
-- {TARGET_LABELS} is injected from model.marketshare_labels.TARGET_LABELS.
SELECT
    WEEK_ENDING_DATE,
    LABEL_NAME,
    ALBUM_EQUIVALENT,
    ALBUM_EQUIVALENT_SHARE
FROM MARKETSHARE_YTD
WHERE
    LABEL_NAME IN ({TARGET_LABELS})
    AND COUNTRY_CODE = 'US'
    AND RELEASE_AGE = 'Current'
    AND YEAR = (SELECT MAX(YEAR) FROM MARKETSHARE_YTD)
ORDER BY date(WEEK_ENDING_DATE) ASC;
