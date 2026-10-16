-- Phase 0 confirmations (not gated to TARGET_LABELS).
-- 1) Distinct Current-US labels in marketshare_weekly (is the pie only the eight?).
-- 2) Latest complete week: AMG-inverted market vs other panel inversions.
-- 3) ICPN CASE: Atlantic Records / 10K / 300 vs Atlantic Music Group.

WITH latest AS (
    SELECT MAX(week_ending_date) AS week_ending_date
    FROM current_dev.data.marketshare_weekly
    WHERE country_code = 'US'
      AND release_age = 'Current'
      AND week_ending_date <= DATEADD(DAY, -1, CURRENT_DATE())
),
week_rows AS (
    SELECT
        w.week_ending_date,
        w.label_name,
        w.album_equivalent,
        w.album_equivalent_share
    FROM current_dev.data.marketshare_weekly w
    JOIN latest l
      ON w.week_ending_date = l.week_ending_date
    WHERE w.country_code = 'US'
      AND w.release_age = 'Current'
),
label_counts AS (
    SELECT COUNT(DISTINCT label_name) AS n_labels_in_weekly
    FROM current_dev.data.marketshare_weekly
    WHERE country_code = 'US'
      AND release_age = 'Current'
      AND week_ending_date >= DATEADD(WEEK, -8, (SELECT week_ending_date FROM latest))
)
SELECT
    'weekly_label_count_8w' AS check_name,
    TO_VARCHAR(c.n_labels_in_weekly) AS value,
    NULL AS label_name,
    NULL AS album_equivalent,
    NULL AS album_equivalent_share
FROM label_counts c

UNION ALL

SELECT
    'week_row' AS check_name,
    TO_VARCHAR(w.week_ending_date) AS value,
    w.label_name,
    w.album_equivalent,
    w.album_equivalent_share
FROM week_rows w
;