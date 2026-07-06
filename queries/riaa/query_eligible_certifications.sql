SELECT
    *
FROM
    current_dev.data.riaa_view_release_summary
WHERE
    predicted_eligibility <> 'NONE'
ORDER BY
    months_eligible ASC,
    predicted_units DESC;
