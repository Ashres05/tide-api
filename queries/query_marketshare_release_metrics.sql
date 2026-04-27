SELECT
    da.week_end_date AS week_ending_date,
    s.mrelg_id,
    ROUND(
        SUM(s.equivalent_quantity),
        0
    ) AS album_equivalent,
    ROUND(
        SUM(
            IFF(
                s.metric_category = 'ProductSales',
                s.equivalent_quantity,
                0
            )
        ),
        0
    ) AS product_sales,
    ROUND(
        SUM(
            IFF(
                s.metric_category = 'RecordingSales',
                s.equivalent_quantity,
                0
            )
        ),
        0
    ) AS song_sale_equivalent,
    ROUND(
        SUM(
            IFF(
                s.metric_category = 'Streams'
                AND s.service_type = 'OnDemand',
                s.equivalent_quantity,
                0
            )
        ),
        0
    ) AS streaming_equivalent
FROM
    luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
    JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
    JOIN luminate_prod.extract_s.vw_musical_release_group_ds m ON m.mrelg_id = s.mrelg_id
WHERE
    s.country_code = 'US'
    AND s.mrelg_id IN ({RELEASE_IDS})
    AND da.week_end_date >= '{MIN_WEEK_END_DATE}'
    AND s.report_date >= m.first_sale_date
    -- Intentionally no upper bound on week_end_date: include the partial
    -- in-progress week so AE YTD totals match Atlantic's expected numbers
    -- (Luminate Connect dashboards include the partial week too). The
    -- previous `week_end_date < CURRENT_DATE - 2d` filter was lopping ~12K
    -- AE off OCTANE-class releases mid-week, causing every per-release card
    -- to display 1 week stale even after refresh.
GROUP BY
    ALL;
