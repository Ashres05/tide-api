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
WHERE
    s.country_code = 'US'
    AND s.mrelg_id IN ({RELEASE_IDS})
GROUP BY
    ALL
