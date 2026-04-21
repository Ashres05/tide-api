WITH target_albums AS (
    SELECT
        mrg.mrelg_id,
        mrg.title,
        mrg.display_artist,
        mrg.genres,
        i.level_2_distributor,
        mrg.release_date,
        mrg.release_type,
        MIN(mrg.first_sale_date) AS first_sale_date
    FROM
        luminate_prod.extract_s.vw_musical_release_group_ds mrg
        JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds ml ON ml.mrelg_id = mrg.mrelg_id
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp ON mp.mrel_id = ml.mrel_id
        JOIN current_dev.data.marketshare_map_icpns i ON i.mp_id = mp.mp_id
    WHERE
        mrg.compilation_type = 'Non Compilation'
        AND mrg.first_sale_date BETWEEN '2018-01-01'
        AND '2025-12-31'
    GROUP BY
        ALL
),
debut_week_qualifiers AS (
    SELECT
        a.mrelg_id,
        a.title,
        a.display_artist,
        a.genres,
        a.first_sale_date
    FROM
        luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
        JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
        JOIN target_albums a ON a.mrelg_id = s.mrelg_id
    WHERE
        s.country_code = 'US'
        AND da.week_end_date BETWEEN '2018-01-01'
        AND DATEADD(MONTH, -18, CURRENT_DATE())
        AND a.first_sale_date BETWEEN DATEADD(DAY, -7, da.week_end_date)
        AND da.week_end_date
    GROUP BY
        ALL
)

SELECT
    dq.mrelg_id,
    dq.title,
    dq.display_artist,
    dq.genres,
    dq.first_sale_date,
    da.week_end_date,
    CEIL(
        DATEDIFF(DAY, dq.first_sale_date, da.week_end_date) / 7.0
    ) AS weeks_since_release,
    COALESCE(
        SUM(
            IFF(
                s.metric_category = 'ProductSales',
                s.equivalent_quantity,
                0
            )
        ),
        0
    ) AS product_sales,
    COALESCE(
        SUM(
            IFF(
                s.metric_category = 'Streams'
                AND s.service_type = 'OnDemand',
                s.equivalent_quantity,
                0
            )
        ),
        0
    ) AS streaming_equivalent,
    COALESCE(
        SUM(
            IFF(
                s.metric_category = 'RecordingSales',
                s.equivalent_quantity,
                0
            )
        ),
        0
    ) AS song_sale_equivalent,
    SUM(s.equivalent_quantity) AS total_album_equivalents
FROM
    debut_week_qualifiers dq
    JOIN luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s ON s.mrelg_id = dq.mrelg_id
    JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
WHERE
    s.country_code = 'US'
    AND da.week_end_date >= dq.first_sale_date
    AND da.week_end_date <= DATEADD(MONTH, 18, dq.first_sale_date)
GROUP BY
    ALL
ORDER BY
    dq.display_artist,
    dq.title,
    da.week_end_date;
