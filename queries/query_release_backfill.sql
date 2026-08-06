WITH mrelg_map AS (
    SELECT
        DISTINCT mrelg.mrelg_id,
        mrelg.release_type,
        i.level_2_distributor AS label_group,
        ROW_NUMBER() OVER (
            PARTITION BY mrelg.mrelg_id
            ORDER BY
                mrelg.release_date DESC
        ) AS rn
    FROM
        current_dev.data.marketshare_map_icpns i
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds m ON m.mp_id = i.mp_id
        JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds mm ON mm.mrel_id = m.mrel_id
        JOIN luminate_prod.extract_s.vw_musical_release_group_ds mrelg ON mrelg.mrelg_id = mm.mrelg_id
        AND mrelg.compilation_type != 'Compilation'
    WHERE
        i.level_2_distributor IN ({TARGET_LABELS})
        AND i.is_current = TRUE
        {RELEASE_DATE_FILTER}
        QUALIFY rn = 1
),
mrelg_metrics AS (
    SELECT
        m.mrelg_id,
        m.release_type,
        da.week_end_date AS week_ending_date,
        SUM(s.equivalent_quantity) AS album_equivalent,
        m.label_group
    FROM
        mrelg_map m
        JOIN luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s ON s.mrelg_id = m.mrelg_id
        AND s.country_code = 'US'
        AND s.report_date >= DATEADD(MONTH, -19, CURRENT_DATE()) -- Current releases cannot have data older than 19 months
        JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
    GROUP BY
        ALL
    HAVING
        album_equivalent >= 75000
)
SELECT
    s.mrelg_id,
    s.release_type,
    s.label_group AS label_name
FROM
    mrelg_metrics s
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY s.mrelg_id
        ORDER BY
            s.album_equivalent DESC
    ) = 1;
