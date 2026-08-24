-- Release backfill roster: one row per mrelg_id that cleared 75k AE.
-- label_name is level_2 for most TARGET_LABELS; IGA / CMG use level_3.
-- Legacy level_2 Interscope/Geffen/A&M maps to IGA. Do not alias
-- Interscope-Capitol (ambiguous IGA vs CMG).
-- {TARGET_LABELS} is injected from model.marketshare_labels.TARGET_LABELS.
-- {RELEASE_DATE_FILTER} is optional (full vs incremental backfill).
-- Excludes Various Artists (same artist set as query_streaming_roster_ytd.sql).
-- Fact window is WEEK -78 to match archetype NUM_WEEKS / FE pull−78w logic
-- (replaces MONTH -19 ≈ 82 weeks).
WITH mrelg_map AS (
    SELECT
        DISTINCT mrelg.mrelg_id,
        mrelg.release_type,
        CASE
            WHEN i.level_3_distributor IN ('IGA', 'CMG') THEN i.level_3_distributor
            WHEN i.level_2_distributor = 'Interscope/Geffen/A&M' THEN 'IGA'
            ELSE i.level_2_distributor
        END AS label_group,
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
        AND mrelg.display_artist NOT IN ('VARIOUS', 'VARIOUS ARTISTS', 'Various Artists', 'various')
    WHERE
        (
            i.level_2_distributor IN ({TARGET_LABELS})
            OR i.level_2_distributor = 'Interscope/Geffen/A&M'
            OR i.level_3_distributor IN ('IGA', 'CMG')
        )
        AND CASE
            WHEN i.level_3_distributor IN ('IGA', 'CMG') THEN i.level_3_distributor
            WHEN i.level_2_distributor = 'Interscope/Geffen/A&M' THEN 'IGA'
            ELSE i.level_2_distributor
        END IN ({TARGET_LABELS})
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
        AND s.report_date >= DATEADD(WEEK, -78, CURRENT_DATE())
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
