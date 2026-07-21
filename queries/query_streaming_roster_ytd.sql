-- YTD streaming revenue roster (Album, Single, EP) with US week AE >= 10k.
-- Used by backfill_streaming_roster() -> STREAMING_ROSTER_2026.
-- Per-MRELG weekly worldwide streams use query_release_global_streaming.sql instead.
--
-- LUMINATE_ARTIST_ID is the first Main Artist from VW_MUSICAL_RELEASE_GROUP_DS.ARTISTS
-- (same ID as CURRENT_DEV.DATA.ARTIST_METADATA.LUMINATE_ARTIST_ID / artist_art/{id}.jpeg).
WITH mrelg_map AS (
    SELECT
        mrelg.mrelg_id,
        mrelg.release_type,
        mrelg.title,
        mrelg.display_artist AS artist,
        mrelg.artists,
        i.level_2_distributor AS label_group,
        i.level_1_distributor AS parent_group,
        COALESCE(mrelg.first_sale_date, mrelg.release_date) AS release_date,
        ROW_NUMBER() OVER (
            PARTITION BY mrelg.mrelg_id
            ORDER BY mrelg.release_date DESC
        ) AS rn
    FROM
        current_dev.data.marketshare_map_icpns i
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds m ON m.mp_id = i.mp_id
        JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds mm ON mm.mrel_id = m.mrel_id
        JOIN luminate_prod.extract_s.vw_musical_release_group_ds mrelg ON mrelg.mrelg_id = mm.mrelg_id
        AND mrelg.compilation_type != 'Compilation'
        AND mrelg.release_type IN ('Album', 'Single', 'EP')
        AND mrelg.display_artist NOT IN ('VARIOUS', 'VARIOUS ARTISTS', 'Various Artists')
        AND i.level_2_distributor IS NOT NULL
    WHERE
        i.is_current = TRUE
        {RELEASE_DATE_FILTER}
        QUALIFY rn = 1
),
mrelg_main_artist AS (
    SELECT
        m.mrelg_id,
        f.value:ARTIST_ID::STRING AS luminate_artist_id
    FROM mrelg_map m,
         LATERAL FLATTEN(input => m.artists) f
    WHERE LOWER(COALESCE(f.value:ROLE::STRING, '')) = 'main artist'
      AND f.value:ARTIST_ID IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY m.mrelg_id
        ORDER BY f.index
    ) = 1
),
mrelg_metrics AS (
    SELECT
        m.mrelg_id,
        m.release_type,
        m.title,
        m.artist,
        a.luminate_artist_id,
        m.label_group,
        m.parent_group,
        m.release_date,
        da.week_end_date AS week_ending_date,
        SUM(s.equivalent_quantity) AS album_equivalent
    FROM
        mrelg_map m
        LEFT JOIN mrelg_main_artist a ON a.mrelg_id = m.mrelg_id
        JOIN luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s ON s.mrelg_id = m.mrelg_id
        AND s.country_code = 'US'
        AND s.report_date >= DATEADD(MONTH, -19, CURRENT_DATE())
        JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
    GROUP BY
        ALL
    HAVING
        album_equivalent >= 10000
)
SELECT
    mrelg_id,
    release_type,
    label_group AS label_name,
    parent_group,
    title,
    artist,
    luminate_artist_id,
    release_date
FROM
    mrelg_metrics
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY mrelg_id
    ORDER BY album_equivalent DESC
) = 1;
