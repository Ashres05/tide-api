-- Singles search snapshot for artist views.
-- LUMINATE_ARTIST_ID is the first Main Artist from
-- VW_MUSICAL_RELEASE_GROUP_DS.ARTISTS (same ID as STREAMING_ROSTER_2026 /
-- ARTIST_METADATA / artist_art/{id}.jpeg).
--
-- Label (IC map) and day-2 worldwide OnDemand streams are LEFT JOINed:
-- missing distributor or a quiet day must not drop the release from the
-- artist page. Rows without a Main Artist are excluded.
--
-- Label pick matches the roster: current ICPNs only. IGA/CMG at level 3;
-- everything else stays level 2. Do not alias Interscope-Capitol (only L3
-- tells IGA from CMG). Do not promote imprint L3 (e.g. Def Jam) into
-- LABEL_NAME — that stays on Republic / UME as L2.
-- Prefer a mapped label when several ICPNs are tied at 100% owned (null L2
-- rows must not win over Republic).
--
-- RELEASE_DATE: street date when Luminate release_date is on or before
-- 2014-01-05; otherwise first sale, then street.
WITH mrelg_labels AS (
    SELECT
        mrel.mrelg_id,
        CASE
            WHEN i.level_3_distributor IN ('IGA', 'CMG') THEN i.level_3_distributor
            WHEN i.level_2_distributor = 'Interscope/Geffen/A&M' THEN 'IGA'
            ELSE i.level_2_distributor
        END AS label
    FROM
        current_dev.data.marketshare_map_icpns i
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp ON mp.mp_id = i.mp_id
        JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel ON mrel.mrel_id = mp.mrel_id
    WHERE
        i.is_current = TRUE
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY mrel.mrelg_id
        ORDER BY
            CASE
                WHEN i.level_3_distributor IN ('IGA', 'CMG') THEN 0
                WHEN i.level_2_distributor IS NOT NULL THEN 0
                ELSE 1
            END,
            i.percent_owned DESC NULLS LAST
    ) = 1
),
mrelg_base AS (
    SELECT
        s.mrelg_id,
        s.title,
        s.display_artist AS artist,
        s.artists,
        s.release_type,
        l.label AS label,
        CASE
            WHEN s.release_date IS NOT NULL
                 AND s.release_date <= DATE '2014-01-05'
                THEN s.release_date
            ELSE COALESCE(s.first_sale_date, s.release_date)
        END AS release_date,
        GET(
            FILTER(genres, x -> x:CLIENT_DOMAIN = 'Billboard'),
            0
        ):MAIN_GENRE::STRING AS genre,
        COALESCE(SUM(ss.quantity), 0) AS daily_streams
    FROM
        luminate_prod.extract_s.vw_musical_release_group_ds s
        LEFT JOIN luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds ss ON ss.mrelg_id = s.mrelg_id
        AND ss.country_code = 'AA'
        AND ss.report_date = DATEADD(DAY, -2, CURRENT_DATE())
        AND ss.metric_category = 'Streams'
        AND ss.service_type = 'OnDemand'
        LEFT JOIN mrelg_labels l ON l.mrelg_id = s.mrelg_id
    WHERE
        s.release_type = 'Single'
    GROUP BY
        ALL
),
mrelg_main_artist AS (
    SELECT
        m.mrelg_id,
        f.value:ARTIST_ID::STRING AS luminate_artist_id
    FROM mrelg_base m,
         LATERAL FLATTEN(input => m.artists) f
    WHERE LOWER(COALESCE(f.value:ROLE::STRING, '')) = 'main artist'
      AND f.value:ARTIST_ID IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY m.mrelg_id
        ORDER BY f.index
    ) = 1
)
SELECT
    m.mrelg_id,
    m.title,
    m.artist,
    a.luminate_artist_id,
    m.release_type,
    m.label,
    m.release_date,
    m.genre,
    m.daily_streams
FROM
    mrelg_base m
    JOIN mrelg_main_artist a ON a.mrelg_id = m.mrelg_id
