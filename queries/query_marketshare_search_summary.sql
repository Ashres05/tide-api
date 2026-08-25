-- Album + EP search snapshot for artist views.
-- LUMINATE_ARTIST_ID is the first Main Artist from
-- VW_MUSICAL_RELEASE_GROUP_DS.ARTISTS (same ID as STREAMING_ROSTER_2026 /
-- ARTIST_METADATA / artist_art/{id}.jpeg).
--
-- Label (IC map) and day-2 worldwide OnDemand streams are LEFT JOINed:
-- missing distributor or a quiet day must not drop the release from the
-- artist page. Compilations (e.g. Single Artist anthologies) are included.
-- Rows without a Main Artist are excluded so the artist index stays keyed.
WITH mrelg_labels AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY mrelg_id
            ORDER BY
                percent_owned DESC
        ) AS rn
    FROM
        (
            SELECT
                DISTINCT mrel.mrelg_id,
                i.level_2_distributor,
                i.percent_owned
            FROM
                current_dev.data.marketshare_map_icpns i
                JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp ON mp.mp_id = i.mp_id
                JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel ON mrel.mrel_id = mp.mrel_id
        ) QUALIFY rn = 1
),
mrelg_base AS (
    SELECT
        s.mrelg_id,
        s.title,
        s.display_artist AS artist,
        s.artists,
        s.release_type,
        l.level_2_distributor AS label,
        COALESCE(s.first_sale_date, s.release_date) AS release_date,
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
        s.release_type IN ('Album', 'EP')
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
