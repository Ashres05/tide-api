-- Singles search-snapshot DELTA (no day-2 streams).
-- Street/first-sale window only: {MIN_STREET_DATE} is YYYY-MM-DD.
-- IC label map is restricted to those MRELGs. Rows without a Main Artist
-- are excluded so the artist index stays keyed.
--
-- RELEASE_DATE: street date when Luminate release_date is on or before
-- 2014-01-05; otherwise first sale, then street.
WITH recent AS (
    SELECT
        s.mrelg_id,
        s.title,
        s.display_artist AS artist,
        s.artists,
        s.release_type,
        CASE
            WHEN s.release_date IS NOT NULL
                 AND s.release_date <= DATE '2014-01-05'
                THEN s.release_date
            ELSE COALESCE(s.first_sale_date, s.release_date)
        END AS release_date,
        GET(
            FILTER(genres, x -> x:CLIENT_DOMAIN = 'Billboard'),
            0
        ):MAIN_GENRE::STRING AS genre
    FROM
        luminate_prod.extract_s.vw_musical_release_group_ds s
    WHERE
        s.release_type = 'Single'
        AND COALESCE(
            CASE
                WHEN s.release_date IS NOT NULL
                     AND s.release_date <= DATE '2014-01-05'
                    THEN s.release_date
                ELSE COALESCE(s.first_sale_date, s.release_date)
            END,
            DATE '1900-01-01'
        ) >= DATE '{MIN_STREET_DATE}'
),
mrelg_labels AS (
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
        JOIN recent r ON r.mrelg_id = mrel.mrelg_id
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
mrelg_main_artist AS (
    SELECT
        r.mrelg_id,
        f.value:ARTIST_ID::STRING AS luminate_artist_id
    FROM recent r,
         LATERAL FLATTEN(input => r.artists) f
    WHERE LOWER(COALESCE(f.value:ROLE::STRING, '')) = 'main artist'
      AND f.value:ARTIST_ID IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY r.mrelg_id
        ORDER BY f.index
    ) = 1
)
SELECT
    r.mrelg_id,
    r.title,
    r.artist,
    a.luminate_artist_id,
    r.release_type,
    l.label,
    r.release_date,
    r.genre
FROM
    recent r
    JOIN mrelg_main_artist a ON a.mrelg_id = r.mrelg_id
    LEFT JOIN mrelg_labels l ON l.mrelg_id = r.mrelg_id
