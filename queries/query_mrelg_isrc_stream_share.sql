-- On-demand US OnDemand stream share by ISRC for one MRELG.
-- Not persisted — worksheet / GET handler only.
--
-- Fast path: resolve the MRELG to a small MR_ID set first, aggregate daily
-- US OnDemand streams at MR x year, then apply IC percent_owned. That keeps
-- VW_DAILY_FACT_MR_SUMMARY_DS from exploding across ISRC and song maps.
-- Duplicate ISRCs from split attribution are expected; us_stream_share is
-- us_streams * (percent_owned / 100).
--
-- Handler substitutes: MRELG_ID (quoted literal), MR_ID_FILTER (optional AND
-- on mm.mr_id, empty when unused), DATE_PREDICATE (on mrd.report_date).
-- Do not put those token names in braces inside comments.
WITH mrelg_mrs AS (
    SELECT DISTINCT
        mm.mr_id
    FROM
        luminate_prod.extract_s.vw_mrel_mrelg_map_ds mmm
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mmd
            ON mmd.mrel_id = mmm.mrel_id
        JOIN luminate_prod.extract_s.vw_mr_mp_map_ds mm
            ON mm.mp_id = mmd.mp_id
    WHERE
        mmm.mrelg_id = {MRELG_ID}
        {MR_ID_FILTER}
),
us_streams AS (
    SELECT
        mrd.mr_id,
        YEAR(mrd.report_date) AS yearid,
        SUM(mrd.quantity) AS us_streams
    FROM
        mrelg_mrs mrs
        JOIN luminate_prod.extract_s.vw_daily_fact_mr_summary_ds mrd
            ON mrd.mr_id = mrs.mr_id
    WHERE
        mrd.country_code = 'US'
        AND mrd.metric_category = 'Streams'
        AND mrd.service_type = 'OnDemand'
        {DATE_PREDICATE}
    GROUP BY
        mrd.mr_id,
        YEAR(mrd.report_date)
),
song AS (
    SELECT
        v2.mr_id,
        s.title,
        s.display_artist
    FROM
        mrelg_mrs mrs
        JOIN luminate_prod.extract_s.vw_mr_song_map_v2_ds v2
            ON v2.mr_id = mrs.mr_id
        JOIN luminate_prod.extract_s.vw_song_v2_ds s
            ON s.song_id = v2.song_id
    QUALIFY ROW_NUMBER() OVER (PARTITION BY v2.mr_id ORDER BY s.song_id) = 1
)
SELECT
    mmi.mr_id,
    mmi.isrc,
    song.title,
    song.display_artist,
    mmi.release_date,
    mmi.is_current,
    mmi.country_code,
    mmi.level_1_distributor,
    mmi.level_2_distributor,
    mmi.level_3_distributor,
    mmi.percent_owned,
    us_streams.yearid,
    us_streams.us_streams * (mmi.percent_owned / 100) AS us_stream_share
FROM
    mrelg_mrs mrs
    JOIN current_dev.data.marketshare_map_isrcs mmi
        ON mmi.mr_id = mrs.mr_id
        AND mmi.country_code = 'US'
    JOIN us_streams
        ON us_streams.mr_id = mmi.mr_id
    LEFT JOIN song
        ON song.mr_id = mmi.mr_id
ORDER BY
    yearid DESC,
    level_1_distributor,
    us_stream_share DESC
;
