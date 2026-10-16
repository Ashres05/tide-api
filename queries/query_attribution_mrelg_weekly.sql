-- All-owner weekly US AE by MRELG (no 75k gate, no TARGET_LABELS filter).
-- Floor {MIN_AE} (production 2500). Window: {MIN_WEEK_END_DATE} .. last
-- complete week. Owner = majority current US ICPN (percent_owned).
-- Atlantic Records / 10K / 300 CASE to Atlantic Music Group (level_3 kept).

WITH weekly_ae AS (
    SELECT
        da.week_end_date,
        m.mrelg_id,
        m.title,
        m.display_artist,
        m.first_sale_date,
        SUM(s.equivalent_quantity) AS ae
    FROM luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
    JOIN luminate_prod.extract_s.vw_musical_release_group_ds m
      ON m.mrelg_id = s.mrelg_id
    JOIN luminate_prod.extract_s.vw_date_ds da
      ON da.datename = s.report_date
    WHERE s.country_code = 'US'
      AND m.compilation_type = 'Non Compilation'
      AND s.report_date >= m.first_sale_date
      AND da.week_end_date >= '{MIN_WEEK_END_DATE}'
      AND da.week_end_date <= DATEADD(DAY, -1, CURRENT_DATE())
    GROUP BY
        da.week_end_date,
        m.mrelg_id,
        m.title,
        m.display_artist,
        m.first_sale_date
    HAVING SUM(s.equivalent_quantity) >= {MIN_AE}
),
owner_map AS (
    SELECT
        mrel.mrelg_id,
        CASE
            WHEN i.level_3_distributor IN ('IGA', 'CMG') THEN i.level_3_distributor
            WHEN i.level_2_distributor = 'Interscope/Geffen/A&M' THEN 'IGA'
            WHEN i.level_2_distributor IN ('Atlantic Music Group', 'Atlantic Records') THEN 'Atlantic Music Group'
            ELSE i.level_2_distributor
        END AS owner_label,
        i.level_2_distributor,
        i.level_3_distributor,
        i.percent_owned
    FROM luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel
    JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp
      ON mp.mrel_id = mrel.mrel_id
    JOIN current_dev.data.marketshare_map_icpns i
      ON i.mp_id = mp.mp_id
    WHERE i.country_code = 'US'
      AND i.is_current = TRUE
      AND EXISTS (
          SELECT 1 FROM weekly_ae w WHERE w.mrelg_id = mrel.mrelg_id
      )
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY mrel.mrelg_id
        ORDER BY i.percent_owned DESC NULLS LAST, i.mp_id
    ) = 1
)
SELECT
    w.week_end_date AS WEEK_END_DATE,
    w.mrelg_id AS MRELG_ID,
    w.title AS TITLE,
    w.display_artist AS ARTIST,
    w.first_sale_date AS FIRST_SALE_DATE,
    w.ae AS AE,
    o.owner_label AS OWNER_LABEL,
    o.level_2_distributor AS LEVEL_2_DISTRIBUTOR,
    o.level_3_distributor AS LEVEL_3_DISTRIBUTOR
FROM weekly_ae w
LEFT JOIN owner_map o
  ON o.mrelg_id = w.mrelg_id
ORDER BY w.week_end_date, w.ae DESC
;