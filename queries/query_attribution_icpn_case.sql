-- ICPN owner CASE vs 75k / streaming roster.
-- Confirms Atlantic Records rolls to Atlantic Music Group at level_2,
-- and how 10K / 300 appear.
SELECT
    CASE
        WHEN i.level_3_distributor IN ('IGA', 'CMG') THEN i.level_3_distributor
        WHEN i.level_2_distributor = 'Interscope/Geffen/A&M' THEN 'IGA'
        WHEN i.level_2_distributor IN ('Atlantic Music Group', 'Atlantic Records') THEN 'Atlantic Music Group'
        ELSE i.level_2_distributor
    END AS owner_label,
    i.level_2_distributor,
    i.level_3_distributor,
    COUNT(*) AS n_icpns
FROM current_dev.data.marketshare_map_icpns i
WHERE i.country_code = 'US'
  AND i.is_current = TRUE
  AND (
        LOWER(COALESCE(i.level_2_distributor, '')) ILIKE '%atlantic%'
     OR LOWER(COALESCE(i.level_3_distributor, '')) ILIKE '%atlantic%'
     OR LOWER(COALESCE(i.level_2_distributor, '')) ILIKE '%10k%'
     OR LOWER(COALESCE(i.level_3_distributor, '')) ILIKE '%10k%'
     OR LOWER(COALESCE(i.level_2_distributor, '')) ILIKE '%300%'
     OR LOWER(COALESCE(i.level_3_distributor, '')) ILIKE '%300%'
  )
GROUP BY 1, 2, 3
ORDER BY n_icpns DESC
;