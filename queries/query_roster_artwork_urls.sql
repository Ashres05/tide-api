-- Artwork URLs for streaming roster mrelg_ids via Apple Feed UPC matching.
-- Called by scripts/fetch_roster_artwork.py.
-- {MRELG_ID_LIST} is replaced at runtime with a comma-separated quoted list.
SELECT
    mrelg_map.mrelg_id,
    MAX(apple.artworks:key_value[0].value[0].url::STRING) AS artwork_url
FROM luminate_prod.extract_s.vw_musical_product_ds mp_base
JOIN LATERAL FLATTEN(input => mp_base.external_ids:ICPN) icpn_flat
JOIN current_dev.data.apple_feed_album_data apple
    ON LTRIM(TRIM(apple.upc::STRING), '0') = LTRIM(TRIM(icpn_flat.value::STRING, '" '), '0')
JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp_map
    ON mp_map.mp_id = mp_base.mp_id
JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrelg_map
    ON mrelg_map.mrel_id = mp_map.mrel_id
WHERE mrelg_map.mrelg_id IN ({MRELG_ID_LIST})
GROUP BY 1
