-- Canonical artist names from Luminate (one row per artist_id).
-- Used by refresh_marketshare_search_artists() to overlay display names
-- on the local billing-line rollup. Fall back to the rollup when a name
-- is missing here.
SELECT
    TRIM(artist_id) AS artist_id,
    TRIM(artist_name) AS artist_name
FROM luminate_prod.extract_s.vw_artist_ds
WHERE artist_id IS NOT NULL
  AND TRIM(artist_id) != ''
  AND artist_name IS NOT NULL
  AND TRIM(artist_name) != ''
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY TRIM(artist_id)
    ORDER BY modified_at DESC NULLS LAST, artist_name
) = 1;
