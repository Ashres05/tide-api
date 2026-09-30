-- Per-title 2025 PPR revenue for one Luminate artist (artist-page list).
-- Grain is one row per MRELG. One scan: window SUM is the full artist total
-- (before QUALIFY); QUALIFY caps rows so huge catalogs cannot dump 100k+ titles.
-- {LUMINATE_ARTIST_ID} quoted id; {LIMIT} positive integer (default 500).
-- Live GET, no local cache. Not for typeahead.
SELECT
    TRIM(TO_VARCHAR(mrelg_id)) AS MRELG_ID,
    "2025_ppr_revenue" AS CATALOG_REVENUE_2025,
    SUM("2025_ppr_revenue") OVER () AS ARTIST_TOTAL
FROM US_LABELS_SANDBOX.RONAN_N.MRELG_REV_2025
WHERE TRIM(TO_VARCHAR(luminate_artist_id)) = {LUMINATE_ARTIST_ID}
QUALIFY ROW_NUMBER() OVER (
    ORDER BY "2025_ppr_revenue" DESC NULLS LAST, TRIM(TO_VARCHAR(mrelg_id))
) <= {LIMIT}
ORDER BY "2025_ppr_revenue" DESC NULLS LAST, MRELG_ID
;
