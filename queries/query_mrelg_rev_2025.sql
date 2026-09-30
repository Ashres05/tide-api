-- 2025 PPR revenue for one title (one row per MRELG in MRELG_REV_2025).
-- {MRELG_ID} is replaced at runtime with a quoted id (same helper as
-- query_release_global_streaming.sql). Live GET, no local cache.
SELECT
    TRIM(TO_VARCHAR(mrelg_id)) AS MRELG_ID,
    TRIM(TO_VARCHAR(luminate_artist_id)) AS LUMINATE_ARTIST_ID,
    "2025_ppr_revenue" AS CATALOG_REVENUE_2025
FROM US_LABELS_SANDBOX.RONAN_N.MRELG_REV_2025
WHERE TRIM(TO_VARCHAR(mrelg_id)) = {MRELG_ID}
QUALIFY ROW_NUMBER() OVER (
    ORDER BY "2025_ppr_revenue" DESC NULLS LAST
) = 1
;
