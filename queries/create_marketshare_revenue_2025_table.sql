-- 2025 catalog revenue per Luminate MRELG release group, loaded from
-- s3://parquetgarage/model/data/2025_revenue_catalog.csv. Live Revenue
-- board only — not joined into MARKETSHARE_RELEASE_METRICS or any
-- model-feeding table, so a stale or missing row here cannot affect the
-- forecast pipeline.
CREATE TABLE IF NOT EXISTS MARKETSHARE_REVENUE_2025 (
    MRELG_ID TEXT PRIMARY KEY,
    REVENUE_2025 REAL,
    UPDATED_AT TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
