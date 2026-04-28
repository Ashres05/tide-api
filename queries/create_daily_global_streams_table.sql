-- Daily worldwide on-demand stream counts per Luminate MRELG release group,
-- cached from Snowflake's vw_daily_fact_mrelg_summary_ds. Owned by the Live
-- Revenue board only — not joined into MARKETSHARE_RELEASE_METRICS or any
-- model-feeding table, so a stale or missing row here cannot affect the
-- forecast pipeline.
CREATE TABLE IF NOT EXISTS MARKETSHARE_DAILY_GLOBAL_STREAMS (
    MRELG_ID TEXT NOT NULL,
    REPORT_DATE DATE NOT NULL,
    GLOBAL_STREAMS REAL,
    UPDATED_AT TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (MRELG_ID, REPORT_DATE)
);
