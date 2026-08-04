-- Fiscal-month AMG group share / MTD metrics from daily US streams.
-- Source: bi_sandbox.aidan_ow.luminate_market_share_revenue_pre_total
-- Fiscal calendar: Sep = month 1 … Aug = month 12; fiscal year ends in August.
WITH daily_mapped AS (
    SELECT
        m.streams,
        m.profit_center_label_group,
        m.p_day,
        CASE
            WHEN MONTH(m.p_day) >= 9 THEN YEAR(m.p_day) + 1
            ELSE YEAR(m.p_day)
        END AS fiscal_year,
        -- Fiscal month number for ordering (Sep = 1, Oct = 2, …, Aug = 12)
        CASE
            WHEN MONTH(m.p_day) >= 9 THEN MONTH(m.p_day) - 8
            ELSE MONTH(m.p_day) + 4
        END AS fiscal_month_num,
        TO_CHAR(m.p_day, 'Mon') AS base_month,
        LAST_DAY(m.p_day) AS month_end_date
    FROM bi_sandbox.aidan_ow.luminate_market_share_revenue_pre_total m
    WHERE m.country_code = 'US'
      AND m.p_day BETWEEN '2023-09-01' AND (
          SELECT MAX(p_day)
          FROM bi_sandbox.aidan_ow.luminate_market_share_revenue_pre_total
          WHERE country_code = 'US'
      )
),
monthly_agg AS (
    SELECT
        fiscal_year,
        fiscal_month_num,
        -- Append the end date if the current month is incomplete
        IFF(
            MAX(p_day) < MAX(month_end_date),
            base_month || ' - (END ' || TO_CHAR(MAX(p_day), 'Mon DD') || ')',
            base_month
        ) AS fiscal_month,
        MAX(p_day) AS p_day,
        COALESCE(SUM(IFF(profit_center_label_group = 'Atlantic Music Group', streams, 0)), 0) AS amg_streams,
        COALESCE(SUM(streams), 0) AS total_universe_streams
    FROM daily_mapped
    GROUP BY fiscal_year, fiscal_month_num, base_month
),
share_calc AS (
    SELECT
        fiscal_year,
        fiscal_month_num,
        fiscal_month,
        p_day,
        amg_streams,
        total_universe_streams,
        (DIV0(amg_streams, total_universe_streams)::FLOAT) * 100 AS amg_share
    FROM monthly_agg
),
yoy_calc AS (
    SELECT
        fiscal_year,
        fiscal_month_num,
        fiscal_month,
        p_day,
        amg_streams,
        total_universe_streams,
        amg_share,
        LAG(amg_share, 1) OVER (
            PARTITION BY fiscal_month_num
            ORDER BY fiscal_year
        ) AS prev_yoy_share,
        LAG(total_universe_streams, 1) OVER (
            PARTITION BY fiscal_month_num
            ORDER BY fiscal_year
        ) AS prev_yoy_market_streams
    FROM share_calc
),
growth_calc AS (
    SELECT
        fiscal_year,
        fiscal_month_num,
        fiscal_month,
        p_day,
        amg_streams,
        total_universe_streams,
        amg_share,
        (DIV0((amg_share - prev_yoy_share), prev_yoy_share)::FLOAT) * 100 AS share_growth_yoy,
        (DIV0((total_universe_streams - prev_yoy_market_streams), prev_yoy_market_streams)::FLOAT) * 100 AS market_growth_yoy
    FROM yoy_calc
)
SELECT
    fiscal_year,
    fiscal_month,
    amg_streams,
    total_universe_streams,
    amg_share,
    share_growth_yoy,
    market_growth_yoy,
    (share_growth_yoy + market_growth_yoy) AS total_growth_yoy,
    p_day
FROM growth_calc
ORDER BY
    fiscal_year,
    fiscal_month_num;
