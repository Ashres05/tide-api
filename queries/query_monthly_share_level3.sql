-- Fiscal-month level-3 profit-center share metrics from daily US streams.
-- Source: bi_sandbox.aidan_ow.luminate_market_share_revenue_pre_total
-- Fiscal calendar: Sep = month 1 … Aug = month 12; fiscal year ends in August.
WITH daily_mapped AS (
    SELECT
        m.streams,
        m.profit_center_label,
        m.p_day,
        CASE
            WHEN MONTH(m.p_day) >= 9 THEN YEAR(m.p_day) + 1
            ELSE YEAR(m.p_day)
        END AS fiscal_year,
        -- Fiscal month number (1 = Sep, 12 = Aug) for ordering and YoY matching
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
market_agg AS (
    SELECT
        fiscal_year,
        fiscal_month_num,
        base_month,
        IFF(
            MAX(p_day) < MAX(month_end_date),
            base_month || ' - (END ' || TO_CHAR(MAX(p_day), 'Mon DD') || ')',
            base_month
        ) AS fiscal_month,
        MAX(p_day) AS p_day,
        COALESCE(SUM(streams), 0) AS total_universe_streams
    FROM daily_mapped
    GROUP BY fiscal_year, fiscal_month_num, base_month
),
label_agg AS (
    SELECT
        fiscal_year,
        fiscal_month_num,
        base_month,
        profit_center_label,
        COALESCE(SUM(streams), 0) AS label_streams
    FROM daily_mapped
    WHERE profit_center_label IN (
        '10K Holdings Project',
        '300 Entertainment',
        'Rhino',
        'Atlantic'
    )
    GROUP BY fiscal_year, fiscal_month_num, base_month, profit_center_label
),
share_calc AS (
    SELECT
        l.fiscal_year,
        l.fiscal_month_num,
        m.fiscal_month,
        m.p_day,
        l.profit_center_label,
        l.label_streams,
        m.total_universe_streams,
        (DIV0(l.label_streams, m.total_universe_streams)::FLOAT) * 100 AS label_share
    FROM label_agg l
    JOIN market_agg m
        ON l.fiscal_year = m.fiscal_year
       AND l.fiscal_month_num = m.fiscal_month_num
),
yoy_calc AS (
    SELECT
        fiscal_year,
        fiscal_month_num,
        fiscal_month,
        p_day,
        profit_center_label,
        label_streams,
        total_universe_streams,
        label_share,
        LAG(label_share, 1) OVER (
            PARTITION BY profit_center_label, fiscal_month_num
            ORDER BY fiscal_year
        ) AS prev_yoy_share,
        LAG(label_streams, 1) OVER (
            PARTITION BY profit_center_label, fiscal_month_num
            ORDER BY fiscal_year
        ) AS prev_yoy_label_streams,
        LAG(total_universe_streams, 1) OVER (
            PARTITION BY profit_center_label, fiscal_month_num
            ORDER BY fiscal_year
        ) AS prev_yoy_market_streams
    FROM share_calc
)
SELECT
    fiscal_year,
    fiscal_month,
    profit_center_label,
    label_streams,
    total_universe_streams,
    label_share,
    ((label_share - prev_yoy_share) * 100)::FLOAT AS share_change_bps,
    (DIV0((label_streams - prev_yoy_label_streams), prev_yoy_label_streams)::FLOAT) * 100 AS label_stream_growth_yoy,
    (DIV0((total_universe_streams - prev_yoy_market_streams), prev_yoy_market_streams)::FLOAT) * 100 AS market_growth_yoy,
    p_day
FROM yoy_calc
ORDER BY
    profit_center_label,
    fiscal_year,
    fiscal_month_num;
