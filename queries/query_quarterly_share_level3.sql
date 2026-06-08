-- Fiscal-quarter level-3 profit-center share metrics from daily US streams.
-- Source: bi_sandbox.aidan_ow.luminate_market_share_revenue_pre_total
WITH daily_mapped AS (
    SELECT
        m.streams,
        m.profit_center_label,
        m.p_day,
        CASE
            WHEN MONTH(m.p_day) >= 9 THEN YEAR(m.p_day) + 1
            ELSE YEAR(m.p_day)
        END AS fiscal_year,
        CASE
            WHEN MONTH(m.p_day) IN (9, 10, 11) THEN 'Q1 (Sep-Nov)'
            WHEN MONTH(m.p_day) IN (12, 1, 2) THEN 'Q2 (Dec-Feb)'
            WHEN MONTH(m.p_day) IN (3, 4, 5) THEN 'Q3 (Mar-May)'
            WHEN MONTH(m.p_day) IN (6, 7, 8) THEN 'Q4 (Jun-Aug)'
        END AS base_quarter,
        CASE
            WHEN MONTH(m.p_day) IN (9, 10, 11) THEN TO_DATE(YEAR(m.p_day) || '-11-30')
            WHEN MONTH(m.p_day) = 12 THEN LAST_DAY(TO_DATE(YEAR(m.p_day) || '-02-01'))
            WHEN MONTH(m.p_day) IN (1, 2) THEN LAST_DAY(TO_DATE(YEAR(m.p_day) || '-02-01'))
            WHEN MONTH(m.p_day) IN (3, 4, 5) THEN TO_DATE(YEAR(m.p_day) || '-05-31')
            WHEN MONTH(m.p_day) IN (6, 7, 8) THEN TO_DATE(YEAR(m.p_day) || '-08-31')
        END AS quarter_end_date
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
        base_quarter,
        IFF(
            MAX(p_day) < MAX(quarter_end_date),
            base_quarter || ' - (END ' || TO_CHAR(MAX(p_day), 'Mon DD') || ')',
            base_quarter
        ) AS fiscal_quarter,
        MAX(p_day) AS p_day,
        COALESCE(SUM(streams), 0) AS total_universe_streams
    FROM daily_mapped
    GROUP BY fiscal_year, base_quarter
),
label_agg AS (
    SELECT
        fiscal_year,
        base_quarter,
        profit_center_label,
        COALESCE(SUM(streams), 0) AS label_streams
    FROM daily_mapped
    WHERE profit_center_label IN (
        '10K Holdings Project',
        '300 Entertainment',
        'Rhino',
        'Atlantic'
    )
    GROUP BY fiscal_year, base_quarter, profit_center_label
),
share_calc AS (
    SELECT
        l.fiscal_year,
        m.fiscal_quarter,
        m.p_day,
        l.profit_center_label,
        l.label_streams,
        m.total_universe_streams,
        (DIV0(l.label_streams, m.total_universe_streams)::FLOAT) * 100 AS label_share
    FROM label_agg l
    JOIN market_agg m
        ON l.fiscal_year = m.fiscal_year
       AND l.base_quarter = m.base_quarter
),
yoy_calc AS (
    SELECT
        fiscal_year,
        fiscal_quarter,
        p_day,
        profit_center_label,
        label_streams,
        total_universe_streams,
        label_share,
        LAG(label_share, 1) OVER (
            PARTITION BY profit_center_label, REGEXP_SUBSTR(fiscal_quarter, 'Q[1-4]')
            ORDER BY fiscal_year
        ) AS prev_yoy_share,
        LAG(label_streams, 1) OVER (
            PARTITION BY profit_center_label, REGEXP_SUBSTR(fiscal_quarter, 'Q[1-4]')
            ORDER BY fiscal_year
        ) AS prev_yoy_label_streams,
        LAG(total_universe_streams, 1) OVER (
            PARTITION BY profit_center_label, REGEXP_SUBSTR(fiscal_quarter, 'Q[1-4]')
            ORDER BY fiscal_year
        ) AS prev_yoy_market_streams
    FROM share_calc
)
SELECT
    fiscal_year,
    fiscal_quarter,
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
    p_day,
    fiscal_year,
    fiscal_quarter;
