-- Latest daily ``p_day`` in the US source table (incremental refresh probe).
SELECT MAX(p_day) AS max_p_day
FROM bi_sandbox.aidan_ow.luminate_market_share_revenue_pre_total
WHERE country_code = 'US';
