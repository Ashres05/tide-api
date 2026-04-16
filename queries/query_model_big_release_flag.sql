WITH current_date AS (
    SELECT
        DATEADD(MONTH, -18, CURRENT_DATE()) AS cur_date
),

target_albums AS (
    SELECT
        m.mrelg_id,
        m.first_sale_date,
        prod.display_artist
    FROM
        luminate_prod.extract_s.vw_musical_release_group_ds m
        JOIN luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel ON mrel.mrelg_id = m.mrelg_id
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp ON mp.mrel_id = mrel.mrel_id
        JOIN luminate_prod.extract_s.vw_musical_product_ds prod ON prod.mp_id = mp.mp_id
    WHERE
        m.compilation_type = 'Non Compilation'
        AND LOWER(prod.display_artist) NOT LIKE '%various%' -- Pulled back slightly just to ensure we catch late-October releases spilling into Nov 7
        AND m.first_sale_date >= (SELECT cur_date FROM current_date) QUALIFY ROW_NUMBER() OVER (
            PARTITION BY m.mrelg_id
            ORDER BY
                prod.mp_id
        ) = 1
),
debut_week_performance AS (
    -- Calculate the total weekly sales, strictly for the album's debut week
    SELECT
        da.week_end_date,
        a.mrelg_id,
        SUM(s.equivalent_quantity) AS equivalent_quantity // should sum week
    FROM
        luminate_prod.extract_s.vw_daily_fact_mrelg_summary_ds s
        JOIN luminate_prod.extract_s.vw_date_ds da ON da.datename = s.report_date
        JOIN target_albums a ON a.mrelg_id = s.mrelg_id
    WHERE
        s.country_code = 'US'
        AND da.week_end_date >= (SELECT cur_date FROM current_date)
        AND a.first_sale_date BETWEEN DATEADD(DAY, -7, da.week_end_date) AND da.week_end_date
    GROUP BY
        da.week_end_date,
        a.mrelg_id
),
mrelg_to_distributor AS (
    SELECT
        mrel.mrelg_id,
        i.level_2_distributor
    FROM
        luminate_prod.extract_s.vw_mrel_mrelg_map_ds mrel
        JOIN luminate_prod.extract_s.vw_mp_mrel_map_ds mp ON mp.mrel_id = mrel.mrel_id
        JOIN CURRENT_DEV.DATA.MARKETSHARE_MAP_ICPNS i ON i.mp_id = mp.mp_id
        // need new releases!!! feb onward
        JOIN target_albums a ON a.mrelg_id = mrel.mrelg_id
    WHERE
        i.country_code = 'US' QUALIFY ROW_NUMBER() OVER (
            PARTITION BY mrel.mrelg_id
            ORDER BY
                i.mp_id
        ) = 1
)
SELECT
    dwp.week_end_date,
    MAX(
        IFF(
            d.level_2_distributor IN ('Atlantic Music Group', 'Atlantic Records')
            AND dwp.equivalent_quantity >= 75000,
            -- changed from 100k
            1,
            0
        )
    ) AS Big_Release_Atlantic,
    MAX(
        IFF(
            d.level_2_distributor = 'Interscope/Geffen/A&M'
            AND dwp.equivalent_quantity >= 75000,
            -- changed from 100k
            1,
            0
        )
    ) AS Big_Release_Interscope
FROM
    debut_week_performance dwp
    LEFT JOIN mrelg_to_distributor d ON dwp.mrelg_id = d.mrelg_id
GROUP BY
    dwp.week_end_date
ORDER BY
    dwp.week_end_date DESC;
