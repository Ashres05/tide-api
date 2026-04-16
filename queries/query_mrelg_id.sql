SELECT
    mrelg_id,
    title,
    display_artist,
    COALESCE(first_sale_date, release_date) AS release_date,
    GET(
        FILTER(genres, x -> x:CLIENT_DOMAIN = 'Billboard'),
        0
    ):MAIN_GENRE::STRING AS genre
FROM luminate_prod.extract_s.vw_musical_release_group_ds
WHERE
    mrelg_id = {MRELG_ID}
