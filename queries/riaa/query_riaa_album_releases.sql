SELECT
    riaa_album_id,
    title,
    artist
FROM current_dev.data.riaa_albums
WHERE
    NOT flagged_as_invalid;
