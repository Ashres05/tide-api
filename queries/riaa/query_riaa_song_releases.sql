SELECT
    riaa_song_id,
    title,
    artist
FROM current_dev.data.riaa_songs
WHERE
    NOT flagged_as_invalid;
