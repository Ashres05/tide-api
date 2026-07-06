SELECT
    cr.riaa_album_id AS release_id,
    'ALBUM' AS release_type,
    a.title,
    a.artist,
    cr.certification,
    cr.certification_date
FROM current_dev.data.riaa_album_certifications_resolved cr
    JOIN current_dev.data.riaa_albums a ON a.riaa_album_id = cr.riaa_album_id
WHERE
    cr.country_code = 'US'
    AND cr.certification != 'NOT FOUND'

UNION ALL

SELECT
    cr.riaa_song_id AS release_id,
    'SINGLE' AS release_type,
    a.title,
    a.artist,
    cr.certification,
    cr.certification_date
FROM current_dev.data.riaa_song_certifications_resolved cr
    JOIN current_dev.data.riaa_songs a ON a.riaa_song_id = cr.riaa_song_id
WHERE
    cr.country_code = 'US'
    AND cr.certification != 'NOT FOUND';
