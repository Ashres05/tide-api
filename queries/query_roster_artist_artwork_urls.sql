-- Artist profile-image URLs for streaming roster MRELG IDs.
-- Called by scripts/fetch_roster_artist_artwork.py.
-- {MRELG_ID_LIST} is replaced at runtime with a comma-separated quoted list.
--
-- VW_MUSICAL_RELEASE_GROUP_DS.ARTISTS is an array such as:
-- [{"ARTIST_ID":"AR...","ROLE":"Main Artist"}].
-- ARTIST_ID is the same identifier as ARTIST_METADATA.LUMINATE_ARTIST_ID.
WITH roster_artists AS (
    SELECT DISTINCT
        artist.value:ARTIST_ID::STRING AS luminate_artist_id
    FROM LUMINATE_PROD.EXTRACT_S.VW_MUSICAL_RELEASE_GROUP_DS mrelg,
         LATERAL FLATTEN(input => mrelg.ARTISTS) artist
    WHERE mrelg.MRELG_ID IN ({MRELG_ID_LIST})
      AND artist.value:ARTIST_ID::STRING IS NOT NULL
      AND LOWER(COALESCE(artist.value:ROLE::STRING, '')) = 'main artist'
)
SELECT
    roster_artists.luminate_artist_id,
    metadata.artist_name,
    metadata.profile_image
FROM roster_artists
JOIN CURRENT_DEV.DATA.ARTIST_METADATA metadata
  ON metadata.LUMINATE_ARTIST_ID = roster_artists.luminate_artist_id
WHERE metadata.PROFILE_IMAGE IS NOT NULL
  AND TRIM(metadata.PROFILE_IMAGE) != ''
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY roster_artists.luminate_artist_id
    ORDER BY metadata.PROFILE_IMAGE DESC
) = 1
ORDER BY metadata.ARTIST_NAME, roster_artists.luminate_artist_id;
