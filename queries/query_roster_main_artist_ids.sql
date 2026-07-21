-- Main Luminate artist ID for a list of MRELG IDs.
-- Used by backfill_streaming_roster() to fill STREAMING_ROSTER_2026.LUMINATE_ARTIST_ID
-- for rows that already exist but were inserted before the artist-id column.
-- {MRELG_ID_LIST} is replaced at runtime with a comma-separated quoted list.
SELECT
    m.mrelg_id,
    f.value:ARTIST_ID::STRING AS luminate_artist_id
FROM LUMINATE_PROD.EXTRACT_S.VW_MUSICAL_RELEASE_GROUP_DS m,
     LATERAL FLATTEN(input => m.ARTISTS) f
WHERE m.MRELG_ID IN ({MRELG_ID_LIST})
  AND LOWER(COALESCE(f.value:ROLE::STRING, '')) = 'main artist'
  AND f.value:ARTIST_ID IS NOT NULL
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY m.mrelg_id
    ORDER BY f.index
) = 1;
