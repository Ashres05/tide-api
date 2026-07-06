from snowflake_conn import get_snowflake_connection, load_sql
import pandas as pd

QUERY_ELIGIBLE_CERTIFICATIONS = "riaa/query_eligible_certifications.sql"
QUERY_OFFICIAL_CERTIFICATIONS = "riaa/query_official_certifications.sql"
QUERY_RIAA_ALBUM_RELEASES = "riaa/query_riaa_album_releases.sql"
QUERY_RIAA_SONG_RELEASES = "riaa/query_riaa_song_releases.sql"


def _df_to_json(df: pd.DataFrame) -> str:
    """
    Converts a pandas DataFrame to a JSON string.
    """
    return df.to_json(orient='records')

def get_eligible_certifications() -> str:
    """
    Returns a JSON string of eligible certifications from the RIAA.
    """
    query = load_sql(QUERY_ELIGIBLE_CERTIFICATIONS)
    with get_snowflake_connection() as sf:
        return _df_to_json(sf.query(query))

def get_official_certifications():
    """
    Returns a JSON string of official certifications from the RIAA.
    """
    query = load_sql(QUERY_OFFICIAL_CERTIFICATIONS)
    with get_snowflake_connection() as sf:
        return _df_to_json(sf.query(query))
