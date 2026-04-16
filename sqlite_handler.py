from snowflake_conn import get_snowflake_connection, load_sql
import sqlite3
import pandas as pd

# TODO: Add a cron job to run this script every week.

# Database name
DATABASE_NAME = 'marketshare_data.db'

# Create table queries
CREATE_EXPECTED_RELEASES_TABLE = 'create_expected_releases_table.sql'
CREATE_MARKETSHARE_RELEASE_METRICS_TABLE = 'create_marketshare_release_metrics.sql'
CREATE_WEEKLY_MARKETSHARE_TABLE = 'create_weekly_marketshare_table.sql'
CREATE_YTD_MARKETSHARE_TABLE = 'create_ytd_marketshare_table.sql'

# Select queries
WEEKLY_MARKETSHARE_QUERY = 'query_weekly_marketshare_query.sql'
YTD_MARKETSHARE_QUERY = 'query_ytd_marketshare_query.sql'
MARKETSHARE_RELEASE_METRICS_QUERY = 'query_marketshare_release_metrics.sql'
EXPECTED_RELEASES_QUERY = 'release_get_all.sql'

# Insert queries
INSERT_WEEKLY_MARKETSHARE = 'insert_weekly_marketshare.sql'
INSERT_YTD_MARKETSHARE = 'insert_ytd_marketshare.sql'
INSERT_MARKETSHARE_RELEASE_METRICS = 'insert_marketshare_release_metrics.sql'


def update_sqlite_main() -> None:
    """
    Loads the weekly and ytd marketshare data from Snowflake and saves it to SQLite database.
    The SQLite database is located in the data folder.
    Other SQLite database tables are currently not being updated by this script.
    """
    # Connect to SQLite database
    sqlite_conn = sqlite3.connect(DATABASE_NAME)
    cursor = sqlite_conn.cursor()

    # Create tables
    cursor.execute(load_sql(CREATE_EXPECTED_RELEASES_TABLE))
    cursor.execute(load_sql(CREATE_WEEKLY_MARKETSHARE_TABLE))
    cursor.execute(load_sql(CREATE_YTD_MARKETSHARE_TABLE))
    cursor.execute(load_sql(CREATE_MARKETSHARE_RELEASE_METRICS_TABLE))

    cursor.execute(load_sql(EXPECTED_RELEASES_QUERY))
    expected_releases_columns = [d[0] for d in cursor.description]
    expected_releases_rows = cursor.fetchall()
    expected_releases_df = pd.DataFrame(expected_releases_rows, columns=expected_releases_columns)

    def _snowflake_string_literal(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    release_ids = (
        expected_releases_df['MRELG_ID']
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.strip() != '']
        .unique()
        .tolist()
    )
    release_ids_str = ','.join(_snowflake_string_literal(rid) for rid in release_ids)

    # Get data from Snowflake
    with get_snowflake_connection() as sf:
        weekly_marketshare_data = sf.query(load_sql(WEEKLY_MARKETSHARE_QUERY))
        ytd_marketshare_data = sf.query(load_sql(YTD_MARKETSHARE_QUERY))

        if release_ids_str:
            metrics_sql = load_sql(MARKETSHARE_RELEASE_METRICS_QUERY).replace(
                '{RELEASE_IDS}', release_ids_str
            )
            marketshare_release_metrics_data = sf.query(metrics_sql)
            marketshare_release_metrics_data = marketshare_release_metrics_data.rename(
                columns=str.upper,
            )
            # After rename(columns=str.upper)
            if "WEEK_ENDING_DATE" in marketshare_release_metrics_data.columns:
                marketshare_release_metrics_data["WEEK_ENDING_DATE"] = marketshare_release_metrics_data["WEEK_ENDING_DATE"].astype(str)

            for col in ("ALBUM_EQUIVALENT", "PRODUCT_SALES", "SONG_SALE_EQUIVALENT", "STREAMING_EQUIVALENT"):
                if col in marketshare_release_metrics_data.columns:
                    s = pd.to_numeric(marketshare_release_metrics_data[col], errors="coerce")
                    s = s.replace([float("inf"), float("-inf")], pd.NA)
                    marketshare_release_metrics_data[col] = s.where(~s.isna(), None).astype(object)
        else:
            marketshare_release_metrics_data = pd.DataFrame(
                columns=[
                    'WEEK_ENDING_DATE', 'MRELG_ID', 'ALBUM_EQUIVALENT', 'PRODUCT_SALES',
                    'SONG_SALE_EQUIVALENT', 'STREAMING_EQUIVALENT',
                ]
            )

    # Update tables
    cursor.executemany(load_sql(INSERT_WEEKLY_MARKETSHARE), weekly_marketshare_data[[
        'WEEK_ENDING_DATE', 'COUNTRY_CODE', 'RELEASE_AGE', 'LABEL_NAME',
        'STREAMING_TOTAL', 'ALBUM_EQUIVALENT', 'PRODUCT_SALES', 'SONG_SALE_EQUIVALENT',
        'STREAMING_EQUIVALENT', 'ALBUM_EQUIVALENT_SHARE', 'PRODUCT_SALES_SHARE',
        'SONG_SALE_EQUIVALENT_SHARE', 'STREAMING_EQUIVALENT_SHARE'
    ]].itertuples(index=False, name=None))

    cursor.executemany(load_sql(INSERT_YTD_MARKETSHARE), ytd_marketshare_data[[
        'WEEK_ENDING_DATE', 'YEAR', 'WEEK_NUM', 'COUNTRY_CODE', 'RELEASE_AGE', 'LABEL_NAME',
        'STREAMING_TOTAL', 'ALBUM_EQUIVALENT', 'PRODUCT_SALES', 'SONG_SALE_EQUIVALENT',
        'STREAMING_EQUIVALENT', 'ALBUM_EQUIVALENT_SHARE', 'PRODUCT_SALES_SHARE',
        'SONG_SALE_EQUIVALENT_SHARE', 'STREAMING_EQUIVALENT_SHARE'
    ]].itertuples(index=False, name=None))

    marketshare_release_metrics_data = pd.merge(
        marketshare_release_metrics_data,
        expected_releases_df[['MRELG_ID', 'RELEASE_ID']],
        on='MRELG_ID',
        how='left',
    )

    metric_cols = [
        'RELEASE_ID', 'MRELG_ID', 'WEEK_ENDING_DATE', 'ALBUM_EQUIVALENT',
        'PRODUCT_SALES', 'SONG_SALE_EQUIVALENT', 'STREAMING_EQUIVALENT',
    ]
    
    marketshare_release_metrics_data = marketshare_release_metrics_data[metric_cols]
    
    cursor.executemany(load_sql(INSERT_MARKETSHARE_RELEASE_METRICS),
        marketshare_release_metrics_data.itertuples(index=False, name=None))

    # Commit and close connection
    sqlite_conn.commit()
    sqlite_conn.close()


def drop_table(table_name: str) -> None:
    """
    Drops a table from the SQLite database.
    """
    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error dropping table: {e}")


if __name__ == "__main__":
    update_sqlite_main()
