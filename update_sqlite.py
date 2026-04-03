from snowflake_conn import get_snowflake_connection, load_sql
import sqlite3

# TODO: Add a cron job to run this script every week.
# Database name
DATABASE_NAME = 'marketshare_data.db'

# Create table queries
CREATE_EXPECTED_RELEASES_TABLE = 'create_expected_releases_table.sql'
CREATE_WEEKLY_MARKETSHARE_TABLE = 'create_weekly_marketshare_table.sql'
CREATE_YTD_MARKETSHARE_TABLE = 'create_ytd_marketshare_table.sql'

# Select queries
WEEKLY_MARKETSHARE_QUERY = 'query_weekly_marketshare_query.sql'
YTD_MARKETSHARE_QUERY = 'query_ytd_marketshare_query.sql'

# Insert queries
INSERT_WEEKLY_MARKETSHARE = 'insert_weekly_marketshare.sql'
INSERT_YTD_MARKETSHARE = 'insert_ytd_marketshare.sql'

def update_sqlite_main() -> None:
    """
    Loads the weekly and ytd marketshare data from Snowflake and saves it to SQLite database.
    The SQLite database is located in the data folder.
    Other SQLite database tables are currently not being updated by this script.
    """
    # Get data from Snowflake
    with get_snowflake_connection() as sf:
        weekly_marketshare_data = sf.query(load_sql(WEEKLY_MARKETSHARE_QUERY))
        ytd_marketshare_data = sf.query(load_sql(YTD_MARKETSHARE_QUERY))
    
    # Connect to SQLite database
    sqlite_conn = sqlite3.connect(DATABASE_NAME)
    cursor = sqlite_conn.cursor()

    # Create tables
    cursor.execute(load_sql(CREATE_EXPECTED_RELEASES_TABLE))
    cursor.execute(load_sql(CREATE_WEEKLY_MARKETSHARE_TABLE))
    cursor.execute(load_sql(CREATE_YTD_MARKETSHARE_TABLE))

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

    # Commit and close connection
    sqlite_conn.commit()
    sqlite_conn.close()


if __name__ == "__main__":
    update_sqlite_main()
