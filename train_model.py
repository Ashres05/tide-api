from model.train_marketshare_artifacts import train_artifacts_main
from snowflake_conn import Snowflake, get_snowflake_connection, load_sql
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Data directory.
DATA_DIR = Path(__file__).resolve().parent / "model" / "data"

# Query names for the database.
CURRENT_DATA_QUERY = "query_model_current_data.sql"
A_LIST_75K_QUERY = "query_model_a_list_75k.sql"
BIG_RELEASE_FLAG_75K_QUERY = "query_model_big_release_flag.sql"

# Query names for the model.
MODEL_PARQUET_METRICS_QUERY = "query_model_parquet_metrics.sql"
MODEL_PARQUET_METRICS_STREAMING_QUERY = "query_model_parquet_metrics_streaming.sql"

def update_data_directory() -> None:
    """
    Updates: Current_Data.csv, A_List_75k.csv, Big_Release_Flag_75k.csv
    """
    
    with get_snowflake_connection() as sf:
        logger.info("train_model.py: Updating data directory")
        _update_current_data(sf)
        logger.info("train_model.py: Updated Current_Data.csv")
        _update_a_list_75k(sf)
        logger.info("train_model.py: Updated alist_75k.csv")
        _update_big_release_flag_75k(sf)
        logger.info("train_model.py: Updated bigreleaseflag_75k.csv")
        

def _update_parquet_metrics(sf: Snowflake) -> None:
    """
    Updates: streams_product_songs_ae_compressed.parquet
    """
    df = sf.query(load_sql(MODEL_PARQUET_METRICS_QUERY))
    df_streaming = sf.query(load_sql(MODEL_PARQUET_METRICS_STREAMING_QUERY))

    if not df.empty:
        logger.info("train_model.py: Updated streams_product_songs_ae_compressed.parquet")
        df.to_parquet(DATA_DIR / "streams_product_songs_ae_compressed.parquet", index=False)
    
    if not df_streaming.empty:
        logger.info("train_model.py: Updated worldwide_streams_compressed.parquet.parquet")
        df_streaming.to_parquet(DATA_DIR / "worldwide_streams_compressed.parquet.parquet", index=False)


def _update_current_data(sf: Snowflake) -> None:
    """
    Updates: Current_Data.csv
    """
    df = sf.query(load_sql(CURRENT_DATA_QUERY))
    if df.empty:
        return
    
    df.to_csv(DATA_DIR / "Current_Data.csv", index=False)


def _update_a_list_75k(sf: Snowflake) -> None:
    """
    Updates: alist_75k.csv
    """
    df = sf.query(load_sql(A_LIST_75K_QUERY))
    if df.empty:
        return
    
    df.to_csv(DATA_DIR / "alist_75k.csv", index=False)


def _update_big_release_flag_75k(sf: Snowflake) -> None:
    """
    Updates: bigreleaseflag_75k.csv
    """
    df = sf.query(load_sql(BIG_RELEASE_FLAG_75K_QUERY))
    if df.empty:
        return
    
    df.to_csv(DATA_DIR / "bigreleaseflag_75k.csv", index=False)


def train_model_main() -> None:
    """
    Calls the main function from train_marketshare_artifacts.py to train the marketshare artifacts.
    Will create necessary parquets and JSON files for model training.
    """
    # TODO: Update the data directory to the new data.
    update_data_directory()

    # Train the model.
    train_artifacts_main()


# TODO: Add a cron job to run this script every week after update_sqlite.py has been run.
if __name__ == "__main__":
    train_model_main()
