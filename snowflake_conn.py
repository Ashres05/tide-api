import pandas as pd
import os
import snowflake.connector
from dotenv import load_dotenv
from pathlib import Path
from cryptography.hazmat.primitives import serialization
import textwrap

SECRETS_DIR_NAME = 'secrets'
SNOWFLAKE_SECRETS_FILE = 'amg_research.env'

class Snowflake:
    """
    Snowflake connection class.
    """
    def __init__(self, creds):
        """
        Initialize the Snowflake connection using provided credentials.
        """
        self._creds = creds
        self.conn = None
        
    def __enter__(self):
        """
        Returns self to build context manager.
        """
        try:
            self.conn = snowflake.connector.connect(**self._creds)
            return self
        except Exception as e:
            raise SnowflakeConnectionError(f"Failed to connect to Snowflake: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensure the Snowflake connection is closed when the object exits a context manager.
        """
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a pandas DataFrame.
        """
        if not self.conn:
            raise SnowflakeConnectionError("Snowflake connection is not established.")

        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            return cursor.fetch_pandas_all()
        except snowflake.connector.errors.ProgrammingError as e:
            raise SnowflakeConnectionError(f"An error occurred while executing the query: {e}")
        except Exception as e:
            raise SnowflakeConnectionError(f"Unexpected error: {e}")


class SnowflakeConnectionError(Exception):
    """
    Custom Snowflake Connection Error.
    """
    pass


def get_snowflake_connection():
    """
    Returns connection to Snowflake based off data from secrets file.
    """
    current_dir = Path(__file__).parent
    env_path = current_dir / SECRETS_DIR_NAME / SNOWFLAKE_SECRETS_FILE

    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        load_dotenv()

    private_key_str = os.getenv("SNOWFLAKE_PRIVATE_KEY")

    if not private_key_str:
        raise ValueError("SNOWFLAKE_PRIVATE_KEY is missing or incorrect.")

    # Extract the header and footer
    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    key_body = private_key_str.replace(header, "").replace(footer, "").strip()

    # Insert line breaks every 64 characters
    key_body_wrapped = "\n".join(textwrap.wrap(key_body, 64))
    private_key_pem = f"{header}\n{key_body_wrapped}\n{footer}".encode()

    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None
    )

    private_key_der = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    creds = {
        "user": os.environ.get("SNOWFLAKE_USER"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
        "private_key": private_key_der,
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "account":os.environ.get("SNOWFLAKE_ACCOUNT"),
        # "password":os.environ.get("PASSWORD")
    }
    
    return Snowflake(creds)


def load_sql(file_name: str) -> str:
    """Takes file name and goes into the queries folder to return the file as text."""
    current_dir = Path(__file__).parent
    path = current_dir / 'queries' / file_name
    return path.read_text(encoding='utf-8')
