import pandas as pd
import os
import snowflake.connector
from snowflake.connector import options as snowflake_options
from dotenv import load_dotenv
from pathlib import Path
from cryptography.hazmat.primitives import serialization
import textwrap
import threading

SECRETS_DIR_NAME = 'secrets'
SNOWFLAKE_SECRETS_FILE = 'amg_research.env'

# Parsing the Snowflake PEM into DER and re-loading it through cryptography
# costs ~30–80ms per call. Forecast endpoints open a fresh session per
# request, so we cache the derived bytes keyed on the raw PEM string. A new
# secrets file (env reload) produces a new cache key automatically; the cache
# never grows beyond a handful of entries.
_PRIVATE_KEY_DER_CACHE: dict[str, bytes] = {}
_PRIVATE_KEY_DER_LOCK = threading.Lock()


def _private_key_der_for(private_key_str: str) -> bytes:
    """Return the cached DER-encoded private key for a Snowflake PEM string."""
    with _PRIVATE_KEY_DER_LOCK:
        cached = _PRIVATE_KEY_DER_CACHE.get(private_key_str)
        if cached is not None:
            return cached

    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    key_body = private_key_str.replace(header, "").replace(footer, "").strip()
    key_body_wrapped = "\n".join(textwrap.wrap(key_body, 64))
    private_key_pem = f"{header}\n{key_body_wrapped}\n{footer}".encode()

    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    private_key_der = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    with _PRIVATE_KEY_DER_LOCK:
        _PRIVATE_KEY_DER_CACHE[private_key_str] = private_key_der
    return private_key_der

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
            if cursor.description is None:
                return pd.DataFrame()

            cols = [col[0] for col in cursor.description]
            # Avoid fetch_pandas_all(): Snowflake sets installed_pandas=False when
            # pandas OR pyarrow fail to import at connector load time, but the error
            # message always says "pandas is not installed" (errno ER_NO_PYARROW).
            if snowflake_options.installed_pandas:
                try:
                    return cursor.fetch_pandas_all()
                except snowflake.connector.errors.ProgrammingError:
                    pass
            return pd.DataFrame(cursor.fetchall(), columns=cols)
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

    private_key_der = _private_key_der_for(private_key_str)

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
