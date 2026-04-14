from model_handler import get_all_releases
import pandas as pd
import sqlite3
from sqlite_handler import update_sqlite_main, DATABASE_NAME

if __name__ == "__main__":
    FINAL_WEEK_ENDING_DATE = "2026-12-31"
    
    json1 = get_all_releases()
    print(json1)