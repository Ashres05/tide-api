from model_handler import get_weekly_release_json, get_weekly_marketshare_json
import pandas as pd
import sqlite3
from sqlite_handler import update_sqlite_main, DATABASE_NAME

if __name__ == "__main__":
    FINAL_WEEK_ENDING_DATE = "2026-12-31"
    
    json1 = get_weekly_release_json(1, FINAL_WEEK_ENDING_DATE)
    json2 = get_weekly_marketshare_json(FINAL_WEEK_ENDING_DATE)
    print("Weekly Release Forecast:")
    print(json1)
    print("Weekly Marketshare Forecast:")
    print(json2)