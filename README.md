## Tide Marketshare API

### How to Run:
#### 1. Clone Git repo.
#### 2. Create new directory inside repo root called "secrets".
#### 3. Place Snowflake Secrets inside secrets
#### 4. Run in repo root using:
#### python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
#### 5. Navigate to interactive documentation:
#### http://127.0.0.1:8000/docs

### Once Inside Documentation:
#### 6. Call data/refresh_data if official marketshare data is outdated (can safely be called every Monday morning at 6 A.M. EST)
#### 7. Call data/train_model if model has not been retrained locally yet (done alongside refresh_data).
#### WARNING: this call takes very long, should not be publicly available for users, should only be called in the back-end rarely.
#### 8. Call releases/backfill if official album data is outdated (can safely be called every Monday morning at 6 A.M. EST alongside data/refresh_data)
#### 9. Create, replace, update, and delete custom releases.
#### 10. Call marketshare/weekly to get marketshare projections from injected releases.
