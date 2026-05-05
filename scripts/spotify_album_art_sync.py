#!/usr/bin/env python3
"""
Sync album cover images to S3 for releases listed in EXPECTED_RELEASES.

Flow (official Spotify Web API — no HTML scraping):
  1. Read (ARTIST, TITLE[, MRELG_ID, RELEASE_ID]) from local SQLite
     ``marketshare_data.db`` → table ``EXPECTED_RELEASES``.
     Optionally pass ``--manual "Artist|Album"`` rows instead of the DB.
  2. ``GET /v1/search?type=album`` with query ``album:{title} artist:{artist}``.
  3. Take the first search hit (or best-effort match), read ``id``.
  4. ``GET /v1/albums/{id}`` → pick the largest image in ``images[]``.
  5. Download the image URL and ``upload_file`` to S3 under a dedicated prefix.

Authentication (pick one; first match wins):
  * **CLI:** ``--client-id`` / ``--client-secret``
  * **Environment:** ``SPOTIFY_CLIENT_ID`` + ``SPOTIFY_CLIENT_SECRET`` (e.g. in ``.env``)
  * **Local-only defaults:** set ``_LOCAL_SPOTIFY_CLIENT_ID`` and
    ``_LOCAL_SPOTIFY_CLIENT_SECRET`` at the top of this script (empty by default;
    do not commit real secrets to git).
  * **Manual token:** ``SPOTIFY_ACCESS_TOKEN`` or ``--access-token`` (short-lived).

S3 layout (default prefix ``album_art/`` under your artifacts bucket):
  ``s3://{bucket}/{prefix}{release_id}_{spotify_album_id}.jpg``

Environment (reuses tide-api conventions where possible):
  * ``TIDE_ARTIFACTS_S3_URI`` / ``TIDE_ARTIFACTS_S3_BUCKET`` + ``TIDE_ARTIFACTS_S3_PREFIX``
  * ``TIDE_ALBUM_ART_S3_PREFIX`` — extra subfolder for covers only, default ``album_art/``
  * ``TIDE_S3_DEFAULT_BUCKET`` — fallback bucket name (default ``parquetgarage``)

Usage examples::

  # All rows in EXPECTED_RELEASES (default DB path = repo root marketshare_data.db).
  # Pull DB from S3 first if needed, then omit --dry-run to upload.
  python3 scripts/spotify_album_art_sync.py --db ./marketshare_data.db

  # Smoke-test first N releases only (still hits Spotify + S3 per row unless --dry-run)
  python3 scripts/spotify_album_art_sync.py --db ./marketshare_data.db --limit 5

  # Re-run full catalog but skip keys that already exist in S3
  python3 scripts/spotify_album_art_sync.py --db ./marketshare_data.db --skip-existing

  # Manual rows only (no DB)
  python3 scripts/spotify_album_art_sync.py --no-db \\
    --manual "Bruno Mars|The Romantic"

  # Preview all rows without S3 writes
  python3 scripts/spotify_album_art_sync.py --db ./marketshare_data.db --dry-run

  # Revenue CSV (MRELG_ID + 2025_revenue, sorted desc). Enrich artist/title from
  # MARKETSHARE_SEARCH_SUMMARY in --db when the CSV omits them. Skip S3 keys that exist.
  python3 scripts/spotify_album_art_sync.py --from-csv ./revenue_by_album.csv \\
    --db ./marketshare_data.db --skip-existing

Requires: boto3 (already in project requirements). Uses stdlib for HTTP.

Spotify pacing (429 avoidance), newest defaults first — override via env:

  * ``SPOTIFY_WINDOW_MAX_REQUESTS`` / ``SPOTIFY_WINDOW_SECONDS`` — rolling cap
    on api.spotify.com + accounts.spotify.com calls (default 20 / 30).
  * ``SPOTIFY_SEARCH_QUERY_SLEEP_SEC`` — pause between search fallback queries
    for the same album (default 0.45).
  * ``--sleep`` — pause after each release row (default 0.85s).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SPOTIFY_ACCOUNTS = "https://accounts.spotify.com"
SPOTIFY_API = "https://api.spotify.com/v1"

# Local testing: paste credentials from https://developer.spotify.com/dashboard
# Leave as "" when using env vars or CLI. Do not commit secrets to version control.
_LOCAL_SPOTIFY_CLIENT_ID = "96848c9e58c4466b83610c1c71578762"
_LOCAL_SPOTIFY_CLIENT_SECRET = "553607f59b314fc0959144cf69521886"

# Spotify applies a rolling ~30s request budget (429 when exceeded). Defaults
# are conservative for long batch runs; raise SPOTIFY_WINDOW_MAX_REQUESTS only
# if you consistently see under-utilization.
_SPOTIFY_WINDOW_MAX_REQUESTS = int(os.environ.get("SPOTIFY_WINDOW_MAX_REQUESTS", "20"))
_SPOTIFY_WINDOW_SECONDS = float(os.environ.get("SPOTIFY_WINDOW_SECONDS", "30"))
_SPOTIFY_WINDOW_TIMESTAMPS: deque[float] = deque()
# Extra gap between search query variants (same release can hit Spotify several times).
_SPOTIFY_SEARCH_QUERY_SLEEP_SEC = float(os.environ.get("SPOTIFY_SEARCH_QUERY_SLEEP_SEC", "0.45"))


@dataclass
class ReleaseRow:
    release_id: Optional[int]
    mrelg_id: Optional[str]
    artist: str
    title: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_repo_root() / ".env")
    except ImportError:
        pass


def _norm_s3_prefix(prefix: str) -> str:
    p = (prefix or "").strip().strip("/")
    return f"{p}/" if p else ""


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    u = uri.strip()
    if not u.lower().startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    rest = u[5:]
    parts = rest.split("/", 1)
    bucket = parts[0].strip()
    if not bucket:
        raise ValueError(f"Invalid S3 URI (empty bucket): {uri!r}")
    key_prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    return bucket, key_prefix


def _resolve_s3_destination() -> tuple[str, str]:
    """Return (bucket, key_prefix_including_album_art_folder)."""
    _load_dotenv()
    uri = os.environ.get("TIDE_ARTIFACTS_S3_URI", "").strip()
    bucket = os.environ.get("TIDE_ARTIFACTS_S3_BUCKET", "").strip()
    prefix = os.environ.get("TIDE_ARTIFACTS_S3_PREFIX", "").strip()
    if uri:
        bucket, prefix = _parse_s3_uri(uri)
    elif not bucket:
        bucket = os.environ.get("TIDE_S3_DEFAULT_BUCKET", "parquetgarage").strip()
        prefix = ""
    art_sub = os.environ.get("TIDE_ALBUM_ART_S3_PREFIX", "album_art/").strip()
    if art_sub and not art_sub.endswith("/"):
        art_sub += "/"
    base = _norm_s3_prefix(prefix) + art_sub
    return bucket, base


def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: Optional[bytes] = None,
    timeout: float = 60.0,
) -> tuple[int, Any]:
    def _throttle_if_needed(target_url: str) -> None:
        if "spotify.com" not in target_url:
            return
        if _SPOTIFY_WINDOW_MAX_REQUESTS <= 0:
            return
        now = time.monotonic()
        while _SPOTIFY_WINDOW_TIMESTAMPS and (now - _SPOTIFY_WINDOW_TIMESTAMPS[0]) >= _SPOTIFY_WINDOW_SECONDS:
            _SPOTIFY_WINDOW_TIMESTAMPS.popleft()
        if len(_SPOTIFY_WINDOW_TIMESTAMPS) >= _SPOTIFY_WINDOW_MAX_REQUESTS:
            oldest = _SPOTIFY_WINDOW_TIMESTAMPS[0]
            wait_sec = max(0.05, _SPOTIFY_WINDOW_SECONDS - (now - oldest))
            logger.info(
                "Spotify pacing: %d requests in %.0fs window; sleeping %.2fs",
                len(_SPOTIFY_WINDOW_TIMESTAMPS),
                _SPOTIFY_WINDOW_SECONDS,
                wait_sec,
            )
            time.sleep(wait_sec)
            now = time.monotonic()
            while _SPOTIFY_WINDOW_TIMESTAMPS and (now - _SPOTIFY_WINDOW_TIMESTAMPS[0]) >= _SPOTIFY_WINDOW_SECONDS:
                _SPOTIFY_WINDOW_TIMESTAMPS.popleft()
        _SPOTIFY_WINDOW_TIMESTAMPS.append(time.monotonic())

    max_attempts = int(os.environ.get("SPOTIFY_HTTP_MAX_ATTEMPTS", "6"))
    base_backoff_sec = float(os.environ.get("SPOTIFY_HTTP_BASE_BACKOFF_SEC", "1.25"))
    max_retry_after_sec = float(os.environ.get("SPOTIFY_HTTP_MAX_RETRY_AFTER_SEC", "120"))
    attempt = 0
    while True:
        attempt += 1
        _throttle_if_needed(url)
        req = urllib.request.Request(url, method=method, headers=headers, data=data)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return resp.status, json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt < max_attempts:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    wait_sec = float(retry_after) if retry_after is not None else 0.0
                except (TypeError, ValueError):
                    wait_sec = 0.0
                if wait_sec > max_retry_after_sec:
                    logger.warning(
                        "Spotify 429 Retry-After %.2fs capped to %.2fs",
                        wait_sec,
                        max_retry_after_sec,
                    )
                    wait_sec = max_retry_after_sec
                if wait_sec <= 0:
                    wait_sec = base_backoff_sec * (2 ** (attempt - 1))
                # Avoid synchronized retries.
                wait_sec += min(0.35, 0.05 * attempt)
                logger.warning(
                    "Spotify rate limited (429). Retrying in %.2fs [attempt %d/%d]: %s",
                    wait_sec,
                    attempt,
                    max_attempts,
                    url,
                )
                time.sleep(wait_sec)
                continue
            raise RuntimeError(f"HTTP {e.code} {url}: {err_body}") from e


def _http_bytes(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def get_access_token(
    *,
    access_token: Optional[str],
    client_id: Optional[str],
    client_secret: Optional[str],
) -> str:
    if access_token and access_token.strip():
        return access_token.strip()
    cid = (
        (client_id or "").strip()
        or os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
        or _LOCAL_SPOTIFY_CLIENT_ID.strip()
    )
    csec = (
        (client_secret or "").strip()
        or os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
        or _LOCAL_SPOTIFY_CLIENT_SECRET.strip()
    )
    if not cid or not csec:
        raise SystemExit(
            "Spotify auth missing: set --client-id/--client-secret, or "
            "SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET, or fill _LOCAL_SPOTIFY_* "
            "in scripts/spotify_album_art_sync.py, or set SPOTIFY_ACCESS_TOKEN."
        )
    basic = base64.b64encode(f"{cid}:{csec}".encode("utf-8")).decode("ascii")
    body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
    status, payload = _http_json(
        "POST",
        f"{SPOTIFY_ACCOUNTS}/api/token",
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data=body,
    )
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(f"Token response unexpected: {payload}")
    tok = payload.get("access_token")
    if not tok:
        raise RuntimeError(f"No access_token in response: {payload}")
    return str(tok)


def search_album_id(token: str, artist: str, album: str) -> Optional[str]:
    def _search_once(q: str) -> Optional[str]:
        params = urllib.parse.urlencode({"q": q, "type": "album", "limit": 5})
        url = f"{SPOTIFY_API}/search?{params}"
        status, data = _http_json(
            "GET",
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        if status != 200 or not isinstance(data, dict):
            return None
        albums = (((data.get("albums") or {}).get("items")) or [])
        if not albums:
            return None
        first = albums[0]
        return str(first.get("id")) if first.get("id") else None

    def _ascii_normalize(s: str) -> str:
        # Helps with artist names containing accents / punctuation variants.
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    q1 = f'album:{album} artist:{artist}'
    album_id = _search_once(q1)
    if album_id:
        return album_id

    artist_ascii = _ascii_normalize(artist)
    album_ascii = _ascii_normalize(album)
    fallback_queries = [
        f'album:{album_ascii} artist:{artist_ascii}',
        f"{album_ascii} {artist_ascii}",
        f"album:{album_ascii}",
    ]
    gap = max(0.0, _SPOTIFY_SEARCH_QUERY_SLEEP_SEC)
    for q in fallback_queries:
        if gap > 0:
            time.sleep(gap)
        album_id = _search_once(q)
        if album_id:
            return album_id
    return None


def fetch_album_largest_image(token: str, album_id: str) -> tuple[str, int, int]:
    url = f"{SPOTIFY_API}/albums/{urllib.parse.quote(album_id)}"
    status, data = _http_json(
        "GET",
        url,
        headers={"Authorization": f"Bearer {token}"},
    )
    if status != 200 or not isinstance(data, dict):
        raise RuntimeError(f"Album fetch failed for {album_id!r}")
    images = data.get("images") or []
    if not images:
        raise RuntimeError(f"No images on album {album_id}")
    best = max(
        images,
        key=lambda im: (int(im.get("width") or 0) * int(im.get("height") or 0)),
    )
    href = best.get("url")
    if not href:
        raise RuntimeError(f"Image URL missing for album {album_id}")
    w = int(best.get("width") or 0)
    h = int(best.get("height") or 0)
    return str(href), w, h


def _slug(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return (s[:max_len] or "release").strip("-")


def _object_key_suffix(row: ReleaseRow, album_id: str, ext: str) -> str:
    """
    Flat key suffix under album_art/ (no nested folders), so all files land in:
      s3://.../album_art/<filename>
    """
    if row.mrelg_id and str(row.mrelg_id).strip():
        # Requested convention: exactly <mrelg_id>.jpg for easy joins.
        return f"{str(row.mrelg_id).strip()}.jpg"
    e = ext.lstrip(".")
    if row.release_id is not None:
        return f"rid_{row.release_id}_{album_id}.{e}"
    return f"{_slug(row.artist)}__{_slug(row.title)}__{album_id}.{e}"


def _guess_ext_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    if path.endswith(".png"):
        return "png"
    if path.endswith(".jpeg"):
        return "jpeg"
    return "jpg"


def _iter_expected_releases(conn: sqlite3.Connection) -> Iterator[ReleaseRow]:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(EXPECTED_RELEASES)")
    cols = {row[1].upper() for row in cur.fetchall()}
    need = {"TITLE", "ARTIST"}
    if not need.issubset(cols):
        raise SystemExit(
            f"EXPECTED_RELEASES must include columns {sorted(need)}; got {sorted(cols)}"
        )
    sel = ["TITLE", "ARTIST"]
    if "RELEASE_ID" in cols:
        sel.insert(0, "RELEASE_ID")
    else:
        sel.insert(0, "NULL AS RELEASE_ID")
    if "MRELG_ID" in cols:
        sel.append("MRELG_ID")
    else:
        sel.append("NULL AS MRELG_ID")
    q = f"SELECT {', '.join(sel)} FROM EXPECTED_RELEASES WHERE TRIM(IFNULL(ARTIST,'')) != '' AND TRIM(IFNULL(TITLE,'')) != ''"
    for tup in cur.execute(q):
        rid, title, artist, mid = tup[0], tup[1], tup[2], tup[3]
        yield ReleaseRow(
            release_id=int(rid) if rid is not None else None,
            mrelg_id=str(mid).strip() if mid else None,
            artist=str(artist).strip(),
            title=str(title).strip(),
        )


def _lookup_search_summary_artist_title(db_path: Path, mrelg_id: str) -> tuple[str, str]:
    """Best-effort ARTIST/TITLE from local MARKETSHARE_SEARCH_SUMMARY."""
    mid = (mrelg_id or "").strip()
    if not mid:
        return "", ""
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error:
        return "", ""
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT ARTIST, TITLE FROM MARKETSHARE_SEARCH_SUMMARY WHERE TRIM(MRELG_ID) = ? LIMIT 1",
            (mid,),
        )
        row = cur.fetchone()
        if row:
            return (str(row[0] or "").strip(), str(row[1] or "").strip())
    finally:
        conn.close()
    return "", ""


def _resolve_revenue_column(df: pd.DataFrame, revenue_col: Optional[str]) -> str:
    if revenue_col:
        c = revenue_col.strip()
        if c not in df.columns:
            raise SystemExit(f"--revenue-col {c!r} not found in CSV columns: {list(df.columns)}")
        return c
    norm = {str(c).strip().upper().replace(" ", "_"): c for c in df.columns}
    for key in (
        "2025_REVENUE",
        "REVENUE_2025",
        "CATALOG_REVENUE_2025",
        "REVENUE",
        "YTD_REVENUE_2025",
    ):
        if key in norm:
            return norm[key]
    raise SystemExit(
        "Could not infer revenue column. Pass --revenue-col (e.g. 2025_revenue). "
        f"Columns seen: {list(df.columns)}"
    )


def load_release_rows_from_revenue_csv(
    csv_path: Path,
    *,
    revenue_col: Optional[str],
    db_path: Optional[Path],
) -> list[ReleaseRow]:
    """
    Read a CSV with MRELG_ID and a numeric revenue column, sort by revenue descending,
    dedupe by MRELG_ID (first row wins). ARTIST/TITLE may come from CSV columns or
    from MARKETSHARE_SEARCH_SUMMARY in ``db_path``.
    """
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    df.columns = [str(c).strip() for c in df.columns]
    norm = {str(c).strip().upper().replace(" ", "_"): c for c in df.columns}
    mrelg_src = norm.get("MRELG_ID") or norm.get("MRELG")
    if not mrelg_src:
        raise SystemExit("CSV must include an MRELG_ID (or MRELG) column.")
    rev_src = _resolve_revenue_column(df, revenue_col)
    artist_src = norm.get("ARTIST") or norm.get("DISPLAY_ARTIST")
    title_src = norm.get("TITLE") or norm.get("ALBUM") or norm.get("ALBUM_TITLE") or norm.get("NAME")

    work = df.copy()
    work["_mrelg"] = work[mrelg_src].astype(str).str.strip()
    work = work[work["_mrelg"] != ""]
    work["_rev"] = pd.to_numeric(work[rev_src], errors="coerce")
    work = work.sort_values("_rev", ascending=False, na_position="last")
    work = work.drop_duplicates(subset=["_mrelg"], keep="first")

    rows: list[ReleaseRow] = []
    for _, r in work.iterrows():
        mid = r["_mrelg"]
        artist = ""
        title = ""
        if artist_src and pd.notna(r.get(artist_src)):
            artist = str(r[artist_src]).strip()
        if title_src and pd.notna(r.get(title_src)):
            title = str(r[title_src]).strip()
        if (not artist or not title) and db_path and db_path.is_file():
            a, t = _lookup_search_summary_artist_title(db_path, mid)
            artist = artist or a
            title = title or t
        if not artist or not title:
            logger.warning("skip %s — missing ARTIST/TITLE (add columns or pass --db for search summary)", mid)
            continue
        rows.append(ReleaseRow(release_id=None, mrelg_id=mid, artist=artist, title=title))
    logger.info(
        "from-csv: %d data rows → %d releases with artist/title (revenue col=%s)",
        len(df),
        len(rows),
        rev_src,
    )
    return rows


def _parse_manual(s: str) -> ReleaseRow:
    """
    Accept:
      --manual "Artist|Album"
      --manual "Artist|Album|MRELG_ID"
    """
    parts = [p.strip() for p in s.split("|")]
    if len(parts) not in (2, 3):
        raise SystemExit(f'--manual must be "Artist|Album" or "Artist|Album|MRELG_ID", got: {s!r}')
    artist, title = parts[0], parts[1]
    mrelg_id = parts[2] if len(parts) == 3 and parts[2] else None
    if not artist or not title:
        raise SystemExit(f"Empty artist or title in --manual {s!r}")
    return ReleaseRow(release_id=None, mrelg_id=mrelg_id, artist=artist, title=title)


def sync_one(
    token: str,
    row: ReleaseRow,
    *,
    bucket: str,
    key_prefix: str,
    dry_run: bool,
    skip_existing: bool,
    boto_client: Any,
) -> dict[str, Any]:
    album_id = search_album_id(token, row.artist, row.title)
    if not album_id:
        return {"ok": False, "error": "no_search_results", "row": row}
    spotify_url = f"https://open.spotify.com/album/{album_id}"
    ext = "jpg"
    key = f"{key_prefix}{_object_key_suffix(row, album_id, ext)}".replace("//", "/")
    if dry_run:
        img_url, w, h = fetch_album_largest_image(token, album_id)
        ext = _guess_ext_from_url(img_url)
        key = f"{key_prefix}{_object_key_suffix(row, album_id, ext)}".replace("//", "/")
        logger.info(
            "DRY-RUN would upload s3://%s/%s | %s — %s | image=%s (%dx%d)",
            bucket,
            key,
            spotify_url,
            f"{row.artist} — {row.title}",
            img_url,
            w,
            h,
        )
        return {
            "ok": True,
            "dry_run": True,
            "spotify_album_url": spotify_url,
            "spotify_album_id": album_id,
        }

    # When mrelg_id sets a fixed S3 filename (.jpg), skip GET /albums/{id} if the
    # object already exists — saves one Spotify call per cached row on long runs.
    if (
        skip_existing
        and boto_client is not None
        and row.mrelg_id
        and str(row.mrelg_id).strip()
    ):
        try:
            boto_client.head_object(Bucket=bucket, Key=key)
            logger.info("SKIP existing s3://%s/%s (no Spotify album fetch)", bucket, key)
            return {
                "ok": True,
                "skipped": True,
                "s3_uri": f"s3://{bucket}/{key}",
                "spotify_album_url": spotify_url,
            }
        except Exception:
            pass

    img_url, w, h = fetch_album_largest_image(token, album_id)
    ext = _guess_ext_from_url(img_url)
    if skip_existing and boto_client is not None:
        key = f"{key_prefix}{_object_key_suffix(row, album_id, ext)}".replace("//", "/")
        try:
            boto_client.head_object(Bucket=bucket, Key=key)
            logger.info("SKIP existing s3://%s/%s", bucket, key)
            return {
                "ok": True,
                "skipped": True,
                "s3_uri": f"s3://{bucket}/{key}",
                "spotify_album_url": spotify_url,
            }
        except Exception:
            pass

    raw = _http_bytes(img_url)
    boto_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=raw,
        ContentType=f"image/{ext}" if ext in ("jpg", "jpeg", "png") else "application/octet-stream",
    )
    logger.info(
        "UPLOAD s3://%s/%s (%d bytes) ← %s (%dx%d)",
        bucket,
        key,
        len(raw),
        spotify_url,
        w,
        h,
    )
    return {
        "ok": True,
        "s3_uri": f"s3://{bucket}/{key}",
        "spotify_album_url": spotify_url,
        "spotify_album_id": album_id,
    }


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(
        description="Upload EXPECTED_RELEASES album art to S3 via Spotify API.",
        epilog=(
            "Default mode reads every non-empty ARTIST+TITLE row from EXPECTED_RELEASES "
            "and uploads one image per row (unless --dry-run). "
            "Use --no-db --manual … for one-offs only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--db",
        type=Path,
        default=_repo_root() / "marketshare_data.db",
        help="Path to SQLite DB containing EXPECTED_RELEASES",
    )
    p.add_argument("--no-db", action="store_true", help="Do not read SQLite; use --manual only.")
    p.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "CSV with MRELG_ID (or MRELG) and a revenue column (default: 2025_revenue / aliases). "
            "Rows sorted by that column descending; deduped by MRELG_ID. "
            "Use ARTIST+TITLE columns or pass --db to fill from MARKETSHARE_SEARCH_SUMMARY."
        ),
    )
    p.add_argument(
        "--revenue-col",
        default="",
        metavar="NAME",
        help="CSV column name for revenue sort (default: auto-detect 2025_revenue, etc.).",
    )
    p.add_argument(
        "--manual",
        action="append",
        default=[],
        metavar='"Artist|Album"',
        help="Manual release line(s). Repeatable. Example: --manual 'Bruno Mars|The Romantic'",
    )
    p.add_argument("--access-token", default="", help="Spotify Bearer token (else env SPOTIFY_ACCESS_TOKEN).")
    p.add_argument(
        "--client-id",
        default="",
        help="Spotify app client id (else SPOTIFY_CLIENT_ID env, else _LOCAL_SPOTIFY_CLIENT_ID in this file).",
    )
    p.add_argument(
        "--client-secret",
        default="",
        help="Spotify app client secret (else SPOTIFY_CLIENT_SECRET env, else _LOCAL_SPOTIFY_CLIENT_SECRET in this file).",
    )
    p.add_argument("--dry-run", action="store_true", help="Only log Spotify URLs and image URLs; no S3 writes.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If object already exists at computed S3 key, skip download/upload.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.85,
        metavar="SEC",
        help=(
            "Seconds to pause after each release row (after all Spotify calls for that row). "
            "Default 0.85s; use 2+ for very strict limits. Also set SPOTIFY_WINDOW_MAX_REQUESTS / "
            "SPOTIFY_SEARCH_QUERY_SLEEP_SEC for finer control."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most the first N rows after loading (DB order + any --manual rows). "
        "Omit for all releases.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    token = get_access_token(
        access_token=args.access_token or os.environ.get("SPOTIFY_ACCESS_TOKEN"),
        client_id=args.client_id or None,
        client_secret=args.client_secret or None,
    )

    rows: list[ReleaseRow] = []
    if args.from_csv is not None:
        if args.manual:
            raise SystemExit("Use either --from-csv or --manual, not both.")
        enrich_db = args.db if args.db.is_file() else None
        rows = load_release_rows_from_revenue_csv(
            args.from_csv,
            revenue_col=(args.revenue_col or None),
            db_path=enrich_db,
        )
    elif not args.no_db:
        if not args.db.is_file():
            raise SystemExit(f"SQLite DB not found: {args.db}")
        conn = sqlite3.connect(str(args.db))
        try:
            rows.extend(_iter_expected_releases(conn))
        finally:
            conn.close()
        for m in args.manual:
            rows.append(_parse_manual(m))
    else:
        for m in args.manual:
            rows.append(_parse_manual(m))
    if not rows:
        raise SystemExit(
            "No releases to process (empty EXPECTED_RELEASES, no --manual, or --from-csv produced no rows)."
        )

    total_before_limit = len(rows)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        rows = rows[: args.limit]
        logger.info(
            "Loaded %d row(s); processing %d due to --limit",
            total_before_limit,
            len(rows),
        )
    else:
        logger.info("Loaded %d release(s) from DB/manual input", len(rows))

    bucket, key_prefix = _resolve_s3_destination()
    logger.info("S3 destination: s3://%s/%s", bucket, key_prefix)

    boto_client = None
    if not args.dry_run:
        try:
            import boto3
        except ImportError as e:
            raise SystemExit("boto3 is required for S3 upload. pip install boto3") from e
        boto_client = boto3.client("s3")

    ok, failed = 0, 0
    for i, row in enumerate(rows):
        try:
            r = sync_one(
                token,
                row,
                bucket=bucket,
                key_prefix=key_prefix,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                boto_client=boto_client,
            )
            if r.get("ok"):
                ok += 1
            else:
                failed += 1
                logger.warning("FAIL %s — %s", r.get("error"), row)
        except Exception as e:
            failed += 1
            logger.exception("FAIL %s — %s: %s", row.artist, row.title, e)
        if args.sleep and i < len(rows) - 1:
            time.sleep(args.sleep)

    logger.info("Done: %d ok, %d failed, %d total", ok, failed, len(rows))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
