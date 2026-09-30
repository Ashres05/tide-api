"""Artist-level current PPR: nightly roster∪expected pull + live one-artist fetch."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from snowflake_conn import get_snowflake_connection, load_sql
from sqlite_handler import _snowflake_str, sqlite_connect

logger = logging.getLogger(__name__)

CREATE_ARTIST_PPR_CURRENT = "create_artist_ppr_current_table.sql"
CREATE_ARTIST_PPR_BY_RELEASE = "create_artist_ppr_by_release_table.sql"
INSERT_ARTIST_PPR_CURRENT = "insert_artist_ppr_current.sql"
INSERT_ARTIST_PPR_BY_RELEASE = "insert_artist_ppr_by_release.sql"
DELETE_ARTIST_PPR_CURRENT = "delete_artist_ppr_current.sql"
DELETE_ARTIST_PPR_BY_RELEASE = "delete_artist_ppr_by_release.sql"
UNIVERSE_QUERY = "query_artist_ppr_universe.sql"
NIGHTLY_QUERY = "query_artist_ppr_current.sql"
LIVE_QUERY = "query_artist_ppr_live.sql"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_id(value: Any) -> str:
    s = str(value or "").strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return ""
    return s


def _date_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            pass
    s = str(value).strip().split(" ")[0][:10]
    if not s or s.lower() in ("none", "nan", "nat", "null"):
        return None
    return s


def _ppr_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _df_cols(df) -> Dict[str, Any]:
    return {str(c).strip().lower(): c for c in df.columns}


def ensure_artist_ppr_tables(conn: sqlite3.Connection) -> None:
    conn.execute(load_sql(CREATE_ARTIST_PPR_CURRENT))
    conn.execute(load_sql(CREATE_ARTIST_PPR_BY_RELEASE))


def _empty_payload(luminate_artist_id: str, *, source: str) -> Dict[str, Any]:
    return {
        "luminate_artist_id": luminate_artist_id,
        "sodatone_artist_id": None,
        "ppr_value": None,
        "calculation_date": None,
        "source": source,
    }


def _row_payload(
    *,
    luminate_artist_id: str,
    sodatone_artist_id: Optional[str],
    ppr_value: Optional[float],
    calculation_date: Optional[str],
    source: str,
) -> Dict[str, Any]:
    return {
        "luminate_artist_id": luminate_artist_id,
        "sodatone_artist_id": sodatone_artist_id or None,
        "ppr_value": ppr_value,
        "calculation_date": calculation_date,
        "source": source,
    }


def universe_mrelg_ids() -> List[str]:
    with sqlite_connect() as conn:
        rows = conn.execute(load_sql(UNIVERSE_QUERY)).fetchall()
    out: List[str] = []
    seen = set()
    for (mid,) in rows:
        key = _norm_id(mid)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def artist_ppr_map() -> Dict[str, Dict[str, Any]]:
    """Uppercased LUMINATE_ARTIST_ID -> record (no source)."""
    with sqlite_connect() as conn:
        ensure_artist_ppr_tables(conn)
        cur = conn.execute(
            "SELECT LUMINATE_ARTIST_ID, SODATONE_ARTIST_ID, PPR_VALUE, "
            "CALCULATION_DATE FROM ARTIST_PPR_CURRENT"
        )
        rows = cur.fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for aid, sod, ppr, calc in rows:
        key = _norm_id(aid)
        if not key:
            continue
        out[key.upper()] = {
            "luminate_artist_id": key,
            "sodatone_artist_id": _norm_id(sod) or None,
            "ppr_value": _ppr_float(ppr),
            "calculation_date": _date_str(calc),
        }
    return out


def ppr_by_release_rows() -> List[Dict[str, Any]]:
    with sqlite_connect() as conn:
        ensure_artist_ppr_tables(conn)
        cur = conn.execute(
            "SELECT MRELG_ID, LUMINATE_ARTIST_ID, ARTIST_INDEX, PPR_VALUE, "
            "CALCULATION_DATE FROM ARTIST_PPR_BY_RELEASE "
            "ORDER BY MRELG_ID, ARTIST_INDEX"
        )
        rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for mid, aid, idx, ppr, calc in rows:
        mrelg = _norm_id(mid)
        artist = _norm_id(aid)
        if not mrelg or not artist:
            continue
        try:
            artist_index = int(idx) if idx is not None else None
        except (TypeError, ValueError):
            artist_index = None
        out.append(
            {
                "mrelg_id": mrelg,
                "luminate_artist_id": artist,
                "artist_index": artist_index,
                "ppr_value": _ppr_float(ppr),
                "calculation_date": _date_str(calc),
            }
        )
    return out


def build_boot_ppr_section() -> Dict[str, Any]:
    artists_raw = artist_ppr_map()
    artists = {
        rec["luminate_artist_id"]: {
            "ppr_value": rec["ppr_value"],
            "calculation_date": rec["calculation_date"],
            "sodatone_artist_id": rec["sodatone_artist_id"],
        }
        for rec in artists_raw.values()
        if rec.get("luminate_artist_id")
    }
    by_release = ppr_by_release_rows()
    return {
        "generated_at": _utc_now(),
        "count": len(artists),
        "artists": artists,
        "by_release": by_release,
        "by_release_count": len(by_release),
    }


def lookup_cached_ppr(luminate_artist_id: str) -> Optional[Dict[str, Any]]:
    aid = _norm_id(luminate_artist_id)
    if not aid:
        return None
    with sqlite_connect() as conn:
        ensure_artist_ppr_tables(conn)
        row = conn.execute(
            "SELECT LUMINATE_ARTIST_ID, SODATONE_ARTIST_ID, PPR_VALUE, "
            "CALCULATION_DATE FROM ARTIST_PPR_CURRENT "
            "WHERE UPPER(LUMINATE_ARTIST_ID) = UPPER(?)",
            (aid,),
        ).fetchone()
    if not row:
        return None
    return _row_payload(
        luminate_artist_id=_norm_id(row[0]) or aid,
        sodatone_artist_id=_norm_id(row[1]),
        ppr_value=_ppr_float(row[2]),
        calculation_date=_date_str(row[3]),
        source="cache",
    )


def _upsert_current(record: Dict[str, Any], *, source: str) -> None:
    aid = _norm_id(record.get("luminate_artist_id"))
    if not aid:
        return
    with sqlite_connect() as conn:
        ensure_artist_ppr_tables(conn)
        conn.execute(
            load_sql(INSERT_ARTIST_PPR_CURRENT),
            (
                aid,
                _norm_id(record.get("sodatone_artist_id")) or None,
                record.get("ppr_value"),
                record.get("calculation_date"),
                source,
                _utc_now(),
            ),
        )


def _fetch_live_snowflake(luminate_artist_id: str) -> Optional[Dict[str, Any]]:
    sql = load_sql(LIVE_QUERY).replace(
        "{LUMINATE_ARTIST_ID}", _snowflake_str(luminate_artist_id)
    )
    with get_snowflake_connection() as sf:
        df = sf.query(sql)
    if df is None or df.empty:
        return None
    cols = _df_cols(df)
    aid_col = cols.get("luminate_artist_id")
    sod_col = cols.get("sodatone_artist_id")
    ppr_col = cols.get("ppr_value")
    calc_col = cols.get("calculation_date")
    if not ppr_col:
        logger.warning("artist_ppr live: unexpected columns %s", list(df.columns))
        return None
    row = df.iloc[0]
    return _row_payload(
        luminate_artist_id=_norm_id(row[aid_col] if aid_col else None)
        or luminate_artist_id,
        sodatone_artist_id=_norm_id(row[sod_col] if sod_col else None),
        ppr_value=_ppr_float(row[ppr_col]),
        calculation_date=_date_str(row[calc_col] if calc_col else None),
        source="live",
    )


def get_artist_ppr(luminate_artist_id: str, *, allow_live: bool = True) -> Dict[str, Any]:
    """
    One current PPR for a Luminate artist. SQLite cache first, then Snowflake.
    """
    aid = _norm_id(luminate_artist_id)
    if not aid:
        raise ValueError("luminate_artist_id is required.")
    cached = lookup_cached_ppr(aid)
    if cached and cached.get("ppr_value") is not None:
        return cached
    if not allow_live:
        return cached or _empty_payload(aid, source="miss")
    try:
        live = _fetch_live_snowflake(aid)
    except Exception:
        logger.exception("artist_ppr live fetch failed id=%s", aid)
        return cached or _empty_payload(aid, source="error")
    if not live or live.get("ppr_value") is None:
        return cached or _empty_payload(aid, source="miss")
    try:
        _upsert_current(live, source="live")
    except Exception:
        logger.exception("artist_ppr live upsert failed id=%s", aid)
    return live


def _fetch_nightly_snowflake(mrelg_ids: List[str]):
    if not mrelg_ids:
        return None
    id_list = ",".join(_snowflake_str(mid) for mid in mrelg_ids)
    sql = load_sql(NIGHTLY_QUERY).replace("{MRELG_ID_LIST}", id_list)
    with get_snowflake_connection() as sf:
        return sf.query(sql)


def refresh_artist_ppr_nightly() -> Dict[str, Any]:
    """
    Full rewrite of local current-PPR tables from roster ∪ expected MRELGs.
    Does not rebuild boot.json; caller patches boot after this succeeds.
    """
    mrelg_ids = universe_mrelg_ids()
    summary: Dict[str, Any] = {
        "ok": True,
        "universe_mrelg": len(mrelg_ids),
        "rows": 0,
        "artists": 0,
        "generated_at": _utc_now(),
    }
    if not mrelg_ids:
        logger.warning("artist_ppr nightly: empty MRELG universe; leaving tables")
        summary["ok"] = False
        summary["reason"] = "empty_universe"
        return summary

    try:
        df = _fetch_nightly_snowflake(mrelg_ids)
    except Exception:
        logger.exception("artist_ppr nightly Snowflake failed")
        summary["ok"] = False
        summary["reason"] = "snowflake"
        return summary

    if df is None or df.empty:
        logger.warning("artist_ppr nightly: Snowflake returned 0 rows; leaving tables")
        summary["ok"] = False
        summary["reason"] = "empty_result"
        return summary

    cols = _df_cols(df)
    mid_col = cols.get("mrelg_id")
    aid_col = cols.get("luminate_artist_id")
    sod_col = cols.get("sodatone_artist_id")
    idx_col = cols.get("artist_index")
    ppr_col = cols.get("ppr_value")
    calc_col = cols.get("calculation_date")
    if not aid_col or not ppr_col:
        summary["ok"] = False
        summary["reason"] = f"unexpected_columns:{list(df.columns)}"
        return summary

    by_release: List[tuple] = []
    current_by_id: Dict[str, tuple] = {}
    now = _utc_now()
    mids = df[mid_col] if mid_col else [None] * len(df)
    aids = df[aid_col]
    sods = df[sod_col] if sod_col else [None] * len(df)
    idxs = df[idx_col] if idx_col else [None] * len(df)
    pprs = df[ppr_col]
    calcs = df[calc_col] if calc_col else [None] * len(df)

    for mid, aid, sod, idx, ppr, calc in zip(mids, aids, sods, idxs, pprs, calcs):
        artist = _norm_id(aid)
        ppr_val = _ppr_float(ppr)
        if not artist or ppr_val is None:
            continue
        calc_s = _date_str(calc)
        sod_s = _norm_id(sod) or None
        mrelg = _norm_id(mid)
        try:
            artist_index = int(idx) if idx is not None else None
        except (TypeError, ValueError):
            artist_index = None
        if mrelg:
            by_release.append((mrelg, artist, artist_index, ppr_val, calc_s))
        current_by_id[artist.upper()] = (
            artist,
            sod_s,
            ppr_val,
            calc_s,
            "nightly",
            now,
        )

    if not current_by_id:
        summary["ok"] = False
        summary["reason"] = "no_parseable_rows"
        return summary

    with sqlite_connect() as conn:
        ensure_artist_ppr_tables(conn)
        cur = conn.cursor()
        cur.execute(load_sql(DELETE_ARTIST_PPR_BY_RELEASE))
        cur.execute(load_sql(DELETE_ARTIST_PPR_CURRENT))
        if by_release:
            cur.executemany(load_sql(INSERT_ARTIST_PPR_BY_RELEASE), by_release)
        cur.executemany(
            load_sql(INSERT_ARTIST_PPR_CURRENT),
            list(current_by_id.values()),
        )

    summary["rows"] = len(by_release)
    summary["artists"] = len(current_by_id)
    logger.info("artist_ppr nightly: %s", summary)
    return summary
