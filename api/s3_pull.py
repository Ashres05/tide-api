"""
Scoped S3 sync helpers for the tide-api pipeline.

Design pattern:
  S3 is the canonical store. Each pipeline stage pulls only the files it
  needs to local disk, runs locally, then pushes only the files it changed.
  This minimizes disk usage on EC2 and avoids unnecessary network transfer.

Scopes:
  - "db"                   marketshare_data.db
  - "csvs"                 model/data/*.csv (current_data_all, bigrelease_alist_75k_all, ytd_fiscal_revenue_by_label, monthly_share_and_qtd, monthly_share_labels, labels_*_marketshare, amg_full_*_marketshare, releases_by_q_amg_labels). Excludes static 2025_revenue_catalog.csv and legacy Current_Data/alist_75k/bigreleaseflag_75k/quarterly_share_*.csv (manual S3 upload only).
  - "parquets"             model/data/*.parquet (heavy; archetype/training inputs)
  - "artifacts_75k"        model/artifacts_75k/** (LGBM/Prophet/spike/df_full/sidecars)
  - "archetypes_artifacts" model/archetypes_artifacts/** (album decay + singles/ subdir; forecast serving)

Convenience entry points:
  - sync_serving_inputs_from_s3()   -> startup: db + csvs + artifacts_75k + archetypes + boot.json
  - sync_weekly_inputs_from_s3()    -> refresh_weekly start: db + csvs + artifacts_75k + boot.json
  - sync_weekly_outputs_to_s3()     -> refresh_weekly end: db + csvs + artifacts_75k (+ boot.json if local)
  - sync_full_inputs_from_s3()      -> full refresh_data: all scopes
  - sync_full_outputs_to_s3()       -> full refresh_data: all scopes
  - upload_boot_json(body)          -> s3://…/boot.json (+ model/data/boot.json mirror)

Configuration (env):
  TIDE_ARTIFACTS_S3_URI=s3://parquetgarage           (preferred)
  TIDE_ARTIFACTS_S3_URI=s3://parquetgarage/tide-api  (with prefix)
  TIDE_ARTIFACTS_S3_BUCKET=parquetgarage             (alt)
  TIDE_ARTIFACTS_S3_PREFIX=                          (alt; empty = bucket root)
  TIDE_S3_DEFAULT_BUCKET=parquetgarage               (fallback when nothing set; '' to disable)
  TIDE_ARTIFACTS_S3_SYNC=1                           (master pull switch)
  TIDE_ARTIFACTS_S3_PUSH=1                           (master push switch)

Uses the EC2 instance IAM role or standard AWS credential env vars.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


SCOPE_DB = "db"
SCOPE_CSVS = "csvs"
SCOPE_PARQUETS = "parquets"
SCOPE_ARTIFACTS_75K = "artifacts_75k"
SCOPE_ARCHETYPES = "archetypes_artifacts"

ALL_SCOPES: tuple[str, ...] = (
    SCOPE_DB,
    SCOPE_CSVS,
    SCOPE_PARQUETS,
    SCOPE_ARTIFACTS_75K,
    SCOPE_ARCHETYPES,
)

_MODEL_DATA_REL = Path("model/data")
_ARTIFACTS_75K_REL = Path("model/artifacts_75k")
_ARCHETYPES_REL = Path("model/archetypes_artifacts")

# 7-label training / actuals CSVs (weekly cron + serving). Kept here so the
# sync scope docs stay explicit; pull/push still uses model/data/*.csv minus
# _CSV_SYNC_EXCLUDE (these names are not excluded).
_CSV_SYNC_CORE_7LABEL: frozenset[str] = frozenset(
    {
        "current_data_all.csv",
        "bigrelease_alist_75k_all.csv",
    }
)

# Static / legacy — never pull/push via automated csv scope (refresh_weekly, startup sync).
_CSV_SYNC_EXCLUDE: frozenset[str] = frozenset(
    {
        "2025_revenue_catalog.csv",
        # Replaced by monthly_share_and_qtd.csv / monthly_share_labels.csv
        "quarterly_share_and_qtd.csv",
        "quarterly_share_level3.csv",
        # Replaced by current_data_all.csv / bigrelease_alist_75k_all.csv
        "Current_Data.csv",
        "current_data.csv",
        "alist_75k.csv",
        "bigreleaseflag_75k.csv",
        "bigrelease_75k.csv",
        "alist_and_flag.csv",
    }
)


def _csv_sync_excluded(path: Path | str) -> bool:
    return Path(path).name in _CSV_SYNC_EXCLUDE


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _norm_s3_prefix(prefix: str) -> str:
    p = (prefix or "").strip().strip("/")
    return f"{p}/" if p else ""


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    u = uri.strip()
    low = u.lower()
    if not low.startswith("s3://"):
        raise ValueError(f"TIDE_ARTIFACTS_S3_URI must start with s3://, got {uri!r}")
    rest = u[5:]
    parts = rest.split("/", 1)
    bucket = parts[0].strip()
    if not bucket:
        raise ValueError(f"Invalid s3 URI (empty bucket): {uri!r}")
    key_prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    return bucket, key_prefix


def _resolve_bucket_prefix(action: str) -> Optional[tuple[str, str]]:
    """Pick (bucket, prefix) from env, or None to skip the operation."""
    uri = os.environ.get("TIDE_ARTIFACTS_S3_URI", "").strip()
    bucket = os.environ.get("TIDE_ARTIFACTS_S3_BUCKET", "").strip()
    prefix = os.environ.get("TIDE_ARTIFACTS_S3_PREFIX", "").strip()
    if uri:
        bucket, prefix = _parse_s3_uri(uri)
    elif bucket:
        pass
    else:
        default_bucket = os.environ.get("TIDE_S3_DEFAULT_BUCKET", "parquetgarage").strip()
        if default_bucket:
            bucket = default_bucket
            prefix = ""
            logger.info(
                "S3 artifact %s: using default bucket=%s prefix=<root> "
                "(set TIDE_S3_DEFAULT_BUCKET= to disable).",
                action,
                bucket,
            )
        else:
            logger.info(
                "S3 artifact %s skipped (no bucket configured and "
                "TIDE_S3_DEFAULT_BUCKET is empty).",
                action,
            )
            return None
    return bucket, prefix


def _normalize_scopes(scopes: Optional[Iterable[str]]) -> set[str]:
    if scopes is None:
        return set(ALL_SCOPES)
    out = {s.strip().lower() for s in scopes if s}
    unknown = out - set(ALL_SCOPES)
    if unknown:
        logger.warning("S3 sync: ignoring unknown scope(s): %s", sorted(unknown))
    return out & set(ALL_SCOPES)


# ---------------------------------------------------------------------------
# PULL
# ---------------------------------------------------------------------------

def sync_artifacts_from_s3_if_configured(
    scopes: Optional[Iterable[str]] = None,
) -> None:
    """Pull selected scopes from S3 to the local repo root."""
    if os.environ.get("TIDE_ARTIFACTS_S3_SYNC", "1").strip().lower() in ("0", "false", "no", "off"):
        logger.info("S3 artifact pull disabled (TIDE_ARTIFACTS_S3_SYNC=0).")
        return

    resolved = _resolve_bucket_prefix("pull")
    if resolved is None:
        return
    bucket, prefix = resolved

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.error("boto3 is required for S3 artifact pull. Install: pip install boto3")
        return

    selected = _normalize_scopes(scopes)
    if not selected:
        logger.info("S3 artifact pull: no scopes selected; nothing to do.")
        return

    pfx = _norm_s3_prefix(prefix)
    root = _repo_root()
    client = boto3.client("s3")

    def download_key(key: str, dest: Path, expected_size: int | None = None) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if expected_size is not None and dest.is_file():
            try:
                if dest.stat().st_size == int(expected_size):
                    logger.info(
                        "S3 pull: skip unchanged object s3://%s/%s (size=%d)",
                        bucket, key, int(expected_size),
                    )
                    return
            except OSError:
                pass
        tmp = dest.with_suffix(dest.suffix + ".partial")
        logger.info("S3 pull: s3://%s/%s -> %s", bucket, key, dest)
        try:
            client.download_file(bucket, key, str(tmp))
            tmp.replace(dest)
        except OSError as e:
            # No-space fallback: in-place download avoids needing 2x file size.
            if getattr(e, "errno", None) == 28:
                logger.warning(
                    "S3 pull: no space left while temp-downloading %s; "
                    "retrying with in-place overwrite.",
                    key,
                )
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                if dest.exists():
                    try:
                        dest.unlink()
                    except OSError:
                        pass
                client.download_file(bucket, key, str(dest))
            else:
                raise

    def _head_exists(key: str) -> bool:
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def _list_keys(s3_sub: str) -> tuple[str, list[dict]]:
        """Return (resolved_prefix, list-of-objects). Falls back to bucket-root prefix when prefixed path is empty."""
        full_prefix = f"{pfx}{s3_sub}".replace("//", "/")

        def _count(p: str) -> int:
            pages = client.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=p, MaxKeys=1
            )
            for page in pages:
                return len(page.get("Contents", []) or [])
            return 0

        if _count(full_prefix) == 0 and pfx:
            root_prefix = s3_sub.lstrip("/")
            if _count(root_prefix) > 0:
                logger.warning(
                    "S3 pull: no objects under prefixed path s3://%s/%s; "
                    "falling back to bucket-root path s3://%s/%s.",
                    bucket, full_prefix, bucket, root_prefix,
                )
                full_prefix = root_prefix

        objs: list[dict] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []) or []:
                if obj["Key"].endswith("/"):
                    continue
                if not obj["Key"].startswith(full_prefix):
                    continue
                objs.append(obj)
        return full_prefix, objs

    def _sync_tree(s3_sub: str, local_rel: Path, suffix_filter: Optional[set[str]] = None) -> int:
        full_prefix, objs = _list_keys(s3_sub)
        if not objs:
            logger.warning("S3 pull: no objects under s3://%s/%s", bucket, full_prefix)
            return 0
        n = 0
        for obj in objs:
            key = obj["Key"]
            rel = key[len(full_prefix):].lstrip("/")
            if suffix_filter is not None:
                ext = Path(rel).suffix.lower()
                if ext not in suffix_filter:
                    continue
            basename = Path(rel).name if rel else Path(key).name
            if suffix_filter == {".csv"} and _csv_sync_excluded(basename):
                logger.info(
                    "S3 pull: skip excluded csv s3://%s/%s (static; not synced by cron)",
                    bucket, key,
                )
                continue
            dest = (root / local_rel / rel) if rel else (root / local_rel / Path(key).name)
            # Do not clobber a locally newer file (e.g. Snowflake refresh in
            # flight, then API OOM-restart pulls stale S3).
            s3_mt = obj.get("LastModified")
            if dest.is_file() and s3_mt is not None:
                local_mt = datetime.fromtimestamp(dest.stat().st_mtime, tz=timezone.utc)
                remote_mt = s3_mt if s3_mt.tzinfo else s3_mt.replace(tzinfo=timezone.utc)
                if local_mt >= remote_mt:
                    logger.info(
                        "S3 pull: skip %s (local mtime %s >= s3 %s)",
                        dest.name,
                        local_mt.isoformat(),
                        remote_mt.isoformat(),
                    )
                    continue
            download_key(key, dest, expected_size=obj.get("Size"))
            n += 1
        logger.info("S3 pull: %d objects under s3://%s/%s (suffixes=%s)",
                    n, bucket, full_prefix, sorted(suffix_filter) if suffix_filter else "*")
        return n

    if SCOPE_DB in selected:
        db_key = f"{pfx}marketshare_data.db".replace("//", "/")
        root_db_key = "marketshare_data.db"
        if not _head_exists(db_key) and pfx and _head_exists(root_db_key):
            logger.warning(
                "S3 pull: %s not found; falling back to bucket-root key %s.",
                db_key, root_db_key,
            )
            db_key = root_db_key
        try:
            db_head = client.head_object(Bucket=bucket, Key=db_key)
            dest = root / "marketshare_data.db"
            s3_mt = db_head.get("LastModified")
            # Same mtime guard as CSV/parquet trees. Size-only skip is not
            # enough: a filled artist index can differ in size from S3, and
            # tmp.replace() would unlink the live file (search wipe).
            if dest.is_file() and s3_mt is not None:
                local_mt = datetime.fromtimestamp(dest.stat().st_mtime, tz=timezone.utc)
                remote_mt = s3_mt if s3_mt.tzinfo else s3_mt.replace(tzinfo=timezone.utc)
                if local_mt >= remote_mt:
                    logger.info(
                        "S3 pull: skip %s (local mtime %s >= s3 %s)",
                        dest.name,
                        local_mt.isoformat(),
                        remote_mt.isoformat(),
                    )
                else:
                    download_key(
                        db_key, dest, expected_size=db_head.get("ContentLength")
                    )
            else:
                download_key(
                    db_key, dest, expected_size=db_head.get("ContentLength")
                )
        except ClientError as e:
            logger.warning("S3 pull: database object missing: s3://%s/%s (%s)",
                           bucket, db_key, e)

    if SCOPE_CSVS in selected:
        _sync_tree("model/data/", _MODEL_DATA_REL, suffix_filter={".csv"})

    if SCOPE_PARQUETS in selected:
        _sync_tree("model/data/", _MODEL_DATA_REL, suffix_filter={".parquet"})

    if SCOPE_ARTIFACTS_75K in selected:
        _sync_tree("model/artifacts_75k/", _ARTIFACTS_75K_REL)

    if SCOPE_ARCHETYPES in selected:
        _sync_tree("model/archetypes_artifacts/", _ARCHETYPES_REL)

    logger.info(
        "S3 artifact pull finished (bucket=%s, prefix=%r, scopes=%s).",
        bucket, pfx, sorted(selected),
    )


# ---------------------------------------------------------------------------
# PUSH
# ---------------------------------------------------------------------------

def sync_artifacts_to_s3_if_configured(
    scopes: Optional[Iterable[str]] = None,
) -> None:
    """Push selected scopes from local repo root to S3."""
    if os.environ.get("TIDE_ARTIFACTS_S3_PUSH", "1").strip().lower() in ("0", "false", "no", "off"):
        logger.info("S3 artifact push disabled (TIDE_ARTIFACTS_S3_PUSH=0).")
        return

    resolved = _resolve_bucket_prefix("push")
    if resolved is None:
        return
    bucket, prefix = resolved

    try:
        import boto3
    except ImportError:
        logger.error("boto3 is required for S3 artifact push. Install: pip install boto3")
        return

    selected = _normalize_scopes(scopes)
    if not selected:
        logger.info("S3 artifact push: no scopes selected; nothing to do.")
        return

    pfx = _norm_s3_prefix(prefix)
    root = _repo_root()
    client = boto3.client("s3")

    def upload_file(path: Path, key: str) -> None:
        if not path.is_file():
            return
        logger.info("S3 push: %s -> s3://%s/%s", path, bucket, key)
        client.upload_file(str(path), bucket, key)

    def _checkpoint_sqlite(path: Path) -> None:
        if not path.is_file():
            return
        try:
            with sqlite3.connect(str(path)) as conn:
                conn.execute("PRAGMA wal_checkpoint(FULL)")
        except sqlite3.Error:
            logger.warning(
                "S3 push: wal_checkpoint failed for %s; uploading anyway",
                path,
                exc_info=True,
            )

    def _upload_tree(local_dir: Path, s3_sub: str, suffix_filter: Optional[set[str]] = None) -> int:
        if not local_dir.exists():
            return 0
        n = 0
        full_prefix = f"{pfx}{s3_sub}".replace("//", "/")
        for fp in local_dir.rglob("*"):
            if not fp.is_file():
                continue
            if suffix_filter is not None and fp.suffix.lower() not in suffix_filter:
                continue
            if suffix_filter == {".csv"} and _csv_sync_excluded(fp):
                logger.info(
                    "S3 push: skip excluded csv %s (static; not synced by cron)",
                    fp,
                )
                continue
            rel = fp.relative_to(local_dir).as_posix()
            key = f"{full_prefix}{rel}".replace("//", "/")
            upload_file(fp, key)
            n += 1
        return n

    if SCOPE_DB in selected:
        db_path = root / "marketshare_data.db"
        _checkpoint_sqlite(db_path)
        upload_file(db_path, f"{pfx}marketshare_data.db".replace("//", "/"))

    if SCOPE_CSVS in selected:
        _upload_tree(root / _MODEL_DATA_REL, "model/data/", suffix_filter={".csv"})

    if SCOPE_PARQUETS in selected:
        _upload_tree(root / _MODEL_DATA_REL, "model/data/", suffix_filter={".parquet"})

    if SCOPE_ARTIFACTS_75K in selected:
        _upload_tree(root / _ARTIFACTS_75K_REL, "model/artifacts_75k/")

    if SCOPE_ARCHETYPES in selected:
        _upload_tree(root / _ARCHETYPES_REL, "model/archetypes_artifacts/")

    logger.info(
        "S3 artifact push finished (bucket=%s, prefix=%r, scopes=%s).",
        bucket, pfx, sorted(selected),
    )


# ---------------------------------------------------------------------------
# Convenience entrypoints used by API/pipeline stages
# ---------------------------------------------------------------------------

def sync_serving_inputs_from_s3() -> None:
    """API startup: db + weekly CSVs + forecast artifacts + boot.json."""
    sync_artifacts_from_s3_if_configured(
        scopes={SCOPE_DB, SCOPE_CSVS, SCOPE_ARTIFACTS_75K, SCOPE_ARCHETYPES},
    )
    download_boot_json_from_s3()


def sync_weekly_inputs_from_s3() -> None:
    """refresh_weekly start: pull only what CSV-only training needs."""
    sync_artifacts_from_s3_if_configured(
        scopes={SCOPE_DB, SCOPE_CSVS, SCOPE_ARTIFACTS_75K},
    )
    download_boot_json_from_s3()


def sync_weekly_outputs_to_s3() -> None:
    """refresh_weekly end: push only what CSV-only training updates."""
    sync_artifacts_to_s3_if_configured(
        scopes={SCOPE_DB, SCOPE_CSVS, SCOPE_ARTIFACTS_75K},
    )
    # boot.json is uploaded by export_boot_json_snapshot / upload_boot_json;
    # re-push local file if present so weekly outputs stay consistent.
    local_boot = _repo_root() / _MODEL_DATA_REL / "boot.json"
    if local_boot.is_file():
        upload_boot_json(local_boot.read_bytes())


def sync_full_inputs_from_s3() -> None:
    """Full refresh_data: pull everything (csv + parquets + artifacts + archetypes + db)."""
    sync_artifacts_from_s3_if_configured(scopes=ALL_SCOPES)


def sync_full_outputs_to_s3() -> None:
    """Full refresh_data: push everything."""
    sync_artifacts_to_s3_if_configured(scopes=ALL_SCOPES)


def sync_db_from_s3() -> None:
    sync_artifacts_from_s3_if_configured(scopes={SCOPE_DB})


def sync_db_to_s3() -> None:
    sync_artifacts_to_s3_if_configured(scopes={SCOPE_DB})


def sync_parquets_from_s3() -> None:
    sync_artifacts_from_s3_if_configured(scopes={SCOPE_PARQUETS})


def sync_parquets_to_s3() -> None:
    sync_artifacts_to_s3_if_configured(scopes={SCOPE_PARQUETS})


# ---------------------------------------------------------------------------
# boot.json (startup payload for the frontend)
# ---------------------------------------------------------------------------

_BOOT_JSON_NAME = "boot.json"
_BOOT_LOCAL_REL = _MODEL_DATA_REL / _BOOT_JSON_NAME
_BOOT_CACHE_CONTROL = "public, max-age=86400, stale-while-revalidate=3600"


def upload_boot_json(body: bytes | str) -> Optional[str]:
    """
    Upload boot.json to the artifacts bucket root (s3://…/boot.json) and mirror
    under model/data/boot.json.

    Returns the primary s3 URI, or None when push is disabled / unconfigured.
    """
    if os.environ.get("TIDE_ARTIFACTS_S3_PUSH", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        logger.info("S3 boot.json push disabled (TIDE_ARTIFACTS_S3_PUSH=0).")
        return None

    resolved = _resolve_bucket_prefix("push")
    if resolved is None:
        return None
    bucket, prefix = resolved
    pfx = _norm_s3_prefix(prefix)

    data = body.encode("utf-8") if isinstance(body, str) else body
    local = _repo_root() / _BOOT_LOCAL_REL
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.is_file() or local.read_bytes() != data:
        local.write_bytes(data)

    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not installed; cannot upload boot.json")
        return None

    client = boto3.client("s3")
    root_key = f"{pfx}{_BOOT_JSON_NAME}".replace("//", "/")
    data_key = f"{pfx}model/data/{_BOOT_JSON_NAME}".replace("//", "/")
    extra = {
        "ContentType": "application/json",
        "CacheControl": _BOOT_CACHE_CONTROL,
    }
    for key in (root_key, data_key):
        client.put_object(Bucket=bucket, Key=key, Body=data, **extra)
        logger.info("S3 push: boot.json -> s3://%s/%s (%d bytes)", bucket, key, len(data))
    return f"s3://{bucket}/{root_key}"


def download_boot_json_from_s3() -> bool:
    """
    Pull boot.json into model/data/boot.json (bucket root key, then model/data/).
    Returns True if a local file was written/updated.
    """
    if os.environ.get("TIDE_ARTIFACTS_S3_SYNC", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return False

    resolved = _resolve_bucket_prefix("pull")
    if resolved is None:
        return False
    bucket, prefix = resolved
    pfx = _norm_s3_prefix(prefix)

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.warning("boto3 not installed; cannot download boot.json")
        return False

    client = boto3.client("s3")
    candidates = [
        f"{pfx}{_BOOT_JSON_NAME}".replace("//", "/"),
        f"{pfx}model/data/{_BOOT_JSON_NAME}".replace("//", "/"),
        _BOOT_JSON_NAME,
        f"model/data/{_BOOT_JSON_NAME}",
    ]
    seen: set[str] = set()
    keys: list[str] = []
    for k in candidates:
        if k not in seen:
            seen.add(k)
            keys.append(k)

    dest = _repo_root() / _BOOT_LOCAL_REL
    for key in keys:
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except ClientError:
            continue
        expected = head.get("ContentLength")
        if dest.is_file() and expected is not None and dest.stat().st_size == int(expected):
            logger.info("S3 pull: skip unchanged boot.json s3://%s/%s", bucket, key)
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(dest))
        logger.info("S3 pull: boot.json <- s3://%s/%s", bucket, key)
        return True

    logger.info("S3 pull: boot.json not found under s3://%s (prefix=%r)", bucket, pfx)
    return False
