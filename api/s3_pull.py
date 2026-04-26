"""
Pull SQLite + model artifacts from S3 into the repo root before the API serves traffic.

SQLite and joblib/pandas expect local paths. Set one of:

  TIDE_ARTIFACTS_S3_URI=s3://parquetgarage
  TIDE_ARTIFACTS_S3_URI=s3://parquetgarage/tide-api

or:

  TIDE_ARTIFACTS_S3_BUCKET=parquetgarage
  TIDE_ARTIFACTS_S3_PREFIX=          # empty = bucket root; or e.g. tide-api

Optional:

  TIDE_ARTIFACTS_S3_SYNC=0            # disable pull entirely
  TIDE_ARTIFACTS_S3_SYNC_MODEL_DATA=0   # skip model/data (large parquets) if you only need artifacts_75k + db

Uses the instance IAM role or standard AWS credential env vars (same as aws CLI).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


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


def sync_artifacts_from_s3_if_configured() -> None:
    if os.environ.get("TIDE_ARTIFACTS_S3_SYNC", "1").strip().lower() in ("0", "false", "no", "off"):
        logger.info("S3 artifact pull disabled (TIDE_ARTIFACTS_S3_SYNC=0).")
        return

    uri = os.environ.get("TIDE_ARTIFACTS_S3_URI", "").strip()
    bucket = os.environ.get("TIDE_ARTIFACTS_S3_BUCKET", "").strip()
    prefix = os.environ.get("TIDE_ARTIFACTS_S3_PREFIX", "").strip()

    if uri:
        bucket, prefix = _parse_s3_uri(uri)
    elif bucket:
        pass
    else:
        logger.info(
            "S3 artifact pull skipped (set TIDE_ARTIFACTS_S3_URI or TIDE_ARTIFACTS_S3_BUCKET)."
        )
        return

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.error(
            "boto3 is required for S3 artifact pull. Install: pip install boto3"
        )
        return

    pfx = _norm_s3_prefix(prefix)
    root = _repo_root()
    client = boto3.client("s3")
    include_data = os.environ.get("TIDE_ARTIFACTS_S3_SYNC_MODEL_DATA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    def download_key(key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        logger.info("S3 pull: s3://%s/%s -> %s", bucket, key, dest)
        client.download_file(bucket, key, str(tmp))
        tmp.replace(dest)

    db_key = f"{pfx}marketshare_data.db".replace("//", "/")
    try:
        client.head_object(Bucket=bucket, Key=db_key)
        download_key(db_key, root / "marketshare_data.db")
    except ClientError as e:
        logger.warning("S3 pull: database object missing or inaccessible: s3://%s/%s (%s)", bucket, db_key, e)

    def sync_tree(s3_sub: str, local_rel: Path) -> int:
        full_prefix = f"{pfx}{s3_sub}".replace("//", "/")
        n = 0
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                if not key.startswith(full_prefix):
                    continue
                rel = key[len(full_prefix) :].lstrip("/")
                dest = (root / local_rel / rel) if rel else (root / local_rel / Path(key).name)
                download_key(key, dest)
                n += 1
        if n:
            logger.info("S3 pull: %d objects under s3://%s/%s", n, bucket, full_prefix)
        else:
            logger.warning("S3 pull: no objects under s3://%s/%s", bucket, full_prefix)
        return n

    sync_tree("model/artifacts_75k/", Path("model/artifacts_75k"))
    if include_data:
        sync_tree("model/data/", Path("model/data"))
    else:
        logger.info("S3 pull: skipping model/data (TIDE_ARTIFACTS_S3_SYNC_MODEL_DATA=0).")
    sync_tree("model/archetypes_artifacts/", Path("model/archetypes_artifacts"))

    logger.info("S3 artifact pull finished (bucket=%s, prefix=%r).", bucket, pfx)
