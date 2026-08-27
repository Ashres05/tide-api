"""
Label-art lookup against ``s3://<bucket>/label_art/``.

Files are uploaded as ``label_art/{filename}`` (e.g. ``atlantic.jpg``).
The API proxies bytes through:

    GET /v1/label_art/{filename}

Same rationale as ``album_art.py`` / ``artist_art.py``: keep the bucket
IAM-locked and proxy through the API. ``Cache-Control`` is set on the
image route. Cache invalidation is hooked into ``reload_artifacts()``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

PREFIX = "label_art/"
_FILENAME_RE = re.compile(
    r"^[A-Za-z0-9._-]+\.(jpg|jpeg|png|webp|gif)$",
    re.IGNORECASE,
)


def _resolve_bucket() -> Optional[str]:
    uri = os.environ.get("TIDE_ARTIFACTS_S3_URI", "").strip()
    if uri.lower().startswith("s3://"):
        return uri[5:].split("/", 1)[0].strip() or None
    bucket = os.environ.get("TIDE_ARTIFACTS_S3_BUCKET", "").strip()
    if bucket:
        return bucket
    return os.environ.get("TIDE_S3_DEFAULT_BUCKET", "parquetgarage").strip() or None


def _safe_filename(value: str) -> Optional[str]:
    raw = (value or "").strip().replace("\\", "/")
    raw = raw.rsplit("/", 1)[-1]
    if not _FILENAME_RE.match(raw):
        return None
    return raw


def clear_cache() -> None:
    """No in-memory index; kept so reload_artifacts() can call it uniformly."""
    return None


def fetch_bytes(filename: str) -> Optional[tuple[bytes, str]]:
    name = _safe_filename(filename)
    if not name:
        return None
    bucket = _resolve_bucket()
    if not bucket:
        return None
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return None
    key = f"{PREFIX}{name}"
    client = boto3.client("s3")
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()
        content_type = resp.get("ContentType") or _guess_content_type(name)
        return body, content_type
    except ClientError as e:
        logger.warning("label_art: S3 get_object failed for %s: %s", key, e)
        return None


def _guess_content_type(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower()
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext, "application/octet-stream")
