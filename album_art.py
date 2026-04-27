"""
Album-art lookup against `s3://<bucket>/album_art/`.

Files are uploaded once per release with the convention
``album_art/mrelg_<mrelg_id_lowercase>_<spotify_id>.jpg``. This module
indexes the prefix at startup (or first request) into an mrelg_id → S3-key
dict, then serves the image bytes through the API. Sits behind Cloudflare
Tunnel + Access — the upstream URL the frontend hits is
``/v1/album_art/{mrelg_id}``, which we set ``Cache-Control: max-age=86400,
immutable`` on so Cloudflare's edge caches each cover for a day.

Why proxy instead of public S3 URLs:
  * The bucket stays IAM-locked (no policy change), consistent with the
    rest of the artifact tree. Album art alone going public would be a
    weird inconsistency, even though the covers themselves are not
    sensitive.
  * Cloudflare cache + browser cache are sufficient for the per-cover
    request volume — cold pull is one S3 HEAD + GET (~30ms), warm hits
    never reach the API.

Cache invalidation is hooked into reload_artifacts() so any new uploads
land within one refresh_weekly cycle without restart.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

PREFIX = "album_art/"
# Filename convention: album_art/mrelg_<mrelg_id_lowercase>_<spotify_id>.jpg
# We accept any extension for forward-compatibility (.png, .webp).
_FILENAME_RE = re.compile(
    r"^album_art/mrelg_(?P<mrelg>mrelg[0-9a-f]+)_[A-Za-z0-9]+\.[A-Za-z]+$"
)

_lock = threading.Lock()
_cache: dict[str, str] | None = None  # mrelg_id_lower → full s3 key


def _resolve_bucket() -> Optional[str]:
    """Same precedence as api/s3_pull.py — single source of truth for bucket."""
    uri = os.environ.get("TIDE_ARTIFACTS_S3_URI", "").strip()
    if uri.lower().startswith("s3://"):
        return uri[5:].split("/", 1)[0].strip() or None
    bucket = os.environ.get("TIDE_ARTIFACTS_S3_BUCKET", "").strip()
    if bucket:
        return bucket
    return os.environ.get("TIDE_S3_DEFAULT_BUCKET", "parquetgarage").strip() or None


def _build_index() -> dict[str, str]:
    """List the album_art/ prefix and parse mrelg_id from each filename."""
    bucket = _resolve_bucket()
    if not bucket:
        logger.info("album_art: no S3 bucket configured; index empty.")
        return {}

    try:
        import boto3
    except ImportError:
        logger.warning("album_art: boto3 not installed; index empty.")
        return {}

    client = boto3.client("s3")
    out: dict[str, str] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=PREFIX):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            m = _FILENAME_RE.match(key)
            if not m:
                # Skip outliers (e.g. slug-based legacy uploads). They tend
                # to duplicate a same-album entry that DOES match the convention.
                continue
            out[m.group("mrelg")] = key
    logger.info("album_art: indexed %d covers from s3://%s/%s", len(out), bucket, PREFIX)
    return out


def _get_index() -> dict[str, str]:
    global _cache
    with _lock:
        if _cache is None:
            _cache = _build_index()
        return _cache


def clear_cache() -> None:
    """Drop the cached index. Called from reload_artifacts() after a refresh
    so newly uploaded covers become visible without an API restart."""
    global _cache
    with _lock:
        _cache = None


def lookup_s3_key(mrelg_id: str) -> Optional[str]:
    """Return the full S3 key for an mrelg_id, or None if no cover exists."""
    if not mrelg_id:
        return None
    return _get_index().get(mrelg_id.strip().lower())


def fetch_bytes(mrelg_id: str) -> Optional[tuple[bytes, str]]:
    """
    Return (image_bytes, content_type) for the given mrelg_id, or None.
    Streams the object body fully into memory — covers are <500KB each so
    this is fine; if we ever serve large images, switch to streaming response.
    """
    key = lookup_s3_key(mrelg_id)
    if not key:
        return None

    bucket = _resolve_bucket()
    if not bucket:
        return None

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return None

    client = boto3.client("s3")
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()
        content_type = resp.get("ContentType") or _guess_content_type(key)
        return body, content_type
    except ClientError as e:
        logger.warning("album_art: S3 get_object failed for %s: %s", key, e)
        return None


def _guess_content_type(key: str) -> str:
    ext = key.rsplit(".", 1)[-1].lower()
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext, "application/octet-stream")
