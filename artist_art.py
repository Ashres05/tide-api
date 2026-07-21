"""
Artist profile-image lookup against ``s3://<bucket>/artist_art/``.

Files are uploaded by ``scripts/fetch_roster_artist_artwork.py`` as:

* ``artist_art/{LUMINATE_ARTIST_ID}.jpeg``

This module indexes the prefix (or first request) into an artist_id → S3-key
dict, then serves image bytes through the API. The frontend hits:

    GET /v1/artist_art/{luminate_artist_id}

with ``Cache-Control: public, max-age=86400, immutable`` so Cloudflare/browser
cache each image for a day.

Same rationale as ``album_art.py``: keep the bucket IAM-locked and proxy
through the API rather than public S3 URLs.

Cache invalidation is hooked into ``reload_artifacts()``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

PREFIX = "artist_art/"
# Uploader writes artist_art/AR….jpeg; accept other image extensions too.
_FILENAME_RE = re.compile(
    r"^artist_art/(?P<artist_id>AR[0-9A-Fa-f]+)\.[A-Za-z]+$"
)

_lock = threading.Lock()
_cache: dict[str, str] | None = None  # artist_id_upper → full s3 key


def _resolve_bucket() -> Optional[str]:
    """Same precedence as api/s3_pull.py / album_art.py."""
    uri = os.environ.get("TIDE_ARTIFACTS_S3_URI", "").strip()
    if uri.lower().startswith("s3://"):
        return uri[5:].split("/", 1)[0].strip() or None
    bucket = os.environ.get("TIDE_ARTIFACTS_S3_BUCKET", "").strip()
    if bucket:
        return bucket
    return os.environ.get("TIDE_S3_DEFAULT_BUCKET", "parquetgarage").strip() or None


def _build_index() -> dict[str, str]:
    """List the artist_art/ prefix and parse LUMINATE_ARTIST_ID from each filename."""
    bucket = _resolve_bucket()
    if not bucket:
        logger.info("artist_art: no S3 bucket configured; index empty.")
        return {}

    try:
        import boto3
    except ImportError:
        logger.warning("artist_art: boto3 not installed; index empty.")
        return {}

    client = boto3.client("s3")
    out: dict[str, str] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=PREFIX):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            m = _FILENAME_RE.match(key)
            if m:
                out[m.group("artist_id").upper()] = key
    logger.info(
        "artist_art: indexed %d images from s3://%s/%s", len(out), bucket, PREFIX
    )
    return out


def _get_index() -> dict[str, str]:
    global _cache
    with _lock:
        if _cache is None:
            _cache = _build_index()
        return _cache


def clear_cache() -> None:
    """Drop the cached index. Called from reload_artifacts()."""
    global _cache
    with _lock:
        _cache = None


def lookup_s3_key(luminate_artist_id: str) -> Optional[str]:
    """Return the full S3 key for an artist id, or None if no image exists."""
    if not luminate_artist_id:
        return None
    aid = luminate_artist_id.strip().upper()
    if not aid:
        return None
    key = _get_index().get(aid)
    if key:
        return key
    # Cold path: try the canonical upload key without waiting for a full reindex.
    candidate = f"{PREFIX}{aid}.jpeg"
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
        client.head_object(Bucket=bucket, Key=candidate)
    except ClientError:
        return None
    with _lock:
        if _cache is not None:
            _cache[aid] = candidate
    return candidate


def fetch_bytes(luminate_artist_id: str) -> Optional[tuple[bytes, str]]:
    """
    Return (image_bytes, content_type) for the given LUMINATE_ARTIST_ID, or None.
    """
    key = lookup_s3_key(luminate_artist_id)
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
        logger.warning("artist_art: S3 get_object failed for %s: %s", key, e)
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
