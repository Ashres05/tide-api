"""Proxy Evergreen album-report onto the Tide API listener.

Cloudflare already publishes this process as api.hightide.fm. Next/Vercel
should set EVERGREEN_URL=https://api.hightide.fm (same Access as other Tide
calls). Direct loopback remains http://127.0.0.1:8080 for SSH tunnels.
"""

from __future__ import annotations

import urllib.error
import urllib.request

from fastapi import HTTPException, Request, Response

_UPSTREAM = "http://127.0.0.1:8080"
_TIMEOUT_S = 600


def forward_evergreen(request: Request, upstream_path: str) -> Response:
    query = request.url.query
    url = f"{_UPSTREAM}{upstream_path}"
    if query:
        url = f"{url}?{query}"
    req = urllib.request.Request(url, method="GET")
    accept = request.headers.get("accept")
    if accept:
        req.add_header("Accept", accept)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            body = resp.read()
            headers = {}
            disposition = resp.headers.get("Content-Disposition")
            if disposition:
                headers["Content-Disposition"] = disposition
            media = resp.headers.get("Content-Type") or "application/json"
            return Response(
                content=body,
                status_code=getattr(resp, "status", 200) or 200,
                media_type=media,
                headers=headers,
            )
    except urllib.error.HTTPError as exc:
        media = exc.headers.get("Content-Type") if exc.headers else None
        return Response(
            content=exc.read(),
            status_code=exc.code,
            media_type=media or "application/json",
        )
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Evergreen album-report API is not reachable at "
                f"{_UPSTREAM} ({exc.reason}). "
                "Start evergreen-api.service on this host."
            ),
        ) from exc
