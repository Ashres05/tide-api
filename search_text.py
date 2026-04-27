"""
Shared text normalization for the artist/title search pipeline.

Both ``model_handler`` (request-time fuzzy scoring) and ``sqlite_handler``
(refresh-time persisted columns) must produce byte-identical normalized
strings; otherwise the SQL prefilter and the Python scorer disagree and
candidates silently drop. Keeping the function in one module guarantees the
two paths can never drift.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def normalize_search_text(value: Any) -> str:
    """Lowercase, strip accents, collapse whitespace, drop punctuation."""
    if value is None:
        return ""
    s = str(value)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = " ".join(s.split())
    return s
