"""
Shared text normalization for artist typeahead and snapshot ARTIST_SEARCH /
TITLE_SEARCH columns.

``model_handler`` (request-time fuzzy scoring) and ``sqlite_handler``
(refresh-time persisted columns) must produce byte-identical normalized
strings. Keeping the function in one module guarantees the two paths
cannot drift.
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
