"""
Canonical marketshare label set (8 entities) and scrub / big-release column maps.

``LABEL_NAME`` values in ``current_data_all.csv`` match these strings exactly —
no aliasing. Scrub + flag volumes come from ``bigrelease_alist_75k_all.csv``
(column names are UPPERCASE on disk / S3).
IGA and CMG are level_3 distributors; other labels remain level_2.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Exact LABEL_NAME values from current_data_all.csv
TARGET_LABELS: List[str] = [
    "Warner Records",
    "Atlantic Music Group",
    "IGA",
    "CMG",
    "REPUBLIC Collective",
    "THE ORCHARD",
    "RCA Records",
    "Columbia Records",
]

TRAINING_CATEGORIES: List[str] = TARGET_LABELS

# Market total is always inverted from this Owner only:
# Total_Market = (AE_Volume * 100) / AE_Share
MARKET_ANCHOR_LABEL: str = "Atlantic Music Group"

# bigrelease_alist_75k_all.csv: WEEK_END_DATE + per-label A-list AE scrub volumes
ALIST_VOL_COLS: Dict[str, str] = {
    "Warner Records": "WARNER_ALBUMS",
    "Atlantic Music Group": "AMG_ALBUMS",
    "IGA": "IGA_ALBUMS",
    "CMG": "CMG_ALBUMS",
    "REPUBLIC Collective": "REPUBLIC_ALBUMS",
    "THE ORCHARD": "ORCHARD_ALBUMS",
    "RCA Records": "RCA_ALBUMS",
    "Columbia Records": "COLUMBIA_ALBUMS",
}

# Same file: per-label big-release flags (0/1)
BIG_RELEASE_COLS: Dict[str, str] = {
    "Warner Records": "BIG_RELEASE_WARNER",
    "Atlantic Music Group": "BIG_RELEASE_ATLANTIC",
    "IGA": "BIG_RELEASE_IGA",
    "CMG": "BIG_RELEASE_CMG",
    "REPUBLIC Collective": "BIG_RELEASE_REPUBLIC",
    "THE ORCHARD": "BIG_RELEASE_ORCHARD",
    "RCA Records": "BIG_RELEASE_RCA",
    "Columbia Records": "BIG_RELEASE_COLUMBIA",
}

MARKET_ALBUMS_COL: str = "MARKET_ALBUMS"
WEEK_END_COL: str = "WEEK_END_DATE"

# Serving / training CSV filenames under model/data/
CURRENT_DATA_ALL_CSV: str = "current_data_all.csv"
BIGRELEASE_ALIST_CSV: str = "bigrelease_alist_75k_all.csv"

# Legacy 2-label filenames (fallback during migration)
LEGACY_CURRENT_DATA_CSV: str = "Current_Data.csv"
LEGACY_ALIST_CSV: str = "alist_75k.csv"
LEGACY_BIGRELEASE_CSV: str = "bigreleaseflag_75k.csv"


def label_alist_pairs() -> List[Tuple[str, str]]:
    """(canonical LABEL_NAME, alist column) in TARGET_LABELS order."""
    return [(lab, ALIST_VOL_COLS[lab]) for lab in TARGET_LABELS]


def label_big_release_pairs() -> List[Tuple[str, str]]:
    """(canonical LABEL_NAME, big_release column) in TARGET_LABELS order."""
    return [(lab, BIG_RELEASE_COLS[lab]) for lab in TARGET_LABELS]


def assert_label_maps_complete() -> None:
    """Fail fast if maps drift from TARGET_LABELS."""
    missing_alist = set(TARGET_LABELS) - set(ALIST_VOL_COLS)
    missing_flag = set(TARGET_LABELS) - set(BIG_RELEASE_COLS)
    if missing_alist or missing_flag:
        raise ValueError(
            f"Incomplete label maps: alist missing {missing_alist}, "
            f"big_release missing {missing_flag}"
        )
    if set(ALIST_VOL_COLS) != set(TARGET_LABELS) or set(BIG_RELEASE_COLS) != set(TARGET_LABELS):
        raise ValueError("ALIST_VOL_COLS / BIG_RELEASE_COLS keys must equal TARGET_LABELS")
