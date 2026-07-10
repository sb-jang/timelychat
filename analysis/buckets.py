"""Shared 6-way time-interval buckets (paper Table 6).

Boundaries in minutes; each bucket is [lo, hi). The last bucket includes 1440.
"""

from bisect import bisect_right
from typing import List

# right edges (exclusive) for buckets 0..5; last edge inclusive of 1440
_EDGES = [5, 30, 120, 360, 720, 1441]
BUCKET_NAMES: List[str] = [
    "0-5 min (instant)",
    "5-30 min (short)",
    "30 min-2 hrs (moderate)",
    "2-6 hrs (long)",
    "6-12 hrs (half-day)",
    "12-24 hrs (full-day)",
]


def bucket_index(minutes: float) -> int:
    """Return 0..5 for a value in minutes. Values >1440 clamp to the last bucket."""
    m = max(0.0, float(minutes))
    idx = bisect_right(_EDGES, m)
    return min(idx, len(BUCKET_NAMES) - 1)
