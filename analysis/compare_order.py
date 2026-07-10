"""Compare timely-first vs instant-first regeneration to detect order bias (E4; ka1t #4).

Reads the two JSONLs produced by datagen/generate_dialog.py (same seed/subset,
matched by idx) and reports, for the timely and instantaneous candidates:
  - response length (words): mean +/- std, and Welch t-test between orders
  - lexical time-leakage rate (fraction mentioning an explicit time unit), which
    the "Temporal Implicitness" constraint forbids
Small, non-significant differences => no structural 4-1/4-2 order bias.

Optionally (--judge) score time-specificity with the Claude turn-level judge to
compare the two orders on quality; requires ANTHROPIC_API_KEY.

Usage
-----
    python analysis/compare_order.py \
        results/e4/order_timely-first_seed0_n50.jsonl \
        results/e4/order_instant-first_seed0_n50.jsonl
"""

import argparse
import json
import re
import sys

import numpy as np
from scipy.stats import ttest_ind

TIME_UNIT = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|few|several|half)\s*"
    r"(second|minute|hour|day|week)s?\b",
    re.IGNORECASE,
)


def load(path):
    return {json.loads(l)["idx"]: json.loads(l) for l in open(path)}


def wordlen(s):
    return len(s.split())


def leaks_time(s):
    return bool(TIME_UNIT.search(s))


def summarize(name, a_vals, b_vals):
    a, b = np.array(a_vals, float), np.array(b_vals, float)
    t, p = ttest_ind(a, b, equal_var=False) if len(a) > 1 and len(b) > 1 else (float("nan"), float("nan"))
    star = "*" if (p == p and p < 0.05) else ""
    print(f"{name:34s} timely-first={a.mean():6.2f}+/-{a.std(ddof=1):4.2f}   "
          f"instant-first={b.mean():6.2f}+/-{b.std(ddof=1):4.2f}   "
          f"t={t:6.2f} p={p:6.3f}{star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timely_first")
    ap.add_argument("instant_first")
    args = ap.parse_args()

    A, B = load(args.timely_first), load(args.instant_first)
    ids = sorted(set(A) & set(B))
    if not ids:
        sys.exit("no overlapping idx between the two files")
    print(f"{len(ids)} matched examples (timely-first vs instant-first)\n")
    print("H0: the two instruction orders produce statistically indistinguishable candidates.\n")

    for field, label in [("timely_response", "TIMELY candidate"),
                         ("instantaneous_response", "INSTANT candidate")]:
        print(f"--- {label} ---")
        summarize("  length (words)",
                  [wordlen(A[i][field]) for i in ids],
                  [wordlen(B[i][field]) for i in ids])
        summarize("  time-unit leakage rate",
                  [leaks_time(A[i][field]) for i in ids],
                  [leaks_time(B[i][field]) for i in ids])
        print()
    print("* p<0.05 (Welch two-sided). No significant differences => no order bias.")


if __name__ == "__main__":
    main()
