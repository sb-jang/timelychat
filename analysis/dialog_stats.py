"""Dispersion + significance for dialog-level LLM-judge scores (strategy E2-b; Fig 3).

Fig 3 reports only mean scores with significance asterisks. Reviewers (ka1t #1,
sV8H #4) ask for the underlying spread. This aggregates per-dialog judge scores
into n / mean / std / 95% CI per (system, metric) and runs a pairwise Welch
t-test of each baseline against a reference system (default TIMER) to reconfirm
the Fig 3 asterisks.

Input: one laaj.py output JSONL per system, each line:
       {"idx": int, "metric": str, "score": int(1-5), "explanation": str}
Pass files as SYSTEM=PATH so the table is labelled, e.g.
    timer=results/claude-sonnet-4-5-eval_dialog-level_seongbo--timer-3b_.jsonl

Usage
-----
    python analysis/dialog_stats.py \
        timer=results/..timer-3b_.jsonl \
        gpt-4o=results/..gpt-4o_.jsonl \
        llama-70b=results/..llama..jsonl \
        --ref timer
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_ind


def load_scores(path):
    """metric -> list[score] from a laaj JSONL."""
    by_metric = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        s = r.get("score")
        if s is not None:
            by_metric[r["metric"]].append(float(s))
    return by_metric


def ci95(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    sd = np.std(xs, ddof=1)
    half = student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="SYSTEM=PATH.jsonl (laaj dialog-level output)")
    ap.add_argument("--ref", default=None, help="reference system for pairwise t-tests (default: first)")
    args = ap.parse_args()

    data = {}
    for spec in args.systems:
        if "=" not in spec:
            sys.exit(f"expected SYSTEM=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        data[name] = load_scores(path)

    ref = args.ref or next(iter(data))
    if ref not in data:
        sys.exit(f"--ref {ref} not among systems: {list(data)}")

    metrics = []
    for d in data.values():
        for m in d:
            if m not in metrics:
                metrics.append(m)

    for metric in metrics:
        print(f"\n===== {metric} =====")
        print(f"{'system':16s} {'n':>4s} {'mean':>6s} {'std':>6s} {'95% CI':>16s} "
              f"{'t vs '+ref:>12s} {'p':>8s}")
        ref_scores = data[ref].get(metric, [])
        for name, d in data.items():
            xs = d.get(metric, [])
            if not xs:
                print(f"{name:16s} {'-':>4s}  (no scores for this metric)")
                continue
            mean, sd = np.mean(xs), (np.std(xs, ddof=1) if len(xs) > 1 else float("nan"))
            half = ci95(xs)
            ci = f"[{mean-half:.2f},{mean+half:.2f}]" if half == half else "n/a"
            if name == ref or not ref_scores:
                tval = pval = float("nan")
            else:
                tval, pval = ttest_ind(ref_scores, xs, equal_var=False)
            star = "*" if (pval == pval and pval < 0.05) else ""
            print(f"{name:16s} {len(xs):>4d} {mean:6.2f} {sd:6.2f} {ci:>16s} "
                  f"{tval:12.2f} {pval:8.3f}{star}")
    print("\n* p<0.05 (Welch two-sided) vs reference; reconfirms Fig 3 asterisks.")


if __name__ == "__main__":
    main()
