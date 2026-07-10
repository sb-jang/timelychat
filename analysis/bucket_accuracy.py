"""Coarse-bucket accuracy for response-timing prediction (strategy E2-c; sV8H #2).

Motivation: a single point estimate (RMSLE) understates performance because the
task is really about picking the right *delay regime*. We report accuracy over
the 6 semantic buckets of Table 6 -- exact-bucket and adjacent (+/-1 bucket) --
which directly answers "you may miss the exact minutes but do you hit the regime?"

Input: a turn-level TIME result JSONL from evaluate_turn-level.py --save-results
       (one row per example): {"generated": <pred minutes float>,
                               "time_elapsed": <GT interval string>, ...}

Usage
-----
    python analysis/bucket_accuracy.py results/turn-level_seongbo--timer-3b_time_zeroshot_*.jsonl
    python analysis/bucket_accuracy.py <file> --delayed-only   # exclude GT instant (0-5min) rows
"""

import argparse
import glob
import json
import sys

from analysis.buckets import BUCKET_NAMES, bucket_index
from utils.postprocess import convert_to_minutes


def load(path_glob: str):
    paths = sorted(glob.glob(path_glob))
    if not paths:
        sys.exit(f"No files match: {path_glob}")
    rows = [json.loads(l) for l in open(paths[-1])]
    return paths[-1], rows


def to_minutes(v):
    return float(v) if isinstance(v, (int, float)) else convert_to_minutes(str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="turn-level time result JSONL (globs ok; latest used)")
    ap.add_argument("--delayed-only", action="store_true",
                    help="exclude rows whose GT is the instant bucket (0-5 min)")
    args = ap.parse_args()

    path, rows = load(args.file)
    n_total = len(rows)

    exact = adj = kept = 0
    confusion = [[0] * len(BUCKET_NAMES) for _ in range(len(BUCKET_NAMES))]
    for r in rows:
        gt_b = bucket_index(to_minutes(r["time_elapsed"]))
        if args.delayed_only and gt_b == 0:
            continue
        pred_b = bucket_index(to_minutes(r["generated"]))
        kept += 1
        confusion[gt_b][pred_b] += 1
        exact += int(pred_b == gt_b)
        adj += int(abs(pred_b - gt_b) <= 1)

    print(f"file: {path}")
    print(f"rows: {n_total} total, {kept} scored"
          + (" (delayed-only)" if args.delayed_only else ""))
    if kept == 0:
        sys.exit("no rows to score")
    print(f"\nExact-bucket accuracy : {exact}/{kept} = {100*exact/kept:5.1f}%")
    print(f"Adjacent(+/-1) accuracy: {adj}/{kept} = {100*adj/kept:5.1f}%")

    print("\nConfusion (rows=GT bucket, cols=pred bucket):")
    header = "GT \\ pred".ljust(26) + "".join(f"{i:>6}" for i in range(len(BUCKET_NAMES)))
    print(header)
    for i, name in enumerate(BUCKET_NAMES):
        row = "".join(f"{confusion[i][j]:>6}" for j in range(len(BUCKET_NAMES)))
        print(f"{i} {name}".ljust(26) + row)


if __name__ == "__main__":
    main()
