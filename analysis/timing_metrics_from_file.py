"""Recompute Table 2 timing-classification metrics from a saved result file
(strategy E2-d; reproducibility for ka1t / general soundness).

evaluate_turn-level.py computes Precision/Recall/F1/FPR only in --do-incremental
mode (FPR needs instant turns as true negatives). This script recomputes the same
metrics from an already-saved *incremental* turn-level TIME JSONL, so Table 2 can
be reproduced without re-running the model on a GPU.

Binary mapping (same as utils.metrics): GT/pred minutes == 0 -> instant, > 0 -> delayed.

Input rows: {"generated": <pred minutes float>, "time_elapsed": <GT string>, ...}

Usage
-----
    python analysis/timing_metrics_from_file.py \
        results/turn-level_seongbo--timer-3b_time_zeroshot_incremental_*.jsonl
"""

import argparse
import glob
import json
import sys

from utils.metrics import f1_score, fpr, precision, recall, rmsle
from utils.postprocess import convert_to_minutes


def to_minutes(v):
    return float(v) if isinstance(v, (int, float)) else convert_to_minutes(str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="incremental turn-level time result JSONL (globs ok; latest used)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.file))
    if not paths:
        sys.exit(f"No files match: {args.file}")
    path = paths[-1]
    rows = [json.loads(l) for l in open(path)]

    y_pred = [to_minutes(r["generated"]) for r in rows]
    y_true = [to_minutes(r["time_elapsed"]) for r in rows]

    n_pos = sum(1 for v in y_true if v > 0)      # delayed (positives)
    n_neg = len(y_true) - n_pos                   # instant (negatives)

    print(f"file: {path}")
    print(f"rows: {len(rows)}  (GT delayed={n_pos}, GT instant={n_neg})")
    if n_neg == 0:
        print("WARNING: no GT-instant rows -> FPR undefined. This looks like a "
              "NON-incremental file; F1/FPR require the --do-incremental output.")
    # utils.metrics returns percentages (x100); Table 2 reports fractions (0-1).
    print("\n===== Table 2 metrics (recomputed, 0-1 scale) =====")
    print(f"Precision: {precision(y_true, y_pred)/100:6.3f}")
    print(f"Recall   : {recall(y_true, y_pred)/100:6.3f}")
    print(f"F1       : {f1_score(y_true, y_pred)/100:6.3f}")
    print(f"FPR      : {fpr(y_true, y_pred)/100:6.3f}")
    print(f"RMSLE    : {rmsle(y_true, y_pred):6.4f}")


if __name__ == "__main__":
    main()
