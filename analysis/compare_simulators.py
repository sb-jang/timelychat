"""E3: does the dialog-level system ranking survive swapping the user simulator?

The paper's Figure 3 ordering was produced with a GPT-4o user simulator. Reviewer
sV8H #4 asks whether that ordering is an artifact of the simulator. We re-simulate
the same dialogs with Claude Sonnet 4.5 as the user, re-judge with the SAME judge,
and compare rankings.

Reads laaj dialog-level outputs for both arms and reports, per metric:
  * per-system mean +/- 95% CI (n = dialogs judged)
  * the induced ranking under each simulator
  * whether the ranking is preserved, and specifically whether the paper's two
    headline claims hold under the alternative simulator:
        - TIMER ranks 1st on Delay-Appropriateness and Time-Specificity
        - GPT-4o ranks 1st on Coherence

A rank swap between two systems whose CIs overlap is reported as NOT MEANINGFUL
(the arms simply cannot separate them at this n) rather than as a contradiction.

Usage
-----
    python -m analysis.compare_simulators \
        --baseline-sim gpt-4o --alt-sim claude-sonnet-4-5 \
        --judge claude-sonnet-4-5 \
        timer=seongbo--timer-3b gpt-4o=gpt-4o gpt-3.5=gpt-3.5-turbo \
        llama-8b=meta-llama--Llama-3.1-8B-Instruct
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

HEADLINE = {
    "Delay-Appropriateness": "timer",
    "Time-Specificity": "timer",
    "Coherence": "gpt-4o",
}


def load_scores(path):
    by_metric = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("score") is not None:
            by_metric[r["metric"]].append(float(r["score"]))
    return by_metric


def ci95(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    return student_t.ppf(0.975, n - 1) * np.std(xs, ddof=1) / np.sqrt(n)


def rank(stats):
    """systems ordered by descending mean."""
    return [name for name, _ in sorted(stats.items(), key=lambda kv: -kv[1]["mean"])]


def overlaps(a, b):
    return not (a["mean"] - a["ci"] > b["mean"] + b["ci"] or b["mean"] - b["ci"] > a["mean"] + a["ci"])


def flipped_pairs(rank_b, rank_a):
    """Pairs (x, y) ordered x>y under the baseline but y>x under the alt simulator."""
    pos_b = {s: i for i, s in enumerate(rank_b)}
    pos_a = {s: i for i, s in enumerate(rank_a)}
    return [
        (x, y)
        for i, x in enumerate(rank_b)
        for y in rank_b[i + 1 :]
        if pos_a[x] > pos_a[y]  # baseline had x above y; alt puts y above x
        and pos_b[x] < pos_b[y]
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="LABEL=MODEL_NAME_AS_IN_FILENAME")
    ap.add_argument("--baseline-sim", default="gpt-4o")
    ap.add_argument("--alt-sim", default="claude-sonnet-4-5")
    ap.add_argument("--judge", default="claude-sonnet-4-5")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    systems = dict(s.split("=", 1) for s in args.systems)
    arms = {}
    for arm, sim in (("baseline", args.baseline_sim), ("alt", args.alt_sim)):
        arms[arm] = {}
        for label, model in systems.items():
            p = os.path.join(args.results_dir, f"{args.judge}-eval_dialog-level_{model}_sim-{sim}.jsonl")
            if not os.path.exists(p):
                raise SystemExit(f"missing judge file: {p}")
            arms[arm][label] = load_scores(p)

    metrics = list(HEADLINE)
    verdict_lines = []
    for metric in metrics:
        print(f"\n===== {metric} =====")
        print(f"{'system':12s} | {'simulator='+args.baseline_sim:>26s} | {'simulator='+args.alt_sim:>26s}")
        stats = {}
        for arm in ("baseline", "alt"):
            stats[arm] = {}
            for label in systems:
                xs = arms[arm][label].get(metric, [])
                if not xs:
                    raise SystemExit(f"no {metric} scores for {label} in {arm} arm")
                stats[arm][label] = {"mean": float(np.mean(xs)), "ci": float(ci95(xs)), "n": len(xs)}
        for label in systems:
            b, a = stats["baseline"][label], stats["alt"][label]
            print(
                f"{label:12s} | {b['mean']:6.2f} +/- {b['ci']:.2f}  (n={b['n']:3d}) "
                f"| {a['mean']:6.2f} +/- {a['ci']:.2f}  (n={a['n']:3d})"
            )

        rb, ra = rank(stats["baseline"]), rank(stats["alt"])
        print(f"  rank ({args.baseline_sim:>16s}): {' > '.join(rb)}")
        print(f"  rank ({args.alt_sim:>16s}): {' > '.join(ra)}")

        if rb == ra:
            print("  ranking PRESERVED")
        else:
            flips = flipped_pairs(rb, ra)
            meaningful = [(x, y) for x, y in flips if not overlaps(stats["alt"][x], stats["alt"][y])]
            if meaningful:
                pairs = ", ".join(f"{y} overtakes {x}" for x, y in meaningful)
                print(f"  ranking CHANGED: MEANINGFUL ({pairs}) -- CIs disjoint under {args.alt_sim}")
            else:
                pairs = ", ".join(f"{x}/{y}" for x, y in flips)
                print(f"  ranking CHANGED: NOT MEANINGFUL -- CIs overlap for {pairs}")

        want = HEADLINE[metric]
        got_b, got_a = rb[0], ra[0]
        ok = got_a == want
        verdict_lines.append(
            (
                ok,
                f"{metric:22s} headline='{want}' 1st  | baseline 1st={got_b:9s} alt 1st={got_a:9s}  "
                f"{'HOLDS' if ok else 'DOES NOT HOLD'}",
            )
        )

    print("\n===== E3 verdict =====")
    for _, line in verdict_lines:
        print(" ", line)
    passed = all(ok for ok, _ in verdict_lines)
    print(f"\nE3 {'PASSES' if passed else 'FAILS'}: under simulator={args.alt_sim}, "
          f"the paper's Figure 3 headline ordering {'is' if passed else 'is NOT'} preserved.")
    print("Report rank changes honestly; do not suppress a DOES NOT HOLD.")


if __name__ == "__main__":
    main()
