"""Significance testing for pairwise human-evaluation win/tie/loss counts (Table 4).

Rebuttal use (strategy E2-a; ka1t #1): we *pre-concede* that the dialog-level
human-eval margins are not individually significant, and report the exact
two-sided sign (binomial) test so the claim is honest and quantified.

Sign test convention
---------------------
Ties are non-informative for a sign test, so the primary test excludes ties and
asks whether wins vs. losses depart from a fair coin:  binomtest(win, win+loss, 0.5).
As a sensitivity check we also split ties evenly between the two systems.

The counts below are derived from Table 4 percentages times the reported sample
sizes (turn-level n=200, dialog-level n=80) and are ROUNDED, so p-values are
approximate. Replace with the exact raw win/tie/loss counts from the annotation
sheets for the number that goes in the response (use --counts or edit TABLE4).

Usage
-----
    python analysis/significance.py                 # all Table 4 rows (approx)
    python analysis/significance.py --counts 40 23 37 --n 80 --label "delay-appr"
"""

import argparse

from scipy.stats import binomtest


def sign_test(win: int, loss: int) -> float:
    """Two-sided exact binomial (sign) test on decisive comparisons only."""
    n = win + loss
    if n == 0:
        return float("nan")
    return binomtest(win, n, 0.5, alternative="two-sided").pvalue


# (setting, metric, win%, tie%, loss%, n) from Table 4 (TIMER 3B vs zero-shot GPT-4o)
TABLE4 = [
    ("turn-level", "Naturalness", 19, 24, 57, 200),
    ("turn-level", "Time-specificity", 54, 24, 22, 200),
    ("dialog-level", "Coherence", 21, 18, 61, 80),
    ("dialog-level", "Delay-appropriateness", 40, 23, 37, 80),
    ("dialog-level", "Time-specificity", 40, 23, 37, 80),
]


def counts_from_pct(win_pct: int, tie_pct: int, loss_pct: int, n: int):
    return round(win_pct / 100 * n), round(tie_pct / 100 * n), round(loss_pct / 100 * n)


def report_row(label: str, win: int, tie: int, loss: int):
    decisive = win + loss
    p_excl = sign_test(win, loss)
    # sensitivity: split ties evenly (round wins up)
    w2, l2 = win + tie / 2, loss + tie / 2
    p_split = sign_test(round(w2), round(l2))
    win_rate = win / decisive * 100 if decisive else float("nan")
    sig = "significant" if p_excl < 0.05 else "NOT significant"
    print(
        f"{label:34s} W/T/L={win:>3}/{tie:>3}/{loss:>3}  "
        f"win-rate(decisive)={win_rate:5.1f}%  "
        f"p(ties excl)={p_excl:.3f}  p(ties split)={p_split:.3f}  -> {sig} at .05"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", nargs=3, type=int, metavar=("WIN", "TIE", "LOSS"),
                    help="exact raw counts; overrides the Table 4 approximation")
    ap.add_argument("--n", type=int, help="sample size (only needed with Table 4 percentages)")
    ap.add_argument("--label", default="custom")
    args = ap.parse_args()

    print("Two-sided exact sign (binomial) test; H0: P(win)=P(loss)=0.5\n")
    if args.counts:
        report_row(args.label, *args.counts)
        return

    print("NOTE: counts below are rounded from Table 4 percentages -> p-values APPROXIMATE.")
    print("      Rerun with --counts WIN TIE LOSS from the raw sheets for the reportable value.\n")
    for setting, metric, wp, tp, lp, n in TABLE4:
        win, tie, loss = counts_from_pct(wp, tp, lp, n)
        report_row(f"[{setting}] {metric}", win, tie, loss)


if __name__ == "__main__":
    main()
