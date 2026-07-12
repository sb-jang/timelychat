"""E1 step 4: aggregate the filled annotation sheets into the rebuttal numbers.

    python -m audit.aggregate_audit audit/sheets/audit_author1.csv audit/sheets/audit_author2.csv

Reports, per constraint and overall:
  * each author's pass rate,
  * the pooled pass rate,
  * inter-annotator agreement (raw + Cohen's kappa) when 2 authors are given,
  * how often the authors agreed with the LLM pre-verdict (if present).

Only rows where every author filled a 1/0 for a constraint are counted; blanks are
reported as skipped so a half-finished sheet cannot silently inflate a pass rate.
"""

import argparse
import csv
import sys
from collections import defaultdict

from audit.constraints import CONSTRAINTS, KEYS


def read_sheet(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[int(r["sample_id"])] = r
    return rows


def parse_cell(v):
    v = (v or "").strip()
    if v in ("1", "0"):
        return int(v)
    return None  # blank / invalid


def cohen_kappa(pairs):
    """pairs: list of (a, b) in {0,1}. Returns kappa or nan."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    # marginals
    pa1 = sum(1 for a, _ in pairs if a == 1) / n
    pb1 = sum(1 for _, b in pairs if b == 1) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+", help="filled author CSVs (1 or 2)")
    args = ap.parse_args()

    sheets = [read_sheet(p) for p in args.sheets]
    common = set.intersection(*[set(s) for s in sheets])
    if not common:
        sys.exit("no shared sample_ids across sheets")

    print(f"{len(args.sheets)} sheet(s), {len(common)} shared samples\n")
    print(f"{'constraint':22s} " + "  ".join(f"A{i+1}_pass" for i in range(len(sheets)))
          + "   pooled   kappa   n   llm_agree")

    overall_pairs = []
    overall_llm = []
    for k in KEYS:
        per_author_pass = [[] for _ in sheets]
        pairs = []          # (a,b) for kappa when 2 authors
        pooled = []         # all author votes pooled
        llm_agree = []      # author vote == llm pre-verdict
        for sid in sorted(common):
            votes = [parse_cell(sheets[i][sid].get(f"{k}__pass")) for i in range(len(sheets))]
            if any(v is None for v in votes):
                continue    # skip rows not fully annotated for this constraint
            for i, v in enumerate(votes):
                per_author_pass[i].append(v)
                pooled.append(v)
            if len(votes) == 2:
                pairs.append((votes[0], votes[1]))
            llm = parse_cell(sheets[0][sid].get(f"{k}__llm_pass"))
            if llm is not None:
                for v in votes:
                    llm_agree.append(int(v == llm))

        n = len(per_author_pass[0])
        rates = [f"{(sum(pa)/len(pa)*100):6.1f}%" if pa else "   n/a" for pa in per_author_pass]
        pooled_rate = f"{(sum(pooled)/len(pooled)*100):6.1f}%" if pooled else "   n/a"
        kappa = cohen_kappa(pairs) if len(sheets) == 2 else float("nan")
        kstr = f"{kappa:5.2f}" if kappa == kappa else "  n/a"
        lstr = f"{(sum(llm_agree)/len(llm_agree)*100):5.1f}%" if llm_agree else "  n/a"
        print(f"{k:22s} " + "  ".join(rates) + f"   {pooled_rate}   {kstr}  {n:3d}   {lstr}")
        overall_pairs += pairs
        overall_llm += llm_agree

    if len(sheets) == 2 and overall_pairs:
        print(f"\noverall Cohen's kappa (all constraints pooled): {cohen_kappa(overall_pairs):.3f}")
    if overall_llm:
        print(f"author agreement with LLM pre-verdict: {sum(overall_llm)/len(overall_llm)*100:.1f}%")

    n_skipped = len(common) * len(KEYS) - sum(
        1 for k in KEYS for sid in common
        if all(parse_cell(sheets[i][sid].get(f"{k}__pass")) is not None for i in range(len(sheets)))
    )
    if n_skipped:
        print(f"\n[warn] {n_skipped} (constraint,sample) cells not fully annotated and were skipped.")


if __name__ == "__main__":
    main()
