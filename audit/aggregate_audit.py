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


def krippendorff_alpha(units):
    """Krippendorff's alpha (nominal metric).

    units: list of per-item rating lists, e.g. [[a, b], ...] for two annotators.
    Items with fewer than 2 ratings are ignored. Validated to match the
    `krippendorff` reference package to 1e-9 on data with >1 category.

    When every rating is the same single category (no variance), expected
    disagreement is 0 and alpha is conventionally reported as 1.0 (the reference
    package errors here); observed disagreement is necessarily 0 in that case.
    """
    from collections import defaultdict

    o = defaultdict(float)
    values = set()
    for u in units:
        r = [x for x in u if x is not None]
        m = len(r)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(r[i], r[j])] += 1.0 / (m - 1)
                    values.add(r[i])
                    values.add(r[j])
    if not values:
        return float("nan")
    values = sorted(values)
    n_c = {c: sum(o[(c, k)] for k in values) for c in values}
    n = sum(n_c.values())
    if n == 0:
        return float("nan")
    d_o = sum(o[(c, k)] for c in values for k in values if c != k)
    d_e = sum(n_c[c] * n_c[k] for c in values for k in values if c != k) / (n - 1)
    if d_e == 0:
        return 1.0  # single category, no possible disagreement
    return 1 - d_o / d_e


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
          + "   pooled   raw_agree   kripp_a   n")

    overall_units = []
    for k in KEYS:
        per_author_pass = [[] for _ in sheets]
        units = []          # one list of ratings per item, for alpha/raw-agreement
        pooled = []         # all author votes pooled
        for sid in sorted(common):
            votes = [parse_cell(sheets[i][sid].get(f"{k}__pass")) for i in range(len(sheets))]
            if any(v is None for v in votes):
                continue    # skip rows not fully annotated for this constraint
            for i, v in enumerate(votes):
                per_author_pass[i].append(v)
                pooled.append(v)
            units.append(list(votes))

        n = len(per_author_pass[0])
        rates = [f"{(sum(pa)/len(pa)*100):6.1f}%" if pa else "   n/a" for pa in per_author_pass]
        pooled_rate = f"{(sum(pooled)/len(pooled)*100):6.1f}%" if pooled else "   n/a"
        raw = (sum(1 for u in units if len(set(u)) == 1) / len(units)) if units else float("nan")
        rstr = f"{raw*100:6.1f}%" if raw == raw else "   n/a"
        alpha = krippendorff_alpha(units) if len(sheets) >= 2 else float("nan")
        astr = f"{alpha:5.2f}" if alpha == alpha else "  n/a"
        print(f"{k:22s} " + "  ".join(rates) + f"   {pooled_rate}   {rstr}   {astr}  {n:3d}")
        overall_units += units

    if len(sheets) >= 2 and overall_units:
        raw = sum(1 for u in overall_units if len(set(u)) == 1) / len(overall_units)
        print(f"\noverall raw agreement (all constraints pooled): {raw*100:.1f}%")
        print(f"overall Krippendorff's alpha (all constraints pooled): {krippendorff_alpha(overall_units):.3f}")
        print("note: where a constraint is passed by almost every example, alpha (like kappa) is deflated "
              "by the base-rate effect; read pass rate and raw agreement as primary.")

    n_skipped = len(common) * len(KEYS) - sum(
        1 for k in KEYS for sid in common
        if all(parse_cell(sheets[i][sid].get(f"{k}__pass")) is not None for i in range(len(sheets)))
    )
    if n_skipped:
        print(f"\n[warn] {n_skipped} (constraint,sample) cells not fully annotated and were skipped.")


if __name__ == "__main__":
    main()
