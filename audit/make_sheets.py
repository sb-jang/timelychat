"""E1 step 3: build per-author annotation sheets (CSV) from samples + pre-screen.

    python -m audit.make_sheets --sample audit/data/sample_n200_seed0.jsonl \
        --prescreen audit/data/prescreen_n200_seed0.jsonl --authors 2

For each author, writes audit/sheets/audit_author<k>.csv with one row per sample:
the readable example, and for each constraint the LLM pre-verdict + rationale plus
two blank columns the author fills — `<key>__pass` (1/0) and `<key>__note`.

The authors work independently; the aggregator reads the filled `<key>__pass`
columns back. Pre-verdicts are provided as a starting point, not an answer key —
the author's job is to confirm or override each one.
"""

import argparse
import csv
import json
import os

from audit.constraints import CONSTRAINTS, KEYS, render_sample


def load(path):
    return {json.loads(l)["sample_id"]: json.loads(l) for l in open(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--prescreen", default=None, help="optional; omit for blank sheets")
    ap.add_argument("--authors", type=int, default=2)
    args = ap.parse_args()

    samples = load(args.sample)
    pre = load(args.prescreen) if args.prescreen else {}

    header = ["sample_id", "example"]
    for k in KEYS:
        header += [f"{k}__llm_pass", f"{k}__llm_rationale", f"{k}__pass", f"{k}__note"]

    os.makedirs("audit/sheets", exist_ok=True)
    for a in range(1, args.authors + 1):
        out = f"audit/sheets/audit_author{a}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for sid in sorted(samples):
                ex = samples[sid]
                p = pre.get(sid, {})
                row = [sid, render_sample(ex)]
                for k in KEYS:
                    llm = p.get(f"{k}__llm_pass")
                    row += [
                        "" if llm is None else int(llm),
                        p.get(f"{k}__llm_rationale", ""),
                        "",  # author fills: 1=pass, 0=fail
                        "",  # author note
                    ]
                w.writerow(row)
        print(f"wrote {out}  ({len(samples)} rows)")

    print("\nConstraint rubric (also shown to the annotators):")
    for k, (label, _def, question) in CONSTRAINTS.items():
        print(f"  [{k}__pass]  {label}: {question}")
    print("\nAuthors: fill the `<key>__pass` column with 1 (satisfies) or 0 (violates) "
          "for every row and constraint, independently. Then run audit.aggregate_audit.")


if __name__ == "__main__":
    main()
