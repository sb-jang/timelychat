"""Force every E3 arm onto the identical set of dialog indices.

E3 compares system rankings across two user simulators. That comparison is only
valid if both arms scored the *same* dialogs -- otherwise a rank change could come
from a different dialog mix rather than from the simulator swap.

A dialog can go missing from one arm: Claude's structured-output API returns
`stop_reason="refusal"` on a small number of conversations (deterministically, so
retries do not help), and `evaluate_dialog-level.py` drops those rather than dying.
This script intersects the idx sets across all given files and rewrites each to the
intersection, reporting exactly what it dropped and why.

Idempotent: running it on already-aligned files changes nothing.

Usage
-----
    python -m analysis.align_subsample 'results/dialog-level_*_T10_n100_seed0.jsonl'
"""

import argparse
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="dialog-level simulation JSONL(s); globs allowed")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not rewrite")
    args = ap.parse_args()

    paths = sorted({p for f in args.files for p in glob.glob(f)})
    if not paths:
        raise SystemExit(f"no files matched: {args.files}")

    rows = {p: [json.loads(line) for line in open(p)] for p in paths}
    idx_sets = {p: {r["idx"] for r in rs} for p, rs in rows.items()}
    common = set.intersection(*idx_sets.values())

    print(f"{len(paths)} arms | common idx: {len(common)}")
    any_drop = False
    for p in paths:
        extra = sorted(idx_sets[p] - common)
        name = p.split("/")[-1]
        if extra:
            any_drop = True
            print(f"  {name}: dropping {len(extra)} idx not present in every arm: {extra}")
        else:
            print(f"  {name}: already aligned ({len(idx_sets[p])} dialogs)")

    missing_from = {}
    for idx in set.union(*idx_sets.values()) - common:
        missing_from[idx] = [p.split("/")[-1] for p in paths if idx not in idx_sets[p]]
    for idx, where in sorted(missing_from.items()):
        print(f"  idx {idx} absent from: {', '.join(where)}")

    if args.dry_run:
        print("\n[dry-run] no files rewritten")
        return
    if not any_drop:
        print("\nnothing to do; all arms already share the same idx set")
        return

    for p in paths:
        kept = [r for r in rows[p] if r["idx"] in common]
        kept.sort(key=lambda r: r["idx"])
        with open(p, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
    print(f"\nrewrote {len(paths)} files to the common subsample (n={len(common)})")


if __name__ == "__main__":
    main()
