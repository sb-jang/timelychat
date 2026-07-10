"""Audit silent time-parse failures in dialog-level simulation outputs.

`convert_to_minutes` (utils/postprocess.py) returns 0.0 when it cannot parse the
agent's predicted delay, which `evaluate_dialog-level.py` then writes as a
perfectly ordinary "0 minutes". A model whose time head emits garbage is
therefore indistinguishable, downstream, from a model that deliberately chose not
to delay -- and Delay-Appropriateness is scored on exactly that distinction.

Each simulation record stores `raw_time_outputs`. This script re-parses them and
flags any output that mapped to 0 minutes without containing a zero-time
expression: those are parse failures wearing a "0 minutes" mask.

Usage
-----
    python -m analysis.audit_time_parse results/dialog-level_*_T10_n100_seed0.jsonl
"""

import argparse
import glob
import json
import re

from utils.postprocess import convert_to_minutes

# A genuine zero delay: "0 minutes", "0.0 min", "zero hours", "immediately",
# "right away", "no delay". Deliberately strict -- a bare "no" or a stray "0"
# inside prose must NOT clear a garbled generation.
ZERO_RE = re.compile(
    r"\b(0+(\.0+)?|zero)\s*(second|minute|hour|day|week)s?\b|\bimmediat\w*|\bright away\b|\bno delay\b",
    re.IGNORECASE,
)


def audit(path: str) -> dict:
    total = zeros = masked = 0
    examples = []
    for line in open(path):
        rec = json.loads(line)
        for raw in rec.get("raw_time_outputs", []):
            total += 1
            if convert_to_minutes(raw) != 0.0:
                continue
            zeros += 1
            if not ZERO_RE.search(raw):
                masked += 1
                if len(examples) < 3:
                    examples.append(raw[:70])
    return {"total": total, "zeros": zeros, "masked": masked, "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="dialog-level simulation JSONL(s); globs allowed")
    ap.add_argument("--threshold", type=float, default=5.0, help="max %% masked parse failures before FAIL")
    ap.add_argument(
        "--exempt",
        action="append",
        default=[],
        help="substring of a filename whose failures are REPORTED but do not fail the gate "
        "(use only for a known, documented limitation)",
    )
    args = ap.parse_args()

    paths = sorted({p for f in args.files for p in glob.glob(f)})
    if not paths:
        raise SystemExit(f"no files matched: {args.files}")

    print(f"{'file':70s} {'n_pred':>7s} {'0-min':>7s} {'masked':>7s} {'%':>7s}  verdict")
    worst_ok = True
    exempted = []
    for p in paths:
        name = p.split("/")[-1]
        r = audit(p)
        pct = 100.0 * r["masked"] / r["total"] if r["total"] else 0.0
        ok = pct <= args.threshold
        is_exempt = any(e in name for e in args.exempt)
        if is_exempt and not ok:
            exempted.append((name, pct))
            verdict = f"EXEMPT ({pct:.1f}% > {args.threshold:.0f}%, reported not gated)"
        else:
            worst_ok &= ok
            verdict = "OK" if ok else "FAIL"
        print(f"{name:70s} {r['total']:>7d} {r['zeros']:>7d} {r['masked']:>7d} {pct:>6.1f}%  {verdict}")
        for e in r["examples"]:
            print(f"{'':70s}   unparsed: {e!r}")
    print("\nmasked = mapped to 0 minutes but contains no zero-time expression -> silent parse failure.")
    for name, pct in exempted:
        print(f"CAVEAT: {name} has {pct:.1f}% silent parse failures; its delay scores understate delay usage.")
    raise SystemExit(0 if worst_ok else 1)


if __name__ == "__main__":
    main()
