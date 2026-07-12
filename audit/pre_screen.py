"""E1 step 2: Claude pre-screens each sampled example against the four constraints.

    python -m audit.pre_screen --in audit/data/sample_n200_seed0.jsonl

For every sample the model returns, per constraint, a pass/fail verdict and a
one-sentence rationale. This is a FIRST PASS to cut author effort, not the verdict:
the authors review and may override every cell. Output:
audit/data/prescreen_<inputname>.jsonl, one row per sample keyed by sample_id.

Uses Claude Sonnet 4.5 with structured outputs, the same judge stack as laaj.py.
"""

import argparse
import asyncio
import json
import os

from anthropic import AsyncAnthropic, DefaultAioHttpClient
from pydantic import BaseModel, create_model
from tqdm import tqdm

from audit.constraints import CONSTRAINTS, KEYS, render_sample


class ConstraintVerdict(BaseModel):
    passed: bool
    rationale: str


# One field per constraint, e.g. spatial_separation: ConstraintVerdict
PrescreenOutput = create_model(
    "PrescreenOutput",
    **{k: (ConstraintVerdict, ...) for k in KEYS},
)

RUBRIC = "\n".join(
    f"- {key} ({label}): {definition}\n    Question: {question}"
    for key, (label, definition, question) in CONSTRAINTS.items()
)

SYSTEM = (
    "You are auditing the quality of a time-aware dialogue dataset. Each example was "
    "generated from a seed narrative and must satisfy four construction constraints. "
    "For each constraint, decide whether this example SATISFIES it (passed=true) or "
    "VIOLATES it (passed=false), and give a one-sentence rationale grounded in the "
    "specific text. Be strict: if a constraint is only partially met, mark it failed "
    "and say why.\n\nConstraints:\n" + RUBRIC
)


async def screen_one(client, sample, sem, max_iter=3):
    async with sem:
        user = render_sample(sample)
        for attempt in range(max_iter):
            try:
                resp = await client.beta.messages.parse(
                    model="claude-sonnet-4-5",
                    max_tokens=1024,
                    betas=["structured-outputs-2025-11-13"],
                    messages=[{"role": "user", "content": SYSTEM},
                              {"role": "user", "content": user}],
                    output_format=PrescreenOutput,
                )
                out = resp.parsed_output
                if out is None:
                    raise AttributeError("null parse")
                row = {"sample_id": sample["sample_id"]}
                for k in KEYS:
                    v = getattr(out, k)
                    row[f"{k}__llm_pass"] = bool(v.passed)
                    row[f"{k}__llm_rationale"] = v.rationale
                return row
            except AttributeError:
                if attempt == max_iter - 1:
                    row = {"sample_id": sample["sample_id"]}
                    for k in KEYS:
                        row[f"{k}__llm_pass"] = None
                        row[f"{k}__llm_rationale"] = "PRESCREEN_FAILED"
                    return row


async def main(args):
    samples = [json.loads(l) for l in open(args.infile)]
    async with AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                              http_client=DefaultAioHttpClient()) as client:
        sem = asyncio.Semaphore(args.concurrency)
        tasks = [screen_one(client, s, sem) for s in samples]
        rows = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="pre-screening"):
            rows.append(await coro)
    rows.sort(key=lambda r: r["sample_id"])

    os.makedirs("audit/data", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.infile))[0].replace("sample_", "")
    out = f"audit/data/prescreen_{base}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    failed = sum(1 for r in rows if r[f"{KEYS[0]}__llm_pass"] is None)
    print(f"wrote {len(rows)} rows -> {out}  ({failed} prescreen failures)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main(args))
