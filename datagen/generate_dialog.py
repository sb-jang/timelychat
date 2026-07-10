"""Order-bias regeneration for the two final-turn candidates (strategy E4; ka1t #4).

Reviewer ka1t asks whether the generation order of instructions 4-1 (timely /
delayed response) and 4-2 (instantaneous response) in the Appendix B "Dialog
Generation" prompt introduces a structural bias. Our reply: 4-1/4-2 produce *two
alternatives for the same final turn*, not a sequential dependency, so swapping
their order should not systematically change the outputs.

This script tests that empirically. For each sampled benchmark dialog we hold the
context and the "[{duration} later]" marker fixed and ask GPT-4o to produce the
two candidate final messages under the ORIGINAL order (timely-first, matching the
paper) and under the FLIPPED order (instant-first). `analysis/compare_order.py`
then compares the two regenerated sets.

Prompt is reconstructed verbatim from Appendix B (only the 4-1/4-2 block order is
swapped between variants; everything else is identical).

Usage
-----
    # inspect the exact prompt without calling the API
    python datagen/generate_dialog.py --order timely-first --n 3 --dry-run

    # real run (needs OPENAI_API_KEY); writes results/e4/order_<variant>.jsonl
    python datagen/generate_dialog.py --order timely-first  --n 50 --seed 0
    python datagen/generate_dialog.py --order instant-first --n 50 --seed 0
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from pydantic import BaseModel
from tqdm import tqdm

INSTR_TIMELY = (
    "4-1. Generate {ev}'s last message which is timely as if {ev} spent time to finish the event.\n"
    "4-2. In contrast, generate {ev}'s last message as if {ev} is responding instantaneously "
    "right before the event to happen."
)
INSTR_INSTANT = (
    "4-1. Generate {ev}'s last message as if {ev} is responding instantaneously right before the "
    "event to happen.\n"
    "4-2. In contrast, generate {ev}'s last message which is timely as if {ev} spent time to finish "
    "the event."
)

PROMPT_TEMPLATE = """You are given an event narrative and the duration. Your task is to complete an instant message dialog between two speakers. The following conditions MUST be met.
[Instructions]
1. Speaker {ev} is in the middle of the event now, while speaker {other} is physically apart from.
2. Do not directly mention the duration in the dialog.
3. The dialog context below already ends right after {other}'s last turn and the marker "[{duration} later]".
{order_block}
Make sure that the timely response and the instantaneous response are time-situationally different.
[End of Instructions]
### Dialog context ###
{context}
[{duration} later]
{ev}:

Narrative: {narrative}
Duration: {duration}"""


class DialogCandidates(BaseModel):
    timely_response: str
    instantaneous_response: str


def render_context(context, speaker_list):
    return "\n".join(f"{spk}: {utt}" for spk, utt in zip(speaker_list, context))


def build_prompt(example, order: str) -> str:
    ev = example["target_speaker"]
    other = "B" if ev == "A" else "A"
    order_block = (INSTR_TIMELY if order == "timely-first" else INSTR_INSTANT).format(ev=ev)
    return PROMPT_TEMPLATE.format(
        ev=ev,
        other=other,
        duration=example["time_elapsed"],
        order_block=order_block,
        context=render_context(example["context"], example["speaker_list"]),
        narrative=example["narrative"],
    )


def make_caller(provider: str, model: str):
    """Return call(prompt)->DialogCandidates for the chosen backend."""
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        def call(prompt):
            c = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=DialogCandidates,
                temperature=1.0,
                top_p=0.95,
            )
            return c.choices[0].message.parsed

    else:
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        def call(prompt):
            r = client.beta.messages.parse(
                model=model,
                max_tokens=1024,
                betas=["structured-outputs-2025-11-13"],
                messages=[{"role": "user", "content": prompt}],
                output_format=DialogCandidates,
            )
            return r.parsed_output

    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=["timely-first", "instant-first"], required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                    help="generator backend. openai=GPT-4o (paper's benchmark generator); "
                         "anthropic=Claude (cross-generator robustness supplement)")
    ap.add_argument("--model", default=None,
                    help="default: gpt-4o (openai) / claude-sonnet-4-5 (anthropic)")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, do not call the API")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.model is None:
        args.model = "gpt-4o" if args.provider == "openai" else "claude-sonnet-4-5"

    data = load_dataset("seongbo/timelychat", split="eval")
    idxs = list(range(len(data)))
    random.Random(args.seed).shuffle(idxs)
    idxs = sorted(idxs[: args.n])  # fixed subset, same across both orders (seed-controlled)

    if args.dry_run:
        for i in idxs[: min(args.n, 3)]:
            print(f"\n########## idx={i}  order={args.order} ##########")
            print(build_prompt(data[i], args.order))
        print(f"\n[dry-run] would generate {len(idxs)} examples for order={args.order}")
        return

    call = make_caller(args.provider, args.model)
    out_path = args.out or f"results/e4/order_{args.order}_{args.provider}_seed{args.seed}_n{args.n}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for i in tqdm(idxs, desc=f"generate ({args.order}/{args.provider})"):
            ex = data[i]
            p = call(build_prompt(ex, args.order))
            f.write(json.dumps({
                "idx": i,
                "order": args.order,
                "provider": args.provider,
                "model": args.model,
                "time_elapsed": ex["time_elapsed"],
                "target_speaker": ex["target_speaker"],
                "narrative": ex["narrative"],
                "timely_response": p.timely_response,
                "instantaneous_response": p.instantaneous_response,
                # keep the originals for reference
                "orig_timely_response": ex["timely_response"],
                "orig_untimely_response": ex["untimely_response"],
            }) + "\n")
    print(f"wrote {len(idxs)} rows -> {out_path}")


if __name__ == "__main__":
    main()
