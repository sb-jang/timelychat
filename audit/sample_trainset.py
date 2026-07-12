"""E1 step 1: draw a fixed, reproducible audit sample from the private training set.

    python -m audit.sample_trainset --n 200 --seed 0

Writes audit/data/sample_n<N>_seed<S>.jsonl, one row per sampled example with a
stable `sample_id` (its index in delayed/valid) so every later step and both
annotators refer to the same items.
"""

import argparse
import json
import os
import random

from datasets import load_dataset

DATASET = "seongbo-research/timelychat-refined"
FIELDS = ["context", "speaker_list", "time_elapsed", "target_speaker",
          "timely_response", "untimely_response", "narrative"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="delayed")
    ap.add_argument("--split", default="valid")
    args = ap.parse_args()

    data = load_dataset(DATASET, name=args.config, split=args.split)
    if args.n > len(data):
        raise SystemExit(f"--n {args.n} exceeds split size {len(data)}")

    idxs = sorted(random.Random(args.seed).sample(range(len(data)), args.n))

    os.makedirs("audit/data", exist_ok=True)
    out = f"audit/data/sample_n{args.n}_seed{args.seed}.jsonl"
    with open(out, "w") as f:
        for i in idxs:
            ex = data[i]
            row = {"sample_id": i, "config": args.config, "split": args.split}
            row.update({k: ex[k] for k in FIELDS})
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(idxs)} samples -> {out}")
    print(f"sample_id range: {idxs[0]}..{idxs[-1]}  (from {len(data)} {args.config}/{args.split} rows)")


if __name__ == "__main__":
    main()
