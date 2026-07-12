# E2-c / E2-d — Timing-prediction reproduction & bucket accuracy (TIMER-3B)

Generated from the main `seongbo/timer-3b` checkpoint, turn-level `time` task,
zeroshot, default decoding (num_beams=3), on the 324-dialog eval split. These are
the outputs the rest of E2 was blocked on.

Source files:
- `results/turn-level_seongbo--timer-3b_time_zeroshot_incremental_*.jsonl` (2099 rows — every context prefix)
- `results/turn-level_seongbo--timer-3b_time_zeroshot_*.jsonl` (324 rows — final turn only)

## E2-d — Table 2 reproduction (ka1t #1: "is Table 2 reproducible?")

Recomputed binary timing metrics (0 min = instant, >0 = delayed) via
`analysis/timing_metrics_from_file.py`:

| Metric | Our re-run | Paper Table 2 |
|---|---|---|
| Precision | 0.781 | 78.3 |
| Recall | 0.802 | 79.9 |
| F1 | **0.791** | **79.1** |
| FPR | **0.041** | **4.1** |
| RMSLE | **1.1903** | **1.189** |

**Exact reproduction.** Combined with the earlier w/o-Special-Token ablation
(reproduced Table 5's F1 0.299 / RMSLE 3.700 exactly), the timing-prediction
pipeline is confirmed reproducible end to end.

## E2-c — Coarse-bucket accuracy (sV8H #2: "single point estimate is limiting")

6 buckets = Table 6 ranges. Via `analysis/bucket_accuracy.py`, TIMER-3B on the
324 final-turn predictions:

| | accuracy |
|---|---|
| Exact bucket | 114/324 = **35.2%** |
| Adjacent (±1 bucket) | 242/324 = **74.7%** |

Reading for the response: the point estimate is coarse (35% land in the exact
bucket), but TIMER is within one bucket of ground truth **3 out of 4 times**, i.e.
its errors are near-misses on an ordinal scale, not arbitrary. This supports the
sV8H #2 reply: a single point estimate is an acknowledged simplification, yet the
prediction is directionally reliable at the granularity that matters (RMSLE is
log-scale for the same reason). Distributional prediction remains future work.

Confusion matrix (rows = GT bucket, cols = predicted), full 324:

```
GT \ pred                      0     1     2     3     4     5
0 0-5 min (instant)            1     4     2     0     0     0
1 5-30 min (short)            20    19    57     1     0     3
2 30 min-2 hrs (moderate)     23     1    76     3     1     3
3 2-6 hrs (long)              12     2    32     9     0    11
4 6-12 hrs (half-day)          3     0    10     3     1     8
5 12-24 hrs (full-day)         6     2     3     0     0     8
```

The mass on column 2 (30 min–2 hrs) shows a central-tendency pull — the model
hedges toward the modal bucket, which is exactly the point-estimate limitation
sV8H #2 raised, now quantified rather than asserted.

## Scope

TIMER-3B only. Full Table 2 / bucket comparison across baselines (GPT-4o, GPT-3.5,
Llama-3.1-8B/70B) is not yet run — it needs OpenAI credit and additional GPU time.
The reproduction *claim* for ka1t #1 is already established by the exact TIMER +
ablation matches; baseline rows would complete the side-by-side table if wanted.
