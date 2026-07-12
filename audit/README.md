# E1 — Training-set quality audit (ka1t #5, sV8H #1)

Audits the private training data `seongbo-research/timelychat-refined` (config
`delayed`, split `valid`, 5,478 rows) against the four construction constraints,
to replace the unquantified "high-quality" claim with measured per-constraint pass
rates and inter-annotator agreement.

## Pipeline

```bash
# 1. fixed 200-sample draw (reproducible; sample_id = index in delayed/valid)
python -m audit.sample_trainset --n 200 --seed 0

# 2. Claude Sonnet 4.5 pre-screens each sample x 4 constraints (pass/fail + rationale)
export ANTHROPIC_API_KEY=...
python -m audit.pre_screen --in audit/data/sample_n200_seed0.jsonl

# 3. build two independent author sheets (LLM pre-verdict prefilled, author columns blank)
python -m audit.make_sheets --sample audit/data/sample_n200_seed0.jsonl \
    --prescreen audit/data/prescreen_n200_seed0.jsonl --authors 2

# 4. authors fill the `<constraint>__pass` columns (1=satisfies, 0=violates), then:
python -m audit.aggregate_audit audit/sheets/audit_author1.csv audit/sheets/audit_author2.csv
```

## Constraints (audit/constraints.py — single source of truth)

1. **Spatial Separation** (paper §4.1) — one speaker experiencing the event while apart from the other.
2. **Temporal Implicitness** (paper §4.1) — timely_response avoids naming the elapsed interval (no lexical shortcut).
3. **Mutual Exclusivity** (paper §4.1) — timely/untimely responses are genuinely time-exclusive, not time-agnostic.
4. **Speaker Consistency** (plan) — speaker labels consistent; target_speaker is the event experiencer.

## Author instructions

- Work **independently** — do not compare sheets until both are complete (Cohen's κ needs independent judgments).
- The `*__llm_pass` / `*__llm_rationale` columns are Claude's first pass, provided to
  save time. **Confirm or override every one** by putting 1 or 0 in `*__pass`.
- The `*__note` columns are optional free text for disagreements or edge cases.
- The full example (narrative, context, both candidate responses, GT interval) is in the `example` column.

## Files

`audit/*.py` and this README are tracked. `audit/data/` (sampled private-dataset
content) and `audit/sheets/` (author sheets) are gitignored — they contain private
training data and must not be committed.

## LLM pre-screen preview (n=200, NOT the final verdict — authors decide)

| Constraint | Claude pass rate |
|---|---|
| Spatial Separation | 85.5% |
| Temporal Implicitness | 86.5% |
| Mutual Exclusivity | **57.0%** |
| Speaker Consistency | 89.5% |
| all four | 39.0% |

Mutual Exclusivity is the weak dimension in the pre-screen — Claude judges ~43% of
delayed examples as not strictly time-exclusive. This is the hardest, most
subjective constraint and the one most likely to shift under author review; it is
also exactly the quality dimension ka1t #5 / sV8H #1 asked about, so the author
verdict on it matters most.
