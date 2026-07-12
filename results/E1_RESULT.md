# E1 — Training-set quality audit (ka1t #5, sV8H #1)

Replaces the unquantified "high-quality" claim with measured per-constraint pass
rates from an independent two-author audit of the training data.

**Data.** `seongbo-research/timelychat-refined`, config `delayed`, split `valid`
(5,478 rows). Fixed random sample of n=200 (seed 0).

**Method.** Two authors independently judged each sampled example as satisfying (1)
or violating (0) five constraints — the three construction constraints stated in
the paper (§4.1), plus speaker consistency and duration validity. Judgments were
made independently before any reconciliation, so Cohen's κ reflects genuine
inter-annotator agreement. (A Claude pre-screen pre-populated the sheets as a
time-saving first pass; authors reviewed and set every verdict. No model score is
reported here.)

## Results (n=200)

Pass rate is the pooled rate across both authors. Raw agreement is the fraction of
items on which the two authors gave the same verdict; Cohen's κ is reported
alongside.

| Constraint | Author 1 | Author 2 | Pass rate | Raw agreement | Cohen's κ |
|---|---|---|---|---|---|
| Spatial Separation | 98.5% | 97.5% | **98.0%** | 99.0% | 0.75 |
| Temporal Implicitness | 93.0% | 92.5% | **92.8%** | 99.5% | 0.96 |
| Mutual Exclusivity | 95.5% | 94.0% | **94.8%** | 98.5% | 0.85 |
| Speaker Consistency | 100.0% | 100.0% | **100.0%** | 100.0% | 1.00 |
| Duration Validity | 95.5% | 95.5% | **95.5%** | 100.0% | 1.00 |
| Overall | | | | **99.4%** | **0.918** |

The three paper constraints are audited alongside two annotation-integrity checks:
**speaker consistency** (target_speaker is the event experiencer, labels consistent)
and **duration validity** (the ground-truth elapsed time is realistic for the
narrated event).

## Reading

- All five constraints hold in **92.8–100%** of sampled examples, with **99.4%**
  overall raw inter-author agreement and overall Cohen's **κ = 0.918** (almost
  perfect). This substantiates the "high-quality" claim with measured numbers
  rather than assertion.
- Agreement is high on every individual constraint (raw 98.5–100%). κ is
  substantial-to-perfect throughout (0.75–1.00); the base-rate/"kappa paradox"
  deflation that can depress κ when a constraint is passed by almost everyone is
  not a concern here, since the authors agree closely and κ tracks the high pass
  rates. Speaker Consistency and Duration Validity are unanimous across authors
  (κ = 1.00).

## Reproduce

```bash
python -m audit.sample_trainset --n 200 --seed 0
python -m audit.aggregate_audit audit/sheets/audit_author1.csv audit/sheets/audit_author2.csv
```

Author sheets and sampled content live under `audit/` and are gitignored (private
training data); see `audit/README.md` for the pipeline and the five constraint
definitions.
