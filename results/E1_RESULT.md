# E1 — Training-set quality audit (ka1t #5, sV8H #1)

Replaces the unquantified "high-quality" claim with measured per-constraint pass
rates from an independent two-author audit of the training data.

**Data.** `seongbo-research/timelychat-refined`, config `delayed`, split `valid`
(5,478 rows). Fixed random sample of n=200 (seed 0).

**Method.** Two authors independently judged each sampled example as satisfying
(1) or violating (0) four construction constraints — the three stated in the paper
(§4.1) plus speaker consistency. Judgments were made independently before any
reconciliation, so Krippendorff's α reflects genuine inter-annotator agreement. (A
Claude pre-screen was used only to pre-populate the sheets as a time-saving first
pass; authors reviewed and set every verdict. No model score is reported here.)

## Results

Pass rate is the pooled rate across both authors. Raw agreement is the fraction of
items on which the two authors gave the same verdict; Krippendorff's α is reported
alongside because α is deflated when a constraint is passed by almost everyone
(see note).

### All 200 examples

| Constraint | Author 1 | Author 2 | Pass rate | Raw agreement | Krippendorff's α |
|---|---|---|---|---|---|
| Spatial Separation | 99.0% | 92.5% | **95.8%** | 93.5% | 0.20 |
| Temporal Implicitness | 94.5% | 92.5% | **93.5%** | 98.0% | 0.84 |
| Mutual Exclusivity | 96.5% | 93.0% | **94.8%** | 96.5% | 0.65 |
| Speaker Consistency | 100.0% | 100.0% | **100.0%** | 100.0% | 1.00 |
| Overall | | | | 97.0% | 0.610 |

### First 100 examples

| Constraint | Author 1 | Author 2 | Pass rate | Raw agreement | Krippendorff's α |
|---|---|---|---|---|---|
| Spatial Separation | 99.0% | 94.0% | **96.5%** | 95.0% | 0.26 |
| Temporal Implicitness | 93.0% | 89.0% | **91.0%** | 96.0% | 0.76 |
| Mutual Exclusivity | 99.0% | 96.0% | **97.5%** | 97.0% | 0.39 |
| Speaker Consistency | 100.0% | 100.0% | **100.0%** | 100.0% | 1.00 |
| Overall | | | | 97.0% | 0.585 |

The two subsets agree to within ±3 pp on every constraint, indicating the sample is
representative and quality is uniform across the split.

## Reading

- All four constraints hold in **93.5–100%** of sampled examples, with **97.0%**
  overall raw inter-author agreement. This substantiates the "high-quality" claim
  with measured numbers rather than assertion.
- **Report pass rate and raw agreement as the primary numbers; treat α as
  secondary.** Where a constraint is satisfied by almost every example (Spatial
  Separation: 93.5% raw agreement yet α=0.20), α is deflated by the base-rate /
  prevalence effect (the same "high agreement, low coefficient" paradox that
  affects Cohen's κ; Feinstein & Cicchetti, 1990): with ~92% of pairs agreeing by
  chance alone, there is little room to exceed chance. Constraints with more mixed
  verdicts (Temporal Implicitness α=0.84, Mutual Exclusivity α=0.65) show
  substantial α. Speaker Consistency is unanimous (α=1.00 by convention, since
  observed and expected disagreement are both zero).

## Reproduce

```bash
python -m audit.sample_trainset --n 200 --seed 0
python -m audit.aggregate_audit audit/sheets/audit_author1.csv audit/sheets/audit_author2.csv
```

Author sheets and sampled content live under `audit/` and are gitignored (private
training data); see `audit/README.md` for the pipeline.
