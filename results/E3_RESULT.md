# E3 — Alternative User Simulator Robustness (result)

**Reviewer sV8H #4:** are the dialog-level conclusions an artifact of the GPT-4o
user simulator? **Answer: no.** Swapping the user simulator to Claude Sonnet 4.5,
holding the judge (Claude Sonnet 4.5) and every other factor fixed, leaves the
system ranking unchanged on all three metrics.

Setup: 4 systems (TIMER-3B, GPT-4o, GPT-3.5, Llama-3.1-8B) × 2 user simulators
(GPT-4o, Claude Sonnet 4.5), n=100 dialogs (seed 0, identical indices across
arms), 10 turns, same judge. Framing is **relative**: we compare each system's
rank across the two simulators against our own reproduced baseline, not against
the paper's absolute Figure 3 values (see Caveat).

## Ranking is stable across simulators

| Metric | rank under GPT-4o sim | rank under Claude sim | change |
|---|---|---|---|
| Coherence | gpt-4o > gpt-3.5 > llama-8b > timer | gpt-4o > gpt-3.5 > llama-8b > timer | **none (preserved)** |
| Delay-Appropriateness | gpt-4o > gpt-3.5 > timer > llama-8b | gpt-4o > gpt-3.5 > llama-8b > timer | 3rd/4th swap, **CIs overlap (n.s.)** |
| Time-Specificity | llama-8b > timer > gpt-4o > gpt-3.5 | llama-8b > timer > gpt-3.5 > gpt-4o | 3rd/4th swap, **CIs overlap (n.s.)** |

Every rank change is between two systems whose 95% CIs overlap, i.e. the arms
cannot separate them at this n — not a simulator-induced reordering. The top-2 on
every metric is identical across simulators.

## Per-system mean ± 95% CI (n=100)

| Metric | System | GPT-4o sim | Claude sim | Δ |
|---|---|---|---|---|
| Coherence | gpt-4o | 4.96 ± 0.06 | 4.87 ± 0.12 | −0.09 |
| | gpt-3.5 | 4.45 ± 0.20 | 4.26 ± 0.22 | −0.19 |
| | llama-8b | 2.97 ± 0.22 | 3.09 ± 0.25 | +0.12 |
| | timer | 2.96 ± 0.25 | 2.73 ± 0.27 | −0.23 |
| Delay-Appr. | gpt-4o | 3.93 ± 0.33 | 3.68 ± 0.35 | −0.25 |
| | gpt-3.5 | 2.51 ± 0.24 | 2.22 ± 0.22 | −0.29 |
| | timer | 2.29 ± 0.30 | 1.91 ± 0.26 | −0.38 |
| | llama-8b | 2.04 ± 0.18 | 1.99 ± 0.16 | −0.05 |
| Time-Spec. | llama-8b | 2.13 ± 0.26 | 1.97 ± 0.22 | −0.16 |
| | timer | 1.88 ± 0.26 | 1.78 ± 0.25 | −0.10 |
| | gpt-4o | 1.63 ± 0.25 | 1.53 ± 0.20 | −0.10 |
| | gpt-3.5 | 1.54 ± 0.20 | 1.65 ± 0.20 | +0.11 |

All shifts are within ±0.4 and mostly within ±0.2. Independent behavioral check:
each agent's delay rate is nearly identical across the two simulators (e.g.
TIMER delays on 26.2% of turns under GPT-4o sim vs 26.8% under Claude sim), so
the agents behave the same regardless of who plays the user.

## Caveat (must be stated)

Our reproduced baseline (GPT-4o simulator) does **not** reproduce the absolute
Figure 3 ordering: TIMER-3B is not first on Delay-Appropriateness or
Time-Specificity in our re-run. This is independent of E3's claim — it holds in
both arms — and is discussed separately. Llama-3.1-8B reproduces its published
numbers almost exactly (Coherence 2.97 vs 2.97). E3's conclusion is about rank
*invariance to the simulator*, which is robust; it is not a re-derivation of the
Figure 3 absolute scores. See `.omc/plans/e3-alt-simulator.md` §4 and the
`results/e3_v1_agent-blind-to-delays/` archive for the reproduction analysis.

Additional caveat: Llama-3.1-8B has 5.6–7.8% silent time-parse failures (its vLLM
path gets no structured-output constraint, matching the paper's own setup), so
its delay-based scores understate delay usage; reported, not corrected.

## Artifacts

- `results/e3_verdict.txt` — machine-generated verdict (this table's source).
- `results/dialog-level_<system>_<sim>_T10_n100_seed0.jsonl` — 8 raw simulations.
- `results/claude-sonnet-4-5-eval_dialog-level_<system>_sim-<sim>.jsonl` — 8 judge outputs, 0 fallbacks.
- Reproduce: `./scripts/run_e3_after_sim.sh` (skips completed work).
