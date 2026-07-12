# E3 Handoff — Alternative User Simulator Re-evaluation

**Status: DONE.** All 8 sims + 8 judge runs complete (0 fallbacks). The result and
its framing are in `results/E3_RESULT.md`; the raw verdict is `results/e3_verdict.txt`.
**Answer to sV8H #4: ranking is robust to the simulator swap** (Coherence preserved;
the two rank changes are between systems with overlapping CIs). E3 is reported with
**relative** framing (rank invariance vs our reproduced baseline), because the
baseline does not reproduce Figure 3's absolute values — see the Caveat in
`results/E3_RESULT.md` and §4/§7 below. No further runs are required for E3 itself;
§7 lists the one open thread (Figure 3 absolute-value reproduction) left for later.

E3 answers reviewer **sV8H #4**: are the dialog-level conclusions an artifact of the
**GPT-4o user simulator**? We re-simulate the same dialogs with **Claude Sonnet 4.5**
as the user, re-judge with the *same* judge, and compare rankings. Together with the
cross-judge check (Appendix C.5) and the cross-generator check (E4), this completes a
3-axis robustness story (generator · simulator · judge).

---

## 0. Repo / branch

- Remote: `git@github.com:sb-jang/timelychat.git`
- Work branch: **`rebuttal-experiments`** (off `origin/anonymize`). Do **not** use `main`.

```bash
git fetch origin && git checkout rebuttal-experiments
```

## 1. Environment

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install scipy          # NOT in requirements.txt; analysis/ needs it
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

> **Rotate both keys.** They were pasted into chat transcripts in two separate sessions.

> **`scripts/*.sh` currently read the keys out of `~/.zshrc`** (`grep '^export OPENAI_API_KEY' ~/.zshrc`).
> That is machine-specific and must be replaced with a plain `${OPENAI_API_KEY:?}` /
> `${ANTHROPIC_API_KEY:?}` read. It was not fixed in place because the scripts were
> executing at the time (bash reads a script by byte offset; editing a running script
> corrupts it). **This is the first thing to do on resume.**

Models / data:
- Agent (TIMER): `seongbo/timer-3b` — `--model-type hf`. Weights are sha256-identical to
  `anonymous17711771/timer-3b` from the README (verified), so either ID works.
- Baselines: `gpt-4o`, `gpt-3.5-turbo` (`--model-type openai`),
  `meta-llama/Llama-3.1-8B-Instruct` (`--model-type vllm`).
- Benchmark: `load_dataset("anonymous17711771/timelychat", split="eval")` — 324 dialogs.

## 2. Scope decisions already made

| Decision | Value | Why |
|---|---|---|
| Systems | timer-3b, gpt-4o, gpt-3.5-turbo, Llama-3.1-8B | **70B dropped**: needs ~4 GPUs, only GPU 0 was free |
| Subsample | `n=100`, `--seed 0`, identical idx across both arms | E3 is a robustness check; matched subsample makes the rank comparison valid |
| Turns | `--num-turns 10` | matches README |
| Judge | `claude-sonnet-4-5` for **both** arms | E3 holds the judge fixed and varies only the simulator |
| Llama parse failures | reproduce the paper, report the rate | see §5 |

## 3. What changed in the code (all committed on this branch)

`evaluate_dialog-level.py`
- `make_simulator(name)` routes on the model name: `gpt-*` → OpenAI, `claude-*` → Anthropic.
  Previously `--simulator` only swapped the model *string* and always called OpenAI, and a
  module-level `OpenAI(api_key=os.environ["OPENAI_API_KEY"])` raised `KeyError` at import
  even for a Claude run. Clients are now constructed lazily inside the factory.
- Saves `{idx, context, alt_speaker_list, time_elapseds, raw_time_outputs}` — the schema
  `laaj.py` actually reads. Output path is now deterministic (no timestamp):
  `results/dialog-level_<model>_<simulator>_T<turns>_n<N>_seed<S>.jsonl`.
- `--num-dialogs`, `--seed`, `--workers`. Dialogs run in a `ThreadPoolExecutor`; for
  `hf`/`vllm` agents `agent.generate` is serialized under a lock so only the simulator's
  API calls overlap.
- `SimulatorRefusal`: Claude returns `stop_reason="refusal"` on some conversations,
  *deterministically*, so retrying cannot help. That dialog is dropped and the run
  continues. A transient null parse (`parsed_output is None`) is still retried 3×,
  mirroring `laaj.py:38`. On retry exhaustion it raises rather than skipping, because
  silently dropping a dialog would break the matched subsample.
- Delays are stored via `format_minutes(...)`, not `f"{int(...)} minutes"` (see §4, defect 9).

`timelychat/models.py`
- New `render_history(example)`: renders a delayed turn as `"A: (30 minutes later) utt"`
  and an immediate one as `"A: utt"`. Used by `VLLMModel` / `OpenAIModel` / `AnthropicModel`.
  When `example` has no `time_elapseds` (i.e. turn-level), the output is **byte-identical to
  the old rendering** — verified, so E2/turn-level results are unaffected.
- `HfModel` still hardcodes `"0 minutes later"` for every context turn. **This is intentional**
  (TIMER's training format) and confirmed by the author. Do not "fix" it.

`utils/postprocess.py`
- New `format_minutes(minutes)` — exact inverse of `convert_to_minutes` (round-trip verified).
  `format_minutes(0) == "0 minutes"`, the literal `laaj.py:92` and `HfModel` depend on.

`laaj.py`
- `--input-file` (bypasses the glob) and `--tag` (output filename suffix). The old glob
  `./results/{setting}_{model}_response_{icl}*.jsonl` requires a `_response_` segment that
  dialog-level filenames never had → `FileNotFoundError`; and the output name omitted the
  simulator, so the two arms overwrote each other. The turn-level glob path is untouched.

New `analysis/`
- `audit_time_parse.py` — gate against silent time-parse failures (§5).
- `align_subsample.py` — force every arm onto the identical idx set; idempotent.
- `compare_simulators.py` — per-metric mean ± 95% CI for both arms, ranking under each,
  and whether a rank flip is *meaningful* (CIs disjoint) or noise. Prints an explicit
  `E3 PASSES/FAILS` verdict and the paper's headline claim per metric.

New `scripts/`
- `run_e3_api.sh` — 4 OpenAI-agent runs (sequential, `--workers 32`).
- `run_e3_gpu.sh` — 4 local-agent runs (sequential, GPU 0, `--workers 8`, `--num-gpus 1`).
- `run_e3_after_sim.sh` — waits for all 8, then align → audit gate → judge → compare.
  All three skip work whose output already exists, so they are safe to re-run.

## 4. Two protocol defects found by comparing against the paper's own README table

v1 reproduced **Llama-8B almost exactly** (Coh 3.03 vs 2.97, DA 2.28 vs 2.38, TS 2.27 vs 2.30)
yet put **TIMER-3B last** on Delay-Appropriateness where the paper has it first. Seeding
(first utterance only), checkpoint, judge, simulator and CLI flags were each confirmed to
match the paper. The cause was two real defects:

8. **Prompted agents never saw the delays they had produced.** `VLLMModel`/`OpenAIModel`/
   `AnthropicModel` built history as `f"{spk}: {utt}"`, with no elapsed time. In dialog-level
   simulation the agent generates its own delays, so it must see them on later turns.
   This is why v1's Time-Specificity was pressed flat at 1.5–2.3 for *every* system — no
   agent could ground a response in elapsed time. Fixed by `render_history()`.
9. **Delays were rendered to the agent in raw minutes.** `"1 week"` → `"10080 minutes"`.
   Harmless in v1 (only the judge saw it); once fix 8 put the delay into the agent's prompt,
   agents echoed it: `"(10080 minutes later) Wow, 10080 minutes later! Time really flies"`.
   Fixed by `format_minutes()`.

**Role mapping is correct as-is:** `agent_speaker = target_speaker` (`evaluate_dialog-level.py`).
The delayed speaker is the agent, consistent with `simulator_prompt` ("agent responds after the
elapsed time"; "user … without any delay") and the Delay-Appropriateness rubric ("the extent to
which *the agent* poses delays"). 157/324 eval dialogs are seeded by the agent's own utterance;
the loop condition already handles who speaks first.

## 5. Known validity caveats (must appear in the write-up)

- **Silent time-parse failures.** `convert_to_minutes` returns `0.0` on no-match
  (`utils/postprocess.py`), and that is written as an ordinary `"0 minutes"` — indistinguishable
  from a deliberate no-delay, which is exactly what Delay-Appropriateness scores.
  `OpenAIModel.generate` even returns the literal string `"Error: Invalid response format"` after
  exhausting retries, which would become `0 minutes`. Every record now stores `raw_time_outputs`;
  `analysis/audit_time_parse.py` measures the rate and gates the judge stage at 5%.
  - timer-3b, gpt-4o, gpt-3.5: **0.0%**.
  - **Llama-3.1-8B: 8.4% (gpt-4o sim) / 11.0% (claude sim)** in v1. Root cause:
    `VLLMModel.make_prompt` passes `output_format=""`, so the vLLM path gets no JSON instruction
    and no schema, unlike the OpenAI/Anthropic paths. `evaluate_turn-level.py` uses the same
    `get_model`, so the paper's published Llama numbers share this behavior. **Decision (author):
    reproduce the paper; do not fix the baseline.** Llama is `--exempt`ed from the gate so its
    rate is *reported*, not hidden. The two arms differ by 2.6pp, so Llama's own cross-arm
    comparison is partly confounded by parse noise.
- **Claude refuses some conversations.** Llama hallucinated a fake Harvard COVID-19 vaccine study
  with a fake *Lancet* publication and a fake lead researcher, and Claude declined to continue it
  (`stop_reason="refusal"`, eval idx 86). Deterministic for a fixed prefix, but the prefix is
  resampled each run (`temperature=1.0`), so it may or may not recur. `align_subsample.py` exists
  precisely so that a dialog lost in one arm is removed from all arms.
- **n=100, not the full 324.** Both arms use the same 100 (seed 0), so the *rank* comparison is
  matched; absolute CIs are wider than the paper's.
- **timer-3b exceeds its 512-token tokenizer limit** on long simulated dialogs. T5 uses relative
  position bias and `HfModel.generate` never passes `truncation=True`, so nothing is cut.
  Verified harmless: all 500 v1 predictions parsed into 8 well-formed time expressions, 0% masked.

## 6. How to run (end to end, ~2h)

```bash
source .venv/bin/activate
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...

./scripts/run_e3_api.sh        # 4 runs, ~40 min  (background-safe)
./scripts/run_e3_gpu.sh        # 4 runs, ~1.5 h   (GPU 0; run concurrently with the above)
./scripts/run_e3_after_sim.sh  # waits for all 8, then align -> audit -> judge -> compare
```

Final verdict lands in `results/e3_verdict.txt`.

Sanity checks the chain already performs:
- every simulation file has 100 lines and identical idx sets across arms;
- each judge file has `100 × 3 = 300` rows (Coherence / Delay-Appropriateness / Time-Specificity);
- fallback rows (`"Failed after 3 attempts"`) < 2% — v1 saw at most 3/300;
- masked parse failures ≤ 5% for every non-exempt system.

## 7. Status and the one open thread

E3 itself is **done and reported** (`results/E3_RESULT.md`). The simulator-robustness
claim — the actual answer to sV8H #4 — is settled: ranking is preserved on Coherence
and the two rank changes are between systems with overlapping CIs. Reported with
relative framing (vs our reproduced baseline). The `scripts/*.sh` now read keys from
the environment (`${VAR:?}`), no longer from `~/.zshrc`.

**Open thread, deliberately deferred by the author — not required for E3:** our
reproduced baseline (GPT-4o simulator) does not reproduce Figure 3's *absolute*
ordering (TIMER-3B is not first on Delay-Appropriateness or Time-Specificity in our
re-run; GPT models score higher than published, TIMER lower; Llama-8B matches almost
exactly). The two protocol fixes in §4 moved TIMER in the right direction but did not
close the gap. Already ruled out: seeding, checkpoint (weights sha256-identical),
judge model, simulator, role mapping, and the two §4 defects. The most likely
remaining suspect is the **judge-facing transcript rendering** in `laaj.py:88-96`
(delay notation / speaker labels / rubric wording) differing from the paper-era
dialog-level script. If that script can be recovered, diff the exact string the judge
receives against it. This affects the paper's Figure 3 reproduction, **not** E3's
simulator-robustness conclusion.

Paper's dialog-level table, for comparison (README §Results):

| Model | Coherence | Delay Appropriateness | Time Specificity |
|---|---|---|---|
| Llama 3.1 8B | 2.97 | 2.38 | 2.30 |
| GPT-3.5 | 3.17 | 1.86 | 1.13 |
| GPT-4o | **4.05** | 2.65 | 1.57 |
| TIMER-3B | 3.30 | **2.91** | **2.76** |

v1 results (superseded, kept as evidence for §4) are in `results/e3_v1_agent-blind-to-delays/`,
which is gitignored — copy it manually if you move machines.

## 8. Rest of the rebuttal (unchanged)

| Exp | Status |
|---|---|
| E2-a sign test | DONE — dialog-level human-eval p≈0.82–0.90 (n.s.); turn-level time-spec p<0.001 |
| E2-b/-c/-d | Aggregators written + verified. E2-b needs the **baseline-arm** dialog-level judge files this experiment produces. E2-c/-d still need the main `timer-3b` turn-level TIME outputs. |
| E4 order bias | DONE — no significant order bias (GPT-4o + Claude) |
| **E3 alt simulator** | **This doc — v2 runs in flight** |
| E1 55K audit | Pending — 55K train split is HF-private, needs a token (`HF_TOKEN` unset here) |
| Response drafts | Pending (no API needed) |

## 9. Bring back to the local machine

- `results/claude-sonnet-4-5-eval_dialog-level_*_sim-*.jsonl` (both arms) — also unblocks **E2-b**.
- `results/dialog-level_*.jsonl` raw simulated dialogs, so the judge can be re-run without GPUs.
- The main `timer-3b` **turn-level** TIME outputs — still needed for **E2-c/-d**.
