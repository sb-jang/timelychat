# E3 Handoff — Alternative User Simulator Re-evaluation (GPU server)

Purpose: resume **E3** (rebuttal experiment) on a GPU server. E3 re-runs the
dialog-level evaluation with an **alternative user simulator** (Claude or
LLaMA-70B instead of GPT-4o) and checks that model **rankings are preserved**.
Combined with the existing cross-judge check (Appendix C.5) and the cross-
generator check (E4), this completes a **3-axis robustness** story
(generator · simulator · judge) for reviewer **sV8H #4**.

This doc is self-contained: branch state, environment, the one code change
needed, exact run commands, and acceptance criteria.

---

## 0. Repo / branch state

- Remote: `git@github.com:sb-jang/timelychat.git`
- Work branch: **`rebuttal-experiments`** (branched off `origin/anonymize`, the
  full pipeline: `laaj.py`, `AnthropicModel`, F1/FPR metrics, ablation flags).
- Do NOT use `main` — it is an older trimmed public branch missing `laaj.py`,
  the Anthropic model class, and the timing-classification metrics.

```bash
git fetch origin
git checkout rebuttal-experiments   # or: git checkout -b rebuttal-experiments origin/rebuttal-experiments
```

New files added this session (all committed on this branch):
- `analysis/significance.py`  — E2-a sign test (done)
- `analysis/bucket_accuracy.py`, `analysis/timing_metrics_from_file.py`,
  `analysis/dialog_stats.py`, `analysis/buckets.py` — E2-b/-c/-d aggregators (verified)
- `analysis/compare_order.py`, `datagen/generate_dialog.py` — E4 (done)
- `results/e4/*.jsonl` — E4 outputs (GPT-4o + Claude, both orders)
- `utils/metrics.py` — heavy imports (torchmetrics/evaluate) made lazy

## 1. Environment (server)

```bash
cd timelychat
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt        # torch, vllm, transformers, datasets, openai, anthropic, ...
# API keys: the local machine kept them in ~/.zshrc. On the server, export:
export ANTHROPIC_API_KEY=...   # valid (verified this session)
export OPENAI_API_KEY=...      # needs an account WITH credit (local key hit 429 insufficient_quota until topped up)
# NOTE: keys were exposed in a chat transcript this session -> ROTATE them.
```

Models / data:
- Agent (TIMER): `seongbo/timer-3b` (HF seq2seq, `--model-type hf`)
- Baselines: `meta-llama/Llama-3.1-8B-Instruct`, `...-70B-Instruct` (vLLM,
  `--model-type vllm`, needs 4×A100 for 70B), `gpt-4o`/`gpt-3.5` (`--model-type openai`)
- Benchmark: `load_dataset("anonymous17711771/timelychat", split="eval")` (324 dialogs; `seongbo/timelychat` is the public mirror, eval split only)

## 2. Overall rebuttal status (context)

| Exp | Status |
|---|---|
| E2-a sign test | DONE — dialog-level human-eval p≈0.82–0.90 (n.s.); turn-level time-spec p<0.001 (sig win) |
| E2-b/-c/-d | Aggregators written+verified. Need real `results/` files: main `timer-3b` time outputs + **dialog-level** laaj judge files (only turn-level ones copied so far) |
| E4 order bias | DONE — GPT-4o + Claude, no significant order bias on any measure (INSTANT length p=0.059 borderline but null on Claude → noise) |
| **E3 alt simulator** | **THIS DOC — needs GPU** |
| E1 55K audit | Pending — 55K train is HF private (needs HF token) |
| Response drafts | Pending (no API needed) |

## 3. E3 objective

Show the dialog-level conclusions (TIMER-3B highest **delay-appropriateness**
and **time-specificity**; GPT-4o highest coherence) are **not an artifact of the
GPT-4o user simulator**. Swap the simulator, re-simulate, re-judge, compare ranks.

## 4. Current code state relevant to E3

- `evaluate_dialog-level.py`
  - Simulates `--num-turns` interactions between an agent and a **user simulator**.
  - **The simulator is hardwired to the OpenAI client** (module-level `client =
    OpenAI(...)` at ~L16, and `client.beta.chat.completions.parse(model=args.simulator, ...)`
    at ~L48). `--simulator` only changes the model *string*, still via OpenAI.
  - Saves `{"context", "speaker_list", "time_elapsed"}` per dialog to
    `results/dialog-level_<model>_<simulator>_T<turns>_<ts>.jsonl`.
- `timelychat/models.py` — has `AnthropicModel` and `--model-type anthropic`
  for the **agent**; simulator is separate (see above).
- `laaj.py` — Claude Sonnet 4.5 dialog-level judge. Reads `--setting dialog-level`
  files and expects fields **`alt_speaker_list`** and **`time_elapseds`** (plural)
  plus `context`.
- `analysis/dialog_stats.py` — aggregates laaj per-example scores into
  n/mean/std/95% CI + Welch t-test vs a reference system (use for rank check).

### ⚠️ Known gap to fix (schema mismatch)
`evaluate_dialog-level.py` writes `speaker_list` / `time_elapsed`, but `laaj.py`
dialog-level reads `alt_speaker_list` / `time_elapseds`. Running laaj on the
dialog output as-is will `KeyError`. Fix by either renaming on save in
`evaluate_dialog-level.py` (preferred) or adding an adapter step.

## 5. Tasks (in order)

1. **Add simulator routing** to `evaluate_dialog-level.py` so `--simulator`
   selects the backend, not just the model name:
   - `gpt-4o` / `gpt-3.5` → OpenAI (existing path)
   - `claude-sonnet-4-5` → Anthropic (`anthropic.Anthropic().beta.messages.parse`,
     `betas=["structured-outputs-2025-11-13"]`, `output_format=Output`) — mirror the
     pattern already used in `laaj.py` and in `datagen/generate_dialog.py::make_caller`.
   - (optional) `meta-llama/Llama-3.1-70B-Instruct` → vLLM.
   Keep `Output` (pydantic `answer: str`) as the structured schema.
2. **Fix the save schema** so laaj can consume it: write `alt_speaker_list`
   (= the user/agent role list) and `time_elapseds` (list) instead of
   `speaker_list` / `time_elapsed`.
3. **Re-simulate** all 5 systems with the ALT simulator (primary run = the
   original used GPT-4o; alt = Claude Sonnet 4.5, and if feasible LLaMA-70B):
   ```bash
   for M in seongbo/timer-3b gpt-4o gpt-3.5 meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B-Instruct; do
     python evaluate_dialog-level.py --model-type <hf|openai|vllm> --model-name $M \
       --simulator claude-sonnet-4-5 --num-turns 10
   done
   ```
   (Match the paper: sample dialogs with ≥1 delayed response as the seed first turn.)
4. **Judge** the simulated dialogs with the SAME judge (Claude Sonnet 4.5):
   ```bash
   python laaj.py --setting dialog-level --model-name <model> --evaluator claude-sonnet-4-5
   ```
5. **Compare ranks** across simulators:
   ```bash
   python -m analysis.dialog_stats \
     timer=results/claude-sonnet-4-5-eval_dialog-level_seongbo--timer-3b_.jsonl \
     gpt-4o=results/claude-sonnet-4-5-eval_dialog-level_gpt-4o_.jsonl \
     ... --ref timer
   ```
   Do this for the GPT-4o-simulator scores and the Claude-simulator scores;
   report whether the ranking (TIMER top on delay-appr / time-spec) holds.

## 6. Acceptance criteria

- E3 passes if, under the alternative simulator, **TIMER-3B still ranks highest
  on delay-appropriateness and time-specificity** and GPT-4o still leads
  coherence (i.e., Figure 3 ordering preserved). Report per-system mean ± 95% CI
  and note any rank changes honestly.

## 7. Bring back to the local machine

- The dialog-level laaj score files (`results/claude-sonnet-4-5-eval_dialog-level_*`)
  — these also unblock **E2-b** (Fig 3 CI) locally via `analysis/dialog_stats.py`.
- The main `timer-3b` turn-level TIME outputs
  (`results/turn-level_seongbo--timer-3b_time_zeroshot_*.jsonl` and `..._incremental_*`)
  — unblock **E2-c/-d** locally (bucket accuracy + Table 2 F1/FPR reproduction).
