#!/usr/bin/env bash
# Wait for all 8 simulation outputs, gate on the time-parse audit, then judge + compare.
# The audit gate matters: convert_to_minutes() maps unparseable text to 0.0, which is
# written as a plausible "0 minutes". Judging corrupted dialogs would spend API budget
# producing numbers that silently understate delay usage.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

PATTERN='results/dialog-level_*_T10_n100_seed0.jsonl'
DEADLINE=$(( $(date +%s) + 4*3600 ))

while :; do
  # `ls` exits 2 on no match, which under `set -e` would abort before the first run lands.
  COUNT=$(find results -maxdepth 1 -name 'dialog-level_*_T10_n100_seed0.jsonl' | wc -l)
  [ "$COUNT" -eq 8 ] && break
  # Guard on the runner scripts, not on `python`: run_e3_gpu.sh invokes python once per
  # (model, simulator) pair, so there are gaps between invocations with no python alive.
  # pgrep -f would also match this script's own command line, so exclude our own PID.
  if ! pgrep -f "run_e3_gpu.sh|run_e3_api.sh" | grep -qv "^$$\$"; then
    echo "ABORT: no simulation runner alive, only ${COUNT}/8 outputs"; exit 1
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "ABORT: 4h deadline, ${COUNT}/8 outputs"; exit 1; fi
  sleep 60
done
echo "All 8 simulation outputs present."

echo; echo "########## align subsample across arms ##########"
# Claude refuses a small number of conversations (stop_reason=refusal), so an arm can be
# short a dialog. Rank comparison requires both arms to have scored the same dialogs.
python -m analysis.align_subsample "$PATTERN"

echo; echo "########## time-parse audit (gate) ##########"
# Llama-8B is exempted, not excused: VLLMModel.make_prompt passes output_format="" so the
# vLLM path gets no JSON instruction and no schema (timelychat/models.py:93). The paper's
# turn-level runs share that path, so we reproduce it rather than "fix" the baseline; the
# audit still prints its exact failure rate for the write-up.
if ! python -m analysis.audit_time_parse "$PATTERN" --threshold 5.0 --exempt meta-llama; then
  echo "ABORT: silent time-parse failures exceed 5% for a non-exempt system; judge stage not run."
  exit 1
fi

echo; echo "########## judge ##########"
./scripts/run_e3_judge.sh

echo; echo "########## E3 rank comparison ##########"
python -m analysis.compare_simulators \
  --baseline-sim gpt-4o --alt-sim claude-sonnet-4-5 --judge claude-sonnet-4-5 \
  timer=seongbo--timer-3b \
  gpt-4o=gpt-4o \
  gpt-3.5=gpt-3.5-turbo \
  llama-8b=meta-llama--Llama-3.1-8B-Instruct \
  | tee results/e3_verdict.txt
