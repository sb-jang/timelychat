#!/usr/bin/env bash
# E3 arm runner: local agents (timer-3b via HF, Llama-3.1-8B via vLLM) x both simulators.
# Sequential: GPU 0 is the only free device (GPUs 1-3 are occupied).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in the environment}"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY in the environment}"
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/logs

N=100
SEED=0
TURNS=10

run() {  # run <model-type> <model-name> <simulator>
  local TYPE=$1 MODEL=$2 SIM=$3
  local SAFE=${MODEL//\//--}
  local OUT="results/dialog-level_${SAFE}_${SIM}_T${TURNS}_n${N}_seed${SEED}.jsonl"
  if [[ -s "$OUT" ]]; then echo "SKIP (exists): $OUT"; return; fi
  echo "=== agent=$MODEL ($TYPE) simulator=$SIM"
  python evaluate_dialog-level.py \
    --model-type "$TYPE" --model-name "$MODEL" --simulator "$SIM" \
    --num-dialogs $N --seed $SEED --num-turns $TURNS --workers 8 \
    --num-gpus 1 --gpu-memory-utilization 0.85 \
    >"results/logs/sim_${SAFE}_${SIM}.log" 2>&1
  echo "    -> $(wc -l < "$OUT") dialogs"
}

for SIM in gpt-4o claude-sonnet-4-5; do
  run hf seongbo/timer-3b "$SIM"
done
for SIM in gpt-4o claude-sonnet-4-5; do
  run vllm meta-llama/Llama-3.1-8B-Instruct "$SIM"
done
echo "GPU arm done."
