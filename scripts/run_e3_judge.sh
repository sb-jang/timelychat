#!/usr/bin/env bash
# E3 judge stage: score every simulated dialog file with the SAME judge
# (Claude Sonnet 4.5) that the paper used, tagging output by simulator arm.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export ANTHROPIC_API_KEY=$(grep '^export ANTHROPIC_API_KEY' ~/.zshrc | cut -d'"' -f2)
mkdir -p results/logs

N=100
SEED=0
TURNS=10
JUDGE=claude-sonnet-4-5

declare -A MODELS=(
  [timer]=seongbo/timer-3b
  [gpt-4o]=gpt-4o
  [gpt-3.5]=gpt-3.5-turbo
  [llama-8b]=meta-llama/Llama-3.1-8B-Instruct
)

for KEY in "${!MODELS[@]}"; do
  MODEL=${MODELS[$KEY]}
  SAFE=${MODEL//\//--}
  for SIM in gpt-4o claude-sonnet-4-5; do
    IN="results/dialog-level_${SAFE}_${SIM}_T${TURNS}_n${N}_seed${SEED}.jsonl"
    OUT="results/${JUDGE}-eval_dialog-level_${SAFE}_sim-${SIM}.jsonl"
    if [[ ! -s "$IN" ]]; then echo "MISSING INPUT: $IN"; exit 1; fi
    if [[ -s "$OUT" ]]; then echo "SKIP (exists): $OUT"; continue; fi
    echo "=== judging $MODEL (simulator=$SIM)"
    python laaj.py --setting dialog-level --model-name "$MODEL" --evaluator "$JUDGE" \
      --input-file "$IN" --tag "sim-${SIM}" \
      >"results/logs/judge_${SAFE}_${SIM}.log" 2>&1
    LINES=$(wc -l < "$OUT")
    FALLBACK=$(grep -c 'Failed after 3 attempts' "$OUT" || true)
    echo "    -> ${LINES} rows (expect $((N*3))), ${FALLBACK} fallbacks"
  done
done
echo "Judge stage done."
