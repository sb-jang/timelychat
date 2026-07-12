#!/usr/bin/env bash
# E3 arm runner: OpenAI-backed agents (gpt-4o, gpt-3.5-turbo) x both simulators.
# Sequential so the four runs stay inside provider rate limits.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in the environment}"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY in the environment}"
mkdir -p results/logs

N=100
SEED=0
TURNS=10

for MODEL in gpt-4o gpt-3.5-turbo; do
  for SIM in gpt-4o claude-sonnet-4-5; do
    OUT="results/dialog-level_${MODEL}_${SIM}_T${TURNS}_n${N}_seed${SEED}.jsonl"
    if [[ -s "$OUT" ]]; then echo "SKIP (exists): $OUT"; continue; fi
    echo "=== agent=$MODEL simulator=$SIM"
    python evaluate_dialog-level.py \
      --model-type openai --model-name "$MODEL" --simulator "$SIM" \
      --num-dialogs $N --seed $SEED --num-turns $TURNS --workers 32 \
      >"results/logs/sim_${MODEL}_${SIM}.log" 2>&1
    echo "    -> $(wc -l < "$OUT") dialogs"
  done
done
echo "API arm done."
