#!/usr/bin/env bash
set -euo pipefail
DATA=${DATA:-/scratch/jialu/workloads/Burst_ShareGPT/sharegpt_prompts_burstgpt_timestamps.jsonl}
SCRIPT=${SCRIPT:-/scratch/jialu/workloads/Burst_ShareGPT/vllm/run.py}
URL=${URL:-https://openrouter.ai/api/v1/chat/completions}
MODEL=${MODEL:-MiniMax-M2.5}
API_KEY_ENV=${API_KEY_ENV:-OPENROUTER_API_KEY1}
NUM_PREDICT=${NUM_PREDICT:-2048}
TIMEOUT_S=${TIMEOUT_S:-1000000}
TEMPERATURE=${TEMPERATURE:-0}
OUT_ROOT=${OUT_ROOT:-results_minimax_m2_5_openrouter2}
SCENARIOS=${SCENARIOS:-"normal"}
#  burst_1200 extreme_burst_1200
if [[ -z "$(printenv "$API_KEY_ENV" || true)" ]]; then
  echo "Missing OpenRouter API key. Export ${API_KEY_ENV} before running." >&2
  exit 1
fi

read -r -a SCENARIO_LIST <<< "$SCENARIOS"

EXTRA_ARGS=()
if [[ -n "${TOP_P:-}" ]]; then
  EXTRA_ARGS+=(--top-p "$TOP_P")
fi
EXTRA_ARGS+=(--max-tokens "$NUM_PREDICT")

for SCENARIO in "${SCENARIO_LIST[@]}"; do
  OUT_DIR="${OUT_ROOT}/${SCENARIO}"
  echo "=== OpenRouter ${MODEL} :: ${SCENARIO} -> ${OUT_DIR} ==="
  python "$SCRIPT" \
    --data "$DATA" \
    --url "$URL" \
    --model "$MODEL" \
    --scenario "$SCENARIO" \
    --api-key-env "$API_KEY_ENV" \
    --temperature "$TEMPERATURE" \
    --timeout-s "$TIMEOUT_S" \
    --out-dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]}"
done
