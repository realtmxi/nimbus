#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-/scratch/jialu/workloads/Burst_ShareGPT/sharegpt_prompts_burstgpt_timestamps.jsonl}
SCRIPT=${SCRIPT:-/scratch/jialu/nimbus/vllm/run.py}
URL=${URL:-http://127.0.0.1:8007/v1/chat/completions}
MODEL=${MODEL:-qwen3-32b}
SCENARIO=${SCENARIO:-burst_1200}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_TOKENS=${MAX_TOKENS:-2048}
TIMEOUT_S=${TIMEOUT_S:-1000000}
OUT_DIR=${OUT_DIR:-vllmresult}
OUT_FILE=${OUT_FILE:-results_qwen3_32b_vllm_${SCENARIO}_max_num_seqs_${MAX_NUM_SEQS}_sharegpt.jsonl}
EXTRA_ARGS=()
[[ -z "${MAX_TOKENS:-}" ]] || EXTRA_ARGS+=(--max-tokens "$MAX_TOKENS")

python "$SCRIPT" \
  --data "$DATA" \
  --url "$URL" \
  --model "$MODEL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --scenario "$SCENARIO" \
  --timeout-s "$TIMEOUT_S" \
  --out-dir "$OUT_DIR" \
  --output "$OUT_FILE" \
  "${EXTRA_ARGS[@]}"
