#!/usr/bin/env bash
set -euo pipefail

# --- vLLM server settings ---
export VLLM_ENABLE_CUDA_COMPATIBILITY=${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}
export VLLM_CUDA_COMPATIBILITY_PATH=${VLLM_CUDA_COMPATIBILITY_PATH:-"/usr/local/cuda-12.9/compat"}

MODEL_DIR=${MODEL_DIR:-/scratch/jialu/models/Qwen3.6-35B-A3B}
MODEL_NAME=${MODEL_NAME:-Qwen3.6-35B-A3B}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8005}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
SPECULATIVE_ENABLED=${SPECULATIVE_ENABLED:-1}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-1}
SPEC_METHOD=${SPEC_METHOD:-mtp}
SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG:-"{\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}"}

export CUDA_VISIBLE_DEVICES

# --- Client workload settings ---
DATA=${DATA:-/scratch/jialu/workloads/Burst_ShareGPT/sharegpt_prompts_burstgpt_timestamps.jsonl}
SCRIPT=${SCRIPT:-/scratch/jialu/nimbus/vllm/run.py}
URL=${URL:-http://$HOST:$PORT/v1/chat/completions}
MODEL=${MODEL:-$MODEL_NAME}
SCENARIO=${SCENARIO:-burst_1200}


OUT_DIR=${OUT_DIR:-vllmresult}
# Add a tag to indicate whether speculative decoding is enabled
if [[ "$SPECULATIVE_ENABLED" == "1" || "$SPECULATIVE_ENABLED" == "true" ]]; then
  # Include explicit param name for clarity in outputs
  SPEC_TAG="spec_num_speculative_tokens_${NUM_SPECULATIVE_TOKENS}"
else
  SPEC_TAG="nospec"
fi
SERVER_LOG=${SERVER_LOG:-vllm_${MODEL_NAME}_${SCENARIO}_max_num_seqs_${MAX_NUM_SEQS}_${SPEC_TAG}.log}
OUT_FILE=${OUT_FILE:-results_qwen3_35b_a3b_vllm_${SCENARIO}_max_num_seqs_${MAX_NUM_SEQS}_${SPEC_TAG}_sharegpt.jsonl}

# --- Start vLLM server if endpoint is not responding ---
if ! curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Starting vLLM server on $HOST:$PORT ..."
  SERVE_ARGS=("$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code)
  if [[ "$SPECULATIVE_ENABLED" == "1" || "$SPECULATIVE_ENABLED" == "true" ]]; then
    SERVE_ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
  fi
  nohup vllm serve "${SERVE_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "vLLM PID: $SERVER_PID; log: $SERVER_LOG"

  # Wait for readiness (up to ~120s)
  for i in {1..120}; do
    if curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
      echo "vLLM server is ready."
      break
    fi
    sleep 1
  done
fi

# --- Run workload ---
# Client accepts only run.py flags; do not pass server-only options here.
EXTRA_ARGS=()
[[ -z "${MAX_TOKENS:-}" ]] || EXTRA_ARGS+=(--max-tokens "$MAX_TOKENS")

python "$SCRIPT" \
  --data "$DATA" \
  --url "$URL" \
  --model "$MODEL" \
  --scenario "$SCENARIO" \
  --out-dir "$OUT_DIR" \
  --output "$OUT_FILE" \
  "${EXTRA_ARGS[@]}"
