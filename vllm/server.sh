#!/usr/bin/env bash
set -euo pipefail
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="/usr/local/cuda-12.9/compat"

MODEL_DIR=${MODEL_DIR:-/scratch/jialu/models/Qwen3.6-35B-A3B}
MODEL_NAME=${MODEL_NAME:-Qwen3.6-35B-A3B}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8007}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
SPECULATIVE_ENABLED=${SPECULATIVE_ENABLED:-1}
SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG:-'{"method":"mtp","num_speculative_tokens":5}'}
SCENARIO=${SCENARIO:-normal}
if [[ "$SPECULATIVE_ENABLED" == "1" || "$SPECULATIVE_ENABLED" == "true" ]]; then
  SPEC_TAG="spec"
else
  SPEC_TAG="nospec"
fi
LOG=${LOG:-vllm_${MODEL_NAME}_${SCENARIO}_max_num_seqs_${MAX_NUM_SEQS}_${SPEC_TAG}.log}

export CUDA_VISIBLE_DEVICES

nohup vllm serve "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  > "$LOG" 2>&1 &
  # --speculative-config "$SPECULATIVE_CONFIG" \
echo "vLLM server starting on http://${HOST}:${PORT}/v1/chat/completions with pid $!"
echo "log: $LOG"
