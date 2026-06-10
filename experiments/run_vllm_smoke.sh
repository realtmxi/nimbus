#!/usr/bin/env bash
# Real vLLM smoke for the online Nimbus harness.
#
# Starts a temporary local vLLM server, runs:
#   1. normal capacity: should mostly stay local
#   2. pressure capacity: should force Nimbus to outsource
#   3. mini policy sweep: Nimbus vs fixed-fraction baselines under one loop
# then stops the temporary server and writes a small manifest.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \
#       bash experiments/run_vllm_smoke.sh logs/vllm_smoke

set -euo pipefail

OUT="${1:-logs/vllm_smoke}"
PY="${PYTHON:-python3}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL="${SERVED_MODEL:-qwen2.5-0.5b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18200}"
GPU="${GPU:-1}"
URL="http://${HOST}:${PORT}"
LOCAL_KV_TOKENS="${LOCAL_KV_TOKENS:-auto}"
PRESSURE_KV_TOKENS="${PRESSURE_KV_TOKENS:-1000}"
NORMAL_N="${NORMAL_N:-6}"
PRESSURE_N="${PRESSURE_N:-8}"
SYNTHETIC_PROMPT_MODE="${SYNTHETIC_PROMPT_MODE:-sized}"
SYNTHETIC_PROMPT_TOKEN_CAP="${SYNTHETIC_PROMPT_TOKEN_CAP:-2048}"
RUN_SWEEP="${RUN_SWEEP:-1}"
SWEEP_N="${SWEEP_N:-8}"
SWEEP_KV_TOKENS="${SWEEP_KV_TOKENS:-$PRESSURE_KV_TOKENS}"
SWEEP_POLICIES="${SWEEP_POLICIES:-nimbus cachedisp_oracle all_local all_cloud}"
SWEEP_FRACTIONS="${SWEEP_FRACTIONS:-0.25 0.50}"
RUN_TPOT_PROFILE="${RUN_TPOT_PROFILE:-0}"
TPOT_BATCH_SIZES="${TPOT_BATCH_SIZES:-1 2 4}"
TPOT_PROMPT_TOKENS="${TPOT_PROMPT_TOKENS:-512}"
TPOT_DECODE_TOKENS="${TPOT_DECODE_TOKENS:-64}"
TPOT_WARMUP="${TPOT_WARMUP:-0}"
TPOT_REPEATS="${TPOT_REPEATS:-1}"
TPOT_PROFILE_PATH="${TPOT_PROFILE_PATH:-$OUT/tpot_profile.json}"
SLO_S="${SLO_S:-5.0}"
CLOUD_TTFT_MS="${CLOUD_TTFT_MS:-900}"
CLOUD_TTFT_GUARD_MULTIPLIER="${CLOUD_TTFT_GUARD_MULTIPLIER:-1.5}"
TIME_SCALE="${TIME_SCALE:-1000}"
MAX_INFLIGHT="${MAX_INFLIGHT:-2}"
TICK_S="${TICK_S:-0.01}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
REUSE_SERVER="${REUSE_SERVER:-0}"

mkdir -p "$OUT"
DRIVER_LOG="$OUT/driver.log"
VLLM_LOG="$OUT/vllm.log"
PID_FILE="$OUT/vllm.pid"
MANIFEST="$OUT/manifest.json"

started_server=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER_LOG"
}

port_is_live() {
    curl -sS --max-time 2 "$URL/v1/models" >/dev/null 2>&1
}

wait_for_server() {
    for _ in $(seq 1 90); do
        if port_is_live; then
            return 0
        fi
        sleep 2
    done
    return 1
}

stop_server() {
    if [[ "$started_server" == "1" && -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if [[ -n "$pid" ]]; then
            log "Stopping vLLM pid=$pid"
            kill "$pid" >/dev/null 2>&1 || true
            sleep 3
            kill -9 "$pid" >/dev/null 2>&1 || true
        fi
    fi
}

trap stop_server EXIT

log "=== Nimbus vLLM Smoke ==="
log "Output: $OUT"
log "URL: $URL"
log "Model: $MODEL_PATH as $SERVED_MODEL"
log "SLO_S: $SLO_S"
log "CLOUD_TTFT_MS: $CLOUD_TTFT_MS"
log "CLOUD_TTFT_GUARD_MULTIPLIER: $CLOUD_TTFT_GUARD_MULTIPLIER"
log "Synthetic prompts: mode=$SYNTHETIC_PROMPT_MODE cap=$SYNTHETIC_PROMPT_TOKEN_CAP"
SLO_MS="$("$PY" -c "print(float('$SLO_S') * 1000.0)")"
TPOT_PROFILE_ARGS=()

if [[ -z "$MODEL_PATH" ]]; then
    log "MODEL_PATH is required; set it to the local vLLM model directory."
    exit 2
fi

if port_is_live; then
    if [[ "$REUSE_SERVER" != "1" ]]; then
        log "Port $PORT already has a live server. Set REUSE_SERVER=1 to use it."
        exit 2
    fi
    log "Reusing existing server on $URL"
else
    log "Starting vLLM on GPU=$GPU attention_backend=$ATTENTION_BACKEND"
    CUDA_VISIBLE_DEVICES="$GPU" nohup "$VLLM_BIN" serve "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --trust-remote-code \
        --enforce-eager \
        --attention-backend "$ATTENTION_BACKEND" \
        > "$VLLM_LOG" 2>&1 &
    echo "$!" > "$PID_FILE"
    started_server=1
    if ! wait_for_server; then
        log "vLLM did not become ready; last log lines:"
        tail -80 "$VLLM_LOG" | tee -a "$DRIVER_LOG"
        exit 3
    fi
fi

if [[ "$LOCAL_KV_TOKENS" == "auto" ]]; then
    parsed_kv="$(
        grep -Eo 'GPU KV cache size: [0-9,]+ tokens' "$VLLM_LOG" 2>/dev/null \
            | tail -1 \
            | sed -E 's/.*: ([0-9,]+) tokens/\1/' \
            | tr -d ','
    )"
    if [[ -z "$parsed_kv" ]]; then
        log "Could not parse KV token capacity from vLLM log; set LOCAL_KV_TOKENS explicitly."
        exit 4
    fi
    LOCAL_KV_TOKENS="$parsed_kv"
fi

log "Using LOCAL_KV_TOKENS=$LOCAL_KV_TOKENS"

if [[ "$RUN_TPOT_PROFILE" == "1" ]]; then
    read -r -a tpot_batch_sizes <<< "$TPOT_BATCH_SIZES"
    log "Profiling TPOT batches=$TPOT_BATCH_SIZES prompt=$TPOT_PROMPT_TOKENS decode=$TPOT_DECODE_TOKENS"
    "$PY" experiments/profile_serving_tpot.py \
        --serving-url "$URL" \
        --model "$SERVED_MODEL" \
        --engine vllm \
        --gpu "$GPU" \
        --batch-sizes "${tpot_batch_sizes[@]}" \
        --prompt-tokens "$TPOT_PROMPT_TOKENS" \
        --decode-tokens "$TPOT_DECODE_TOKENS" \
        --warmup "$TPOT_WARMUP" \
        --repeats "$TPOT_REPEATS" \
        --output "$TPOT_PROFILE_PATH" 2>&1 | tee -a "$DRIVER_LOG"
    TPOT_PROFILE_ARGS=(--tpot-profile "$TPOT_PROFILE_PATH")
fi

log "Running normal-capacity real vLLM smoke"
"$PY" experiments/run_engine.py \
    --synthetic-burst \
    --synthetic-n "$NORMAL_N" \
    --synthetic-prompt-mode "$SYNTHETIC_PROMPT_MODE" \
    --synthetic-prompt-token-cap "$SYNTHETIC_PROMPT_TOKEN_CAP" \
    --policy nimbus \
    --weight v2 \
    --local real \
    --serving-engine vllm \
    --serving-url "$URL" \
    --model "$SERVED_MODEL" \
    --local-kv-tokens "$LOCAL_KV_TOKENS" \
    --slo-s "$SLO_S" \
    --cloud-ttft-ms "$CLOUD_TTFT_MS" \
    --cloud-ttft-guard-multiplier "$CLOUD_TTFT_GUARD_MULTIPLIER" \
    "${TPOT_PROFILE_ARGS[@]}" \
    --tick-s "$TICK_S" \
    --time-scale "$TIME_SCALE" \
    --max-inflight "$MAX_INFLIGHT" \
    --output-dir "$OUT/normal" 2>&1 | tee -a "$DRIVER_LOG"

log "Running pressure real vLLM smoke"
"$PY" experiments/run_engine.py \
    --synthetic-burst \
    --synthetic-n "$PRESSURE_N" \
    --synthetic-prompt-mode "$SYNTHETIC_PROMPT_MODE" \
    --synthetic-prompt-token-cap "$SYNTHETIC_PROMPT_TOKEN_CAP" \
    --policy nimbus \
    --weight v2 \
    --local real \
    --serving-engine vllm \
    --serving-url "$URL" \
    --model "$SERVED_MODEL" \
    --local-kv-tokens "$PRESSURE_KV_TOKENS" \
    --slo-s "$SLO_S" \
    --cloud-ttft-ms "$CLOUD_TTFT_MS" \
    --cloud-ttft-guard-multiplier "$CLOUD_TTFT_GUARD_MULTIPLIER" \
    "${TPOT_PROFILE_ARGS[@]}" \
    --tick-s "$TICK_S" \
    --time-scale "$TIME_SCALE" \
    --max-inflight "$MAX_INFLIGHT" \
    --output-dir "$OUT/pressure" 2>&1 | tee -a "$DRIVER_LOG"

"$PY" scripts/analysis/plot_engine_sweep.py "$OUT/pressure" \
    --ttft-slo-ms "$SLO_MS" \
    --no-plot 2>&1 | tee -a "$DRIVER_LOG"

if [[ "$RUN_SWEEP" == "1" ]]; then
    read -r -a sweep_policies <<< "$SWEEP_POLICIES"
    read -r -a sweep_fractions <<< "$SWEEP_FRACTIONS"

    log "Running mini policy sweep"
    "$PY" experiments/run_engine_sweep.py \
        --synthetic-burst \
        --synthetic-n "$SWEEP_N" \
        --synthetic-prompt-mode "$SYNTHETIC_PROMPT_MODE" \
        --synthetic-prompt-token-cap "$SYNTHETIC_PROMPT_TOKEN_CAP" \
        --policies "${sweep_policies[@]}" \
        --fractions "${sweep_fractions[@]}" \
        --nimbus-weights v2 \
        --local real \
        --serving-engine vllm \
        --serving-url "$URL" \
        --model "$SERVED_MODEL" \
        --local-kv-tokens "$SWEEP_KV_TOKENS" \
        --slo-s "$SLO_S" \
        --cloud-ttft-ms "$CLOUD_TTFT_MS" \
        --cloud-ttft-guard-multiplier "$CLOUD_TTFT_GUARD_MULTIPLIER" \
        "${TPOT_PROFILE_ARGS[@]}" \
        --tick-s "$TICK_S" \
        --time-scale "$TIME_SCALE" \
        --max-inflight "$MAX_INFLIGHT" \
        --output-dir "$OUT/sweep" 2>&1 | tee -a "$DRIVER_LOG"

    "$PY" scripts/analysis/plot_engine_sweep.py "$OUT/sweep" \
        --ttft-slo-ms "$SLO_MS" \
        --no-plot 2>&1 | tee -a "$DRIVER_LOG"
fi

"$PY" - "$MANIFEST" <<PY
import json
import pathlib
import time

manifest = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "serving_engine": "vllm",
    "serving_url": "$URL",
    "model_path": "$MODEL_PATH",
    "served_model": "$SERVED_MODEL",
    "gpu": "$GPU",
    "attention_backend": "$ATTENTION_BACKEND",
    "local_kv_tokens": float("$LOCAL_KV_TOKENS"),
    "pressure_kv_tokens": float("$PRESSURE_KV_TOKENS"),
    "synthetic_prompt_mode": "$SYNTHETIC_PROMPT_MODE",
    "synthetic_prompt_token_cap": int("$SYNTHETIC_PROMPT_TOKEN_CAP"),
    "slo_s": float("$SLO_S"),
    "cloud_ttft_ms": float("$CLOUD_TTFT_MS"),
    "cloud_ttft_guard_multiplier": float("$CLOUD_TTFT_GUARD_MULTIPLIER"),
    "run_tpot_profile": "$RUN_TPOT_PROFILE" == "1",
    "tpot_profile_path": "$TPOT_PROFILE_PATH" if "$RUN_TPOT_PROFILE" == "1" else None,
    "tpot_batch_sizes": [int(x) for x in "$TPOT_BATCH_SIZES".split()],
    "tpot_prompt_tokens": int("$TPOT_PROMPT_TOKENS"),
    "tpot_decode_tokens": int("$TPOT_DECODE_TOKENS"),
    "run_policy_sweep": "$RUN_SWEEP" == "1",
    "sweep_kv_tokens": float("$SWEEP_KV_TOKENS"),
    "sweep_policies": "$SWEEP_POLICIES".split(),
    "sweep_fractions": [float(x) for x in "$SWEEP_FRACTIONS".split()],
    "normal_summary": str(pathlib.Path("$OUT/normal/engine_summary.csv")),
    "pressure_summary": str(pathlib.Path("$OUT/pressure/engine_summary.csv")),
    "sweep_summary": str(pathlib.Path("$OUT/sweep/engine_summary.csv")) if "$RUN_SWEEP" == "1" else None,
}
path = pathlib.Path("$MANIFEST")
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY

log "Completed. Manifest: $MANIFEST"
