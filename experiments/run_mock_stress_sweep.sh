#!/usr/bin/env bash
# No-GPU policy stress sweep for Nimbus algorithm development.
#
# This runs the online accounting loop in mock-local mode, where synthetic token
# metadata directly controls local service time and KV pressure. Use this for
# quick policy comparisons; use run_vllm_smoke.sh only for serving integration.

set -euo pipefail

OUT="${1:-logs/mock_stress_sweep}"
PY="${PYTHON:-python3}"
SYNTHETIC_N="${SYNTHETIC_N:-200}"
SYNTHETIC_PROMPT_MODE="${SYNTHETIC_PROMPT_MODE:-stub}"
SYNTHETIC_PROMPT_TOKEN_CAP="${SYNTHETIC_PROMPT_TOKEN_CAP:-2048}"
POLICIES="${POLICIES:-nimbus cachedisp_oracle random all_local all_cloud}"
FRACTIONS="${FRACTIONS:-0.10 0.20 0.30 0.40 0.50 0.70}"
SLO_S="${SLO_S:-1.0}"
CLOUD_TTFT_MS="${CLOUD_TTFT_MS:-300}"
CLOUD_TTFT_GUARD_MULTIPLIER="${CLOUD_TTFT_GUARD_MULTIPLIER:-1.5}"
LOCAL_KV_TOKENS="${LOCAL_KV_TOKENS:-200000}"
TIME_SCALE="${TIME_SCALE:-50}"
MAX_INFLIGHT="${MAX_INFLIGHT:-16}"
TICK_S="${TICK_S:-0.01}"
PREFILL_TPUT="${PREFILL_TPUT:-50000}"
TPOT_S="${TPOT_S:-0.03}"
MOCK_SERVICE_MODEL="${MOCK_SERVICE_MODEL:-v2}"
SEED="${SEED:-42}"

mkdir -p "$OUT"
DRIVER_LOG="$OUT/driver.log"
MANIFEST="$OUT/manifest.json"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER_LOG"
}

read -r -a policy_args <<< "$POLICIES"
read -r -a fraction_args <<< "$FRACTIONS"
SLO_MS="$("$PY" -c "print(float('$SLO_S') * 1000.0)")"

log "=== Nimbus Mock Stress Sweep ==="
log "Output: $OUT"
log "Policies: $POLICIES"
log "Fractions: $FRACTIONS"
log "SLO_S: $SLO_S"
log "CLOUD_TTFT_MS: $CLOUD_TTFT_MS"
log "CLOUD_TTFT_GUARD_MULTIPLIER: $CLOUD_TTFT_GUARD_MULTIPLIER"
log "LOCAL_KV_TOKENS: $LOCAL_KV_TOKENS"
log "MOCK_SERVICE_MODEL: $MOCK_SERVICE_MODEL"

"$PY" experiments/run_engine_sweep.py \
    --synthetic-burst \
    --synthetic-n "$SYNTHETIC_N" \
    --synthetic-prompt-mode "$SYNTHETIC_PROMPT_MODE" \
    --synthetic-prompt-token-cap "$SYNTHETIC_PROMPT_TOKEN_CAP" \
    --policies "${policy_args[@]}" \
    --fractions "${fraction_args[@]}" \
    --nimbus-weights v2 \
    --local mock \
    --local-kv-tokens "$LOCAL_KV_TOKENS" \
    --slo-s "$SLO_S" \
    --cloud-ttft-ms "$CLOUD_TTFT_MS" \
    --cloud-ttft-guard-multiplier "$CLOUD_TTFT_GUARD_MULTIPLIER" \
    --time-scale "$TIME_SCALE" \
    --max-inflight "$MAX_INFLIGHT" \
    --tick-s "$TICK_S" \
    --mock-service-model "$MOCK_SERVICE_MODEL" \
    --prefill-tput "$PREFILL_TPUT" \
    --tpot-s "$TPOT_S" \
    --seed "$SEED" \
    --output-dir "$OUT" 2>&1 | tee -a "$DRIVER_LOG"

"$PY" scripts/analysis/plot_engine_sweep.py "$OUT" \
    --ttft-slo-ms "$SLO_MS" \
    --no-plot 2>&1 | tee -a "$DRIVER_LOG"

"$PY" - <<PY
import json
import pathlib
import time

manifest = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "mode": "mock_stress_sweep",
    "synthetic_n": int("$SYNTHETIC_N"),
    "synthetic_prompt_mode": "$SYNTHETIC_PROMPT_MODE",
    "synthetic_prompt_token_cap": int("$SYNTHETIC_PROMPT_TOKEN_CAP"),
    "policies": "$POLICIES".split(),
    "fractions": [float(x) for x in "$FRACTIONS".split()],
    "slo_s": float("$SLO_S"),
    "cloud_ttft_ms": float("$CLOUD_TTFT_MS"),
    "cloud_ttft_guard_multiplier": float("$CLOUD_TTFT_GUARD_MULTIPLIER"),
    "local_kv_tokens": float("$LOCAL_KV_TOKENS"),
    "time_scale": float("$TIME_SCALE"),
    "max_inflight": int("$MAX_INFLIGHT"),
    "mock_service_model": "$MOCK_SERVICE_MODEL",
    "prefill_tput": float("$PREFILL_TPUT"),
    "tpot_s": float("$TPOT_S"),
    "seed": int("$SEED"),
    "summary": str(pathlib.Path("$OUT/engine_summary.csv")),
    "normalized": str(pathlib.Path("$OUT/engine_sweep_normalized.csv")),
}
pathlib.Path("$MANIFEST").write_text(json.dumps(manifest, indent=2) + "\n")
PY

log "Completed. Manifest: $MANIFEST"
