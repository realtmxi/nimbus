#!/usr/bin/env bash
# Fair trigger/selector matrix for router.run.  The server lifecycle is owned
# by the caller; this script only runs clients and never starts or kills a GPU
# process.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${DATA:?set DATA to the trace JSONL}"
: "${URL:?set URL to the local OpenAI-compatible endpoint}"
: "${MODEL:?set MODEL to the served model name}"
: "${KV_CAP:?set KV_CAP from the server startup log for the kv_gap arm}"
: "${OUT_DIR:?set OUT_DIR for raw results, summaries, decisions, and manifest}"

PYBIN=${PYBIN:-python3}
SCENARIO=${SCENARIO:-extreme_burst_1200}
MAX_INFLIGHT=${MAX_INFLIGHT:-128}
PREFILL_TPUT=${PREFILL_TPUT:-2000}
TPOT_MS=${TPOT_MS:-103}
SLO_S=${SLO_S:-5}
TTFT_GUARD_MS=${TTFT_GUARD_MS:-300}
NIMBUS_TICK_MS=${NIMBUS_TICK_MS:-250}
COOLDOWN_S=${COOLDOWN_S:-20}

mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/matrix_manifest.txt"
{
  printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'trace_sha256=%s\n' "$(shasum -a 256 "$DATA" | awk '{print $1}')"
  printf 'python=%s\n' "$($PYBIN --version 2>&1)"
  printf 'scenario=%s max_inflight=%s kv_cap=%s\n' "$SCENARIO" "$MAX_INFLIGHT" "$KV_CAP"
  printf 'prefill_tput=%s tpot_ms=%s slo_s=%s guard_ms=%s tick_ms=%s\n' \
    "$PREFILL_TPUT" "$TPOT_MS" "$SLO_S" "$TTFT_GUARD_MS" "$NIMBUS_TICK_MS"
} >> "$MANIFEST"

if (( $# == 0 )); then
  set -- \
    kv_gap:cost_disp_current:0 \
    ttft_pred:newest:0 \
    ttft_pred:waiting_random:0 \
    ttft_pred:waiting_random:1 \
    ttft_pred:waiting_random:2 \
    ttft_pred:max_cachedisp_old:0 \
    ttft_pred:cost_cachedisp_old:0 \
    ttft_pred:cost_disp_current:0
fi

for arm in "$@"; do
  IFS=: read -r trigger selector seed <<< "$arm"
  if [[ -z "$trigger" || -z "$selector" || -z "$seed" ]]; then
    printf 'invalid arm %q; expected trigger:selector:seed\n' "$arm" >&2
    exit 2
  fi

  stem="${SCENARIO}_${trigger}_${selector}_seed${seed}"
  raw="$OUT_DIR/${stem}.jsonl"
  summary="$OUT_DIR/${stem}.summary.json"
  decisions="$OUT_DIR/${stem}.decisions.jsonl"
  if [[ -e "$summary" ]]; then
    printf 'skip completed arm: %s\n' "$arm"
    continue
  fi
  if [[ -e "$raw" || -e "$decisions" ]]; then
    printf 'refusing to overwrite incomplete arm artifacts for %s\n' "$arm" >&2
    exit 3
  fi

  cmd=(
    "$PYBIN" -m router.run
    --data "$DATA" --scenario "$SCENARIO" --policy nimbus
    --local-url "$URL" --local-model "$MODEL" --max-inflight "$MAX_INFLIGHT"
    --nimbus-trigger "$trigger" --nimbus-selector "$selector" --seed "$seed"
    --kv-capacity-tokens "$KV_CAP"
    --prefill-tput "$PREFILL_TPUT" --tpot-ms "$TPOT_MS"
    --slo-s "$SLO_S" --ttft-guard-ms "$TTFT_GUARD_MS"
    --nimbus-tick-ms "$NIMBUS_TICK_MS"
    --out-dir "$OUT_DIR" --output "${stem}.jsonl"
    --decision-log "${stem}.decisions.jsonl"
  )
  {
    printf 'arm_started_at=%s arm=%s command=' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$arm"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >> "$MANIFEST"
  "${cmd[@]}"
  printf 'arm_finished_at=%s arm=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$arm" \
    >> "$MANIFEST"
  sleep "$COOLDOWN_S"
done

printf 'matrix_finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"
