#!/usr/bin/env bash
# Fair trigger/selector matrix for router.run.  The server lifecycle is owned
# by the caller; this script only runs clients and never starts or kills a GPU
# process.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'matrix requires a clean checkout; commit or remove all changes first\n' >&2
  git status --short >&2
  exit 3
fi

: "${DATA:?set DATA to the trace JSONL}"
: "${BASE_URL:?set BASE_URL to the local server root, e.g. http://127.0.0.1:8010}"
: "${MODEL:?set MODEL to the served model name}"
: "${OUT_DIR:?set OUT_DIR for raw results, summaries, decisions, and manifest}"
: "${PROFILE:?set PROFILE to the same-server profile_ttft_batch artifact}"
: "${SERVER_LOG:?set SERVER_LOG to the active no-cache vLLM log}"
: "${SERVER_PID:?set SERVER_PID to the recorded parent vLLM PID}"

PYBIN=${PYBIN:-python3}
CHAT_URL=${CHAT_URL:-${BASE_URL%/}/v1/chat/completions}
MAX_INFLIGHT=${MAX_INFLIGHT:-128}
SLO_S=${SLO_S:-5}
NIMBUS_TICK_MS=${NIMBUS_TICK_MS:-250}
COOLDOWN_S=${COOLDOWN_S:-20}
TEMPERATURE=${TEMPERATURE:-0}
IGNORE_EOS=${IGNORE_EOS:-1}
IN_PRICE=${IN_PRICE:-0.15}
OUT_PRICE=${OUT_PRICE:-1.20}
ARM_ORDER_SEED=${ARM_ORDER_SEED:-0}
DATA_MANIFEST="${DATA}.manifest.json"

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  printf 'recorded SERVER_PID is not alive: %s\n' "$SERVER_PID" >&2
  exit 4
fi
if [[ ! -f "$DATA_MANIFEST" || ! -f "$PROFILE" || ! -f "$SERVER_LOG" ]]; then
  printf 'DATA manifest, PROFILE, and SERVER_LOG must all exist\n' >&2
  exit 4
fi

# Bind the trace, calibration, server log prefix, engine capacity/concurrency,
# model endpoint, and current process before any arm can run.  The profile log
# may have grown since calibration, so compare its recorded prefix rather than
# an unstable whole-file hash.
PREFLIGHT="$($PYBIN - "$DATA" "$DATA_MANIFEST" "$PROFILE" "$SERVER_LOG" \
  "$SERVER_PID" "$BASE_URL" "$MODEL" "$MAX_INFLIGHT" "${KV_CAP:-}" \
  "$NIMBUS_TICK_MS" "$SLO_S" "$TEMPERATURE" "$IGNORE_EOS" <<'PY'
import hashlib
import json
import os
import re
import sys
import urllib.request

(
    data_path, manifest_path, profile_path, log_path, server_pid, url, model,
    max_inflight, requested_kv, tick_ms, slo_s, temperature, ignore_eos,
) = sys.argv[1:]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = json.load(open(manifest_path, encoding="utf-8"))
profile = json.load(open(profile_path, encoding="utf-8"))
data_sha = sha(data_path)
profile_sha = sha(profile_path)
if manifest.get("tool_sha256") != sha(os.path.join("tools", "materialize_token_aligned_trace.py")):
    raise SystemExit("trace was not materialized by the current checkout tool")
if profile.get("tool_sha256") != sha(os.path.join("tools", "profile_ttft_batch.py")):
    raise SystemExit("PROFILE was not produced by the current checkout tool")
if profile.get("schema_version") != 2:
    raise SystemExit("PROFILE is not a shared-prefill-lane schema-v2 artifact")
if profile.get("predictor_model") != "seq_slots_shared_prefill_lane_v1":
    raise SystemExit("PROFILE predictor model does not match this matrix")
if profile.get("predictor_module_sha256") != sha(os.path.join("router", "nimbus.py")):
    raise SystemExit("PROFILE was not calibrated against the current predictor")
for dependency in (
    os.path.join("router", "common.py"),
    os.path.join("tools", "materialize_token_aligned_trace.py"),
):
    if profile.get("dependency_sha256", {}).get(dependency) != sha(dependency):
        raise SystemExit(f"PROFILE dependency differs from checkout: {dependency}")
if manifest.get("tokenizer_fingerprint") != profile.get("tokenizer_fingerprint"):
    raise SystemExit("trace and PROFILE tokenizer fingerprints differ")
if manifest.get("cache_mode") != "none":
    raise SystemExit("matrix requires a cache_mode=none trace manifest")
if data_sha != manifest.get("output_sha256"):
    raise SystemExit("trace hash does not match its manifest")
with open(data_path, "rb") as f:
    line_n = sum(1 for line in f if line.strip())
if line_n != int(manifest.get("n", -1)):
    raise SystemExit(f"trace line count {line_n} != manifest n {manifest.get('n')}")
if not profile.get("valid") or profile.get("cache_mode_required") != "none":
    raise SystemExit("profile is not a valid no-cache calibration artifact")
if profile.get("model") != model:
    raise SystemExit(f"profile model {profile.get('model')!r} != MODEL {model!r}")
if profile.get("base_url", "").rstrip("/") != url.rstrip("/"):
    raise SystemExit("PROFILE base_url does not match BASE_URL")
if int(profile.get("server_pid", -1)) != int(server_pid):
    raise SystemExit("PROFILE server_pid does not match active SERVER_PID")
proc_cmdline_path = f"/proc/{server_pid}/cmdline"
if not os.path.isfile(proc_cmdline_path):
    raise SystemExit("active SERVER_PID has no Linux /proc cmdline evidence")
if sha(proc_cmdline_path) != profile.get("server_proc_cmdline_sha256_at_start"):
    raise SystemExit("active SERVER_PID cmdline differs from PROFILE")
profile_tick = float(profile.get("profile_config", {}).get("nimbus_tick_ms", -1))
if profile_tick != float(tick_ms):
    raise SystemExit(f"PROFILE tick {profile_tick} != NIMBUS_TICK_MS {tick_ms}")
sampling = profile.get("sampling", {})
if float(temperature) != float(sampling.get("temperature", -1)):
    raise SystemExit("matrix temperature does not match PROFILE sampling")
if (ignore_eos == "1") != bool(sampling.get("ignore_eos")):
    raise SystemExit("matrix IGNORE_EOS does not match PROFILE sampling")

log = open(log_path, "rb").read()
prefix_n = int(profile.get("server_log_bytes_at_start", -1))
if prefix_n <= 0 or len(log) < prefix_n:
    raise SystemExit("active server log is shorter than the profile log prefix")
if hashlib.sha256(log[:prefix_n]).hexdigest() != profile.get("server_log_sha256_at_start"):
    raise SystemExit("PROFILE was not produced from this server-log prefix")
text = log.decode("utf-8", errors="replace")
if "enable_prefix_caching=False" not in text:
    raise SystemExit("active server is not proven to have prefix caching disabled")
capacities = re.findall(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text)
if not capacities:
    raise SystemExit("GPU KV cache token capacity missing from server log")
log_kv = int(capacities[-1].replace(",", ""))
profile_kv = int(profile.get("kv_capacity_tokens", -1))
if log_kv != profile_kv:
    raise SystemExit(f"server-log KV {log_kv} != profile KV {profile_kv}")
if requested_kv and int(float(requested_kv)) != profile_kv:
    raise SystemExit(f"requested KV_CAP {requested_kv} != profile KV {profile_kv}")
max_seq_hits = re.findall(r"max_num_seqs(?:=|['\": ]+)\s*([0-9]+)", text)
if not max_seq_hits:
    raise SystemExit("max_num_seqs missing from server log")
if int(max_seq_hits[-1]) != int(max_inflight):
    raise SystemExit(
        f"server max_num_seqs {max_seq_hits[-1]} != MAX_INFLIGHT {max_inflight}"
    )
max_model_hits = re.findall(
    r"(?:max_model_len|max_seq_len)(?:=|['\": ]+)\s*([0-9]+)", text
)
manifest_context = manifest.get("max_context_tokens")
if not max_model_hits or manifest_context is None:
    raise SystemExit("max_model_len/context cap evidence is missing")
if int(max_model_hits[-1]) != int(manifest_context):
    raise SystemExit(
        f"server max_model_len/max_seq_len {max_model_hits[-1]} "
        f"!= trace context cap {manifest_context}"
    )

with urllib.request.urlopen(url.rstrip("/") + "/version", timeout=10) as response:
    version_raw = response.read()
with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10) as response:
    models_raw = response.read()
models = json.loads(models_raw)
model_ids = [row.get("id") for row in models.get("data", [])]
if model not in model_ids:
    raise SystemExit(f"MODEL {model!r} not served by active endpoint: {model_ids}")
# vLLM generates a fresh ``created`` timestamp and model-permission id for
# every /v1/models response.  Hash only the stable serving identity; hashing
# the raw body makes an otherwise identical completed matrix impossible to
# validate or resume.
model_identity = [
    {
        "id": row.get("id"),
        "owned_by": row.get("owned_by"),
        "root": row.get("root"),
        "parent": row.get("parent"),
        "max_model_len": row.get("max_model_len"),
    }
    for row in models.get("data", [])
]
model_identity.sort(key=lambda row: json.dumps(row, sort_keys=True))
models_identity_raw = json.dumps(
    model_identity, sort_keys=True, separators=(",", ":")
).encode()

cal = profile.get("predictor_calibration", {})
prefill = float(cal.get("recommended_prefill_tput_tokens_per_s", 0))
tpot = float(cal.get("recommended_tpot_ms", -1))
first_token = float(cal.get("recommended_first_token_overhead_ms", -1))
guard = float(cal.get("recommended_ttft_guard_ms", -1))
if (prefill <= 0 or tpot < 0 or first_token < 0 or guard < 0
        or not cal.get("heldout_n")):
    raise SystemExit("PROFILE lacks a usable held-out predictor calibration")
if float(cal.get("target_slo_s", -1)) != float(slo_s):
    raise SystemExit("PROFILE target SLO does not match SLO_S")
if cal.get("heldout_violation_confusion", {}).get("false_negative") != 0:
    raise SystemExit("PROFILE held-out classifier has false negatives")

fields = [
    data_sha,
    sha(manifest_path),
    str(line_n),
    str(manifest.get("scenario")),
    profile_sha,
    str(profile_kv),
    repr(prefill),
    repr(tpot),
    repr(first_token),
    repr(guard),
    profile.get("server_log_sha256_at_start"),
    hashlib.sha256(log).hexdigest(),
    hashlib.sha256(version_raw).hexdigest(),
    hashlib.sha256(models_identity_raw).hexdigest(),
]
print("\t".join(fields))
PY
)"
IFS=$'\t' read -r TRACE_SHA TRACE_MANIFEST_SHA TRACE_N TRACE_SCENARIO PROFILE_SHA KV_CAP \
  PREFILL_TPUT TPOT_MS FIRST_TOKEN_OVERHEAD_MS TTFT_GUARD_MS \
  SERVER_LOG_PREFIX_SHA SERVER_LOG_SHA \
  ENDPOINT_VERSION_SHA ENDPOINT_MODELS_IDENTITY_SHA <<< "$PREFLIGHT"
SCENARIO=${SCENARIO:-$TRACE_SCENARIO}
if [[ "$SCENARIO" != "$TRACE_SCENARIO" ]]; then
  printf 'SCENARIO %s does not match trace manifest scenario %s\n' \
    "$SCENARIO" "$TRACE_SCENARIO" >&2
  exit 4
fi
if ! "$PYBIN" - "$TTFT_GUARD_MS" "$SLO_S" <<'PY'
import sys
guard_ms, slo_s = map(float, sys.argv[1:])
raise SystemExit(0 if 0 <= guard_ms < slo_s * 1000 else 1)
PY
then
  printf 'profile-recommended guard %s ms is incompatible with SLO %s s\n' \
    "$TTFT_GUARD_MS" "$SLO_S" >&2
  exit 4
fi

mkdir -p "$OUT_DIR"
if (( $# == 0 )); then
  default_arms=( \
    kv_gap:cost_disp_current:0 \
    ttft_pred:newest:0 \
    ttft_pred:waiting_random:0 \
    ttft_pred:waiting_random:1 \
    ttft_pred:waiting_random:2 \
    ttft_pred:max_cachedisp_old:0 \
    ttft_pred:cost_cachedisp_old:0 \
    ttft_pred:cost_disp_current:0 \
  )
  ordered_arms=()
  while IFS= read -r arm; do
    ordered_arms+=("$arm")
  done < <("$PYBIN" - "$ARM_ORDER_SEED" "${default_arms[@]}" <<'PY'
import random
import sys
seed = int(sys.argv[1])
arms = sys.argv[2:]
random.Random(seed).shuffle(arms)
print("\n".join(arms))
PY
  )
  set -- "${ordered_arms[@]}"
  ARM_ORDER_MODE=seeded_shuffle
else
  ARM_ORDER_MODE=explicit
fi

# Validate every arm before creating a manifest or launching router.run.  The
# only non-Nimbus arm is the exact audited anchor spelling below; near-misses
# are rejected rather than leaking "anchor" into argparse as a trigger name.
for arm in "$@"; do
  "$PYBIN" -m tools.ttft_matrix_evidence describe-arm "$arm" >/dev/null
done

COMMIT=$(git rev-parse HEAD)
RUN_FINGERPRINT="$($PYBIN - "$TRACE_SHA" "$TRACE_MANIFEST_SHA" "$PROFILE_SHA" \
  "$SERVER_LOG_PREFIX_SHA" "$SERVER_PID" "$ENDPOINT_VERSION_SHA" \
  "$ENDPOINT_MODELS_IDENTITY_SHA" "$BASE_URL" "$CHAT_URL" "$COMMIT" "$SCENARIO" \
  "$MAX_INFLIGHT" "$KV_CAP" \
  "$PREFILL_TPUT" "$TPOT_MS" "$FIRST_TOKEN_OVERHEAD_MS" \
  "$SLO_S" "$TTFT_GUARD_MS" \
  "$NIMBUS_TICK_MS" "$TEMPERATURE" "$IGNORE_EOS" "$IN_PRICE" \
  "$OUT_PRICE" "$ARM_ORDER_MODE" "$ARM_ORDER_SEED" "$@" <<'PY'
import hashlib
import json
import sys
payload = json.dumps(sys.argv[1:], ensure_ascii=False, separators=(",", ":"))
print(hashlib.sha256(payload.encode()).hexdigest())
PY
)"

MANIFEST="$OUT_DIR/matrix_manifest.txt"
EVENTS="$OUT_DIR/matrix_events.log"
if [[ -e "$MANIFEST" ]]; then
  if ! grep -qx "run_fingerprint=$RUN_FINGERPRINT" "$MANIFEST"; then
    printf 'OUT_DIR contains a matrix manifest for different inputs/config\n' >&2
    exit 3
  fi
else
  manifest_tmp="$MANIFEST.tmp.$$"
  {
    printf 'run_fingerprint=%s\n' "$RUN_FINGERPRINT"
    printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'commit=%s\n' "$COMMIT"
    printf 'trace_sha256=%s trace_manifest_sha256=%s trace_n=%s\n' \
      "$TRACE_SHA" "$TRACE_MANIFEST_SHA" "$TRACE_N"
    printf 'profile_sha256=%s\n' "$PROFILE_SHA"
    printf 'cache_mode=none server_pid=%s server_log=%s server_log_prefix_sha256=%s\n' \
      "$SERVER_PID" "$SERVER_LOG" "$SERVER_LOG_PREFIX_SHA"
    printf 'server_log_sha256_at_manifest=%s\n' "$SERVER_LOG_SHA"
    printf 'endpoint_version_sha256=%s endpoint_models_identity_sha256=%s\n' \
      "$ENDPOINT_VERSION_SHA" "$ENDPOINT_MODELS_IDENTITY_SHA"
    printf 'base_url=%s chat_url=%s model=%s\n' \
      "$BASE_URL" "$CHAT_URL" "$MODEL"
    printf 'python=%s\n' "$($PYBIN --version 2>&1)"
    printf 'scenario=%s max_inflight=%s kv_cap=%s\n' \
      "$SCENARIO" "$MAX_INFLIGHT" "$KV_CAP"
    printf 'prefill_tput=%s tpot_ms=%s first_token_overhead_ms=%s slo_s=%s guard_ms=%s tick_ms=%s\n' \
      "$PREFILL_TPUT" "$TPOT_MS" "$FIRST_TOKEN_OVERHEAD_MS" \
      "$SLO_S" "$TTFT_GUARD_MS" "$NIMBUS_TICK_MS"
    printf 'temperature=%s ignore_eos=%s in_price=%s out_price=%s\n' \
      "$TEMPERATURE" "$IGNORE_EOS" "$IN_PRICE" "$OUT_PRICE"
    printf 'arm_order_mode=%s arm_order_seed=%s\n' \
      "$ARM_ORDER_MODE" "$ARM_ORDER_SEED"
    printf 'arms='
    printf '%s ' "$@"
    printf '\n'
    printf 'evidence_scope=single-pass exploratory; repeat full matrices before paper claims\n'
  } > "$manifest_tmp"
  mv "$manifest_tmp" "$MANIFEST"
fi

for arm in "$@"; do
  read -r _arm_kind _policy trigger selector seed < <(
    "$PYBIN" -m tools.ttft_matrix_evidence describe-arm "$arm"
  )

  stem="${SCENARIO}_${trigger}_${selector}_seed${seed}"
  raw="$OUT_DIR/${stem}.jsonl"
  summary="$OUT_DIR/${stem}.summary.json"
  decisions="$OUT_DIR/${stem}.decisions.jsonl"
  marker="$OUT_DIR/${stem}.complete.json"
  evidence_cmd=(
    "$PYBIN" -m tools.ttft_matrix_evidence validate
    --raw "$raw" --summary "$summary" --decisions "$decisions"
    --marker "$marker" --fingerprint "$RUN_FINGERPRINT" --arm "$arm"
    --trace-n "$TRACE_N"
    --prefill-tput "$PREFILL_TPUT" --tpot-ms "$TPOT_MS"
    --first-token-overhead-ms "$FIRST_TOKEN_OVERHEAD_MS"
    --ttft-guard-ms "$TTFT_GUARD_MS"
    --in-price "$IN_PRICE" --out-price "$OUT_PRICE"
    --slo-s "$SLO_S" --nimbus-tick-ms "$NIMBUS_TICK_MS"
    --max-inflight "$MAX_INFLIGHT" --temperature "$TEMPERATURE"
    --ignore-eos "$IGNORE_EOS" --kv-capacity-tokens "$KV_CAP"
    --model "$MODEL" --chat-url "$CHAT_URL" --scenario "$SCENARIO"
  )
  if [[ -e "$marker" ]]; then
    "${evidence_cmd[@]}"
    printf 'skip completed arm: %s\n' "$arm"
    continue
  fi
  if [[ -e "$raw" || -e "$summary" || -e "$decisions" ]]; then
    printf 'refusing to overwrite incomplete arm artifacts for %s\n' "$arm" >&2
    exit 3
  fi

  policy_args=()
  while IFS= read -r arg; do
    policy_args+=("$arg")
  done < <("$PYBIN" -m tools.ttft_matrix_evidence policy-cli "$arm")
  cmd=(
    "$PYBIN" -m router.run
    --data "$DATA" --scenario "$SCENARIO" "${policy_args[@]}"
    --local-url "$CHAT_URL" --local-model "$MODEL" --max-inflight "$MAX_INFLIGHT"
    --seed "$seed"
    --kv-capacity-tokens "$KV_CAP"
    --prefill-tput "$PREFILL_TPUT" --tpot-ms "$TPOT_MS"
    --first-token-overhead-ms "$FIRST_TOKEN_OVERHEAD_MS"
    --slo-s "$SLO_S" --ttft-guard-ms "$TTFT_GUARD_MS"
    --nimbus-tick-ms "$NIMBUS_TICK_MS"
    --in-price "$IN_PRICE" --out-price "$OUT_PRICE"
    --temperature "$TEMPERATURE"
    --out-dir "$OUT_DIR" --output "${stem}.jsonl"
    --decision-log "${stem}.decisions.jsonl"
  )
  if [[ "$IGNORE_EOS" == 1 ]]; then
    cmd+=(--ignore-eos)
  fi
  {
    printf 'arm_started_at=%s arm=%s command=' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$arm"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >> "$EVENTS"
  "${cmd[@]}"
  "${evidence_cmd[@]}" --write-marker
  printf 'arm_finished_at=%s arm=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$arm" \
    >> "$EVENTS"
  sleep "$COOLDOWN_S"
done

printf 'matrix_finished_at=%s run_fingerprint=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_FINGERPRINT" >> "$EVENTS"
