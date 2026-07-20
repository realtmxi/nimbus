#!/usr/bin/env bash
# Fair trigger/selector matrix for router.run.  The server lifecycle is owned
# by the caller; this script only runs clients and never starts or kills a GPU
# process.
set -euo pipefail
set +x

# Capture the frozen OpenRouter secret before the first external command, then
# remove it from the matrix environment.  The value remains a non-exported
# shell variable and is scoped only to the usage capture and router.run below;
# git, evidence, budget, launch-verification, and other child processes never
# inherit it.
MATRIX_OPENROUTER_API_KEY_WAS_SET=${OPENROUTER_API_KEY+x}
MATRIX_OPENROUTER_API_KEY=${OPENROUTER_API_KEY-}
unset OPENROUTER_API_KEY
MATRIX_CLOUD_API_KEY_SECRET=
trap 'MATRIX_CLOUD_API_KEY_SECRET=; MATRIX_OPENROUTER_API_KEY=' EXIT

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
TIMEOUT_S=${TIMEOUT_S:-600}
NIMBUS_TICK_MS=${NIMBUS_TICK_MS:-250}
COOLDOWN_S=${COOLDOWN_S:-20}
TEMPERATURE=${TEMPERATURE:-0}
IGNORE_EOS=${IGNORE_EOS:-1}
IN_PRICE=${IN_PRICE:-0.15}
OUT_PRICE=${OUT_PRICE:-1.20}
ARM_ORDER_SEED=${ARM_ORDER_SEED:-}
DATA_MANIFEST="${DATA}.manifest.json"

# The same runner supports the historical NullCloud matrix and an observed
# OpenRouter TTFT-cancel matrix.  CLOUD_API_KEY_ENV is the *name* of the secret
# environment variable.  Its value is checked for presence below, but is never
# passed to fingerprint/manifest/evidence commands or printed.
CLOUD=${CLOUD:-null}
if [[ -z "${CLOUD_MAX_CONCURRENCY+x}" ]]; then
  [[ "$CLOUD" == real ]] && CLOUD_MAX_CONCURRENCY=16 \
    || CLOUD_MAX_CONCURRENCY=32
fi
if [[ -z "$ARM_ORDER_SEED" ]]; then
  [[ "$CLOUD" == real ]] && ARM_ORDER_SEED=20260716 || ARM_ORDER_SEED=0
fi
if [[ -z "${CLOUD_NO_FALLBACKS+x}" ]]; then
  [[ "$CLOUD" == real ]] && CLOUD_NO_FALLBACKS=1 || CLOUD_NO_FALLBACKS=0
fi
if [[ -z "${CLOUD_STOP_AFTER_FIRST_TOKEN+x}" ]]; then
  [[ "$CLOUD" == real ]] && CLOUD_STOP_AFTER_FIRST_TOKEN=1 \
    || CLOUD_STOP_AFTER_FIRST_TOKEN=0
fi
LOCAL_IGNORE_EOS=${LOCAL_IGNORE_EOS:-$IGNORE_EOS}
if [[ -z "${CLOUD_IGNORE_EOS+x}" ]]; then
  if [[ "$CLOUD" == real ]]; then
    CLOUD_IGNORE_EOS=0
  else
    CLOUD_IGNORE_EOS=$IGNORE_EOS
  fi
fi
CLOUD_URL=${CLOUD_URL:-}
CLOUD_MODEL=${CLOUD_MODEL:-}
CLOUD_API_KEY_ENV=${CLOUD_API_KEY_ENV:-}
CLOUD_PROVIDER=${CLOUD_PROVIDER:-}
REAL_CLOUD_EXPECT_N_WAS_SET=${REAL_CLOUD_EXPECT_N+x}
REAL_CLOUD_EXPECT_N=${REAL_CLOUD_EXPECT_N:-11605}

# E11 remains the default real-cloud contract.  E12 is deliberately opt-in and
# stage-scoped: setting any E12 contract variable requires the complete
# contract, and one invocation can run only its exact A or C arm.  The frozen
# values are environment-visible so an operator can state them explicitly,
# but cannot silently change them and still call the run E12.
E12_LIVE_CONTRACT_ID_WAS_SET=${E12_LIVE_CONTRACT_ID+x}
E12_LIVE_STAGE_WAS_SET=${E12_LIVE_STAGE+x}
E12_LIVE_EXPECT_N_WAS_SET=${E12_LIVE_EXPECT_N+x}
E12_LIVE_ARM_ORDER_SEED_WAS_SET=${E12_LIVE_ARM_ORDER_SEED+x}
E12_LIVE_AUTHORIZED_BUDGET_USD_WAS_SET=${E12_LIVE_AUTHORIZED_BUDGET_USD+x}
E12_LIVE_BUDGET_ATTESTATION_SHA256_WAS_SET=${E12_LIVE_BUDGET_ATTESTATION_SHA256+x}
E12_LIVE_KEY_LIMIT_MAX_USD_WAS_SET=${E12_LIVE_KEY_LIMIT_MAX_USD+x}
E12_LIVE_KEY_CONTRACT_MODE_WAS_SET=${E12_LIVE_KEY_CONTRACT_MODE+x}
E12_LIVE_STAGE_LAUNCH_ATTESTATION_WAS_SET=${E12_LIVE_STAGE_LAUNCH_ATTESTATION+x}
E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256_WAS_SET=${E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256+x}
E12_LIVE_PRICE_SNAPSHOT_WAS_SET=${E12_LIVE_PRICE_SNAPSHOT+x}
E12_LIVE_BUDGET_ATTESTATION_WAS_SET=${E12_LIVE_BUDGET_ATTESTATION+x}
E12_LIVE_CANARY_ATTESTATION_WAS_SET=${E12_LIVE_CANARY_ATTESTATION+x}
E12_LIVE_BASELINE_USAGE_WAS_SET=${E12_LIVE_BASELINE_USAGE+x}
E12_LIVE_SETTLEMENT_PREVIOUS_USAGE_WAS_SET=${E12_LIVE_SETTLEMENT_PREVIOUS_USAGE+x}
E12_LIVE_SETTLEMENT_CURRENT_USAGE_WAS_SET=${E12_LIVE_SETTLEMENT_CURRENT_USAGE+x}
E12_LIVE_STAGE_BUDGET_GATE_WAS_SET=${E12_LIVE_STAGE_BUDGET_GATE+x}
E12_LIVE_A_DIR_WAS_SET=${E12_LIVE_A_DIR+x}
E12_LIVE_A_STAGE_LAUNCH_ATTESTATION_WAS_SET=${E12_LIVE_A_STAGE_LAUNCH_ATTESTATION+x}
E12_LIVE_CONTRACT_ID=${E12_LIVE_CONTRACT_ID:-}
E12_LIVE_STAGE=${E12_LIVE_STAGE:-}
E12_LIVE_EXPECT_N=${E12_LIVE_EXPECT_N:-11604}
E12_LIVE_ARM_ORDER_SEED=${E12_LIVE_ARM_ORDER_SEED:-20260716}
E12_LIVE_AUTHORIZED_BUDGET_USD=${E12_LIVE_AUTHORIZED_BUDGET_USD:-}
E12_LIVE_BUDGET_ATTESTATION_SHA256=${E12_LIVE_BUDGET_ATTESTATION_SHA256:-}
E12_LIVE_KEY_LIMIT_MAX_USD=${E12_LIVE_KEY_LIMIT_MAX_USD:-}
E12_LIVE_KEY_CONTRACT_MODE=${E12_LIVE_KEY_CONTRACT_MODE:-strict_server_cap_v1}
E12_LIVE_STAGE_LAUNCH_ATTESTATION=${E12_LIVE_STAGE_LAUNCH_ATTESTATION:-}
E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256=${E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256:-}
E12_LIVE_PRICE_SNAPSHOT=${E12_LIVE_PRICE_SNAPSHOT:-}
E12_LIVE_BUDGET_ATTESTATION=${E12_LIVE_BUDGET_ATTESTATION:-}
E12_LIVE_CANARY_ATTESTATION=${E12_LIVE_CANARY_ATTESTATION:-}
E12_LIVE_BASELINE_USAGE=${E12_LIVE_BASELINE_USAGE:-}
E12_LIVE_SETTLEMENT_PREVIOUS_USAGE=${E12_LIVE_SETTLEMENT_PREVIOUS_USAGE:-}
E12_LIVE_SETTLEMENT_CURRENT_USAGE=${E12_LIVE_SETTLEMENT_CURRENT_USAGE:-}
E12_LIVE_STAGE_BUDGET_GATE=${E12_LIVE_STAGE_BUDGET_GATE:-}
E12_LIVE_A_DIR=${E12_LIVE_A_DIR:-}
E12_LIVE_A_STAGE_LAUNCH_ATTESTATION=${E12_LIVE_A_STAGE_LAUNCH_ATTESTATION:-}
E12_LIVE_CONTRACT=e12_current_turn_v1
if [[ "$E12_LIVE_KEY_CONTRACT_MODE" == e12_marketplace_deepinfra_no_byok_v1 ]]; then
  E12_LIVE_CONTRACT=e12_current_turn_marketplace_v2
fi
E12_LIVE_COOLDOWN_S=20
E12_LIVE_ACTIVE=0
if [[ -n "$E12_LIVE_CONTRACT_ID_WAS_SET" || \
      -n "$E12_LIVE_STAGE_WAS_SET" || \
      -n "$E12_LIVE_EXPECT_N_WAS_SET" || \
      -n "$E12_LIVE_ARM_ORDER_SEED_WAS_SET" || \
      -n "$E12_LIVE_AUTHORIZED_BUDGET_USD_WAS_SET" || \
      -n "$E12_LIVE_BUDGET_ATTESTATION_SHA256_WAS_SET" || \
      -n "$E12_LIVE_KEY_LIMIT_MAX_USD_WAS_SET" || \
      -n "$E12_LIVE_KEY_CONTRACT_MODE_WAS_SET" || \
      -n "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_WAS_SET" || \
      -n "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256_WAS_SET" || \
      -n "$E12_LIVE_PRICE_SNAPSHOT_WAS_SET" || \
      -n "$E12_LIVE_BUDGET_ATTESTATION_WAS_SET" || \
      -n "$E12_LIVE_CANARY_ATTESTATION_WAS_SET" || \
      -n "$E12_LIVE_BASELINE_USAGE_WAS_SET" || \
      -n "$E12_LIVE_SETTLEMENT_PREVIOUS_USAGE_WAS_SET" || \
      -n "$E12_LIVE_SETTLEMENT_CURRENT_USAGE_WAS_SET" || \
      -n "$E12_LIVE_STAGE_BUDGET_GATE_WAS_SET" || \
      -n "$E12_LIVE_A_DIR_WAS_SET" || \
      -n "$E12_LIVE_A_STAGE_LAUNCH_ATTESTATION_WAS_SET" ]]; then
  E12_LIVE_ACTIVE=1
  if [[ -z "$E12_LIVE_CONTRACT_ID_WAS_SET" || \
        -z "$E12_LIVE_STAGE_WAS_SET" || \
        -z "$E12_LIVE_AUTHORIZED_BUDGET_USD_WAS_SET" || \
        -z "$E12_LIVE_BUDGET_ATTESTATION_SHA256_WAS_SET" || \
        -z "$E12_LIVE_KEY_LIMIT_MAX_USD_WAS_SET" || \
        -z "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_WAS_SET" || \
        -z "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256_WAS_SET" || \
        -z "$E12_LIVE_PRICE_SNAPSHOT_WAS_SET" || \
        -z "$E12_LIVE_BUDGET_ATTESTATION_WAS_SET" || \
        -z "$E12_LIVE_CANARY_ATTESTATION_WAS_SET" || \
        -z "$E12_LIVE_BASELINE_USAGE_WAS_SET" || \
        -z "$E12_LIVE_SETTLEMENT_PREVIOUS_USAGE_WAS_SET" || \
        -z "$E12_LIVE_SETTLEMENT_CURRENT_USAGE_WAS_SET" || \
        -z "$E12_LIVE_STAGE_BUDGET_GATE_WAS_SET" || \
        -z "$E12_LIVE_CONTRACT_ID" || -z "$E12_LIVE_STAGE" || \
        -z "$E12_LIVE_BUDGET_ATTESTATION_SHA256" || \
        -z "$E12_LIVE_STAGE_LAUNCH_ATTESTATION" || \
        -z "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256" || \
        -z "$E12_LIVE_PRICE_SNAPSHOT" || \
        -z "$E12_LIVE_BUDGET_ATTESTATION" || \
        -z "$E12_LIVE_CANARY_ATTESTATION" || \
        -z "$E12_LIVE_BASELINE_USAGE" || \
        -z "$E12_LIVE_SETTLEMENT_PREVIOUS_USAGE" || \
        -z "$E12_LIVE_SETTLEMENT_CURRENT_USAGE" || \
        -z "$E12_LIVE_STAGE_BUDGET_GATE" ]]; then
    printf 'E12 live requires explicit contract, budget, and stage-launch attestations\n' >&2
    exit 4
  fi
  if [[ "$CLOUD" != real ]]; then
    printf 'E12 live contract is valid only with CLOUD=real\n' >&2
    exit 4
  fi
  if [[ ! "$E12_LIVE_CONTRACT_ID" =~ ^[A-Za-z0-9._-]{1,128}$ ]]; then
    printf 'E12_LIVE_CONTRACT_ID must be a safe one-line identifier\n' >&2
    exit 4
  fi
  if [[ "$E12_LIVE_EXPECT_N" != 11604 ]]; then
    printf 'E12 live freezes E12_LIVE_EXPECT_N=11604\n' >&2
    exit 4
  fi
  if [[ "$E12_LIVE_ARM_ORDER_SEED" != 20260716 ]]; then
    printf 'E12 live freezes E12_LIVE_ARM_ORDER_SEED=20260716\n' >&2
    exit 4
  fi
  if [[ "$E12_LIVE_AUTHORIZED_BUDGET_USD" != 3 ]]; then
    printf 'E12 live freezes E12_LIVE_AUTHORIZED_BUDGET_USD=3\n' >&2
    exit 4
  fi
  if [[ ! "$E12_LIVE_BUDGET_ATTESTATION_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'E12_LIVE_BUDGET_ATTESTATION_SHA256 must be exactly 64 lowercase hex characters\n' >&2
    exit 4
  fi
  case "$E12_LIVE_KEY_CONTRACT_MODE" in
    strict_server_cap_v1)
      E12_LIVE_EXPECTED_KEY_LIMIT_MAX_USD=3
      ;;
    e12_marketplace_deepinfra_no_byok_v1)
      E12_LIVE_EXPECTED_KEY_LIMIT_MAX_USD=5
      ;;
    *)
      printf 'unknown E12_LIVE_KEY_CONTRACT_MODE\n' >&2
      exit 4
      ;;
  esac
  if [[ "$E12_LIVE_KEY_LIMIT_MAX_USD" != "$E12_LIVE_EXPECTED_KEY_LIMIT_MAX_USD" ]]; then
    printf 'E12 key-limit maximum does not match the selected key contract\n' >&2
    exit 4
  fi
  if [[ ! "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256 must be exactly 64 lowercase hex characters\n' >&2
    exit 4
  fi
  case "$E12_LIVE_STAGE" in
    A)
      E12_LIVE_EXACT_ARM=ttft_pred:cost_cachedisp_old:0
      if [[ -n "$E12_LIVE_A_DIR_WAS_SET" || \
            -n "$E12_LIVE_A_STAGE_LAUNCH_ATTESTATION_WAS_SET" ]]; then
        printf 'E12 stage A must not accept prior-A evidence paths\n' >&2
        exit 4
      fi
      ;;
    C)
      E12_LIVE_EXACT_ARM=ttft_pred:cost_disp_current:0
      if [[ -z "$E12_LIVE_A_DIR_WAS_SET" || \
            -z "$E12_LIVE_A_STAGE_LAUNCH_ATTESTATION_WAS_SET" || \
            -z "$E12_LIVE_A_DIR" || \
            -z "$E12_LIVE_A_STAGE_LAUNCH_ATTESTATION" ]]; then
        printf 'E12 stage C requires completed stage-A directory and launch attestation\n' >&2
        exit 4
      fi
      ;;
    *)
      printf 'E12_LIVE_STAGE must be exactly A or C (got %s)\n' \
        "$E12_LIVE_STAGE" >&2
      exit 4
      ;;
  esac
  if [[ -n "$REAL_CLOUD_EXPECT_N_WAS_SET" && \
        "$REAL_CLOUD_EXPECT_N" != "$E12_LIVE_EXPECT_N" ]]; then
    printf 'REAL_CLOUD_EXPECT_N conflicts with frozen E12 expected n=11604\n' >&2
    exit 4
  fi
  REAL_CLOUD_EXPECT_N=$E12_LIVE_EXPECT_N
fi

if [[ "$E12_LIVE_ACTIVE" == 0 && "$CLOUD" == real && \
      "$REAL_CLOUD_EXPECT_N" == 11604 ]]; then
  printf 'REAL_CLOUD_EXPECT_N=11604 is reserved for the explicit E12 live contract\n' >&2
  exit 4
fi

for bit_name in IGNORE_EOS LOCAL_IGNORE_EOS CLOUD_IGNORE_EOS \
  CLOUD_NO_FALLBACKS CLOUD_STOP_AFTER_FIRST_TOKEN; do
  bit_value=${!bit_name}
  if [[ "$bit_value" != 0 && "$bit_value" != 1 ]]; then
    printf '%s must be 0 or 1 (got %s)\n' "$bit_name" "$bit_value" >&2
    exit 4
  fi
done
if [[ "$CLOUD" == real ]]; then
  CLOUD_URL=${CLOUD_URL:-https://openrouter.ai/api/v1/chat/completions}
  CLOUD_MODEL=${CLOUD_MODEL:-qwen/qwen3-32b}
  CLOUD_API_KEY_ENV=${CLOUD_API_KEY_ENV:-OPENROUTER_API_KEY}
  CLOUD_PROVIDER=${CLOUD_PROVIDER:-deepinfra}
  if [[ ! "$CLOUD_API_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    printf 'CLOUD_API_KEY_ENV must be an environment-variable name\n' >&2
    exit 4
  fi
  if [[ "$CLOUD_API_KEY_ENV" == OPENROUTER_API_KEY && \
        -n "$MATRIX_OPENROUTER_API_KEY_WAS_SET" ]]; then
    MATRIX_CLOUD_API_KEY_SECRET=$MATRIX_OPENROUTER_API_KEY
  else
    MATRIX_CLOUD_API_KEY_SECRET=${!CLOUD_API_KEY_ENV:-}
    unset "$CLOUD_API_KEY_ENV"
  fi
  MATRIX_OPENROUTER_API_KEY=
  unset MATRIX_OPENROUTER_API_KEY_WAS_SET
  if [[ -z "$MATRIX_CLOUD_API_KEY_SECRET" ]]; then
    printf 'real cloud requires nonempty secret env %s\n' \
      "$CLOUD_API_KEY_ENV" >&2
    exit 4
  fi
  if [[ "$CLOUD_URL" == *'?'* || "$CLOUD_URL" == *'#'* || \
        "$CLOUD_URL" == *'@'* ]]; then
    printf 'CLOUD_URL must not contain query, fragment, or userinfo secrets\n' >&2
    exit 4
  fi
  if [[ "$CLOUD_NO_FALLBACKS" != 1 || \
        "$CLOUD_STOP_AFTER_FIRST_TOKEN" != 1 ]]; then
    printf 'real-cloud matrix requires no fallbacks and first-token abort\n' >&2
    exit 4
  fi
  if [[ "$LOCAL_IGNORE_EOS" != 1 || "$CLOUD_IGNORE_EOS" != 0 ]]; then
    printf 'real-cloud matrix requires local_ignore_eos=1 and cloud_ignore_eos=0\n' >&2
    exit 4
  fi
  if [[ "$CLOUD_URL" != https://openrouter.ai/api/v1/chat/completions || \
        "$CLOUD_MODEL" != qwen/qwen3-32b || \
        "$CLOUD_PROVIDER" != deepinfra || \
        "$CLOUD_MAX_CONCURRENCY" != 16 || \
        "$ARM_ORDER_SEED" != 20260716 ]]; then
    printf 'real-cloud freezes OpenRouter/qwen3-32b/deepinfra, concurrency=16, arm-order-seed=20260716\n' >&2
    exit 4
  fi
  if [[ "$E12_LIVE_ACTIVE" == 1 ]] && \
      { [[ "$IN_PRICE" != 0.08 ]] || [[ "$OUT_PRICE" != 0.28 ]] || \
        [[ "$SLO_S" != 5 ]] || [[ "$TIMEOUT_S" != 600 ]] || \
        [[ "$NIMBUS_TICK_MS" != 250 ]] || [[ "$COOLDOWN_S" != 20 ]] || \
        [[ "$MAX_INFLIGHT" != 128 ]] || [[ "$TEMPERATURE" != 0 ]] || \
        [[ "$IGNORE_EOS" != 1 ]] || \
        [[ "$CLOUD_API_KEY_ENV" != OPENROUTER_API_KEY ]]; }; then
    printf 'E12 live freezes prices=0.08/0.28, SLO=5, timeout=600, tick=250, cooldown=20, max-inflight=128, temperature=0, ignore-eos=1, and secret env name OPENROUTER_API_KEY\n' >&2
    exit 4
  fi
elif [[ "$CLOUD" != null ]]; then
  printf 'CLOUD must be null or real (got %s)\n' "$CLOUD" >&2
  exit 4
else
  MATRIX_OPENROUTER_API_KEY=
  unset MATRIX_OPENROUTER_API_KEY_WAS_SET
fi

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
  "$NIMBUS_TICK_MS" "$SLO_S" "$TEMPERATURE" "$LOCAL_IGNORE_EOS" <<'PY'
import hashlib
import json
import os
import re
import sys
import urllib.request

from tools.ttft_matrix_evidence import (
    MatrixEvidenceError,
    validate_trace_materializer_manifest,
)

(
    data_path, manifest_path, profile_path, log_path, server_pid, url, model,
    max_inflight, requested_kv, tick_ms, slo_s, temperature, local_ignore_eos,
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
try:
    validate_trace_materializer_manifest(manifest, os.curdir)
except MatrixEvidenceError as exc:
    raise SystemExit(str(exc)) from None
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
if (local_ignore_eos == "1") != bool(sampling.get("ignore_eos")):
    raise SystemExit("matrix LOCAL_IGNORE_EOS does not match PROFILE sampling")

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
    str(manifest.get("payload_mode")),
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
IFS=$'\t' read -r TRACE_SHA TRACE_MANIFEST_SHA TRACE_N TRACE_SCENARIO TRACE_PAYLOAD_MODE PROFILE_SHA KV_CAP \
  PREFILL_TPUT TPOT_MS FIRST_TOKEN_OVERHEAD_MS TTFT_GUARD_MS \
  SERVER_LOG_PREFIX_SHA SERVER_LOG_SHA \
  ENDPOINT_VERSION_SHA ENDPOINT_MODELS_IDENTITY_SHA <<< "$PREFLIGHT"
if [[ "$CLOUD" == real && "$TRACE_N" != "$REAL_CLOUD_EXPECT_N" ]]; then
  printf 'real-cloud trace_n=%s, expected %s (set REAL_CLOUD_EXPECT_N intentionally to override)\n' \
    "$TRACE_N" "$REAL_CLOUD_EXPECT_N" >&2
  exit 4
fi
if [[ "$E12_LIVE_ACTIVE" == 0 ]] && \
    { [[ "$TRACE_SHA" == e838016a8e55660c565dadb1ad019770f6b88f878d8ca29f165c30887d2cb410 ]] || \
      [[ "$TRACE_PAYLOAD_MODE" == sharegpt_current_turn_retokenized ]]; }; then
  printf 'the E12 current-turn trace/payload requires the explicit E12 live contract\n' >&2
  exit 4
fi
SCENARIO=${SCENARIO:-$TRACE_SCENARIO}
if [[ "$SCENARIO" != "$TRACE_SCENARIO" ]]; then
  printf 'SCENARIO %s does not match trace manifest scenario %s\n' \
    "$SCENARIO" "$TRACE_SCENARIO" >&2
  exit 4
fi
if ! "$PYBIN" - "$TTFT_GUARD_MS" "$SLO_S" "$TIMEOUT_S" <<'PY'
import sys
guard_ms, slo_s, timeout_s = map(float, sys.argv[1:])
raise SystemExit(
    0 if 0 <= guard_ms < slo_s * 1000 and timeout_s > 0 else 1
)
PY
then
  printf 'guard=%s ms, SLO=%s s, or timeout=%s s is invalid\n' \
    "$TTFT_GUARD_MS" "$SLO_S" "$TIMEOUT_S" >&2
  exit 4
fi

mkdir -p "$OUT_DIR"
if [[ "$E12_LIVE_ACTIVE" == 1 ]]; then
  if (( $# == 0 )); then
    set -- "$E12_LIVE_EXACT_ARM"
  elif (( $# != 1 )) || [[ "$1" != "$E12_LIVE_EXACT_ARM" ]]; then
    printf 'E12 live stage %s requires the single exact arm %s\n' \
      "$E12_LIVE_STAGE" "$E12_LIVE_EXACT_ARM" >&2
    exit 4
  fi
  ARM_ORDER_MODE=e12_contract_stage
elif (( $# == 0 )); then
  if [[ "$CLOUD" == real ]]; then
    # Full real-cloud default is the frozen A/C comparison only.  Accidentally
    # running every exploratory selector would spend credits and confound the
    # paired comparison with a much longer wall-clock window.
    default_arms=( \
      ttft_pred:cost_cachedisp_old:0 \
      ttft_pred:cost_disp_current:0 \
    )
  else
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
  fi
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
if [[ "$E12_LIVE_ACTIVE" == 1 ]]; then
  if (( $# != 1 )) || [[ "$1" != "$E12_LIVE_EXACT_ARM" ]]; then
    printf 'E12 live stage %s requires the single exact arm %s\n' \
      "$E12_LIVE_STAGE" "$E12_LIVE_EXACT_ARM" >&2
    exit 4
  fi
elif [[ "$CLOUD" == real ]]; then
  if (( $# != 2 )) \
      || [[ "$1" != ttft_pred:cost_cachedisp_old:0 ]] \
      || [[ "$2" != ttft_pred:cost_disp_current:0 ]]; then
    printf 'real-cloud E11 requires exact A then C arms\n' >&2
    exit 4
  fi
fi

COMMIT=$(git rev-parse HEAD)
if [[ "$E12_LIVE_ACTIVE" == 1 ]]; then
  if [[ ! -f "$E12_LIVE_STAGE_LAUNCH_ATTESTATION" ]]; then
    printf 'E12 stage-launch attestation is not a regular file\n' >&2
    exit 4
  fi
  E12_LIVE_CURRENT_USAGE_ARTIFACT="$OUT_DIR/e12_live_current_usage.json"
  E12_LIVE_VERIFY_RECEIPT_ARTIFACT="$OUT_DIR/e12_stage_launch_verify_receipt.json"
  if [[ -e "$E12_LIVE_CURRENT_USAGE_ARTIFACT" || \
        -e "$E12_LIVE_VERIFY_RECEIPT_ARTIFACT" ]]; then
    printf 'E12 launch usage/receipt artifact already exists; refusing overwrite\n' >&2
    exit 4
  fi
  (
    umask 077
    export "$CLOUD_API_KEY_ENV=$MATRIX_CLOUD_API_KEY_SECRET"
    "$PYBIN" -m tools.openrouter_usage_snapshot \
      --api-key-env "$CLOUD_API_KEY_ENV" \
      --key-contract-mode "$E12_LIVE_KEY_CONTRACT_MODE" \
      --output "$E12_LIVE_CURRENT_USAGE_ARTIFACT" \
      --no-overwrite
  )
  chmod 600 "$E12_LIVE_CURRENT_USAGE_ARTIFACT"
  E12_LIVE_VERIFY_A_ARGS=()
  if [[ "$E12_LIVE_STAGE" == C ]]; then
    E12_LIVE_VERIFY_A_ARGS=(
      --a-dir "$E12_LIVE_A_DIR"
      --a-launch-attestation "$E12_LIVE_A_STAGE_LAUNCH_ATTESTATION"
    )
  fi
  "$PYBIN" -m tools.check_e12_stage_launch verify \
    --attestation "$E12_LIVE_STAGE_LAUNCH_ATTESTATION" \
    --expected-sha256 "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256" \
    --stage "$E12_LIVE_STAGE" \
    --key-contract-mode "$E12_LIVE_KEY_CONTRACT_MODE" \
    --contract-id "$E12_LIVE_CONTRACT_ID" \
    --trace-manifest-sha256 "$TRACE_MANIFEST_SHA" \
    --profile-sha256 "$PROFILE_SHA" \
    --budget-attestation-sha256 "$E12_LIVE_BUDGET_ATTESTATION_SHA256" \
    --trace-manifest "$DATA_MANIFEST" \
    --profile "$PROFILE" \
    --price-snapshot "$E12_LIVE_PRICE_SNAPSHOT" \
    --budget-attestation "$E12_LIVE_BUDGET_ATTESTATION" \
    --canary "$E12_LIVE_CANARY_ATTESTATION" \
    --baseline-usage "$E12_LIVE_BASELINE_USAGE" \
    --settlement-previous "$E12_LIVE_SETTLEMENT_PREVIOUS_USAGE" \
    --settlement-current "$E12_LIVE_SETTLEMENT_CURRENT_USAGE" \
    --stage-budget-gate "$E12_LIVE_STAGE_BUDGET_GATE" \
    --commit "$COMMIT" \
    --server-pid "$SERVER_PID" \
    --server-log-prefix-sha256 "$SERVER_LOG_PREFIX_SHA" \
    --endpoint-version-sha256 "$ENDPOINT_VERSION_SHA" \
    --endpoint-models-identity-sha256 "$ENDPOINT_MODELS_IDENTITY_SHA" \
    --base-url "$BASE_URL" \
    --chat-url "$CHAT_URL" \
    --model "$MODEL" \
    "${E12_LIVE_VERIFY_A_ARGS[@]}" \
    --live-current-usage "$E12_LIVE_CURRENT_USAGE_ARTIFACT" \
    --output "$E12_LIVE_VERIFY_RECEIPT_ARTIFACT"
  chmod 600 "$E12_LIVE_VERIFY_RECEIPT_ARTIFACT"
  E12_LIVE_ARTIFACT_HASHES="$($PYBIN - \
    "$E12_LIVE_CURRENT_USAGE_ARTIFACT" \
    "$E12_LIVE_VERIFY_RECEIPT_ARTIFACT" <<'PY'
import hashlib
import pathlib
import sys

print(" ".join(
    hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    for path in sys.argv[1:]
))
PY
)"
  read -r E12_LIVE_CURRENT_USAGE_SHA256 \
    E12_LIVE_VERIFY_RECEIPT_SHA256 <<< "$E12_LIVE_ARTIFACT_HASHES"
fi
LIVE_FINGERPRINT_ARGS=()
E12_LIVE_MANIFEST_LINE=
if [[ "$E12_LIVE_ACTIVE" == 1 ]]; then
  LIVE_FINGERPRINT_ARGS=(
    "$E12_LIVE_CONTRACT_ID"
    "$E12_LIVE_CONTRACT"
    "$E12_LIVE_STAGE"
    "$E12_LIVE_EXPECT_N"
    "$E12_LIVE_ARM_ORDER_SEED"
    "$E12_LIVE_EXACT_ARM"
    "$E12_LIVE_AUTHORIZED_BUDGET_USD"
    "$E12_LIVE_BUDGET_ATTESTATION_SHA256"
    "$E12_LIVE_KEY_LIMIT_MAX_USD"
    "$E12_LIVE_KEY_CONTRACT_MODE"
    "$E12_LIVE_COOLDOWN_S"
    "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256"
    "$E12_LIVE_CURRENT_USAGE_SHA256"
    "$E12_LIVE_VERIFY_RECEIPT_SHA256"
  )
  printf -v E12_LIVE_MANIFEST_LINE \
    'live_contract_id=%s live_contract=%s live_stage=%s live_expected_trace_n=%s live_arm_order_seed=%s live_exact_arm=%s live_authorized_budget_usd=%s live_budget_attestation_sha256=%s live_key_limit_max_usd=%s live_key_contract_mode=%s live_cooldown_s=%s live_stage_launch_attestation_sha256=%s live_current_usage_sha256=%s live_stage_launch_verify_receipt_sha256=%s' \
    "${LIVE_FINGERPRINT_ARGS[@]}"
fi
RUN_FINGERPRINT="$($PYBIN - "$TRACE_SHA" "$TRACE_MANIFEST_SHA" "$PROFILE_SHA" \
  "$SERVER_LOG_PREFIX_SHA" "$SERVER_PID" "$ENDPOINT_VERSION_SHA" \
  "$ENDPOINT_MODELS_IDENTITY_SHA" "$BASE_URL" "$CHAT_URL" "$COMMIT" "$SCENARIO" \
  "$MAX_INFLIGHT" "$KV_CAP" \
  "$PREFILL_TPUT" "$TPOT_MS" "$FIRST_TOKEN_OVERHEAD_MS" \
  "$SLO_S" "$TIMEOUT_S" "$TTFT_GUARD_MS" \
  "$NIMBUS_TICK_MS" "$TEMPERATURE" "$IGNORE_EOS" "$IN_PRICE" \
  "$OUT_PRICE" "$CLOUD" "$CLOUD_URL" "$CLOUD_MODEL" \
  "$CLOUD_API_KEY_ENV" "$CLOUD_MAX_CONCURRENCY" "$CLOUD_PROVIDER" \
  "$CLOUD_NO_FALLBACKS" "$CLOUD_STOP_AFTER_FIRST_TOKEN" \
  "$LOCAL_IGNORE_EOS" "$CLOUD_IGNORE_EOS" "$REAL_CLOUD_EXPECT_N" \
  "$ARM_ORDER_MODE" "$ARM_ORDER_SEED" "$@" \
  "${LIVE_FINGERPRINT_ARGS[@]}" <<'PY'
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
  if [[ "$E12_LIVE_ACTIVE" == 1 ]] && \
      ! grep -Fqx "$E12_LIVE_MANIFEST_LINE" "$MANIFEST"; then
    printf 'OUT_DIR matrix manifest lacks the exact E12 live contract\n' >&2
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
    printf 'prefill_tput=%s tpot_ms=%s first_token_overhead_ms=%s slo_s=%s timeout_s=%s guard_ms=%s tick_ms=%s\n' \
      "$PREFILL_TPUT" "$TPOT_MS" "$FIRST_TOKEN_OVERHEAD_MS" \
      "$SLO_S" "$TIMEOUT_S" "$TTFT_GUARD_MS" "$NIMBUS_TICK_MS"
    printf 'temperature=%s ignore_eos=%s in_price=%s out_price=%s\n' \
      "$TEMPERATURE" "$IGNORE_EOS" "$IN_PRICE" "$OUT_PRICE"
    printf 'local_ignore_eos=%s cloud_ignore_eos=%s\n' \
      "$LOCAL_IGNORE_EOS" "$CLOUD_IGNORE_EOS"
    printf 'cloud=%s cloud_url=%s cloud_model=%s cloud_api_key_env=%s\n' \
      "$CLOUD" "$CLOUD_URL" "$CLOUD_MODEL" "$CLOUD_API_KEY_ENV"
    printf 'cloud_max_concurrency=%s cloud_provider=%s cloud_no_fallbacks=%s cloud_stop_after_first_token=%s\n' \
      "$CLOUD_MAX_CONCURRENCY" "$CLOUD_PROVIDER" \
      "$CLOUD_NO_FALLBACKS" "$CLOUD_STOP_AFTER_FIRST_TOKEN"
    printf 'real_cloud_expected_trace_n=%s secret_value_recorded=false\n' \
      "$REAL_CLOUD_EXPECT_N"
    if [[ "$E12_LIVE_ACTIVE" == 1 ]]; then
      printf '%s\n' "$E12_LIVE_MANIFEST_LINE"
    fi
    printf 'arm_order_mode=%s arm_order_seed=%s\n' \
      "$ARM_ORDER_MODE" "$ARM_ORDER_SEED"
    printf 'arms='
    printf '%s ' "$@"
    printf '\n'
    printf 'evidence_scope=single-pass exploratory; repeat full matrices before paper claims\n'
  } > "$manifest_tmp"
  mv "$manifest_tmp" "$MANIFEST"
fi

# A real endpoint can return a complete result file while every request was
# rejected by authentication, billing, permission, or a shared bad request
# configuration.  router.run stops early on the deterministic online subset;
# this persisted gate also catches complete but unusable arms (for example,
# zero successful cloud TTFTs) before a completion marker can authorize a later
# stage.  Re-checking persisted markers closes the same gate on resume; the
# post-marker check also prevents a marker from being treated as permission to
# continue if raw evidence changes at that boundary.
check_systemic_cloud_errors() {
  local raw_path=$1
  local arm_name=$2
  local checkpoint=$3
  "$PYBIN" - "$raw_path" "$arm_name" "$checkpoint" <<'PY_SYSTEMIC_CLOUD_CHECK'
import collections
import json
import sys

raw_path, arm, checkpoint = sys.argv[1:]
cloud_rows = []
try:
    with open(raw_path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"raw line {line_number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"raw line {line_number} is not an object")
            if row.get("endpoint") == "cloud":
                cloud_rows.append(row)
except (OSError, ValueError) as exc:
    print(
        f"cloud status gate failed at {checkpoint} for {arm}: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(5) from None

status_counts = collections.Counter()
cloud_success_n = 0
non429_http_failure_n = 0
for row in cloud_rows:
    status = row.get("http_status")
    if isinstance(status, int) and not isinstance(status, bool):
        status_counts[status] += 1
        success = row.get("success") is True and 200 <= status < 300
        cloud_success_n += int(success)
        if not success and status != 429:
            non429_http_failure_n += 1

immediate_statuses = (401, 402, 403, 404, 405, 422)
cloud_n = len(cloud_rows)
# A complete arm with no successful cloud TTFT is not usable evidence, even
# when its failures (including 429s) remain valid measured rows.  With at least
# one success, isolated HTTP failures remain measured.  Reject only an
# immediate fixed-endpoint/config status, a >=80% 400 pattern after 3 rows, or
# a >=80% non-429 HTTP-failure pattern after 10 rows.  Transport failures with
# no HTTP status do not count toward the dominant-HTTP threshold.
hard_stop = (
    any(status_counts[status] for status in immediate_statuses)
    or any(
        count for status, count in status_counts.items()
        if 300 <= status < 400
    )
)
zero_success = cloud_n > 0 and cloud_success_n == 0
dominant_400 = (
    cloud_n >= 3 and status_counts[400] * 5 >= cloud_n * 4
)
dominant_non429_http = (
    cloud_n >= 10 and non429_http_failure_n * 5 >= cloud_n * 4
)
if hard_stop or zero_success or dominant_400 or dominant_non429_http:
    rendered = ",".join(
        f"{status}:{status_counts[status]}"
        for status in sorted(status_counts)
    )
    print(
        "systemic cloud authentication/configuration/transport failure "
        f"at {checkpoint} for {arm}: cloud_n={cloud_n} "
        f"cloud_success_n={cloud_success_n} "
        f"non429_http_failure_n={non429_http_failure_n} statuses={rendered}",
        file=sys.stderr,
    )
    raise SystemExit(5)
PY_SYSTEMIC_CLOUD_CHECK
}

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
    --slo-s "$SLO_S" --timeout-s "$TIMEOUT_S" \
    --nimbus-tick-ms "$NIMBUS_TICK_MS"
    --max-inflight "$MAX_INFLIGHT" --temperature "$TEMPERATURE"
    --ignore-eos "$IGNORE_EOS" --kv-capacity-tokens "$KV_CAP"
    --model "$MODEL" --chat-url "$CHAT_URL" --scenario "$SCENARIO"
    --cloud "$CLOUD" --cloud-max-concurrency "$CLOUD_MAX_CONCURRENCY"
    --cloud-no-fallbacks "$CLOUD_NO_FALLBACKS"
    --cloud-stop-after-first-token "$CLOUD_STOP_AFTER_FIRST_TOKEN"
    --local-ignore-eos "$LOCAL_IGNORE_EOS"
    --cloud-ignore-eos "$CLOUD_IGNORE_EOS"
  )
  if [[ "$CLOUD" == real ]]; then
    evidence_cmd+=(
      --cloud-url "$CLOUD_URL" --cloud-model "$CLOUD_MODEL"
      --cloud-api-key-env "$CLOUD_API_KEY_ENV"
      --cloud-provider "$CLOUD_PROVIDER"
    )
  fi
  if [[ -e "$marker" ]]; then
    "${evidence_cmd[@]}"
    if [[ "$CLOUD" == real ]]; then
      check_systemic_cloud_errors "$raw" "$arm" resume_marker
    fi
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
    --data "$DATA" --expected-trace-sha256 "$TRACE_SHA"
    --scenario "$SCENARIO" "${policy_args[@]}"
    --local-url "$CHAT_URL" --local-model "$MODEL" --max-inflight "$MAX_INFLIGHT"
    --seed "$seed"
    --kv-capacity-tokens "$KV_CAP"
    --prefill-tput "$PREFILL_TPUT" --tpot-ms "$TPOT_MS"
    --first-token-overhead-ms "$FIRST_TOKEN_OVERHEAD_MS"
    --slo-s "$SLO_S" --timeout-s "$TIMEOUT_S" \
    --ttft-guard-ms "$TTFT_GUARD_MS"
    --nimbus-tick-ms "$NIMBUS_TICK_MS"
    --in-price "$IN_PRICE" --out-price "$OUT_PRICE"
    --temperature "$TEMPERATURE"
    --cloud "$CLOUD" --cloud-max-concurrency "$CLOUD_MAX_CONCURRENCY"
    --out-dir "$OUT_DIR" --output "${stem}.jsonl"
    --decision-log "${stem}.decisions.jsonl"
  )
  if [[ "$IGNORE_EOS" == 1 ]]; then
    cmd+=(--ignore-eos)
  fi
  if [[ "$LOCAL_IGNORE_EOS" == 1 ]]; then
    cmd+=(--local-ignore-eos)
  else
    cmd+=(--no-local-ignore-eos)
  fi
  if [[ "$CLOUD_IGNORE_EOS" == 1 ]]; then
    cmd+=(--cloud-ignore-eos)
  else
    cmd+=(--no-cloud-ignore-eos)
  fi
  if [[ "$CLOUD" == real ]]; then
    cmd+=(
      --cloud-url "$CLOUD_URL" --cloud-model "$CLOUD_MODEL"
      --cloud-api-key-env "$CLOUD_API_KEY_ENV"
      --cloud-provider "$CLOUD_PROVIDER"
      --cloud-no-fallbacks --cloud-stop-after-first-token
    )
  fi
  {
    if [[ "$E12_LIVE_ACTIVE" == 1 ]]; then
      printf 'arm_started_at=%s arm=%s live_contract_id=%s live_stage=%s live_authorized_budget_usd=%s live_budget_attestation_sha256=%s live_key_limit_max_usd=%s live_key_contract_mode=%s live_cooldown_s=%s live_stage_launch_attestation_sha256=%s live_current_usage_sha256=%s live_stage_launch_verify_receipt_sha256=%s command=router.run_config_bound_by_fingerprint\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$arm" \
        "$E12_LIVE_CONTRACT_ID" "$E12_LIVE_STAGE" \
        "$E12_LIVE_AUTHORIZED_BUDGET_USD" \
        "$E12_LIVE_BUDGET_ATTESTATION_SHA256" \
        "$E12_LIVE_KEY_LIMIT_MAX_USD" \
        "$E12_LIVE_KEY_CONTRACT_MODE" \
        "$E12_LIVE_COOLDOWN_S" \
        "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256" \
        "$E12_LIVE_CURRENT_USAGE_SHA256" \
        "$E12_LIVE_VERIFY_RECEIPT_SHA256"
    else
      printf 'arm_started_at=%s arm=%s command=' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$arm"
      printf '%q ' "${cmd[@]}"
      printf '\n'
    fi
  } >> "$EVENTS"
  (
    export "$CLOUD_API_KEY_ENV=$MATRIX_CLOUD_API_KEY_SECRET"
    "${cmd[@]}"
  )
  if [[ "$CLOUD" == real ]]; then
    check_systemic_cloud_errors "$raw" "$arm" pre_marker
  fi
  "${evidence_cmd[@]}" --write-marker
  if [[ "$CLOUD" == real ]]; then
    check_systemic_cloud_errors "$raw" "$arm" post_marker
  fi
  if [[ "$E12_LIVE_ACTIVE" == 1 ]]; then
    printf 'arm_finished_at=%s arm=%s live_contract_id=%s live_stage=%s live_authorized_budget_usd=%s live_budget_attestation_sha256=%s live_key_limit_max_usd=%s live_key_contract_mode=%s live_cooldown_s=%s live_stage_launch_attestation_sha256=%s live_current_usage_sha256=%s live_stage_launch_verify_receipt_sha256=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$arm" \
      "$E12_LIVE_CONTRACT_ID" "$E12_LIVE_STAGE" \
      "$E12_LIVE_AUTHORIZED_BUDGET_USD" \
      "$E12_LIVE_BUDGET_ATTESTATION_SHA256" \
      "$E12_LIVE_KEY_LIMIT_MAX_USD" \
      "$E12_LIVE_KEY_CONTRACT_MODE" \
      "$E12_LIVE_COOLDOWN_S" \
      "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256" \
      "$E12_LIVE_CURRENT_USAGE_SHA256" \
      "$E12_LIVE_VERIFY_RECEIPT_SHA256" >> "$EVENTS"
  else
    printf 'arm_finished_at=%s arm=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$arm" >> "$EVENTS"
  fi
  sleep "$COOLDOWN_S"
done

if [[ "$E12_LIVE_ACTIVE" == 1 ]]; then
  printf 'matrix_finished_at=%s run_fingerprint=%s live_contract_id=%s live_stage=%s live_authorized_budget_usd=%s live_budget_attestation_sha256=%s live_key_limit_max_usd=%s live_key_contract_mode=%s live_cooldown_s=%s live_stage_launch_attestation_sha256=%s live_current_usage_sha256=%s live_stage_launch_verify_receipt_sha256=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_FINGERPRINT" \
    "$E12_LIVE_CONTRACT_ID" "$E12_LIVE_STAGE" \
    "$E12_LIVE_AUTHORIZED_BUDGET_USD" \
    "$E12_LIVE_BUDGET_ATTESTATION_SHA256" \
    "$E12_LIVE_KEY_LIMIT_MAX_USD" \
    "$E12_LIVE_KEY_CONTRACT_MODE" \
    "$E12_LIVE_COOLDOWN_S" \
    "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256" \
    "$E12_LIVE_CURRENT_USAGE_SHA256" \
    "$E12_LIVE_VERIFY_RECEIPT_SHA256" >> "$EVENTS"
else
  printf 'matrix_finished_at=%s run_fingerprint=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_FINGERPRINT" >> "$EVENTS"
fi
