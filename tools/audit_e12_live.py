#!/usr/bin/env python3
"""Offline, text-free audit for the staged E12 live current-turn A/C run.

The auditor never opens the original trace JSONL and never contacts a local or
remote service.  It binds two independently completed matrix directories to
the frozen text-free trace manifest, fresh profile, price/budget/canary launch
chain, settled usage gates, and final spend gate.  Only aggregate counts and
cryptographic identities are emitted.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ttft_matrix_evidence import (
    MatrixEvidenceError,
    MatrixExpectations,
    parse_arm,
    validate_or_write_marker,
)
from tools.check_openrouter_stage_budget import (
    StageGateError,
    build_stage_gate_attestation,
)
from tools.check_e12_stage_launch import (
    LaunchCheckError,
    LaunchContext,
    create_launch_attestation,
    validate_verify_receipt,
)
from tools.check_e12_live_budget import (
    BudgetCheckError,
    validate_materializer_manifest_schema,
)


LIVE_CONTRACT = "e12_current_turn_v1"
PAYLOAD_MODE = "sharegpt_current_turn_retokenized"
CACHE_MODE = "none"
TRACE_N = 11_604
TRACE_SHA256 = (
    "e838016a8e55660c565dadb1ad019770f6b88f878d8ca29f165c30887d2cb410"
)
PROMPT_TOKEN_SUM = 1_289_405
DECODE_TOKEN_SUM = 3_038_796
AUTHORIZED_BUDGET_USD = Decimal("3")
KEY_LIMIT_MAX_USD = Decimal("3")
MIN_STAGE_COOLDOWN_S = Decimal("20")
INPUT_PRICE = Decimal("0.08")
OUTPUT_PRICE = Decimal("0.28")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CLOUD_MODEL = "qwen/qwen3-32b"
CLOUD_PROVIDER_SLUG = "deepinfra"
CLOUD_API_KEY_ENV = "OPENROUTER_API_KEY"
ARM_ORDER_SEED = "20260716"
EXPECTED_ARMS = {
    "A": "ttft_pred:cost_cachedisp_old:0",
    "C": "ttft_pred:cost_disp_current:0",
}
EXPECTED_SELECTORS = {
    "A": "cost_cachedisp_old",
    "C": "cost_disp_current",
}
PREDICTION_SCOPE = "waiting_only"
PREDICTION_MODEL = "seq_slots_shared_prefill_lane_v1"
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
CONTRACT_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
GENERATION_ID_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SNAPSHOT_HASH_RE = re.compile(r"[0-9a-f]{16}\Z")
EPSILON = 1e-9


@dataclass(frozen=True)
class TraceIdentity:
    """Frozen E12 identity, including the checkout-dependent manifest hash."""

    n: int
    trace_sha256: str
    manifest_sha256: str
    prompt_token_sum: int
    decode_token_sum: int


class EvidenceError(ValueError):
    """The supplied artifacts do not satisfy the frozen live contract."""


@dataclass
class EventAudit:
    sha256: str
    arm_started_at: datetime
    arm_finished_at: datetime
    matrix_finished_at: datetime


@dataclass
class StageAudit:
    label: str
    manifest: dict[str, Any]
    config: dict[str, Any]
    fingerprint: str
    raw_sha256: str
    summary_sha256: str
    decisions_sha256: str
    marker_sha256: str
    events: EventAudit
    local_n: int
    cloud_n: int
    local_success_n: int
    cloud_success_n: int
    cloud_failure_n: int
    cloud_slo_violation_n: int
    overall_slo_violation_n: int
    cloud_http_status_counts: dict[str, int]
    cloud_error_type_counts: dict[str, int]
    decision_n: int
    applied_victim_n: int


@dataclass(frozen=True)
class BudgetAudit:
    sha256: str
    price_snapshot_sha256: str
    request_price_source: str
    estimated_cost_usd: Decimal
    budget_usd: Decimal


@dataclass(frozen=True)
class UsageSnapshot:
    sha256: str
    captured_at: datetime
    key_usage: Decimal
    key_limit: Decimal
    key_remaining: Decimal
    key_fingerprint_sha256: str
    expires_at: datetime | None
    limit_reset: None
    include_byok_in_limit: bool
    is_management_key: bool
    is_provisioning_key: bool
    is_free_tier: bool
    account_status: int
    account_usage: Decimal | None
    account_credits: Decimal | None
    account_remaining: Decimal | None


def _sha256(path: Path, label: str) -> str:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"{label}: unreadable") from None
    return hashlib.sha256(content).hexdigest()


def _artifact(path: Path, label: str) -> dict[str, Any]:
    try:
        content = path.read_bytes()
    except OSError:
        raise EvidenceError(f"{label}: unreadable") from None
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "nonempty_line_n": sum(bool(line.strip()) for line in content.splitlines()),
    }


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise EvidenceError(f"{label}: unreadable UTF-8 JSON object") from None
    if not isinstance(payload, dict):
        raise EvidenceError(f"{label}: JSON root must be an object")
    return payload


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        raise EvidenceError(f"{label}: unreadable UTF-8 JSONL") from None
    result: list[dict[str, Any]] = []
    for line_n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise EvidenceError(f"{label} line {line_n}: invalid JSON") from None
        if not isinstance(row, dict):
            raise EvidenceError(f"{label} line {line_n}: row must be an object")
        result.append(row)
    return result


def _object(value: Any, field: str, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{source}: {field} must be an object")
    return value


def _integer(value: Any, field: str, source: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{source}: {field} must be an integer")
    if minimum is not None and value < minimum:
        raise EvidenceError(f"{source}: {field} must be >= {minimum}")
    return value


def _number(value: Any, field: str, source: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{source}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{source}: {field} must be finite")
    if minimum is not None and result < minimum:
        raise EvidenceError(f"{source}: {field} must be >= {minimum}")
    return result


def _optional_number(
    value: Any, field: str, source: str, *, minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _number(value, field, source, minimum=minimum)


def _decimal(value: Any, field: str, source: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise EvidenceError(f"{source}: {field} must be a decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise EvidenceError(f"{source}: {field} must be a finite decimal") from None
    if not result.is_finite():
        raise EvidenceError(f"{source}: {field} must be a finite decimal")
    return result


def _close(left: float, right: float, *, tolerance: float = EPSILON) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _parse_iso(value: Any, source: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EvidenceError(f"{source}: timestamp must be UTC ISO-8601 Z form")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise EvidenceError(f"{source}: invalid timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvidenceError(f"{source}: timestamp must be UTC")
    return parsed


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _manifest_number(manifest: dict[str, Any], key: str, source: str) -> float:
    try:
        value = float(manifest[key])
    except (KeyError, TypeError, ValueError):
        raise EvidenceError(f"{source}: manifest {key} must be numeric") from None
    if not math.isfinite(value):
        raise EvidenceError(f"{source}: manifest {key} must be finite")
    return value


def _manifest_int(manifest: dict[str, Any], key: str, source: str) -> int:
    value = _manifest_number(manifest, key, source)
    if not value.is_integer():
        raise EvidenceError(f"{source}: manifest {key} must be an integer")
    return int(value)


REQUIRED_MANIFEST_FIELDS = frozenset({
    "run_fingerprint", "started_at", "commit", "trace_sha256",
    "trace_manifest_sha256", "trace_n", "profile_sha256", "cache_mode",
    "server_pid", "server_log", "server_log_prefix_sha256",
    "server_log_sha256_at_manifest", "endpoint_version_sha256",
    "endpoint_models_identity_sha256", "base_url", "chat_url", "model",
    "python", "scenario", "max_inflight", "kv_cap", "prefill_tput",
    "tpot_ms", "first_token_overhead_ms", "slo_s", "timeout_s",
    "guard_ms", "tick_ms", "temperature", "ignore_eos", "in_price",
    "out_price", "local_ignore_eos", "cloud_ignore_eos", "cloud",
    "cloud_url", "cloud_model", "cloud_api_key_env",
    "cloud_max_concurrency", "cloud_provider", "cloud_no_fallbacks",
    "cloud_stop_after_first_token", "real_cloud_expected_trace_n",
    "secret_value_recorded", "arm_order_mode", "arm_order_seed", "arms",
    "evidence_scope", "live_contract_id", "live_contract", "live_stage",
    "live_expected_trace_n", "live_arm_order_seed", "live_exact_arm",
    "live_authorized_budget_usd", "live_budget_attestation_sha256",
    "live_key_limit_max_usd", "live_cooldown_s",
    "live_stage_launch_attestation_sha256",
    "live_current_usage_sha256",
    "live_stage_launch_verify_receipt_sha256",
})


def _parse_manifest(path: Path, label: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        raise EvidenceError(f"{label}: unreadable matrix manifest") from None
    result: dict[str, Any] = {}
    remainder_fields = {"python", "evidence_scope", "arms"}
    for line_n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        first_key = line.split("=", 1)[0]
        tokens = [line] if first_key in remainder_fields else line.split()
        for token in tokens:
            if "=" not in token:
                raise EvidenceError(f"{label} line {line_n}: malformed field")
            key, value = token.split("=", 1)
            if not key or key in result:
                raise EvidenceError(f"{label} line {line_n}: duplicate/empty field")
            result[key] = tuple(value.split()) if key == "arms" else value
    missing = sorted(REQUIRED_MANIFEST_FIELDS - result.keys())
    if missing:
        raise EvidenceError(f"{label}: missing fields: {', '.join(missing)}")
    return result


FINGERPRINT_BASE_KEYS = (
    "trace_sha256", "trace_manifest_sha256", "profile_sha256",
    "server_log_prefix_sha256", "server_pid", "endpoint_version_sha256",
    "endpoint_models_identity_sha256", "base_url", "chat_url", "commit",
    "scenario", "max_inflight", "kv_cap", "prefill_tput", "tpot_ms",
    "first_token_overhead_ms", "slo_s", "timeout_s", "guard_ms",
    "tick_ms", "temperature", "ignore_eos", "in_price", "out_price",
    "cloud", "cloud_url", "cloud_model", "cloud_api_key_env",
    "cloud_max_concurrency", "cloud_provider", "cloud_no_fallbacks",
    "cloud_stop_after_first_token", "local_ignore_eos", "cloud_ignore_eos",
    "real_cloud_expected_trace_n", "arm_order_mode", "arm_order_seed",
)
FINGERPRINT_LIVE_KEYS = (
    "live_contract_id", "live_contract", "live_stage",
    "live_expected_trace_n", "live_arm_order_seed", "live_exact_arm",
    "live_authorized_budget_usd", "live_budget_attestation_sha256",
    "live_key_limit_max_usd", "live_cooldown_s",
    "live_stage_launch_attestation_sha256",
    "live_current_usage_sha256", "live_stage_launch_verify_receipt_sha256",
)


def recompute_run_fingerprint(manifest: dict[str, Any]) -> str:
    """Reproduce the live runner's exact ``base + arms + extras`` hash."""
    try:
        base = [manifest[key] for key in FINGERPRINT_BASE_KEYS]
        arms = manifest["arms"]
        extras = [manifest[key] for key in FINGERPRINT_LIVE_KEYS]
    except KeyError as exc:
        raise EvidenceError(f"manifest lacks fingerprint input {exc.args[0]}") from None
    if not all(isinstance(value, str) for value in base + extras):
        raise EvidenceError("manifest fingerprint inputs must be strings")
    if not isinstance(arms, tuple) or not all(isinstance(arm, str) for arm in arms):
        raise EvidenceError("manifest arms must be an exact tuple")
    payload = json.dumps(
        base + list(arms) + extras,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_trace_manifest(
    path: Path, identity: TraceIdentity,
) -> tuple[dict[str, Any], str]:
    source = "trace manifest"
    actual_sha = _sha256(path, source)
    if actual_sha != identity.manifest_sha256:
        raise EvidenceError("trace manifest: SHA256 does not match frozen E12 identity")
    manifest = _json_object(path, source)
    try:
        validate_materializer_manifest_schema(manifest)
    except BudgetCheckError:
        # The budget validator deliberately never reflects unknown keys or
        # values, but keep the audit boundary generic as defense in depth.
        raise EvidenceError(
            "trace manifest: fields do not match the text-free materializer schema"
        ) from None
    exact = {
        "schema_version": 1,
        "output_sha256": identity.trace_sha256,
        "n": identity.n,
        "payload_mode": PAYLOAD_MODE,
        "cache_mode": CACHE_MODE,
        "semantic_scope": "verbatim_current_user_turn_only",
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise EvidenceError(f"trace manifest: {key} does not match frozen identity")
    prompt = _object(manifest.get("actual_prompt_tokens"), "actual_prompt_tokens", source)
    decode = _object(manifest.get("output_decode_tokens"), "output_decode_tokens", source)
    if (
        _integer(prompt.get("n"), "actual_prompt_tokens.n", source) != identity.n
        or _integer(prompt.get("sum"), "actual_prompt_tokens.sum", source)
        != identity.prompt_token_sum
    ):
        raise EvidenceError("trace manifest: prompt-token population/sum mismatch")
    if (
        _integer(decode.get("n"), "output_decode_tokens.n", source) != identity.n
        or _integer(decode.get("sum"), "output_decode_tokens.sum", source)
        != identity.decode_token_sum
    ):
        raise EvidenceError("trace manifest: decode-token population/sum mismatch")
    preservation = _object(
        manifest.get("preservation_checks"), "preservation_checks", source
    )
    for key in (
        "prompt_text_exact_n", "arrived_at_exact_n",
        "decode_source_provenance_n", "token_metadata_aligned_n",
    ):
        if _integer(preservation.get(key), f"preservation_checks.{key}", source) != identity.n:
            raise EvidenceError(f"trace manifest: preservation check {key} is incomplete")
    return manifest, actual_sha


BUDGET_FIELDS = frozenset({
    "schema_version", "status", "manifest_sha256", "price_snapshot_sha256",
    "trace_sha256",
    "trace_n", "payload_mode", "cache_mode", "prompt_token_sum",
    "decode_token_sum", "arm_count", "input_price_per_million_usd",
    "output_price_per_million_usd", "request_price_per_request_usd",
    "request_price_source", "prompt_cost_usd", "decode_cost_usd",
    "estimated_cost_usd", "budget_usd", "budget_headroom_usd",
})


def _validate_budget_attestation(
    path: Path, identity: TraceIdentity, trace_manifest_sha: str,
) -> BudgetAudit:
    source = "budget attestation"
    sha = _sha256(path, source)
    payload = _json_object(path, source)
    if set(payload) != BUDGET_FIELDS:
        raise EvidenceError("budget attestation: fields do not match the text-free schema")
    exact = {
        "schema_version": 1,
        "status": "pass",
        "manifest_sha256": trace_manifest_sha,
        "trace_sha256": identity.trace_sha256,
        "trace_n": identity.n,
        "payload_mode": PAYLOAD_MODE,
        "cache_mode": CACHE_MODE,
        "prompt_token_sum": identity.prompt_token_sum,
        "decode_token_sum": identity.decode_token_sum,
        "arm_count": 2,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise EvidenceError(f"budget attestation: {key} mismatch")
    price_snapshot_sha = payload.get("price_snapshot_sha256")
    if not isinstance(price_snapshot_sha, str) or not SHA256_RE.fullmatch(
        price_snapshot_sha
    ):
        raise EvidenceError("budget attestation: price snapshot SHA256 is invalid")
    values = {
        key: _decimal(payload.get(key), key, source)
        for key in (
            "input_price_per_million_usd", "output_price_per_million_usd",
            "request_price_per_request_usd",
            "prompt_cost_usd", "decode_cost_usd", "estimated_cost_usd",
            "budget_usd", "budget_headroom_usd",
        )
    }
    if values["input_price_per_million_usd"] != INPUT_PRICE:
        raise EvidenceError("budget attestation: input price is not frozen at 0.08")
    if values["output_price_per_million_usd"] != OUTPUT_PRICE:
        raise EvidenceError("budget attestation: output price is not frozen at 0.28")
    if values["request_price_per_request_usd"] != 0:
        raise EvidenceError("budget attestation: request fee is not frozen at zero")
    request_price_source = payload.get("request_price_source")
    if request_price_source not in {"absent_not_advertised", "explicit_zero"}:
        raise EvidenceError("budget attestation: request-price source is invalid")
    prompt_cost = (
        Decimal(2) * Decimal(identity.prompt_token_sum) * INPUT_PRICE
        / Decimal(1_000_000)
    )
    decode_cost = (
        Decimal(2) * Decimal(identity.decode_token_sum) * OUTPUT_PRICE
        / Decimal(1_000_000)
    )
    estimated = prompt_cost + decode_cost
    budget = values["budget_usd"]
    if values["prompt_cost_usd"] != prompt_cost:
        raise EvidenceError("budget attestation: prompt cost arithmetic mismatch")
    if values["decode_cost_usd"] != decode_cost:
        raise EvidenceError("budget attestation: decode cost arithmetic mismatch")
    if values["estimated_cost_usd"] != estimated:
        raise EvidenceError("budget attestation: total cost arithmetic mismatch")
    if budget != AUTHORIZED_BUDGET_USD or estimated > budget:
        raise EvidenceError("budget attestation: run is not within the exact $3 authorization")
    if values["budget_headroom_usd"] != budget - estimated:
        raise EvidenceError("budget attestation: headroom arithmetic mismatch")
    return BudgetAudit(
        sha256=sha,
        price_snapshot_sha256=price_snapshot_sha.lower(),
        request_price_source=request_price_source,
        estimated_cost_usd=estimated,
        budget_usd=budget,
    )


def _validate_profile(
    path: Path,
    trace_manifest: dict[str, Any],
    stage_manifest: dict[str, Any],
) -> str:
    source = "profile"
    sha = _sha256(path, source)
    if sha != stage_manifest["profile_sha256"]:
        raise EvidenceError("profile: SHA256 does not match stage manifest")
    profile = _json_object(path, source)
    exact = {
        "schema_version": 2,
        "predictor_model": PREDICTION_MODEL,
        "valid": True,
        "cache_mode_required": CACHE_MODE,
        "model": stage_manifest["model"],
    }
    for key, expected in exact.items():
        if profile.get(key) != expected:
            raise EvidenceError(f"profile: {key} mismatch")
    if profile.get("tokenizer_fingerprint") != trace_manifest.get("tokenizer_fingerprint"):
        raise EvidenceError("profile: tokenizer fingerprint differs from trace manifest")
    if str(profile.get("base_url", "")).rstrip("/") != str(
        stage_manifest["base_url"]
    ).rstrip("/"):
        raise EvidenceError("profile: base URL differs from stage manifest")
    if _integer(profile.get("server_pid"), "server_pid", source) != _manifest_int(
        stage_manifest, "server_pid", source
    ):
        raise EvidenceError("profile: server PID differs from stage lifecycle")
    if profile.get("server_log_sha256_at_start") != stage_manifest["server_log_prefix_sha256"]:
        raise EvidenceError("profile: server log prefix differs from stage lifecycle")
    if not isinstance(profile.get("server_proc_cmdline_sha256_at_start"), str) or not SHA256_RE.fullmatch(
        profile["server_proc_cmdline_sha256_at_start"]
    ):
        raise EvidenceError("profile: server process identity hash is missing")
    if not _close(
        _number(profile.get("kv_capacity_tokens"), "kv_capacity_tokens", source),
        _manifest_number(stage_manifest, "kv_cap", source),
    ):
        raise EvidenceError("profile: KV capacity differs from stage manifest")
    sampling = _object(profile.get("sampling"), "sampling", source)
    if sampling.get("ignore_eos") is not True or sampling.get("continuous_usage_stats") is not True:
        raise EvidenceError("profile: sampling is not exact no-cache local profiling mode")
    if not _close(
        _number(sampling.get("temperature"), "sampling.temperature", source),
        _manifest_number(stage_manifest, "temperature", source),
    ):
        raise EvidenceError("profile: sampling temperature differs from stage manifest")
    profile_config = _object(profile.get("profile_config"), "profile_config", source)
    for profile_key, manifest_key in (("nimbus_tick_ms", "tick_ms"), ("slo_s", "slo_s")):
        if not _close(
            _number(profile_config.get(profile_key), f"profile_config.{profile_key}", source),
            _manifest_number(stage_manifest, manifest_key, source),
        ):
            raise EvidenceError(f"profile: {profile_key} differs from stage manifest")
    calibration = _object(profile.get("predictor_calibration"), "predictor_calibration", source)
    calibration_fields = (
        ("recommended_prefill_tput_tokens_per_s", "prefill_tput"),
        ("recommended_tpot_ms", "tpot_ms"),
        ("recommended_first_token_overhead_ms", "first_token_overhead_ms"),
        ("recommended_ttft_guard_ms", "guard_ms"),
        ("target_slo_s", "slo_s"),
    )
    for profile_key, manifest_key in calibration_fields:
        if not _close(
            _number(calibration.get(profile_key), profile_key, source),
            _manifest_number(stage_manifest, manifest_key, source),
        ):
            raise EvidenceError(f"profile: calibration {profile_key} mismatch")
    if _integer(calibration.get("heldout_n"), "heldout_n", source, minimum=1) <= 0:
        raise EvidenceError("profile: held-out population is empty")
    confusion = _object(
        calibration.get("heldout_violation_confusion"),
        "heldout_violation_confusion",
        source,
    )
    if _integer(confusion.get("false_negative"), "false_negative", source) != 0:
        raise EvidenceError("profile: held-out classifier has false negatives")
    return sha


def _validate_stage_manifest(
    manifest: dict[str, Any],
    label: str,
    identity: TraceIdentity,
    trace_manifest_sha: str,
    budget_sha: str,
    launch_sha: str,
    live_current_usage_sha: str,
    launch_verify_receipt_sha: str,
) -> None:
    source = f"stage {label} manifest"
    arm = EXPECTED_ARMS[label]
    try:
        parsed_arm = parse_arm(arm)
    except MatrixEvidenceError as exc:  # pragma: no cover - frozen constant defense
        raise EvidenceError(f"{source}: invalid frozen arm") from exc
    if parsed_arm.trigger != "ttft_pred" or parsed_arm.selector != EXPECTED_SELECTORS[label]:
        raise EvidenceError(f"{source}: frozen arm parser identity mismatch")
    exact = {
        "arms": (arm,),
        "trace_sha256": identity.trace_sha256,
        "trace_manifest_sha256": trace_manifest_sha,
        "trace_n": str(identity.n),
        "cache_mode": CACHE_MODE,
        "cloud": "real",
        "cloud_url": OPENROUTER_URL,
        "cloud_model": CLOUD_MODEL,
        "cloud_api_key_env": CLOUD_API_KEY_ENV,
        "cloud_max_concurrency": "16",
        "cloud_provider": CLOUD_PROVIDER_SLUG,
        "cloud_no_fallbacks": "1",
        "cloud_stop_after_first_token": "1",
        "local_ignore_eos": "1",
        "cloud_ignore_eos": "0",
        "in_price": str(INPUT_PRICE),
        "out_price": str(OUTPUT_PRICE),
        "real_cloud_expected_trace_n": str(identity.n),
        "secret_value_recorded": "false",
        "arm_order_mode": "e12_contract_stage",
        "arm_order_seed": ARM_ORDER_SEED,
        "live_contract": LIVE_CONTRACT,
        "live_stage": label,
        "live_expected_trace_n": str(identity.n),
        "live_arm_order_seed": ARM_ORDER_SEED,
        "live_exact_arm": arm,
        "live_authorized_budget_usd": "3",
        "live_key_limit_max_usd": "3",
        "live_cooldown_s": "20",
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise EvidenceError(f"{source}: frozen field mismatch")
    contract_id = manifest.get("live_contract_id")
    if not isinstance(contract_id, str) or not CONTRACT_ID_RE.fullmatch(contract_id):
        raise EvidenceError(f"{source}: invalid live_contract_id")
    attestation_spelling = manifest.get("live_budget_attestation_sha256")
    if (
        not isinstance(attestation_spelling, str)
        or not SHA256_RE.fullmatch(attestation_spelling)
        or attestation_spelling.lower() != budget_sha
    ):
        raise EvidenceError(f"{source}: budget attestation SHA256 mismatch")
    launch_spelling = manifest.get("live_stage_launch_attestation_sha256")
    if (
        not isinstance(launch_spelling, str)
        or not SHA256_RE.fullmatch(launch_spelling)
        or launch_spelling.lower() != launch_sha
    ):
        raise EvidenceError(f"{source}: launch attestation SHA256 mismatch")
    persisted_hashes = {
        "live_current_usage_sha256": live_current_usage_sha,
        "live_stage_launch_verify_receipt_sha256": launch_verify_receipt_sha,
    }
    for key, expected_sha in persisted_hashes.items():
        spelling = manifest.get(key)
        if (
            not isinstance(spelling, str)
            or not SHA256_RE.fullmatch(spelling)
            or spelling.lower() != expected_sha
        ):
            raise EvidenceError(f"{source}: persisted launch evidence SHA256 mismatch")
    for key in (
        "run_fingerprint", "profile_sha256", "server_log_prefix_sha256",
        "server_log_sha256_at_manifest", "endpoint_version_sha256",
        "endpoint_models_identity_sha256",
    ):
        value = manifest.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise EvidenceError(f"{source}: {key} is not a SHA256")
    recomputed = recompute_run_fingerprint(manifest)
    if manifest["run_fingerprint"] != recomputed:
        raise EvidenceError(f"{source}: run fingerprint does not bind exact inputs")


def _event_regexes() -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    token = r"(\S+)"
    common = (
        r" live_contract_id=" + token
        + r" live_stage=" + token
        + r" live_authorized_budget_usd=" + token
        + r" live_budget_attestation_sha256=" + token
        + r" live_key_limit_max_usd=" + token
        + r" live_cooldown_s=" + token
        + r" live_stage_launch_attestation_sha256=" + token
        + r" live_current_usage_sha256=" + token
        + r" live_stage_launch_verify_receipt_sha256=" + token
    )
    return (
        re.compile(r"arm_started_at=" + token + r" arm=" + token + common + r" command=(.+)\Z"),
        re.compile(r"arm_finished_at=" + token + r" arm=" + token + common + r"\Z"),
        re.compile(r"matrix_finished_at=" + token + r" run_fingerprint=" + token + common + r"\Z"),
    )


def _validate_events(path: Path, label: str, manifest: dict[str, Any]) -> EventAudit:
    source = f"stage {label} events"
    try:
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        raise EvidenceError(f"{source}: unreadable UTF-8") from None
    if len(lines) != 3 or any(not line for line in lines):
        raise EvidenceError(f"{source}: require exactly three complete event lines")
    start_re, finish_re, matrix_re = _event_regexes()
    matches = (start_re.fullmatch(lines[0]), finish_re.fullmatch(lines[1]), matrix_re.fullmatch(lines[2]))
    if any(match is None for match in matches):
        raise EvidenceError(f"{source}: event line does not match live schema")
    start, finish, matrix = matches
    assert start is not None and finish is not None and matrix is not None
    # timestamp/arm-or-fingerprint precede nine identical live fields.
    expected_live = (
        manifest["live_contract_id"], label, "3",
        manifest["live_budget_attestation_sha256"], "3", "20",
        manifest["live_stage_launch_attestation_sha256"],
        manifest["live_current_usage_sha256"],
        manifest["live_stage_launch_verify_receipt_sha256"],
    )
    if start.group(2) != EXPECTED_ARMS[label] or tuple(start.group(i) for i in range(3, 12)) != expected_live:
        raise EvidenceError(f"{source}: start event contract/stage/arm mismatch")
    if finish.group(2) != EXPECTED_ARMS[label] or tuple(finish.group(i) for i in range(3, 12)) != expected_live:
        raise EvidenceError(f"{source}: finish event contract/stage/arm mismatch")
    if matrix.group(2) != manifest["run_fingerprint"] or tuple(matrix.group(i) for i in range(3, 12)) != expected_live:
        raise EvidenceError(f"{source}: matrix-finish contract/fingerprint mismatch")
    if start.group(12) != "router.run_config_bound_by_fingerprint":
        raise EvidenceError(f"{source}: start command attestation is invalid")
    started_at = _parse_iso(start.group(1), source)
    finished_at = _parse_iso(finish.group(1), source)
    matrix_at = _parse_iso(matrix.group(1), source)
    if not started_at <= finished_at <= matrix_at:
        raise EvidenceError(f"{source}: require start <= arm finish <= matrix finish")
    if _parse_iso(manifest["started_at"], source) > started_at:
        raise EvidenceError(f"{source}: manifest was created after arm start")
    return EventAudit(
        sha256=hashlib.sha256(raw).hexdigest(),
        arm_started_at=started_at,
        arm_finished_at=finished_at,
        matrix_finished_at=matrix_at,
    )


FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "prompt", "prompt_text", "messages", "message", "input", "inputs",
    "content", "reasoning", "reasoning_content", "output", "output_text",
    "response_text", "response_body", "provider_body", "error_body",
    "raw", "raw_error", "raw_response", "body", "api_key", "api_key_value",
    "authorization", "authorization_header", "key_fingerprint_sha256",
    "secret", "generation_id", "sse_id", "response_id",
})
RAW_STRING_KEYS = frozenset({
    "endpoint", "model", "payload_mode", "cache_mode", "error", "error_type",
    "probe_mode", "first_token_kind", "generation_id_sha256", "provider",
    "response_model", "requested_provider_order",
})
DECISION_STRING_KEYS = frozenset({
    "status", "trigger", "selector", "prediction_scope", "prediction_model",
    "snapshot_hash",
})
SUMMARY_STRING_KEYS = frozenset({
    "policy", "scenario", "nimbus_trigger", "nimbus_selector",
    "nimbus_prediction_scope", "nimbus_prediction_model", "local_url",
    "local_model", "cloud", "cloud_url", "cloud_model",
    "cloud_api_key_env", "cloud_provider_order",
})
SUMMARY_TOP_FIELDS = frozenset({
    "policy", "target_fraction", "actual_fraction", "slo_s", "overall",
    "local", "cloud", "pessimistic_combined", "queue", "token_alignment",
    "config",
})
SUMMARY_SIDE_FIELDS = frozenset({
    "n", "success", "routed_only", "errors", "ttft_p50_ms",
    "ttft_p95_ms", "ttft_p99_ms", "tpot_p50_ms", "slo_violations",
    "slo_measured_n", "slo_violation_pct", "cost_usd", "known_cost_usd",
    "cost_measured_n", "cost_pending_n",
})
SUMMARY_PESSIMISTIC_FIELDS = frozenset({
    "slo_violations", "slo_n", "slo_violation_pct",
    "cloud_assumed_violations", "cost_usd", "known_cost_usd",
    "cost_measured_n", "cost_pending_n",
})
SUMMARY_TOKEN_ALIGNMENT_FIELDS = frozenset({
    "local_success_n", "measured_n", "missing_prompt_usage_n",
    "prompt_exact_n", "prompt_exact_fraction", "absolute_error_p50_tokens",
    "absolute_error_p95_tokens", "absolute_error_max_tokens",
    "relative_error_p50", "relative_error_p95", "actual_over_scheduler_p50",
    "actual_over_scheduler_p95", "decode_measured_n",
    "missing_completion_usage_n", "decode_cap_hit_n",
    "decode_cap_hit_fraction", "completion_over_scheduler_p50",
    "completion_over_scheduler_p05",
})
SUMMARY_CONFIG_FIELDS = frozenset({
    "scenario", "seed", "time_scale", "max_inflight",
    "max_tokens_override", "temperature", "ignore_eos",
    "local_ignore_eos", "cloud_ignore_eos", "nimbus_trigger",
    "nimbus_selector", "prefill_tput", "tpot_ms",
    "first_token_overhead_ms", "slo_s", "timeout_s", "ttft_guard_ms",
    "nimbus_tick_ms", "kv_capacity_tokens", "kv_hysteresis_fraction",
    "cloud", "cloud_url", "cloud_model", "cloud_api_key_env",
    "cloud_max_concurrency", "cloud_provider_order", "cloud_no_fallbacks",
    "cloud_stop_after_first_token", "local_url", "local_model", "in_price",
    "out_price",
})
SUMMARY_QUEUE_FIELDS = frozenset({
    "queue_delay_p50_ms", "queue_delay_p99_ms", "queue_delay_max_ms",
    "peak_inflight", "peak_waiting", "nimbus_ticks", "nimbus_kick_rounds",
    "nimbus_kicked", "nimbus_trigger", "nimbus_selector",
    "nimbus_applied_kick_rounds", "nimbus_stale_decisions",
    "nimbus_decision_calls", "nimbus_decision_mean_ms",
    "nimbus_decision_max_ms", "nimbus_prediction_scope",
    "nimbus_prediction_model", "nimbus_max_waiting_predicted_ttft_s",
    "nimbus_max_waiting_post_kick_ttft_s",
})


def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")


def _reject_payload_keys(value: Any, source: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in FORBIDDEN_PAYLOAD_KEYS:
                raise EvidenceError(f"{source}: payload-bearing field is forbidden")
            _reject_payload_keys(child, source)
    elif isinstance(value, list):
        for child in value:
            _reject_payload_keys(child, source)


def _validate_string_locations(
    value: Any, source: str, allowed_keys: frozenset[str], parent_key: str | None = None,
) -> None:
    if isinstance(value, str):
        if parent_key not in allowed_keys:
            raise EvidenceError(f"{source}: unexpected string-bearing field")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_string_locations(child, source, allowed_keys, str(key))
    elif isinstance(value, list):
        for child in value:
            _validate_string_locations(child, source, allowed_keys, parent_key)


def _validate_finite_tree(value: Any, source: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise EvidenceError(f"{source}: contains non-finite numeric evidence")
    elif isinstance(value, dict):
        for child in value.values():
            _validate_finite_tree(child, source)
    elif isinstance(value, list):
        for child in value:
            _validate_finite_tree(child, source)
    else:
        raise EvidenceError(f"{source}: unsupported JSON value")


def _require_exact_fields(
    value: Any, expected: frozenset[str], source: str,
) -> dict[str, Any]:
    obj = _object(value, "object", source)
    if set(obj) != expected:
        raise EvidenceError(f"{source}: fields differ from frozen text-free schema")
    return obj


def _validate_summary_text_free(summary: dict[str, Any], label: str) -> None:
    source = f"stage {label} summary"
    _reject_payload_keys(summary, source)
    _validate_string_locations(summary, source, SUMMARY_STRING_KEYS)
    _validate_finite_tree(summary, source)
    _require_exact_fields(summary, SUMMARY_TOP_FIELDS, source)
    for side in ("overall", "local", "cloud"):
        side_obj = _require_exact_fields(
            summary.get(side), SUMMARY_SIDE_FIELDS, source
        )
        errors = _object(side_obj.get("errors"), "errors", source)
        if any(
            not isinstance(key, str)
            or not re.fullmatch(
                r"HTTP [0-9]{3}|StreamError|NoToken|ProgressUnavailable|TimeoutError",
                key,
            )
            for key in errors
        ):
            raise EvidenceError(f"{source}: error keys are not sanitized")
    _require_exact_fields(
        summary.get("pessimistic_combined"),
        SUMMARY_PESSIMISTIC_FIELDS,
        source,
    )
    _require_exact_fields(
        summary.get("token_alignment"),
        SUMMARY_TOKEN_ALIGNMENT_FIELDS,
        source,
    )
    _require_exact_fields(summary.get("config"), SUMMARY_CONFIG_FIELDS, source)
    _require_exact_fields(summary.get("queue"), SUMMARY_QUEUE_FIELDS, source)


def _validate_safe_error(row: dict[str, Any], source: str, timeout_s: float) -> None:
    success = row.get("success")
    error_type = row.get("error_type")
    error = row.get("error")
    if success is True:
        if error_type is not None or error is not None:
            raise EvidenceError(f"{source}: successful row carries an error string")
        return
    if not isinstance(error_type, str) or not isinstance(error, str):
        raise EvidenceError(f"{source}: failed row lacks a sanitized error pair")
    status_match = re.fullmatch(r"HTTP ([0-9]{3})", error_type)
    if status_match:
        expected = f"{error_type}: provider response body omitted"
        if error != expected:
            raise EvidenceError(f"{source}: HTTP failure contains an unsafe provider error")
        return
    safe_exact = {
        "StreamError": "provider stream error message omitted",
        "NoToken": "stream ended before a generated token",
        "ProtocolMismatch": (
            "provider response metadata mismatch; details omitted"
        ),
        "ProgressUnavailable": (
            "endpoint did not return continuous completion-token usage requested by Nimbus"
        ),
    }
    if error_type in safe_exact:
        if error != safe_exact[error_type]:
            raise EvidenceError(f"{source}: failure contains an unsafe error string")
        return
    if error_type == "TimeoutError":
        match = re.fullmatch(r"timeout after ([0-9]+(?:\.[0-9]+)?)s", error)
        if not match or not _close(float(match.group(1)), timeout_s):
            raise EvidenceError(f"{source}: timeout error is not the safe configured template")
        return
    raise EvidenceError(f"{source}: unapproved raw error type could leak provider data")


def _token_exact(row: dict[str, Any], source: str) -> None:
    prompt = _integer(row.get("prompt_tokens"), "prompt_tokens", source, minimum=0)
    decode = _integer(row.get("completion_tokens"), "completion_tokens", source, minimum=0)
    scheduler_prompt = _integer(
        row.get("scheduler_prompt_tokens"), "scheduler_prompt_tokens", source, minimum=0
    )
    scheduler_uncached = _integer(
        row.get("scheduler_uncached_prompt_tokens"),
        "scheduler_uncached_prompt_tokens", source, minimum=0,
    )
    scheduler_decode = _integer(
        row.get("scheduler_decode_tokens"), "scheduler_decode_tokens", source, minimum=0
    )
    if prompt != scheduler_prompt or scheduler_uncached != scheduler_prompt:
        raise EvidenceError(f"{source}: no-cache local prompt tokens are not exact")
    if decode != scheduler_decode:
        raise EvidenceError(f"{source}: local decode tokens are not exact")


def _audit_raw_rows(
    rows: list[dict[str, Any]], label: str, manifest: dict[str, Any], identity: TraceIdentity,
) -> tuple[set[int], set[int], dict[str, Any]]:
    ids: list[int] = []
    local_rows: list[dict[str, Any]] = []
    cloud_rows: list[dict[str, Any]] = []
    timeout_s = _manifest_number(manifest, "timeout_s", f"stage {label}")
    slo_ms = _manifest_number(manifest, "slo_s", f"stage {label}") * 1000.0
    for line_n, row in enumerate(rows, 1):
        source = f"stage {label} raw line {line_n}"
        _reject_payload_keys(row, source)
        _validate_string_locations(row, source, RAW_STRING_KEYS)
        _validate_finite_tree(row, source)
        request_id = _integer(row.get("request_id"), "request_id", source)
        if not 0 <= request_id < identity.n:
            raise EvidenceError(f"{source}: request_id outside frozen trace range")
        ids.append(request_id)
        endpoint = row.get("endpoint")
        if endpoint not in {"local", "cloud"}:
            raise EvidenceError(f"{source}: unsupported endpoint")
        if row.get("payload_mode") != PAYLOAD_MODE or row.get("cache_mode") != CACHE_MODE:
            raise EvidenceError(f"{source}: payload/cache identity mismatch")
        if not isinstance(row.get("success"), bool):
            raise EvidenceError(f"{source}: success must be boolean")
        if row.get("routed_only", False) is not False:
            raise EvidenceError(f"{source}: real-cloud evidence may not be routed_only")
        generation_id_sha256 = row.get("generation_id_sha256")
        if generation_id_sha256 is not None and (
            not isinstance(generation_id_sha256, str)
            or not GENERATION_ID_SHA256_RE.fullmatch(generation_id_sha256)
        ):
            raise EvidenceError(f"{source}: unsafe generation identifier hash")
        first_token_kind = row.get("first_token_kind")
        if first_token_kind not in {None, "reasoning", "content"}:
            raise EvidenceError(f"{source}: invalid first_token_kind")
        _validate_safe_error(row, source, timeout_s)
        if endpoint == "local":
            local_rows.append(row)
            if row.get("model") != manifest["model"]:
                raise EvidenceError(f"{source}: local model mismatch")
            if row.get("success") is not True:
                raise EvidenceError(f"{source}: retained-local failure invalidates live gate")
            if row.get("probe_mode") is not None:
                raise EvidenceError(f"{source}: local row unexpectedly claims probe mode")
            if row.get("provider") is not None:
                raise EvidenceError(f"{source}: local row unexpectedly names a provider")
            if row.get("response_model") not in {None, manifest["model"]}:
                raise EvidenceError(f"{source}: local response model mismatch")
            if row.get("requested_provider_order", []) != []:
                raise EvidenceError(f"{source}: local row carries cloud provider preferences")
            _token_exact(row, source)
            _number(row.get("ttft_ms"), "ttft_ms", source, minimum=0.0)
            status = _integer(row.get("http_status"), "http_status", source)
            if not 200 <= status < 300:
                raise EvidenceError(f"{source}: successful local HTTP status is not 2xx")
        else:
            cloud_rows.append(row)
            if row.get("model") != CLOUD_MODEL:
                raise EvidenceError(f"{source}: requested cloud model mismatch")
            if row.get("requested_provider_order") != [CLOUD_PROVIDER_SLUG]:
                raise EvidenceError(f"{source}: requested provider is not exactly deepinfra")
            if row.get("probe_mode") != "ttft_cancel":
                raise EvidenceError(f"{source}: cloud probe mode is not ttft_cancel")
            status_value = row.get("http_status")
            if status_value is not None:
                _integer(status_value, "http_status", source)
            for field in (
                "scheduler_prompt_tokens", "scheduler_uncached_prompt_tokens",
                "scheduler_decode_tokens",
            ):
                _integer(row.get(field), field, source, minimum=0)
            if row["success"]:
                if status_value is None or not 200 <= int(status_value) < 300:
                    raise EvidenceError(f"{source}: successful cloud status is not 2xx")
                if generation_id_sha256 is None:
                    raise EvidenceError(
                        f"{source}: successful cloud row lacks generation identifier hash"
                    )
                parts = [
                    _number(row.get(key), key, source, minimum=0.0)
                    for key in (
                        "pre_route_queue_ms", "cloud_gate_wait_ms", "service_ttft_ms"
                    )
                ]
                ttft = _number(row.get("ttft_ms"), "ttft_ms", source, minimum=0.0)
                if not _close(ttft, sum(parts), tolerance=1e-6):
                    raise EvidenceError(f"{source}: TTFT is not the exact three-part sum")
                if row.get("stream_abort_requested") is not True:
                    raise EvidenceError(f"{source}: successful cloud stream was not aborted")
                if row.get("response_completed") is not False:
                    raise EvidenceError(f"{source}: successful cloud response completed unexpectedly")
                provider = row.get("provider")
                if provider != CLOUD_PROVIDER_SLUG:
                    raise EvidenceError(f"{source}: observed provider is not exactly deepinfra")
                if row.get("first_token_kind") not in {"reasoning", "content"}:
                    raise EvidenceError(f"{source}: invalid first_token_kind")
                if row.get("response_model") != CLOUD_MODEL:
                    raise EvidenceError(f"{source}: response model mismatch")
                if row.get("cost_usd") is not None or row.get("cost_pending") is not True:
                    raise EvidenceError(f"{source}: aborted cloud cost must remain pending")
            else:
                provider = row.get("provider")
                if provider not in {None, CLOUD_PROVIDER_SLUG}:
                    raise EvidenceError(f"{source}: failed row names an unexpected provider")
                response_model = row.get("response_model")
                if response_model not in {None, CLOUD_MODEL}:
                    raise EvidenceError(f"{source}: failed row names an unexpected response model")
                if row.get("error_type") == "ProtocolMismatch" and (
                    provider is not None or response_model is not None
                ):
                    raise EvidenceError(
                        f"{source}: protocol mismatch retained response metadata"
                    )
                if row.get("error_type") == "ProtocolMismatch":
                    raise EvidenceError(
                        f"{source}: provider response metadata did not conform"
                    )
    request_ids = set(ids)
    if len(ids) != identity.n or len(request_ids) != identity.n:
        raise EvidenceError(f"stage {label}: raw request IDs are not unique for all N rows")
    if request_ids != set(range(identity.n)):
        raise EvidenceError(f"stage {label}: request IDs are not exactly 0..{identity.n - 1}")
    if not cloud_rows:
        raise EvidenceError(f"stage {label}: live arm routed no cloud rows")
    cloud_ids = {int(row["request_id"]) for row in cloud_rows}
    status_counts = collections.Counter(
        str(row["http_status"])
        for row in cloud_rows
        if row.get("http_status") is not None
    )
    error_counts = collections.Counter(
        str(row["error_type"])
        for row in cloud_rows
        if row.get("error_type") is not None
    )
    cloud_success_n = sum(
        row.get("success") is True
        and isinstance(row.get("http_status"), int)
        and not isinstance(row.get("http_status"), bool)
        and 200 <= int(row["http_status"]) < 300
        for row in cloud_rows
    )
    non429_http_failure_n = sum(
        row.get("success") is not True
        and isinstance(row.get("http_status"), int)
        and not isinstance(row.get("http_status"), bool)
        and int(row["http_status"]) != 429
        for row in cloud_rows
    )
    hard_stop = (
        any(
            status_counts[str(status)]
            for status in (401, 402, 403, 404, 405, 422)
        )
        or any(
            count
            for status, count in status_counts.items()
            if 300 <= int(status) < 400
        )
    )
    zero_success = cloud_success_n == 0
    dominant_400 = (
        len(cloud_rows) >= 3
        and status_counts["400"] * 5 >= len(cloud_rows) * 4
    )
    dominant_non429_http = (
        len(cloud_rows) >= 10
        and non429_http_failure_n * 5 >= len(cloud_rows) * 4
    )
    if hard_stop or zero_success or dominant_400 or dominant_non429_http:
        raise EvidenceError(f"stage {label}: systematic cloud failure")
    overall_violations = sum(
        not row["success"] or row.get("ttft_ms") is None or float(row["ttft_ms"]) > slo_ms
        for row in rows
    )
    cloud_violations = sum(
        not row["success"] or row.get("ttft_ms") is None or float(row["ttft_ms"]) > slo_ms
        for row in cloud_rows
    )
    return request_ids, cloud_ids, {
        "local_rows": local_rows,
        "cloud_rows": cloud_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "overall_violations": overall_violations,
        "cloud_violations": cloud_violations,
    }


def _audit_decisions(
    decisions: list[dict[str, Any]], label: str, request_ids: set[int], cloud_ids: set[int],
) -> int:
    applied: list[int] = []
    previous_at = -math.inf
    statuses = {
        "applied", "no_op", "stale_retry", "stale_future_arrival",
        "applied_stale_bounded",
    }
    for line_n, decision in enumerate(decisions, 1):
        source = f"stage {label} decisions line {line_n}"
        _reject_payload_keys(decision, source)
        _validate_string_locations(decision, source, DECISION_STRING_KEYS)
        _validate_finite_tree(decision, source)
        if _integer(decision.get("decision_id"), "decision_id", source) != line_n:
            raise EvidenceError(f"{source}: decision IDs must be consecutive")
        if decision.get("trigger") != "ttft_pred" or decision.get("selector") != EXPECTED_SELECTORS[label]:
            raise EvidenceError(f"{source}: trigger/selector mismatch")
        status = decision.get("status")
        if status not in statuses:
            raise EvidenceError(f"{source}: unsupported decision status")
        if decision.get("prediction_scope") != PREDICTION_SCOPE:
            raise EvidenceError(f"{source}: prediction scope mismatch")
        if decision.get("prediction_model") != PREDICTION_MODEL:
            raise EvidenceError(f"{source}: prediction model mismatch")
        at_s = _number(decision.get("at_s"), "at_s", source, minimum=0.0)
        if at_s < previous_at:
            raise EvidenceError(f"{source}: decision time is not monotone")
        previous_at = at_s
        _number(decision.get("decision_ms"), "decision_ms", source, minimum=0.0)
        snapshot_hash = decision.get("snapshot_hash")
        if snapshot_hash is not None and (
            not isinstance(snapshot_hash, str) or not SNAPSHOT_HASH_RE.fullmatch(snapshot_hash)
        ):
            raise EvidenceError(f"{source}: snapshot hash is invalid")
        proposed_value = decision.get("proposed_victim_ids")
        applied_value = decision.get("applied_victim_ids", [])
        if not isinstance(proposed_value, list) or not isinstance(applied_value, list):
            raise EvidenceError(f"{source}: victim IDs must be lists")
        proposed = [_integer(value, "proposed_victim_id", source) for value in proposed_value]
        applied_row = [_integer(value, "applied_victim_id", source) for value in applied_value]
        if len(proposed) != len(set(proposed)) or len(applied_row) != len(set(applied_row)):
            raise EvidenceError(f"{source}: victim list contains duplicates")
        if any(value not in request_ids for value in proposed + applied_row):
            raise EvidenceError(f"{source}: victim ID is outside raw request IDs")
        if status in {"applied", "applied_stale_bounded"}:
            if applied_row != proposed:
                raise EvidenceError(f"{source}: applied victims do not equal proposed victims")
            if status == "applied" and not applied_row:
                raise EvidenceError(f"{source}: applied decision has no victims")
        elif applied_row:
            raise EvidenceError(f"{source}: non-applied decision carries victims")
        if status == "no_op" and proposed:
            raise EvidenceError(f"{source}: no-op decision proposes victims")
        applied.extend(applied_row)
    if not decisions:
        raise EvidenceError(f"stage {label}: decision log is empty")
    if len(applied) != len(set(applied)):
        raise EvidenceError(f"stage {label}: applied victim IDs contain duplicates")
    if set(applied) != cloud_ids:
        raise EvidenceError(f"stage {label}: applied victim IDs do not equal cloud IDs")
    return len(applied)


def _validate_summary_counts(
    summary: dict[str, Any], label: str, rows: list[dict[str, Any]], stats: dict[str, Any],
) -> tuple[int, int]:
    source = f"stage {label} summary"
    local_rows = stats["local_rows"]
    cloud_rows = stats["cloud_rows"]
    slo_ms = _number(summary.get("slo_s"), "slo_s", source) * 1000.0

    def check_side(name: str, side_rows: list[dict[str, Any]]) -> tuple[int, int]:
        obj = _object(summary.get(name), name, source)
        success_n = sum(bool(row["success"]) for row in side_rows)
        violations = sum(
            not row["success"] or row.get("ttft_ms") is None or float(row["ttft_ms"]) > slo_ms
            for row in side_rows
        )
        expected = {
            "n": len(side_rows), "success": success_n, "routed_only": 0,
            "slo_measured_n": len(side_rows), "slo_violations": violations,
        }
        for key, value in expected.items():
            if _integer(obj.get(key), f"{name}.{key}", source) != value:
                raise EvidenceError(f"{source}: {name}.{key} does not match raw")
        expected_pct = 100.0 * violations / max(len(side_rows), 1)
        if not _close(
            _number(obj.get("slo_violation_pct"), f"{name}.slo_violation_pct", source),
            expected_pct,
        ):
            raise EvidenceError(f"{source}: {name} SLO percentage does not match raw")
        errors = collections.Counter(
            str(row["error_type"])
            for row in side_rows
            if row.get("error_type") is not None
        )
        if obj.get("errors") != dict(errors):
            raise EvidenceError(f"{source}: {name} error counts do not match raw")
        return success_n, violations

    overall_success, overall_violations = check_side("overall", rows)
    local_success, _ = check_side("local", local_rows)
    cloud_success, cloud_violations = check_side("cloud", cloud_rows)
    if overall_success != local_success + cloud_success:
        raise EvidenceError(f"{source}: success partitions do not add up")
    if overall_violations != stats["overall_violations"] or cloud_violations != stats["cloud_violations"]:
        raise EvidenceError(f"{source}: failure denominator was not retained")
    return cloud_success, cloud_violations


def _matrix_expectations(manifest: dict[str, Any], identity: TraceIdentity) -> MatrixExpectations:
    return MatrixExpectations(
        trace_n=identity.n,
        prefill_tput=_manifest_number(manifest, "prefill_tput", "manifest"),
        tpot_ms=_manifest_number(manifest, "tpot_ms", "manifest"),
        first_token_overhead_ms=_manifest_number(
            manifest, "first_token_overhead_ms", "manifest"
        ),
        ttft_guard_ms=_manifest_number(manifest, "guard_ms", "manifest"),
        in_price=_manifest_number(manifest, "in_price", "manifest"),
        out_price=_manifest_number(manifest, "out_price", "manifest"),
        slo_s=_manifest_number(manifest, "slo_s", "manifest"),
        timeout_s=_manifest_number(manifest, "timeout_s", "manifest"),
        nimbus_tick_ms=_manifest_number(manifest, "tick_ms", "manifest"),
        max_inflight=_manifest_int(manifest, "max_inflight", "manifest"),
        temperature=_manifest_number(manifest, "temperature", "manifest"),
        ignore_eos=manifest["ignore_eos"] == "1",
        kv_capacity_tokens=_manifest_number(manifest, "kv_cap", "manifest"),
        model=manifest["model"],
        chat_url=manifest["chat_url"],
        scenario=manifest["scenario"],
        cloud="real",
        cloud_url=OPENROUTER_URL,
        cloud_model=CLOUD_MODEL,
        cloud_api_key_env=CLOUD_API_KEY_ENV,
        cloud_max_concurrency=16,
        cloud_provider_order=(CLOUD_PROVIDER_SLUG,),
        cloud_no_fallbacks=True,
        cloud_stop_after_first_token=True,
        local_ignore_eos=True,
        cloud_ignore_eos=False,
    )


def _audit_stage(
    directory: Path,
    label: str,
    identity: TraceIdentity,
    trace_manifest_sha: str,
    budget_sha: str,
    launch_sha: str,
    live_current_usage_sha: str,
    launch_verify_receipt_sha: str,
) -> StageAudit:
    if not directory.is_dir():
        raise EvidenceError(f"stage {label}: directory does not exist")
    manifest = _parse_manifest(directory / "matrix_manifest.txt", f"stage {label} manifest")
    _validate_stage_manifest(
        manifest, label, identity, trace_manifest_sha, budget_sha, launch_sha,
        live_current_usage_sha, launch_verify_receipt_sha,
    )
    markers = sorted(directory.glob("*.complete.json"))
    if len(markers) != 1:
        raise EvidenceError(f"stage {label}: require exactly one completion marker")
    marker_path = markers[0]
    stem = marker_path.name.removesuffix(".complete.json")
    if stem == marker_path.name:
        raise EvidenceError(f"stage {label}: invalid completion marker name")
    raw_path = directory / f"{stem}.jsonl"
    summary_path = directory / f"{stem}.summary.json"
    decisions_path = directory / f"{stem}.decisions.jsonl"
    expected = _matrix_expectations(manifest, identity)
    try:
        validate_or_write_marker(
            raw_path=raw_path,
            summary_path=summary_path,
            decisions_path=decisions_path,
            marker_path=marker_path,
            fingerprint=manifest["run_fingerprint"],
            arm_text=EXPECTED_ARMS[label],
            expected=expected,
            write_marker=False,
        )
    except (MatrixEvidenceError, OSError):
        raise EvidenceError(f"stage {label}: matrix evidence invalid") from None
    events = _validate_events(directory / "matrix_events.log", label, manifest)
    rows = _jsonl(raw_path, f"stage {label} raw")
    summary = _json_object(summary_path, f"stage {label} summary")
    _validate_summary_text_free(summary, label)
    decisions = _jsonl(decisions_path, f"stage {label} decisions")
    request_ids, cloud_ids, stats = _audit_raw_rows(rows, label, manifest, identity)
    applied_n = _audit_decisions(decisions, label, request_ids, cloud_ids)
    cloud_success, cloud_violations = _validate_summary_counts(
        summary, label, rows, stats
    )
    config = _object(summary.get("config"), "config", f"stage {label} summary")
    artifacts = {
        "raw": _artifact(raw_path, f"stage {label} raw"),
        "summary": _artifact(summary_path, f"stage {label} summary"),
        "decisions": _artifact(decisions_path, f"stage {label} decisions"),
    }
    return StageAudit(
        label=label,
        manifest=manifest,
        config=config,
        fingerprint=manifest["run_fingerprint"],
        raw_sha256=artifacts["raw"]["sha256"],
        summary_sha256=artifacts["summary"]["sha256"],
        decisions_sha256=artifacts["decisions"]["sha256"],
        marker_sha256=_sha256(marker_path, f"stage {label} marker"),
        events=events,
        local_n=len(stats["local_rows"]),
        cloud_n=len(stats["cloud_rows"]),
        local_success_n=sum(bool(row["success"]) for row in stats["local_rows"]),
        cloud_success_n=cloud_success,
        cloud_failure_n=len(stats["cloud_rows"]) - cloud_success,
        cloud_slo_violation_n=cloud_violations,
        overall_slo_violation_n=stats["overall_violations"],
        cloud_http_status_counts=stats["status_counts"],
        cloud_error_type_counts=stats["error_counts"],
        decision_n=len(decisions),
        applied_victim_n=applied_n,
    )


MANIFEST_STAGE_DIFFERENCES = frozenset({
    "run_fingerprint", "started_at", "server_log_sha256_at_manifest", "arms",
    "live_stage", "live_exact_arm", "live_stage_launch_attestation_sha256",
    "live_current_usage_sha256",
    "live_stage_launch_verify_receipt_sha256",
})


def _validate_common_stages(a: StageAudit, c: StageAudit) -> None:
    a_keys = set(a.manifest) - MANIFEST_STAGE_DIFFERENCES
    c_keys = set(c.manifest) - MANIFEST_STAGE_DIFFERENCES
    if a_keys != c_keys:
        raise EvidenceError("A/C manifests do not have the same contract fields")
    for key in sorted(a_keys):
        if a.manifest[key] != c.manifest[key]:
            raise EvidenceError("A/C manifest contract fields differ")
    if a.config.keys() != c.config.keys():
        raise EvidenceError("A/C summary config fields differ")
    for key in sorted(a.config):
        if key == "nimbus_selector":
            continue
        if a.config[key] != c.config[key]:
            raise EvidenceError("A/C summary config fields differ")
    if a.config.get("nimbus_selector") != EXPECTED_SELECTORS["A"]:
        raise EvidenceError("A summary selector is not the frozen arm")
    if c.config.get("nimbus_selector") != EXPECTED_SELECTORS["C"]:
        raise EvidenceError("C summary selector is not the frozen arm")
    if a.events.matrix_finished_at > c.events.arm_started_at:
        raise EvidenceError("stage order invalid: C started before A matrix completed")
    observed_cooldown = Decimal(
        str((c.events.arm_started_at - a.events.matrix_finished_at).total_seconds())
    )
    if observed_cooldown < MIN_STAGE_COOLDOWN_S:
        raise EvidenceError("stage order invalid: A-to-C cooldown is below 20 seconds")


USAGE_TOP_FIELDS = frozenset({"schema_version", "captured_at_utc", "key", "account"})
USAGE_KEY_FIELDS = frozenset({
    "http_status", "usage", "limit", "limit_remaining",
    "key_fingerprint_sha256", "limit_reset", "include_byok_in_limit",
    "is_management_key", "is_provisioning_key", "is_free_tier",
    "expires_at_utc",
})
USAGE_ACCOUNT_FIELDS = frozenset({
    "http_status", "total_usage", "total_credits", "remaining_credits",
})
USAGE_ARITHMETIC_TOLERANCE = Decimal("0.0000001")
MIN_SETTLEMENT_INTERVAL = timedelta(seconds=60)


def _nonnegative_decimal(value: Any, field: str, source: str) -> Decimal:
    result = _decimal(value, field, source)
    if result < 0:
        raise EvidenceError(f"{source}: {field} must be nonnegative")
    return result


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _load_usage_snapshot(path: Path, label: str) -> UsageSnapshot:
    source = f"usage snapshot {label}"
    sha = _sha256(path, source)
    payload = _json_object(path, source)
    if set(payload) != USAGE_TOP_FIELDS or payload.get("schema_version") != 1:
        raise EvidenceError(f"{source}: top-level schema mismatch")
    key = _object(payload.get("key"), "key", source)
    account = _object(payload.get("account"), "account", source)
    if set(key) != USAGE_KEY_FIELDS or set(account) != USAGE_ACCOUNT_FIELDS:
        raise EvidenceError(f"{source}: whitelist schema mismatch")
    exact_key = {
        "http_status": 200,
        "limit_reset": None,
        "include_byok_in_limit": True,
        "is_management_key": False,
        "is_provisioning_key": False,
        "is_free_tier": False,
    }
    for field, expected in exact_key.items():
        if key.get(field) != expected:
            raise EvidenceError(f"{source}: paid dedicated-key contract mismatch")
    fingerprint = key.get("key_fingerprint_sha256")
    if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
        raise EvidenceError(f"{source}: key fingerprint is invalid")
    expires_at = key.get("expires_at_utc")
    if expires_at is not None:
        expiry = _parse_iso(expires_at, source)
    else:
        expiry = None
    captured_at = _parse_iso(payload.get("captured_at_utc"), source)
    if expiry is not None and expiry <= captured_at:
        raise EvidenceError(f"{source}: key was expired when captured")

    key_usage = _nonnegative_decimal(key.get("usage"), "key.usage", source)
    key_limit = _nonnegative_decimal(key.get("limit"), "key.limit", source)
    key_remaining = _nonnegative_decimal(
        key.get("limit_remaining"), "key.limit_remaining", source
    )
    if key_limit <= 0 or key_limit > KEY_LIMIT_MAX_USD:
        raise EvidenceError(f"{source}: key limit is not within the exact $3 guard")
    if abs(key_usage + key_remaining - key_limit) > USAGE_ARITHMETIC_TOLERANCE:
        raise EvidenceError(f"{source}: key remaining arithmetic mismatch")

    account_status = _integer(
        account.get("http_status"), "account.http_status", source
    )
    if account_status not in {200, 403}:
        raise EvidenceError(f"{source}: account status is not 200 or safe 403")
    if account_status == 200:
        account_usage = _nonnegative_decimal(
            account.get("total_usage"), "account.total_usage", source
        )
        account_credits = _nonnegative_decimal(
            account.get("total_credits"), "account.total_credits", source
        )
        account_remaining = _nonnegative_decimal(
            account.get("remaining_credits"), "account.remaining_credits", source
        )
        if account_usage + account_remaining != account_credits:
            raise EvidenceError(f"{source}: account remaining arithmetic mismatch")
    else:
        if any(
            account.get(field) is not None
            for field in ("total_usage", "total_credits", "remaining_credits")
        ):
            raise EvidenceError(
                f"{source}: safe account HTTP 403 must carry only null numerics"
            )
        account_usage = account_credits = account_remaining = None
    return UsageSnapshot(
        sha256=sha,
        captured_at=captured_at,
        key_usage=key_usage,
        key_limit=key_limit,
        key_remaining=key_remaining,
        key_fingerprint_sha256=fingerprint,
        expires_at=expiry,
        limit_reset=None,
        include_byok_in_limit=True,
        is_management_key=False,
        is_provisioning_key=False,
        is_free_tier=False,
        account_status=account_status,
        account_usage=account_usage,
        account_credits=account_credits,
        account_remaining=account_remaining,
    )


def _usage_delta(after: Decimal, before: Decimal, label: str) -> Decimal:
    delta = after - before
    if delta < 0:
        raise EvidenceError(f"usage audit: {label} delta is negative")
    return delta


def _same_settled_snapshot(left: UsageSnapshot, right: UsageSnapshot) -> bool:
    return (
        left.key_usage == right.key_usage
        and left.key_limit == right.key_limit
        and left.key_remaining == right.key_remaining
        and left.key_fingerprint_sha256 == right.key_fingerprint_sha256
        and left.expires_at == right.expires_at
        and left.account_status == right.account_status
        and left.account_usage == right.account_usage
        and left.account_credits == right.account_credits
        and left.account_remaining == right.account_remaining
    )


def _validate_final_gate(
    *, gate_path: Path, baseline_path: Path, previous_path: Path,
    current_path: Path, current_at: datetime,
) -> str:
    actual = _json_object(gate_path, "final budget gate")
    try:
        expected = build_stage_gate_attestation(
            baseline_path=baseline_path,
            current_path=current_path,
            settlement_previous_path=previous_path,
            final=True,
            now=current_at,
        )
    except StageGateError:
        raise EvidenceError("final budget gate sources are invalid") from None
    if actual != expected:
        raise EvidenceError("final budget gate is stale or does not bind exact snapshots")
    return _sha256(gate_path, "final budget gate")


def _launch_context(stage: StageAudit) -> LaunchContext:
    manifest = stage.manifest
    try:
        context = LaunchContext(
            commit=manifest["commit"],
            server_pid=_manifest_int(manifest, "server_pid", "stage manifest"),
            server_log_prefix_sha256=manifest["server_log_prefix_sha256"],
            endpoint_version_sha256=manifest["endpoint_version_sha256"],
            endpoint_models_identity_sha256=manifest[
                "endpoint_models_identity_sha256"
            ],
            base_url=manifest["base_url"],
            chat_url=manifest["chat_url"],
            model=manifest["model"],
        )
        context.validate()
    except (KeyError, LaunchCheckError):
        raise EvidenceError("stage launch lifecycle context is invalid") from None
    return context


def _revalidate_launch_attestation(
    *, label: str, attestation_path: Path, trace_manifest_path: Path,
    profile_path: Path, price_snapshot_path: Path, budget_attestation_path: Path,
    canary_path: Path, baseline_usage_path: Path,
    settlement_previous_path: Path, settlement_current_path: Path,
    stage_budget_gate_path: Path, stages: dict[str, StageAudit],
    a_dir: Path | None = None, a_launch_attestation_path: Path | None = None,
) -> dict[str, Any]:
    source = f"stage {label} launch attestation"
    actual = _json_object(attestation_path, source)
    actual_sha = _sha256(attestation_path, source)
    authorized_at = _parse_iso(actual.get("authorized_at_utc"), source)
    try:
        expected = create_launch_attestation(
            stage=label,
            contract_id=stages[label].manifest["live_contract_id"],
            trace_manifest_path=trace_manifest_path,
            profile_path=profile_path,
            price_snapshot_path=price_snapshot_path,
            budget_attestation_path=budget_attestation_path,
            canary_path=canary_path,
            baseline_usage_path=baseline_usage_path,
            settlement_previous_path=settlement_previous_path,
            settlement_current_path=settlement_current_path,
            stage_budget_gate_path=stage_budget_gate_path,
            context=_launch_context(stages[label]),
            a_dir=a_dir,
            a_launch_attestation_path=a_launch_attestation_path,
            now=authorized_at,
        )
    except (LaunchCheckError, KeyError):
        raise EvidenceError(f"stage {label} launch evidence is invalid") from None
    if actual != expected:
        raise EvidenceError(
            f"stage {label} launch attestation is stale or not exactly reproducible"
        )
    if stages[label].manifest["live_stage_launch_attestation_sha256"].lower() != actual_sha:
        raise EvidenceError(f"stage {label} manifest does not bind launch attestation")
    started_at = stages[label].events.arm_started_at
    if not authorized_at <= started_at:
        raise EvidenceError(f"stage {label} started before launch authorization")
    if started_at - authorized_at > timedelta(minutes=10):
        raise EvidenceError(f"stage {label} launch authorization was stale at start")
    bindings = _object(actual.get("bindings"), "bindings", source)
    timing = _object(actual.get("timing"), "timing", source)
    return {
        "sha256": actual_sha,
        "authorized_at": authorized_at,
        "canary_at": _parse_iso(timing.get("canary_at_utc"), source),
        "price_snapshot_sha256": bindings.get("price_snapshot_sha256"),
        "canary_sha256": bindings.get("canary_sha256"),
        "baseline_usage_sha256": bindings.get("baseline_usage_sha256"),
        "settlement_previous_usage_sha256": bindings.get(
            "settlement_previous_usage_sha256"
        ),
        "settlement_current_usage_sha256": bindings.get(
            "settlement_current_usage_sha256"
        ),
        "stage_budget_gate_sha256": bindings.get("stage_budget_gate_sha256"),
    }


def _revalidate_launch_verify_receipt(
    *, label: str, receipt_path: Path, live_current_usage_path: Path,
    launch_attestation_path: Path, stage: StageAudit,
) -> dict[str, Any]:
    """Validate the persisted last-moment quota check without exposing key IDs."""
    expected_receipt_sha = stage.manifest[
        "live_stage_launch_verify_receipt_sha256"
    ]
    try:
        receipt = validate_verify_receipt(
            receipt_path=receipt_path,
            expected_sha256=expected_receipt_sha,
            launch_attestation_path=launch_attestation_path,
            live_current_usage_path=live_current_usage_path,
        )
    except LaunchCheckError:
        raise EvidenceError(
            f"stage {label} persisted launch verification is invalid"
        ) from None
    live = _load_usage_snapshot(live_current_usage_path, f"stage {label} launch")
    receipt_sha = _sha256(receipt_path, f"stage {label} launch verify receipt")
    if live.sha256 != stage.manifest["live_current_usage_sha256"].lower():
        raise EvidenceError(f"stage {label} manifest does not bind live usage")
    if receipt_sha != expected_receipt_sha.lower():
        raise EvidenceError(f"stage {label} manifest does not bind verify receipt")

    authorized_at = _parse_iso(
        receipt.get("authorization_at_utc"),
        f"stage {label} launch verify receipt",
    )
    live_at = _parse_iso(
        receipt.get("live_usage_at_utc"),
        f"stage {label} launch verify receipt",
    )
    verified_at = _parse_iso(
        receipt.get("verified_at_utc"),
        f"stage {label} launch verify receipt",
    )
    if live_at != live.captured_at:
        raise EvidenceError(f"stage {label} receipt live time differs from snapshot")
    if verified_at - live_at > timedelta(minutes=2):
        raise EvidenceError(f"stage {label} live launch usage was stale at verification")
    if not authorized_at <= live_at <= verified_at <= stage.events.arm_started_at:
        raise EvidenceError(f"stage {label} launch verification times are not ordered")
    expires_at = receipt.get("key_expires_at_utc")
    if expires_at is not None:
        expiry = _parse_iso(expires_at, f"stage {label} launch verify receipt")
        if expiry - verified_at < timedelta(hours=6):
            raise EvidenceError(f"stage {label} launch key expires too soon")
    return {
        "receipt_sha256": receipt_sha,
        "live_current_usage_sha256": live.sha256,
        "authorized_at": authorized_at,
        "live_at": live_at,
        "verified_at": verified_at,
        "snapshot": live,
    }


def _audit_usage(
    *, baseline_path: Path, pre_a_previous_path: Path,
    pre_a_current_path: Path, post_a_previous_path: Path,
    post_a_current_path: Path, final_previous_path: Path,
    final_current_path: Path, final_gate_path: Path, canary_at: datetime,
    canary_key_fingerprint_sha256: str, stages: dict[str, StageAudit],
    budget: BudgetAudit, a_launch_usage: UsageSnapshot,
    c_launch_usage: UsageSnapshot,
) -> dict[str, Any]:
    paths = {
        "baseline": baseline_path,
        "pre_A_previous": pre_a_previous_path,
        "pre_A_current": pre_a_current_path,
        "post_A_previous": post_a_previous_path,
        "post_A_current": post_a_current_path,
        "final_previous": final_previous_path,
        "final_current": final_current_path,
    }
    snapshots = {
        label: _load_usage_snapshot(path, label) for label, path in paths.items()
    }
    ordered_labels = tuple(paths)
    ordered = [snapshots[label] for label in ordered_labels]
    if any(
        left.captured_at >= right.captured_at
        for left, right in zip(ordered, ordered[1:])
    ):
        raise EvidenceError("usage audit: snapshot timestamps are not strictly ordered")
    first = snapshots["baseline"]
    if first.key_fingerprint_sha256 != canary_key_fingerprint_sha256:
        raise EvidenceError("usage audit: canary and usage key fingerprints differ")
    for snapshot in ordered[1:]:
        if snapshot.key_fingerprint_sha256 != first.key_fingerprint_sha256:
            raise EvidenceError("usage audit: API key fingerprint changed")
        if snapshot.key_limit != first.key_limit:
            raise EvidenceError("usage audit: key limit changed")
        if snapshot.expires_at != first.expires_at:
            raise EvidenceError("usage audit: key expiry changed")
        if snapshot.account_status != first.account_status:
            raise EvidenceError("usage audit: account metadata availability changed")
        if snapshot.account_credits != first.account_credits:
            raise EvidenceError("usage audit: account credit identity changed")
    for snapshot in (a_launch_usage, c_launch_usage):
        if snapshot.key_fingerprint_sha256 != first.key_fingerprint_sha256:
            raise EvidenceError("usage audit: launch API key fingerprint changed")
        if snapshot.key_limit != first.key_limit:
            raise EvidenceError("usage audit: launch key limit changed")
        if snapshot.expires_at != first.expires_at:
            raise EvidenceError("usage audit: launch key expiry changed")
    if first.expires_at is not None and first.expires_at <= snapshots["final_current"].captured_at:
        raise EvidenceError("usage audit: key expires before final evidence")
    for left, right in zip(ordered, ordered[1:]):
        _usage_delta(right.key_usage, left.key_usage, "key sequence")
        if left.account_usage is not None and right.account_usage is not None:
            _usage_delta(right.account_usage, left.account_usage, "account sequence")
    for previous_label, current_label in (
        ("pre_A_previous", "pre_A_current"),
        ("post_A_previous", "post_A_current"),
        ("final_previous", "final_current"),
    ):
        previous = snapshots[previous_label]
        current = snapshots[current_label]
        if current.captured_at - previous.captured_at < MIN_SETTLEMENT_INTERVAL:
            raise EvidenceError("usage audit: settled pair interval is below 60 seconds")
        if not _same_settled_snapshot(previous, current):
            raise EvidenceError("usage audit: settled pair counters differ")

    a = stages["A"].events
    c = stages["C"].events
    if not (
        first.captured_at <= canary_at <= snapshots["pre_A_previous"].captured_at
        < snapshots["pre_A_current"].captured_at
        <= a_launch_usage.captured_at <= a.arm_started_at
    ):
        raise EvidenceError("usage audit: baseline/canary/pre-A snapshots do not bracket A")
    if not (
        a.matrix_finished_at <= snapshots["post_A_previous"].captured_at
        < snapshots["post_A_current"].captured_at
        <= c_launch_usage.captured_at <= c.arm_started_at
    ):
        raise EvidenceError("usage audit: post-A snapshots do not bracket C")
    if not (
        c.matrix_finished_at <= snapshots["final_previous"].captured_at
        < snapshots["final_current"].captured_at
    ):
        raise EvidenceError("usage audit: final snapshots do not follow C")
    launch_pairs = (
        (snapshots["pre_A_current"], a_launch_usage, "A"),
        (snapshots["post_A_current"], c_launch_usage, "C"),
    )
    for settled, live, label in launch_pairs:
        if (
            settled.key_usage != live.key_usage
            or settled.key_limit != live.key_limit
            or settled.key_remaining != live.key_remaining
            or settled.key_fingerprint_sha256 != live.key_fingerprint_sha256
            or settled.expires_at != live.expires_at
        ):
            raise EvidenceError(
                f"usage audit: stage {label} launch counters differ from settlement"
            )

    final_gate_sha = _validate_final_gate(
        gate_path=final_gate_path,
        baseline_path=baseline_path,
        previous_path=final_previous_path,
        current_path=final_current_path,
        current_at=snapshots["final_current"].captured_at,
    )
    key_canary = _usage_delta(
        snapshots["pre_A_current"].key_usage, first.key_usage, "key canary"
    )
    key_a = _usage_delta(
        snapshots["post_A_current"].key_usage,
        snapshots["pre_A_current"].key_usage,
        "key A",
    )
    key_c = _usage_delta(
        snapshots["final_current"].key_usage,
        snapshots["post_A_current"].key_usage,
        "key C",
    )
    key_total = _usage_delta(
        snapshots["final_current"].key_usage, first.key_usage, "key total"
    )
    if key_canary + key_a + key_c != key_total:
        raise EvidenceError("usage audit: key stage deltas do not add to total")

    account_available = first.account_status == 200
    if account_available:
        assert first.account_usage is not None
        pre_a_account = snapshots["pre_A_current"].account_usage
        post_a_account = snapshots["post_A_current"].account_usage
        final_account = snapshots["final_current"].account_usage
        assert pre_a_account is not None and post_a_account is not None
        assert final_account is not None
        account_canary: Decimal | None = _usage_delta(
            pre_a_account, first.account_usage, "account canary"
        )
        account_a: Decimal | None = _usage_delta(
            post_a_account, pre_a_account, "account A"
        )
        account_c: Decimal | None = _usage_delta(
            final_account, post_a_account, "account C"
        )
        account_total: Decimal | None = _usage_delta(
            final_account, first.account_usage, "account total"
        )
        if account_canary + account_a + account_c != account_total:
            raise EvidenceError("usage audit: account stage deltas do not add to total")
    else:
        account_canary = account_a = account_c = account_total = None
    limit = min(budget.budget_usd, AUTHORIZED_BUDGET_USD)
    if key_total > limit or (account_total is not None and account_total > limit):
        raise EvidenceError("usage audit: observed total usage exceeds $3 authorization")
    if first.key_remaining < budget.estimated_cost_usd:
        raise EvidenceError("usage audit: baseline key headroom is insufficient")
    if account_available:
        assert first.account_remaining is not None
        if first.account_remaining < budget.estimated_cost_usd:
            raise EvidenceError("usage audit: baseline account headroom is insufficient")

    def maybe_text(value: Decimal | None) -> str | None:
        return None if value is None else _decimal_text(value)

    return {
        "snapshot_sha256": {
            label: snapshots[label].sha256 for label in ordered_labels
        },
        "final_gate_sha256": final_gate_sha,
        "captured_at_utc": {
            label: _iso(snapshots[label].captured_at) for label in ordered_labels
        },
        "key_limit_usd": _decimal_text(first.key_limit),
        "same_key_fingerprint_verified": True,
        "no_reset_verified": True,
        "byok_in_limit_verified": True,
        "paid_dedicated_inference_key_verified": True,
        "key_valid_through_final": True,
        "all_settled_pairs_verified": True,
        "launch_time_usage_verified": True,
        "account_balance_verified": account_available,
        "deltas_usd": {
            "key": {
                "canary": _decimal_text(key_canary),
                "A": _decimal_text(key_a),
                "C": _decimal_text(key_c),
                "total": _decimal_text(key_total),
            },
            "account": {
                "canary": maybe_text(account_canary),
                "A": maybe_text(account_a),
                "C": maybe_text(account_c),
                "total": maybe_text(account_total),
            },
        },
        "total_within_authorized_budget": True,
    }


def audit(
    *,
    a_dir: str | Path,
    c_dir: str | Path,
    trace_manifest: str | Path,
    profile: str | Path,
    price_snapshot: str | Path,
    budget_attestation: str | Path,
    canary_attestation: str | Path,
    a_launch_attestation: str | Path,
    c_launch_attestation: str | Path,
    a_live_current_usage: str | Path,
    a_launch_verify_receipt: str | Path,
    c_live_current_usage: str | Path,
    c_launch_verify_receipt: str | Path,
    usage_baseline: str | Path,
    usage_pre_a_previous: str | Path,
    usage_pre_a_current: str | Path,
    pre_a_gate: str | Path,
    usage_post_a_previous: str | Path,
    usage_post_a_current: str | Path,
    pre_c_gate: str | Path,
    usage_final_previous: str | Path,
    usage_final_current: str | Path,
    final_gate: str | Path,
    trace_identity: TraceIdentity,
) -> dict[str, Any]:
    """Audit all E12 live artifacts without writing files or using network I/O."""
    if trace_identity.n <= 0:
        raise EvidenceError("trace identity N must be positive")
    for value, label in (
        (trace_identity.trace_sha256, "trace SHA256"),
        (trace_identity.manifest_sha256, "trace manifest SHA256"),
    ):
        if not SHA256_RE.fullmatch(value):
            raise EvidenceError(f"{label} is invalid")
    trace_payload, trace_manifest_sha = _validate_trace_manifest(
        Path(trace_manifest), trace_identity
    )
    budget = _validate_budget_attestation(
        Path(budget_attestation), trace_identity, trace_manifest_sha
    )
    price_snapshot_sha = _sha256(Path(price_snapshot), "price snapshot")
    if price_snapshot_sha != budget.price_snapshot_sha256:
        raise EvidenceError("price snapshot SHA256 does not match budget attestation")
    launch_paths = {
        "A": Path(a_launch_attestation),
        "C": Path(c_launch_attestation),
    }
    launch_shas = {
        label: _sha256(path, f"stage {label} launch attestation")
        for label, path in launch_paths.items()
    }
    live_current_usage_paths = {
        "A": Path(a_live_current_usage),
        "C": Path(c_live_current_usage),
    }
    launch_verify_receipt_paths = {
        "A": Path(a_launch_verify_receipt),
        "C": Path(c_launch_verify_receipt),
    }
    live_current_usage_shas = {
        label: _sha256(path, f"stage {label} live current usage")
        for label, path in live_current_usage_paths.items()
    }
    launch_verify_receipt_shas = {
        label: _sha256(path, f"stage {label} launch verify receipt")
        for label, path in launch_verify_receipt_paths.items()
    }
    stages = {
        "A": _audit_stage(
            Path(a_dir), "A", trace_identity, trace_manifest_sha, budget.sha256,
            launch_shas["A"], live_current_usage_shas["A"],
            launch_verify_receipt_shas["A"],
        ),
        "C": _audit_stage(
            Path(c_dir), "C", trace_identity, trace_manifest_sha, budget.sha256,
            launch_shas["C"], live_current_usage_shas["C"],
            launch_verify_receipt_shas["C"],
        ),
    }
    _validate_common_stages(stages["A"], stages["C"])
    profile_sha = _validate_profile(Path(profile), trace_payload, stages["A"].manifest)
    if profile_sha != stages["C"].manifest["profile_sha256"]:
        raise EvidenceError("C manifest does not bind the same profile")
    launches = {
        "A": _revalidate_launch_attestation(
            label="A",
            attestation_path=launch_paths["A"],
            trace_manifest_path=Path(trace_manifest),
            profile_path=Path(profile),
            price_snapshot_path=Path(price_snapshot),
            budget_attestation_path=Path(budget_attestation),
            canary_path=Path(canary_attestation),
            baseline_usage_path=Path(usage_baseline),
            settlement_previous_path=Path(usage_pre_a_previous),
            settlement_current_path=Path(usage_pre_a_current),
            stage_budget_gate_path=Path(pre_a_gate),
            stages=stages,
        ),
        "C": _revalidate_launch_attestation(
            label="C",
            attestation_path=launch_paths["C"],
            trace_manifest_path=Path(trace_manifest),
            profile_path=Path(profile),
            price_snapshot_path=Path(price_snapshot),
            budget_attestation_path=Path(budget_attestation),
            canary_path=Path(canary_attestation),
            baseline_usage_path=Path(usage_baseline),
            settlement_previous_path=Path(usage_post_a_previous),
            settlement_current_path=Path(usage_post_a_current),
            stage_budget_gate_path=Path(pre_c_gate),
            stages=stages,
            a_dir=Path(a_dir),
            a_launch_attestation_path=launch_paths["A"],
        ),
    }
    if any(
        launch["price_snapshot_sha256"] != price_snapshot_sha
        for launch in launches.values()
    ):
        raise EvidenceError("stage launch does not bind the budget price snapshot")
    canary_sha = _sha256(Path(canary_attestation), "synthetic canary")
    if any(launch["canary_sha256"] != canary_sha for launch in launches.values()):
        raise EvidenceError("stage launch does not bind the same synthetic canary")
    if launches["A"]["canary_at"] != launches["C"]["canary_at"]:
        raise EvidenceError("stage launches do not bind the same canary time")
    launch_verifications = {
        label: _revalidate_launch_verify_receipt(
            label=label,
            receipt_path=launch_verify_receipt_paths[label],
            live_current_usage_path=live_current_usage_paths[label],
            launch_attestation_path=launch_paths[label],
            stage=stages[label],
        )
        for label in ("A", "C")
    }
    for label in ("A", "C"):
        if launch_verifications[label]["authorized_at"] != launches[label][
            "authorized_at"
        ]:
            raise EvidenceError(
                f"stage {label} verify receipt authorization differs from launch"
            )
    canary_payload = _json_object(Path(canary_attestation), "synthetic canary")
    canary_key_fingerprint = canary_payload.get("key_fingerprint_sha256")
    if (
        not isinstance(canary_key_fingerprint, str)
        or not SHA256_RE.fullmatch(canary_key_fingerprint)
    ):
        raise EvidenceError("synthetic canary key fingerprint is invalid")
    usage = _audit_usage(
        baseline_path=Path(usage_baseline),
        pre_a_previous_path=Path(usage_pre_a_previous),
        pre_a_current_path=Path(usage_pre_a_current),
        post_a_previous_path=Path(usage_post_a_previous),
        post_a_current_path=Path(usage_post_a_current),
        final_previous_path=Path(usage_final_previous),
        final_current_path=Path(usage_final_current),
        final_gate_path=Path(final_gate),
        canary_at=launches["A"]["canary_at"],
        canary_key_fingerprint_sha256=canary_key_fingerprint,
        stages=stages,
        budget=budget,
        a_launch_usage=launch_verifications["A"]["snapshot"],
        c_launch_usage=launch_verifications["C"]["snapshot"],
    )
    result = {
        "schema_version": 1,
        "analysis": "E12 live ShareGPT current-turn A/C audit",
        "integrity_valid": True,
        "text_payload_in_output": False,
        "contract": {
            "id": stages["A"].manifest["live_contract_id"],
            "name": LIVE_CONTRACT,
            "stage_order": ["A", "C"],
            "authorized_budget_usd": "3",
            "key_limit_max_usd": "3",
        },
        "evidence": {
            "trace_n": trace_identity.n,
            "trace_sha256": trace_identity.trace_sha256,
            "trace_manifest_sha256": trace_manifest_sha,
            "profile_sha256": profile_sha,
            "budget_attestation_sha256": budget.sha256,
            "price_snapshot_sha256": price_snapshot_sha,
            "request_price_source": budget.request_price_source,
            "canary_attestation_sha256": canary_sha,
            "stage_launch_attestation_sha256": {
                label: launches[label]["sha256"] for label in ("A", "C")
            },
            "live_current_usage_sha256": {
                label: launch_verifications[label][
                    "live_current_usage_sha256"
                ]
                for label in ("A", "C")
            },
            "stage_launch_verify_receipt_sha256": {
                label: launch_verifications[label]["receipt_sha256"]
                for label in ("A", "C")
            },
            "stage_budget_gate_sha256": {
                "pre_A": launches["A"]["stage_budget_gate_sha256"],
                "pre_C": launches["C"]["stage_budget_gate_sha256"],
                "final": usage["final_gate_sha256"],
            },
            "commit": stages["A"].manifest["commit"],
            "server_pid": _manifest_int(stages["A"].manifest, "server_pid", "manifest"),
            "server_log_prefix_sha256": stages["A"].manifest[
                "server_log_prefix_sha256"
            ],
            "common_lifecycle_and_config": True,
            "request_ids_exact": True,
            "marker_hashes_and_fingerprints_valid": True,
            "launch_chain_revalidated": True,
            "launch_time_usage_receipts_revalidated": True,
        },
        "stages": {
            label: {
                "arm": EXPECTED_ARMS[label],
                "run_fingerprint": stage.fingerprint,
                "artifacts": {
                    "raw_sha256": stage.raw_sha256,
                    "summary_sha256": stage.summary_sha256,
                    "decisions_sha256": stage.decisions_sha256,
                    "marker_sha256": stage.marker_sha256,
                    "events_sha256": stage.events.sha256,
                },
                "timing": {
                    "launch_authorized_at": _iso(
                        launches[label]["authorized_at"]
                    ),
                    "launch_usage_at": _iso(
                        launch_verifications[label]["live_at"]
                    ),
                    "launch_verified_at": _iso(
                        launch_verifications[label]["verified_at"]
                    ),
                    "arm_started_at": _iso(stage.events.arm_started_at),
                    "arm_finished_at": _iso(stage.events.arm_finished_at),
                    "matrix_finished_at": _iso(stage.events.matrix_finished_at),
                },
                "n": trace_identity.n,
                "local_n": stage.local_n,
                "cloud_n": stage.cloud_n,
                "local_success_n": stage.local_success_n,
                "cloud_success_n": stage.cloud_success_n,
                "cloud_failure_n": stage.cloud_failure_n,
                "cloud_slo_violation_n": stage.cloud_slo_violation_n,
                "overall_slo_violation_n": stage.overall_slo_violation_n,
                "cloud_http_status_counts": stage.cloud_http_status_counts,
                "cloud_error_type_counts": stage.cloud_error_type_counts,
                "decision_n": stage.decision_n,
                "applied_victim_n": stage.applied_victim_n,
                "applied_victims_equal_cloud_ids": True,
                "local_token_exact": True,
            }
            for label, stage in stages.items()
        },
        "usage": usage,
        "verdict": "pass",
    }
    # Defense in depth: output is assembled only from whitelisted aggregates,
    # but never let future edits accidentally add one of the payload field names.
    _reject_payload_keys(result, "audit output")
    return result


def render_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--c-dir", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--price-snapshot", type=Path, required=True)
    parser.add_argument("--budget-attestation", type=Path, required=True)
    parser.add_argument("--canary-attestation", type=Path, required=True)
    parser.add_argument("--a-launch-attestation", type=Path, required=True)
    parser.add_argument("--c-launch-attestation", type=Path, required=True)
    parser.add_argument("--a-live-current-usage", type=Path, required=True)
    parser.add_argument("--a-launch-verify-receipt", type=Path, required=True)
    parser.add_argument("--c-live-current-usage", type=Path, required=True)
    parser.add_argument("--c-launch-verify-receipt", type=Path, required=True)
    parser.add_argument("--usage-baseline", type=Path, required=True)
    parser.add_argument("--usage-pre-a-previous", type=Path, required=True)
    parser.add_argument("--usage-pre-a-current", type=Path, required=True)
    parser.add_argument("--pre-a-gate", type=Path, required=True)
    parser.add_argument("--usage-post-a-previous", type=Path, required=True)
    parser.add_argument("--usage-post-a-current", type=Path, required=True)
    parser.add_argument("--pre-c-gate", type=Path, required=True)
    parser.add_argument("--usage-final-previous", type=Path, required=True)
    parser.add_argument("--usage-final-current", type=Path, required=True)
    parser.add_argument("--final-gate", type=Path, required=True)
    parser.add_argument(
        "--expected-trace-manifest-sha256",
        required=True,
        help=(
            "SHA256 of the post-checkout, text-free materialization manifest; "
            "required so a stale dependency hash cannot be silently accepted"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        trace_identity = TraceIdentity(
            n=TRACE_N,
            trace_sha256=TRACE_SHA256,
            manifest_sha256=args.expected_trace_manifest_sha256,
            prompt_token_sum=PROMPT_TOKEN_SUM,
            decode_token_sum=DECODE_TOKEN_SUM,
        )
        result = audit(
            a_dir=args.a_dir,
            c_dir=args.c_dir,
            trace_manifest=args.trace_manifest,
            profile=args.profile,
            price_snapshot=args.price_snapshot,
            budget_attestation=args.budget_attestation,
            canary_attestation=args.canary_attestation,
            a_launch_attestation=args.a_launch_attestation,
            c_launch_attestation=args.c_launch_attestation,
            a_live_current_usage=args.a_live_current_usage,
            a_launch_verify_receipt=args.a_launch_verify_receipt,
            c_live_current_usage=args.c_live_current_usage,
            c_launch_verify_receipt=args.c_launch_verify_receipt,
            usage_baseline=args.usage_baseline,
            usage_pre_a_previous=args.usage_pre_a_previous,
            usage_pre_a_current=args.usage_pre_a_current,
            pre_a_gate=args.pre_a_gate,
            usage_post_a_previous=args.usage_post_a_previous,
            usage_post_a_current=args.usage_post_a_current,
            pre_c_gate=args.pre_c_gate,
            usage_final_previous=args.usage_final_previous,
            usage_final_current=args.usage_final_current,
            final_gate=args.final_gate,
            trace_identity=trace_identity,
        )
    except EvidenceError as exc:
        print(f"E12 live evidence invalid: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(render_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
