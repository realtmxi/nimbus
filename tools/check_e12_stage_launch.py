#!/usr/bin/env python3
"""Create or verify a fail-closed, text-free E12 stage launch attestation.

``create`` is deliberately offline.  It validates the immutable E12 trace,
profile, pricing, full-pair budget, synthetic canary, a settled usage pair,
and the exact before-stage budget gate.  Stage C additionally validates the
completed stage-A matrix and its stage-launch authorization.

``verify`` binds that attestation to the matrix runner's current preflight and
compares a freshly captured, secret-free ``/api/v1/key`` snapshot with the
settled key identity and counters authorized by ``create``.  This module never
reads an API key and never writes paths, labels, prompts, provider bodies, or
plaintext credentials to its output.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_e12_live_budget import (
    ARM_COUNT,
    AUTHORIZED_BUDGET_USD,
    CACHE_MODE,
    DECODE_TOKEN_SUM,
    INPUT_PRICE_PER_MILLION_USD,
    OUTPUT_PRICE_PER_MILLION_USD,
    PAYLOAD_MODE,
    PROMPT_TOKEN_SUM,
    TRACE_N,
    TRACE_SHA256,
    build_budget_attestation,
)
from tools.check_openrouter_stage_budget import (
    ARITHMETIC_TOLERANCE,
    StageGateError,
    build_stage_gate_attestation,
)
from tools.ttft_matrix_evidence import (
    MatrixEvidenceError,
    MatrixExpectations,
    validate_or_write_marker,
)


LIVE_CONTRACT = "e12_current_turn_v1"
EXPECTED_ARMS = {
    "A": "ttft_pred:cost_cachedisp_old:0",
    "C": "ttft_pred:cost_disp_current:0",
}
EXPECTED_SELECTORS = {"A": "cost_cachedisp_old", "C": "cost_disp_current"}
STAGE_FUTURE_BOUND_USD = {
    "A": Decimal("1.90803056"),
    "C": Decimal("0.95401528"),
}
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "qwen/qwen3-32b"
OPENROUTER_PROVIDER = "deepinfra"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
ARM_ORDER_SEED = "20260716"
SLO_S = Decimal("5")
TIMEOUT_S = Decimal("600")
TICK_MS = Decimal("250")
COOLDOWN_S = Decimal("20")
MAX_INFLIGHT = 128
TEMPERATURE = Decimal("0")
MIN_SETTLEMENT_S = Decimal("60")
MAX_ATTESTATION_AGE = timedelta(minutes=10)
MAX_JSON_BYTES = 2_000_000
LIVE_CURRENT_USAGE_FILENAME = "e12_live_current_usage.json"
VERIFY_RECEIPT_FILENAME = "e12_stage_launch_verify_receipt.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
CONTRACT_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")

CANARY_FIELDS = frozenset({
    "schema_version", "status", "captured_at_utc", "request_count",
    "retry_count", "trace_data_used", "http_status", "ttft_ms",
    "provider", "response_model", "first_token_kind",
    "stream_abort_requested", "response_completed", "cost_pending",
    "generation_id_sha256", "key_fingerprint_sha256",
})
USAGE_KEY_FIELDS = frozenset({
    "http_status", "usage", "limit", "limit_remaining", "limit_reset",
    "key_fingerprint_sha256", "include_byok_in_limit", "is_management_key",
    "is_provisioning_key", "is_free_tier", "expires_at_utc",
})
USAGE_ACCOUNT_FIELDS = frozenset({
    "http_status", "total_usage", "total_credits", "remaining_credits",
})
USAGE_TOP_LEVEL_FIELDS = frozenset({
    "schema_version", "captured_at_utc", "key", "account",
})

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

VERIFY_RECEIPT_FIELDS = frozenset({
    "schema_version", "status", "stage", "attestation_sha256",
    "live_current_usage_sha256", "authorization_at_utc",
    "settled_usage_at_utc", "live_usage_at_utc", "verified_at_utc",
    "key_fingerprint_sha256", "key_limit_usd", "key_usage_usd",
    "key_remaining_usd", "key_expires_at_utc",
})

TOP_LEVEL_FIELDS = frozenset({
    "schema_version", "status", "stage", "authorized_at_utc",
    "contract_id_sha256", "bindings", "frozen_contract", "settled_key",
    "timing", "prior_a",
})
BINDING_FIELDS = frozenset({
    "trace_manifest_sha256", "trace_sha256", "trace_n",
    "prompt_token_sum", "decode_token_sum", "profile_sha256",
    "price_snapshot_sha256", "budget_attestation_sha256", "canary_sha256",
    "baseline_usage_sha256", "settlement_previous_usage_sha256",
    "settlement_current_usage_sha256", "stage_budget_gate_sha256",
    "commit", "lifecycle_sha256",
})
FROZEN_FIELDS = frozenset({
    "input_price_per_million_usd", "output_price_per_million_usd",
    "request_price_per_request_usd", "authorized_budget_usd",
    "stage_future_bound_usd", "slo_s",
    "timeout_s", "tick_ms", "cooldown_s", "max_inflight",
    "temperature", "cloud_max_concurrency",
})
SETTLED_KEY_FIELDS = frozenset({
    "key_fingerprint_sha256", "limit_usd", "usage_usd", "remaining_usd",
    "expires_at_utc",
})
TIMING_FIELDS = frozenset({
    "baseline_usage_at_utc", "canary_at_utc",
    "settlement_previous_at_utc", "settlement_current_at_utc",
    "settlement_interval_s", "minimum_settlement_interval_s",
    "minimum_a_to_c_authorization_interval_s",
})
PRIOR_A_FIELDS = frozenset({
    "launch_attestation_sha256", "matrix_manifest_sha256",
    "completion_marker_sha256", "raw_sha256", "raw_n", "summary_sha256",
    "decisions_sha256", "decision_n", "events_sha256", "run_fingerprint",
    "matrix_finished_at_utc", "authorization_interval_s",
    "key_fingerprint_sha256", "expires_at_utc",
    "live_current_usage_sha256", "launch_verify_receipt_sha256",
})


class LaunchCheckError(ValueError):
    """A sanitized E12 launch-evidence failure."""


@dataclass(frozen=True)
class LaunchContext:
    commit: str
    server_pid: int
    server_log_prefix_sha256: str
    endpoint_version_sha256: str
    endpoint_models_identity_sha256: str
    base_url: str
    chat_url: str
    model: str

    def validate(self) -> None:
        if not COMMIT_RE.fullmatch(self.commit):
            raise LaunchCheckError("commit must be a lowercase git object hash")
        if isinstance(self.server_pid, bool) or self.server_pid <= 0:
            raise LaunchCheckError("server PID must be a positive integer")
        for field, value in (
            ("server log prefix", self.server_log_prefix_sha256),
            ("endpoint version", self.endpoint_version_sha256),
            ("endpoint model identity", self.endpoint_models_identity_sha256),
        ):
            _sha256(value, field)
        for field, value in (
            ("base URL", self.base_url),
            ("chat URL", self.chat_url),
            ("model", self.model),
        ):
            if not isinstance(value, str) or not value or any(
                character.isspace() for character in value
            ):
                raise LaunchCheckError(f"{field} must be a nonempty token")

    @property
    def lifecycle_sha256(self) -> str:
        self.validate()
        return _canonical_sha256([
            str(self.server_pid),
            self.server_log_prefix_sha256,
            self.endpoint_version_sha256,
            self.endpoint_models_identity_sha256,
            self.base_url.rstrip("/"),
            self.chat_url,
            self.model,
        ])


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path, label: str) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        raise LaunchCheckError(f"unable to read {label}") from None


def _validate_private_regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        raise LaunchCheckError(f"unable to read {label}") from None
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise LaunchCheckError(f"{label} must be a mode-0600 regular file")


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_JSON_BYTES + 1)
    except OSError:
        raise LaunchCheckError(f"unable to read {label}") from None
    if len(raw) > MAX_JSON_BYTES:
        raise LaunchCheckError(f"{label} is too large")
    try:
        payload = json.loads(
            raw,
            parse_float=Decimal,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise LaunchCheckError(f"{label} is not valid finite UTF-8 JSON") from None
    if not isinstance(payload, dict):
        raise LaunchCheckError(f"{label} root must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise LaunchCheckError(f"{label} must be a lowercase SHA256")
    return value


def _decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise LaunchCheckError(f"{label} must be a finite decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise LaunchCheckError(f"{label} must be a finite decimal") from None
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise LaunchCheckError(f"{label} must be a finite nonnegative decimal")
    return result


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LaunchCheckError(f"{label} must be an integer >= {minimum}")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise LaunchCheckError(f"{label} must be a UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise LaunchCheckError(f"{label} must be a UTC Z timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise LaunchCheckError(f"{label} must be a UTC Z timestamp")
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now(value: datetime | None) -> datetime:
    current = datetime.now(timezone.utc) if value is None else value
    if current.tzinfo is None or current.utcoffset() is None:
        raise LaunchCheckError("current time must be timezone-aware")
    return current.astimezone(timezone.utc)


def _contract_id_hash(contract_id: str) -> str:
    if not isinstance(contract_id, str) or not CONTRACT_ID_RE.fullmatch(contract_id):
        raise LaunchCheckError("contract id is invalid")
    return hashlib.sha256(contract_id.encode("utf-8")).hexdigest()


def _validate_trace_manifest(payload: dict[str, Any], sha: str) -> None:
    exact = {
        "schema_version": 1,
        "output_sha256": TRACE_SHA256,
        "n": TRACE_N,
        "payload_mode": PAYLOAD_MODE,
        "cache_mode": CACHE_MODE,
        "semantic_scope": "verbatim_current_user_turn_only",
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise LaunchCheckError(f"trace manifest {key} is not frozen E12")
    for field, expected_sum in (
        ("actual_prompt_tokens", PROMPT_TOKEN_SUM),
        ("output_decode_tokens", DECODE_TOKEN_SUM),
    ):
        aggregate = payload.get(field)
        if not isinstance(aggregate, dict):
            raise LaunchCheckError(f"trace manifest {field} is missing")
        if aggregate.get("n") != TRACE_N or aggregate.get("sum") != expected_sum:
            raise LaunchCheckError(f"trace manifest {field} identity differs")
    preservation = payload.get("preservation_checks")
    if not isinstance(preservation, dict):
        raise LaunchCheckError("trace manifest preservation checks are missing")
    for field in (
        "prompt_text_exact_n", "arrived_at_exact_n",
        "decode_source_provenance_n", "token_metadata_aligned_n",
    ):
        if preservation.get(field) != TRACE_N:
            raise LaunchCheckError("trace manifest preservation check is incomplete")
    _sha256(sha, "trace manifest SHA256")


def _validate_profile(
    profile: dict[str, Any], *, trace_manifest: dict[str, Any], context: LaunchContext
) -> None:
    exact = {
        "schema_version": 2,
        "predictor_model": "seq_slots_shared_prefill_lane_v1",
        "valid": True,
        "cache_mode_required": CACHE_MODE,
        "model": context.model,
        "server_pid": context.server_pid,
        "server_log_sha256_at_start": context.server_log_prefix_sha256,
    }
    for key, expected in exact.items():
        if profile.get(key) != expected:
            raise LaunchCheckError(f"profile {key} does not match launch lifecycle")
    if profile.get("tokenizer_fingerprint") != trace_manifest.get(
        "tokenizer_fingerprint"
    ):
        raise LaunchCheckError("profile tokenizer differs from trace")
    if str(profile.get("base_url", "")).rstrip("/") != context.base_url.rstrip("/"):
        raise LaunchCheckError("profile base URL differs from launch lifecycle")
    _sha256(profile.get("server_proc_cmdline_sha256_at_start"), "profile process")
    sampling = profile.get("sampling")
    if (
        not isinstance(sampling, dict)
        or sampling.get("ignore_eos") is not True
        or sampling.get("continuous_usage_stats") is not True
    ):
        raise LaunchCheckError("profile local ignore_eos is not frozen")
    if _decimal(sampling.get("temperature"), "profile temperature") != TEMPERATURE:
        raise LaunchCheckError("profile temperature is not frozen")
    config = profile.get("profile_config")
    if not isinstance(config, dict):
        raise LaunchCheckError("profile config is missing")
    if _decimal(config.get("nimbus_tick_ms"), "profile tick") != TICK_MS:
        raise LaunchCheckError("profile tick is not frozen")
    profile_slo = config.get("slo_s", config.get("target_slo_s"))
    if _decimal(profile_slo, "profile SLO") != SLO_S:
        raise LaunchCheckError("profile SLO is not frozen")
    calibration = profile.get("predictor_calibration")
    if not isinstance(calibration, dict):
        raise LaunchCheckError("profile calibration is missing")
    if _decimal(calibration.get("target_slo_s"), "calibration SLO") != SLO_S:
        raise LaunchCheckError("calibration SLO is not frozen")
    heldout = calibration.get("heldout_violation_confusion")
    if not isinstance(heldout, dict) or heldout.get("false_negative") != 0:
        raise LaunchCheckError("profile calibration has held-out false negatives")


def _validate_canary(path: Path) -> tuple[str, datetime, Decimal, str]:
    payload, sha = _load_json(path, "synthetic canary")
    if set(payload) != CANARY_FIELDS:
        raise LaunchCheckError("synthetic canary fields differ from schema")
    exact = {
        "schema_version": 1,
        "status": "pass",
        "request_count": 1,
        "retry_count": 0,
        "trace_data_used": False,
        "http_status": 200,
        "provider": OPENROUTER_PROVIDER,
        "response_model": OPENROUTER_MODEL,
        "stream_abort_requested": True,
        "response_completed": False,
        "cost_pending": True,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise LaunchCheckError(f"synthetic canary {key} is invalid")
    if payload.get("first_token_kind") not in {"content", "reasoning"}:
        raise LaunchCheckError("synthetic canary first-token kind is invalid")
    generation = payload.get("generation_id_sha256")
    if generation is not None:
        _sha256(generation, "synthetic canary generation id")
    ttft = _decimal(payload.get("ttft_ms"), "synthetic canary TTFT")
    key_fingerprint = _sha256(
        payload.get("key_fingerprint_sha256"),
        "synthetic canary key fingerprint",
    )
    return (
        sha,
        _timestamp(payload.get("captured_at_utc"), "canary time"),
        ttft,
        key_fingerprint,
    )


def _usage_key(payload: dict[str, Any], label: str) -> dict[str, Any]:
    if set(payload) != USAGE_TOP_LEVEL_FIELDS or payload.get("schema_version") != 1:
        raise LaunchCheckError(f"{label} schema is invalid")
    key = payload.get("key")
    account = payload.get("account")
    if not isinstance(key, dict) or set(key) != USAGE_KEY_FIELDS:
        raise LaunchCheckError(f"{label} key fields differ from E12 schema")
    if not isinstance(account, dict) or set(account) != USAGE_ACCOUNT_FIELDS:
        raise LaunchCheckError(f"{label} account fields differ from E12 schema")
    exact = {
        "http_status": 200,
        "limit_reset": None,
        "include_byok_in_limit": True,
        "is_management_key": False,
        "is_provisioning_key": False,
        "is_free_tier": False,
    }
    for field, expected in exact.items():
        if key.get(field) != expected:
            raise LaunchCheckError(f"{label} key contract is invalid")
    fingerprint = _sha256(key.get("key_fingerprint_sha256"), f"{label} key fingerprint")
    usage = _decimal(key.get("usage"), f"{label} usage")
    limit = _decimal(key.get("limit"), f"{label} limit", positive=True)
    remaining = _decimal(key.get("limit_remaining"), f"{label} remaining")
    if limit > AUTHORIZED_BUDGET_USD or remaining > limit:
        raise LaunchCheckError(f"{label} key limit is invalid")
    if abs(usage + remaining - limit) > ARITHMETIC_TOLERANCE:
        raise LaunchCheckError(f"{label} key arithmetic is inconsistent")
    expires = key.get("expires_at_utc")
    if expires is not None:
        expires = _timestamp_text(_timestamp(expires, f"{label} key expiration"))
    return {
        "usage": usage,
        "limit": limit,
        "remaining": remaining,
        "fingerprint": fingerprint,
        "expires_at_utc": expires,
    }


def _parse_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        raise LaunchCheckError("unable to read stage A matrix manifest") from None
    result: dict[str, Any] = {}
    remainder_fields = {"python", "evidence_scope", "arms"}
    for line in lines:
        if not line.strip():
            continue
        first_key = line.split("=", 1)[0]
        tokens = [line] if first_key in remainder_fields else line.split()
        for token in tokens:
            if "=" not in token:
                raise LaunchCheckError("stage A matrix manifest is malformed")
            key, value = token.split("=", 1)
            if not key or key in result:
                raise LaunchCheckError("stage A matrix manifest has duplicate fields")
            result[key] = tuple(value.split()) if key == "arms" else value
    return result, hashlib.sha256(raw).hexdigest()


def _manifest_decimal(manifest: dict[str, Any], key: str) -> Decimal:
    return _decimal(manifest.get(key), f"stage A manifest {key}")


def _manifest_int(manifest: dict[str, Any], key: str) -> int:
    value = _manifest_decimal(manifest, key)
    if value != value.to_integral_value():
        raise LaunchCheckError(f"stage A manifest {key} must be integral")
    return int(value)


def recompute_matrix_fingerprint(manifest: dict[str, Any]) -> str:
    """Recompute the runner fingerprint including E12 launch authorization."""
    try:
        values = [manifest[key] for key in FINGERPRINT_BASE_KEYS]
        arms = manifest["arms"]
        extras = [manifest[key] for key in FINGERPRINT_LIVE_KEYS]
    except KeyError as exc:
        raise LaunchCheckError(
            f"stage A manifest lacks fingerprint field {exc.args[0]}"
        ) from None
    if not all(isinstance(value, str) for value in values + extras):
        raise LaunchCheckError("stage A fingerprint fields must be strings")
    if not isinstance(arms, tuple) or not all(isinstance(x, str) for x in arms):
        raise LaunchCheckError("stage A arms field is invalid")
    return _canonical_sha256(values + list(arms) + extras)


def _validate_stage_a_manifest(
    manifest: dict[str, Any], *, contract_id: str, context: LaunchContext,
    trace_manifest_sha256: str, profile_sha256: str,
    budget_attestation_sha256: str, a_launch_sha256: str,
    a_live_usage_sha256: str, a_verify_receipt_sha256: str,
) -> None:
    exact = {
        "arms": (EXPECTED_ARMS["A"],),
        "commit": context.commit,
        "trace_sha256": TRACE_SHA256,
        "trace_manifest_sha256": trace_manifest_sha256,
        "trace_n": str(TRACE_N),
        "profile_sha256": profile_sha256,
        "server_pid": str(context.server_pid),
        "server_log_prefix_sha256": context.server_log_prefix_sha256,
        "endpoint_version_sha256": context.endpoint_version_sha256,
        "endpoint_models_identity_sha256": context.endpoint_models_identity_sha256,
        "base_url": context.base_url,
        "chat_url": context.chat_url,
        "model": context.model,
        "cache_mode": CACHE_MODE,
        "max_inflight": str(MAX_INFLIGHT),
        "slo_s": _decimal_text(SLO_S),
        "timeout_s": _decimal_text(TIMEOUT_S),
        "tick_ms": _decimal_text(TICK_MS),
        "temperature": _decimal_text(TEMPERATURE),
        "ignore_eos": "1",
        "in_price": _decimal_text(INPUT_PRICE_PER_MILLION_USD),
        "out_price": _decimal_text(OUTPUT_PRICE_PER_MILLION_USD),
        "local_ignore_eos": "1",
        "cloud_ignore_eos": "0",
        "cloud": "real",
        "cloud_url": OPENROUTER_URL,
        "cloud_model": OPENROUTER_MODEL,
        "cloud_api_key_env": OPENROUTER_API_KEY_ENV,
        "cloud_max_concurrency": "16",
        "cloud_provider": OPENROUTER_PROVIDER,
        "cloud_no_fallbacks": "1",
        "cloud_stop_after_first_token": "1",
        "real_cloud_expected_trace_n": str(TRACE_N),
        "secret_value_recorded": "false",
        "arm_order_mode": "e12_contract_stage",
        "arm_order_seed": ARM_ORDER_SEED,
        "live_contract_id": contract_id,
        "live_contract": LIVE_CONTRACT,
        "live_stage": "A",
        "live_expected_trace_n": str(TRACE_N),
        "live_arm_order_seed": ARM_ORDER_SEED,
        "live_exact_arm": EXPECTED_ARMS["A"],
        "live_authorized_budget_usd": "3",
        "live_budget_attestation_sha256": budget_attestation_sha256,
        "live_key_limit_max_usd": "3",
        "live_cooldown_s": "20",
        "live_stage_launch_attestation_sha256": a_launch_sha256,
        "live_current_usage_sha256": a_live_usage_sha256,
        "live_stage_launch_verify_receipt_sha256": a_verify_receipt_sha256,
    }
    for field, expected in exact.items():
        if manifest.get(field) != expected:
            raise LaunchCheckError(f"stage A manifest {field} is not exact")
    fingerprint = _sha256(manifest.get("run_fingerprint"), "stage A fingerprint")
    if fingerprint != recompute_matrix_fingerprint(manifest):
        raise LaunchCheckError("stage A run fingerprint is invalid")


def _matrix_expectations(manifest: dict[str, Any]) -> MatrixExpectations:
    return MatrixExpectations(
        trace_n=TRACE_N,
        prefill_tput=float(_manifest_decimal(manifest, "prefill_tput")),
        tpot_ms=float(_manifest_decimal(manifest, "tpot_ms")),
        first_token_overhead_ms=float(
            _manifest_decimal(manifest, "first_token_overhead_ms")
        ),
        ttft_guard_ms=float(_manifest_decimal(manifest, "guard_ms")),
        in_price=float(INPUT_PRICE_PER_MILLION_USD),
        out_price=float(OUTPUT_PRICE_PER_MILLION_USD),
        slo_s=float(SLO_S),
        timeout_s=float(TIMEOUT_S),
        nimbus_tick_ms=float(TICK_MS),
        max_inflight=MAX_INFLIGHT,
        temperature=float(TEMPERATURE),
        ignore_eos=True,
        kv_capacity_tokens=float(_manifest_decimal(manifest, "kv_cap")),
        model=str(manifest["model"]),
        chat_url=str(manifest["chat_url"]),
        scenario=str(manifest["scenario"]),
        cloud="real",
        cloud_url=OPENROUTER_URL,
        cloud_model=OPENROUTER_MODEL,
        cloud_api_key_env=OPENROUTER_API_KEY_ENV,
        cloud_max_concurrency=16,
        cloud_provider_order=(OPENROUTER_PROVIDER,),
        cloud_no_fallbacks=True,
        cloud_stop_after_first_token=True,
        local_ignore_eos=True,
        cloud_ignore_eos=False,
    )


def _validate_stage_a_events(
    path: Path, *, manifest: dict[str, Any], launch_authorized_at: datetime,
    launch_verified_at: datetime,
) -> tuple[str, datetime]:
    try:
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        raise LaunchCheckError("unable to read stage A events") from None
    if len(lines) != 3 or any(not line for line in lines):
        raise LaunchCheckError("stage A events must contain three lines")

    def fields(line: str, prefix: str, *, has_command: bool) -> dict[str, str]:
        if not line.startswith(prefix):
            raise LaunchCheckError("stage A event order/schema is invalid")
        if has_command:
            if line.count(" command=") != 1:
                raise LaunchCheckError("stage A start command is invalid")
            head, command = line.split(" command=", 1)
            if command != "router.run_config_bound_by_fingerprint":
                raise LaunchCheckError("stage A start command is invalid")
        else:
            if " command=" in line:
                raise LaunchCheckError("stage A finish event contains a command")
            head = line
        result: dict[str, str] = {}
        for token in head.split():
            if "=" not in token:
                raise LaunchCheckError("stage A event field is malformed")
            key, value = token.split("=", 1)
            if key in result:
                raise LaunchCheckError("stage A event has duplicate fields")
            result[key] = value
        return result

    started = fields(lines[0], "arm_started_at=", has_command=True)
    finished = fields(lines[1], "arm_finished_at=", has_command=False)
    matrix = fields(lines[2], "matrix_finished_at=", has_command=False)
    common = {
        "live_contract_id": manifest["live_contract_id"],
        "live_stage": "A",
        "live_authorized_budget_usd": "3",
        "live_budget_attestation_sha256": manifest[
            "live_budget_attestation_sha256"
        ],
        "live_key_limit_max_usd": "3",
        "live_cooldown_s": "20",
        "live_stage_launch_attestation_sha256": manifest[
            "live_stage_launch_attestation_sha256"
        ],
        "live_current_usage_sha256": manifest["live_current_usage_sha256"],
        "live_stage_launch_verify_receipt_sha256": manifest[
            "live_stage_launch_verify_receipt_sha256"
        ],
    }
    if set(started) != {"arm_started_at", "arm", *common}:
        raise LaunchCheckError("stage A start event fields are not exact")
    if set(finished) != {"arm_finished_at", "arm", *common}:
        raise LaunchCheckError("stage A finish event fields are not exact")
    if set(matrix) != {"matrix_finished_at", "run_fingerprint", *common}:
        raise LaunchCheckError("stage A matrix event fields are not exact")
    for event in (started, finished, matrix):
        for key, expected in common.items():
            if event.get(key) != expected:
                raise LaunchCheckError("stage A event contract binding differs")
    if started.get("arm") != EXPECTED_ARMS["A"] or finished.get("arm") != EXPECTED_ARMS["A"]:
        raise LaunchCheckError("stage A event arm is invalid")
    if matrix.get("run_fingerprint") != manifest["run_fingerprint"]:
        raise LaunchCheckError("stage A event fingerprint is invalid")
    started_at = _timestamp(started.get("arm_started_at"), "stage A start")
    finished_at = _timestamp(finished.get("arm_finished_at"), "stage A finish")
    matrix_at = _timestamp(matrix.get("matrix_finished_at"), "stage A matrix finish")
    manifest_at = _timestamp(manifest.get("started_at"), "stage A manifest start")
    if not (
        launch_authorized_at <= launch_verified_at <= manifest_at
        <= started_at <= finished_at <= matrix_at
    ):
        raise LaunchCheckError("stage A launch/event times are not ordered")
    return hashlib.sha256(raw).hexdigest(), matrix_at


def _artifact_hash(marker: dict[str, Any], name: str) -> tuple[str, int]:
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(name), dict):
        raise LaunchCheckError("stage A completion marker artifact is missing")
    item = artifacts[name]
    return (
        _sha256(item.get("sha256"), f"stage A {name} artifact"),
        _integer(item.get("nonempty_line_n"), f"stage A {name} line count"),
    )


def _validate_no_systemic_cloud_errors(path: Path) -> None:
    counts: collections.Counter[int] = collections.Counter()
    cloud_n = 0
    cloud_success_n = 0
    non429_http_failure_n = 0
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError
                if row.get("endpoint") != "cloud":
                    continue
                cloud_n += 1
                status = row.get("http_status")
                if isinstance(status, int) and not isinstance(status, bool):
                    counts[status] += 1
                    success = row.get("success") is True and 200 <= status < 300
                    cloud_success_n += int(success)
                    if not success and status != 429:
                        non429_http_failure_n += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise LaunchCheckError("stage A raw cloud-status evidence is invalid") from None
    immediate = (
        any(counts[status] for status in (401, 402, 403, 404, 405, 422))
        or any(count for status, count in counts.items() if 300 <= status < 400)
    )
    zero_success = cloud_n > 0 and cloud_success_n == 0
    dominant_400 = cloud_n >= 3 and counts[400] * 5 >= cloud_n * 4
    dominant_non429_http = (
        cloud_n >= 10 and non429_http_failure_n * 5 >= cloud_n * 4
    )
    if immediate or zero_success or dominant_400 or dominant_non429_http:
        raise LaunchCheckError("stage A has a systemic cloud failure")


def _validate_prior_a(
    *, directory: Path, a_launch_path: Path, contract_id: str,
    context: LaunchContext, trace_manifest_sha256: str, profile_sha256: str,
    budget_attestation_sha256: str,
) -> tuple[dict[str, Any], datetime]:
    a_launch, a_launch_sha = _load_json(a_launch_path, "stage A launch attestation")
    _validate_launch_schema(a_launch)
    _validate_attestation_bindings(
        a_launch,
        stage="A",
        contract_id=contract_id,
        trace_manifest_sha256=trace_manifest_sha256,
        profile_sha256=profile_sha256,
        budget_attestation_sha256=budget_attestation_sha256,
        context=context,
        check_freshness=False,
        now=None,
    )
    a_live_path = directory / LIVE_CURRENT_USAGE_FILENAME
    a_receipt_path = directory / VERIFY_RECEIPT_FILENAME
    a_live_sha = _file_sha256(a_live_path, "stage A live current usage")
    a_receipt_sha = _file_sha256(a_receipt_path, "stage A verify receipt")
    receipt = validate_verify_receipt(
        receipt_path=a_receipt_path,
        expected_sha256=a_receipt_sha,
        launch_attestation_path=a_launch_path,
        live_current_usage_path=a_live_path,
    )
    if receipt.get("stage") != "A":
        raise LaunchCheckError("stage A verify receipt stage is invalid")
    manifest, manifest_sha = _parse_manifest(directory / "matrix_manifest.txt")
    _validate_stage_a_manifest(
        manifest,
        contract_id=contract_id,
        context=context,
        trace_manifest_sha256=trace_manifest_sha256,
        profile_sha256=profile_sha256,
        budget_attestation_sha256=budget_attestation_sha256,
        a_launch_sha256=a_launch_sha,
        a_live_usage_sha256=a_live_sha,
        a_verify_receipt_sha256=a_receipt_sha,
    )
    markers = sorted(directory.glob("*.complete.json"))
    if len(markers) != 1:
        raise LaunchCheckError("stage A must have exactly one completion marker")
    marker_path = markers[0]
    stem = marker_path.name.removesuffix(".complete.json")
    raw_path = directory / f"{stem}.jsonl"
    summary_path = directory / f"{stem}.summary.json"
    decisions_path = directory / f"{stem}.decisions.jsonl"
    try:
        marker = validate_or_write_marker(
            raw_path=raw_path,
            summary_path=summary_path,
            decisions_path=decisions_path,
            marker_path=marker_path,
            fingerprint=manifest["run_fingerprint"],
            arm_text=EXPECTED_ARMS["A"],
            expected=_matrix_expectations(manifest),
            write_marker=False,
        )
    except (MatrixEvidenceError, OSError, KeyError, ValueError):
        raise LaunchCheckError("stage A matrix evidence is invalid") from None
    raw_sha, raw_n = _artifact_hash(marker, "raw")
    summary_sha, _ = _artifact_hash(marker, "summary")
    decisions_sha, decision_n = _artifact_hash(marker, "decisions")
    marker_sha = _file_sha256(marker_path, "stage A completion marker")
    _validate_no_systemic_cloud_errors(raw_path)
    events_sha, matrix_finished = _validate_stage_a_events(
        directory / "matrix_events.log",
        manifest=manifest,
        launch_authorized_at=_timestamp(
            a_launch["authorized_at_utc"], "stage A authorization"
        ),
        launch_verified_at=_timestamp(
            receipt["verified_at_utc"], "stage A launch verification"
        ),
    )
    return {
        "launch_attestation_sha256": a_launch_sha,
        "matrix_manifest_sha256": manifest_sha,
        "completion_marker_sha256": marker_sha,
        "raw_sha256": raw_sha,
        "raw_n": raw_n,
        "summary_sha256": summary_sha,
        "decisions_sha256": decisions_sha,
        "decision_n": decision_n,
        "events_sha256": events_sha,
        "run_fingerprint": manifest["run_fingerprint"],
        "matrix_finished_at_utc": _timestamp_text(matrix_finished),
        "authorization_interval_s": None,
        "key_fingerprint_sha256": a_launch["settled_key"][
            "key_fingerprint_sha256"
        ],
        "expires_at_utc": a_launch["settled_key"]["expires_at_utc"],
        "live_current_usage_sha256": a_live_sha,
        "launch_verify_receipt_sha256": a_receipt_sha,
    }, matrix_finished


def _frozen_contract(stage: str) -> dict[str, Any]:
    return {
        "input_price_per_million_usd": "0.08",
        "output_price_per_million_usd": "0.28",
        "request_price_per_request_usd": "0",
        "authorized_budget_usd": "3",
        "stage_future_bound_usd": _decimal_text(STAGE_FUTURE_BOUND_USD[stage]),
        "slo_s": "5",
        "timeout_s": "600",
        "tick_ms": "250",
        "cooldown_s": "20",
        "max_inflight": 128,
        "temperature": "0",
        "cloud_max_concurrency": 16,
    }


def _validate_launch_schema(payload: dict[str, Any]) -> None:
    if set(payload) != TOP_LEVEL_FIELDS:
        raise LaunchCheckError("launch attestation fields differ from schema")
    if payload.get("schema_version") != 1 or payload.get("status") != "pass":
        raise LaunchCheckError("launch attestation header is invalid")
    stage = payload.get("stage")
    if stage not in EXPECTED_ARMS:
        raise LaunchCheckError("launch attestation stage is invalid")
    mappings = (
        ("bindings", BINDING_FIELDS),
        ("frozen_contract", FROZEN_FIELDS),
        ("settled_key", SETTLED_KEY_FIELDS),
        ("timing", TIMING_FIELDS),
    )
    for name, fields in mappings:
        value = payload.get(name)
        if not isinstance(value, dict) or set(value) != fields:
            raise LaunchCheckError(f"launch attestation {name} schema is invalid")
    prior = payload.get("prior_a")
    if stage == "A" and prior is not None:
        raise LaunchCheckError("stage A launch cannot contain prior-A evidence")
    if stage == "C" and (not isinstance(prior, dict) or set(prior) != PRIOR_A_FIELDS):
        raise LaunchCheckError("stage C launch lacks exact prior-A evidence")
    authorized = _timestamp(payload.get("authorized_at_utc"), "launch authorization")
    _sha256(payload.get("contract_id_sha256"), "contract id hash")
    for key in BINDING_FIELDS - {"trace_n", "prompt_token_sum", "decode_token_sum"}:
        if key == "commit":
            if not COMMIT_RE.fullmatch(str(payload["bindings"].get(key, ""))):
                raise LaunchCheckError("launch commit binding is invalid")
        else:
            _sha256(payload["bindings"].get(key), f"launch binding {key}")
    for key, expected in (
        ("trace_n", TRACE_N),
        ("prompt_token_sum", PROMPT_TOKEN_SUM),
        ("decode_token_sum", DECODE_TOKEN_SUM),
    ):
        if payload["bindings"].get(key) != expected:
            raise LaunchCheckError(f"launch binding {key} is invalid")
    if payload["frozen_contract"] != _frozen_contract(stage):
        raise LaunchCheckError("launch frozen contract is invalid")
    for field in SETTLED_KEY_FIELDS:
        if field in {"limit_usd", "usage_usd", "remaining_usd"}:
            _decimal(payload["settled_key"].get(field), f"settled key {field}")
    _sha256(
        payload["settled_key"].get("key_fingerprint_sha256"),
        "settled key fingerprint",
    )
    expires = payload["settled_key"].get("expires_at_utc")
    if expires is not None:
        _timestamp(expires, "settled key expiration")
    limit = _decimal(payload["settled_key"].get("limit_usd"), "settled key limit")
    usage = _decimal(payload["settled_key"].get("usage_usd"), "settled key usage")
    remaining = _decimal(
        payload["settled_key"].get("remaining_usd"), "settled key remaining"
    )
    if limit <= 0 or limit > AUTHORIZED_BUDGET_USD or remaining > limit:
        raise LaunchCheckError("settled key limit is invalid")
    if abs(usage + remaining - limit) > ARITHMETIC_TOLERANCE:
        raise LaunchCheckError("settled key arithmetic is inconsistent")
    timing = payload["timing"]
    baseline_at = _timestamp(timing["baseline_usage_at_utc"], "baseline time")
    canary_at = _timestamp(timing["canary_at_utc"], "canary time")
    previous_at = _timestamp(
        timing["settlement_previous_at_utc"], "settlement previous time"
    )
    current_at = _timestamp(
        timing["settlement_current_at_utc"], "settlement current time"
    )
    if not baseline_at <= canary_at <= previous_at < current_at <= authorized:
        raise LaunchCheckError("launch evidence times are not ordered")
    observed = _decimal(timing["settlement_interval_s"], "settlement interval")
    actual = Decimal(str((current_at - previous_at).total_seconds()))
    if observed != actual or observed < MIN_SETTLEMENT_S:
        raise LaunchCheckError("settlement interval binding is invalid")
    for field in (
        "baseline_usage_at_utc", "canary_at_utc",
        "settlement_previous_at_utc", "settlement_current_at_utc",
    ):
        _timestamp(payload["timing"].get(field), field)
    if _decimal(payload["timing"].get("minimum_settlement_interval_s"), "settlement minimum") != MIN_SETTLEMENT_S:
        raise LaunchCheckError("launch settlement minimum is invalid")
    if _decimal(payload["timing"].get("minimum_a_to_c_authorization_interval_s"), "cooldown minimum") != COOLDOWN_S:
        raise LaunchCheckError("launch A-to-C minimum is invalid")
    if stage == "C":
        assert isinstance(prior, dict)
        for key in (
            "launch_attestation_sha256", "matrix_manifest_sha256",
            "completion_marker_sha256", "raw_sha256", "summary_sha256",
            "decisions_sha256", "events_sha256", "run_fingerprint",
        ):
            _sha256(prior.get(key), f"prior A {key}")
        if prior.get("raw_n") != TRACE_N:
            raise LaunchCheckError("prior A raw population is invalid")
        _integer(prior.get("decision_n"), "prior A decision count", minimum=1)
        matrix_at = _timestamp(
            prior.get("matrix_finished_at_utc"), "prior A matrix finish"
        )
        interval = _decimal(
            prior.get("authorization_interval_s"), "prior A authorization interval"
        )
        actual_interval = Decimal(str((authorized - matrix_at).total_seconds()))
        if interval != actual_interval or interval < COOLDOWN_S:
            raise LaunchCheckError("prior A authorization interval is invalid")
        if matrix_at > previous_at:
            raise LaunchCheckError("stage C settlement predates prior A completion")
        _sha256(prior.get("key_fingerprint_sha256"), "prior A key fingerprint")
        prior_expiration = prior.get("expires_at_utc")
        if prior_expiration is not None:
            _timestamp(prior_expiration, "prior A key expiration")


def _validate_attestation_bindings(
    payload: dict[str, Any], *, stage: str, contract_id: str,
    trace_manifest_sha256: str, profile_sha256: str,
    budget_attestation_sha256: str, context: LaunchContext,
    check_freshness: bool, now: datetime | None,
) -> None:
    _validate_launch_schema(payload)
    if payload["stage"] != stage:
        raise LaunchCheckError("launch stage does not match runner stage")
    if payload["contract_id_sha256"] != _contract_id_hash(contract_id):
        raise LaunchCheckError("launch contract id binding differs")
    expected = {
        "trace_manifest_sha256": _sha256(
            trace_manifest_sha256, "expected trace manifest"
        ),
        "trace_sha256": TRACE_SHA256,
        "trace_n": TRACE_N,
        "prompt_token_sum": PROMPT_TOKEN_SUM,
        "decode_token_sum": DECODE_TOKEN_SUM,
        "profile_sha256": _sha256(profile_sha256, "expected profile"),
        "budget_attestation_sha256": _sha256(
            budget_attestation_sha256, "expected budget attestation"
        ),
        "commit": context.commit,
        "lifecycle_sha256": context.lifecycle_sha256,
    }
    bindings = payload["bindings"]
    for key, value in expected.items():
        if bindings.get(key) != value:
            raise LaunchCheckError(f"launch binding {key} differs from preflight")
    if check_freshness:
        current = _now(now)
        authorized = _timestamp(payload["authorized_at_utc"], "authorization")
        if authorized > current:
            raise LaunchCheckError("launch authorization is in the future")
        if current - authorized > MAX_ATTESTATION_AGE:
            raise LaunchCheckError("launch authorization is stale")


def create_launch_attestation(
    *, stage: str, contract_id: str, trace_manifest_path: Path,
    profile_path: Path, price_snapshot_path: Path,
    budget_attestation_path: Path, canary_path: Path,
    baseline_usage_path: Path, settlement_previous_path: Path,
    settlement_current_path: Path, stage_budget_gate_path: Path,
    context: LaunchContext, a_dir: Path | None = None,
    a_launch_attestation_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate all launch sources and return a whitelisted attestation."""
    if stage not in EXPECTED_ARMS:
        raise LaunchCheckError("stage must be exactly A or C")
    contract_hash = _contract_id_hash(contract_id)
    context.validate()
    # The shell matrix records manifest/events at whole-second resolution.
    # Floor authorization to the same resolution so an immediate legitimate
    # launch in the same second cannot appear to predate its authorization.
    current = _now(now).replace(microsecond=0)

    trace_manifest, trace_manifest_sha = _load_json(
        trace_manifest_path, "trace manifest"
    )
    _validate_trace_manifest(trace_manifest, trace_manifest_sha)
    profile, profile_sha = _load_json(profile_path, "profile")
    _validate_profile(profile, trace_manifest=trace_manifest, context=context)

    price_snapshot, price_sha = _load_json(price_snapshot_path, "price snapshot")
    budget, budget_sha = _load_json(
        budget_attestation_path, "budget attestation"
    )
    try:
        rebuilt_budget = build_budget_attestation(
            manifest_path=trace_manifest_path,
            price_snapshot_path=price_snapshot_path,
            expected_manifest_sha256=trace_manifest_sha,
            expected_trace_sha256=TRACE_SHA256,
            expected_n=TRACE_N,
            expected_prompt_token_sum=PROMPT_TOKEN_SUM,
            expected_decode_token_sum=DECODE_TOKEN_SUM,
            expected_payload_mode=PAYLOAD_MODE,
            expected_cache_mode=CACHE_MODE,
            arm_count=ARM_COUNT,
            input_price_per_million_usd=INPUT_PRICE_PER_MILLION_USD,
            output_price_per_million_usd=OUTPUT_PRICE_PER_MILLION_USD,
            budget_usd=AUTHORIZED_BUDGET_USD,
            now=current,
        )
    except ValueError:
        raise LaunchCheckError("budget attestation sources are invalid") from None
    if budget != rebuilt_budget:
        raise LaunchCheckError("budget attestation is not the exact E12 result")
    if budget.get("estimated_cost_usd") != "1.90803056":
        raise LaunchCheckError("budget attestation full-pair bound is not exact")
    if budget.get("price_snapshot_sha256") != price_sha:
        raise LaunchCheckError("budget does not bind the current price snapshot")
    # Keep the loaded public snapshot live and ensure it was not merely hashed
    # from a JSON value that the budget validator ignored.
    if not price_snapshot or len(price_snapshot) < 1:
        raise LaunchCheckError("price snapshot is empty")

    canary_sha, canary_at, _, canary_key_fingerprint = _validate_canary(
        canary_path
    )
    baseline, baseline_sha = _load_json(baseline_usage_path, "baseline usage")
    previous, previous_sha = _load_json(
        settlement_previous_path, "settlement-previous usage"
    )
    settled, settled_sha = _load_json(
        settlement_current_path, "settlement-current usage"
    )
    baseline_key = _usage_key(baseline, "baseline usage")
    previous_key = _usage_key(previous, "settlement-previous usage")
    settled_key = _usage_key(settled, "settlement-current usage")
    baseline_at = _timestamp(baseline.get("captured_at_utc"), "baseline time")
    previous_at = _timestamp(previous.get("captured_at_utc"), "settlement previous time")
    settled_at = _timestamp(settled.get("captured_at_utc"), "settlement current time")
    if not baseline_at <= canary_at <= previous_at < settled_at <= current:
        raise LaunchCheckError("baseline/canary/settlement times are not ordered")
    if settled_at - previous_at < timedelta(seconds=60):
        raise LaunchCheckError("settlement pair is less than 60 seconds apart")
    if previous_key != settled_key:
        raise LaunchCheckError("settlement key counters are not equal")
    if baseline_key["fingerprint"] != settled_key["fingerprint"]:
        raise LaunchCheckError("baseline and settled snapshots use different keys")
    if canary_key_fingerprint != settled_key["fingerprint"]:
        raise LaunchCheckError("canary and budget snapshots use different keys")
    if baseline_key["expires_at_utc"] != settled_key["expires_at_utc"]:
        raise LaunchCheckError("key expiration changed during launch evidence")
    if settled_key["expires_at_utc"] is not None:
        expiration = _timestamp(settled_key["expires_at_utc"], "key expiration")
        if expiration - current < timedelta(hours=6):
            raise LaunchCheckError("key expires less than six hours after authorization")

    gate, gate_sha = _load_json(stage_budget_gate_path, "stage budget gate")
    try:
        rebuilt_gate = build_stage_gate_attestation(
            baseline_path=baseline_usage_path,
            current_path=settlement_current_path,
            next_stage_full_upper_bound_usd=STAGE_FUTURE_BOUND_USD[stage],
            final=False,
            settlement_previous_path=settlement_previous_path,
            now=current,
        )
    except StageGateError:
        raise LaunchCheckError("stage budget gate sources are invalid") from None
    if gate != rebuilt_gate:
        raise LaunchCheckError("stage budget gate is not the exact settled result")
    if gate.get("next_stage_full_upper_bound_usd") != _decimal_text(
        STAGE_FUTURE_BOUND_USD[stage]
    ):
        raise LaunchCheckError("stage budget gate future bound is not exact")

    prior_a: dict[str, Any] | None = None
    if stage == "A":
        if a_dir is not None or a_launch_attestation_path is not None:
            raise LaunchCheckError("stage A must not accept prior-A evidence")
    else:
        if a_dir is None or a_launch_attestation_path is None:
            raise LaunchCheckError("stage C requires prior-A evidence")
        prior_a, matrix_finished = _validate_prior_a(
            directory=a_dir,
            a_launch_path=a_launch_attestation_path,
            contract_id=contract_id,
            context=context,
            trace_manifest_sha256=trace_manifest_sha,
            profile_sha256=profile_sha,
            budget_attestation_sha256=budget_sha,
        )
        if previous_at < matrix_finished:
            raise LaunchCheckError("stage C settled pair predates stage A completion")
        interval = Decimal(str((current - matrix_finished).total_seconds()))
        if interval < COOLDOWN_S:
            raise LaunchCheckError("stage C authorization is less than 20s after A")
        prior_a["authorization_interval_s"] = _decimal_text(interval)
        if prior_a["key_fingerprint_sha256"] != settled_key["fingerprint"]:
            raise LaunchCheckError("stage A and stage C use different API keys")
        if prior_a["expires_at_utc"] != settled_key["expires_at_utc"]:
            raise LaunchCheckError("stage A and stage C key expiration differs")

    settlement_interval = Decimal(str((settled_at - previous_at).total_seconds()))
    result = {
        "schema_version": 1,
        "status": "pass",
        "stage": stage,
        "authorized_at_utc": _timestamp_text(current),
        "contract_id_sha256": contract_hash,
        "bindings": {
            "trace_manifest_sha256": trace_manifest_sha,
            "trace_sha256": TRACE_SHA256,
            "trace_n": TRACE_N,
            "prompt_token_sum": PROMPT_TOKEN_SUM,
            "decode_token_sum": DECODE_TOKEN_SUM,
            "profile_sha256": profile_sha,
            "price_snapshot_sha256": price_sha,
            "budget_attestation_sha256": budget_sha,
            "canary_sha256": canary_sha,
            "baseline_usage_sha256": baseline_sha,
            "settlement_previous_usage_sha256": previous_sha,
            "settlement_current_usage_sha256": settled_sha,
            "stage_budget_gate_sha256": gate_sha,
            "commit": context.commit,
            "lifecycle_sha256": context.lifecycle_sha256,
        },
        "frozen_contract": _frozen_contract(stage),
        "settled_key": {
            "key_fingerprint_sha256": settled_key["fingerprint"],
            "limit_usd": _decimal_text(settled_key["limit"]),
            "usage_usd": _decimal_text(settled_key["usage"]),
            "remaining_usd": _decimal_text(settled_key["remaining"]),
            "expires_at_utc": settled_key["expires_at_utc"],
        },
        "timing": {
            "baseline_usage_at_utc": _timestamp_text(baseline_at),
            "canary_at_utc": _timestamp_text(canary_at),
            "settlement_previous_at_utc": _timestamp_text(previous_at),
            "settlement_current_at_utc": _timestamp_text(settled_at),
            "settlement_interval_s": _decimal_text(settlement_interval),
            "minimum_settlement_interval_s": "60",
            "minimum_a_to_c_authorization_interval_s": "20",
        },
        "prior_a": prior_a,
    }
    _validate_launch_schema(result)
    return result


def verify_launch_attestation(
    *, attestation_path: Path, expected_sha256: str, stage: str,
    contract_id: str, trace_manifest_sha256: str, profile_sha256: str,
    budget_attestation_sha256: str, context: LaunchContext,
    trace_manifest_path: Path, profile_path: Path, price_snapshot_path: Path,
    budget_attestation_path: Path, canary_path: Path,
    baseline_usage_path: Path, settlement_previous_path: Path,
    settlement_current_path: Path, stage_budget_gate_path: Path,
    live_current_usage_path: Path, a_dir: Path | None = None,
    a_launch_attestation_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify a launch artifact against current runner and live-key evidence."""
    expected_sha256 = _sha256(expected_sha256, "expected launch attestation")
    payload, actual_sha = _load_json(attestation_path, "launch attestation")
    if actual_sha != expected_sha256:
        raise LaunchCheckError("launch attestation SHA256 differs")
    _validate_attestation_bindings(
        payload,
        stage=stage,
        contract_id=contract_id,
        trace_manifest_sha256=trace_manifest_sha256,
        profile_sha256=profile_sha256,
        budget_attestation_sha256=budget_attestation_sha256,
        context=context,
        check_freshness=True,
        now=now,
    )
    # Do not trust a self-described pass JSON.  Re-run the complete offline
    # source validation at its recorded authorization time and require byte-
    # independent semantic equality.  In particular, stage C cannot be
    # authorized by hand-writing prior-A hashes without presenting the exact
    # marker/raw/summary/decisions/events that those hashes describe.
    rebuilt = create_launch_attestation(
        stage=stage,
        contract_id=contract_id,
        trace_manifest_path=trace_manifest_path,
        profile_path=profile_path,
        price_snapshot_path=price_snapshot_path,
        budget_attestation_path=budget_attestation_path,
        canary_path=canary_path,
        baseline_usage_path=baseline_usage_path,
        settlement_previous_path=settlement_previous_path,
        settlement_current_path=settlement_current_path,
        stage_budget_gate_path=stage_budget_gate_path,
        context=context,
        a_dir=a_dir,
        a_launch_attestation_path=a_launch_attestation_path,
        now=_timestamp(payload["authorized_at_utc"], "authorization"),
    )
    if rebuilt != payload:
        raise LaunchCheckError("launch attestation differs from current source evidence")
    _validate_private_regular_file(live_current_usage_path, "live current usage")
    live, live_sha = _load_json(live_current_usage_path, "live current usage")
    live_key = _usage_key(live, "live current usage")
    live_at = _timestamp(live.get("captured_at_utc"), "live current usage time")
    current = _now(now)
    authorized = _timestamp(payload["authorized_at_utc"], "authorization")
    if not authorized <= live_at <= current:
        raise LaunchCheckError("live current usage time is outside launch window")
    if current - live_at > timedelta(minutes=2):
        raise LaunchCheckError("live current usage snapshot is stale")
    settled = payload["settled_key"]
    if live_key["fingerprint"] != settled["key_fingerprint_sha256"]:
        raise LaunchCheckError("live API key differs from authorized key")
    if live_key["expires_at_utc"] != settled["expires_at_utc"]:
        raise LaunchCheckError("live API key expiration differs from authorization")
    if live_key["expires_at_utc"] is not None:
        expiration = _timestamp(live_key["expires_at_utc"], "live key expiration")
        if expiration - current < timedelta(hours=6):
            raise LaunchCheckError("live API key expires in less than six hours")
    for key, attestation_key in (
        ("limit", "limit_usd"),
        ("usage", "usage_usd"),
        ("remaining", "remaining_usd"),
    ):
        expected = _decimal(settled[attestation_key], f"settled {key}")
        if abs(live_key[key] - expected) > ARITHMETIC_TOLERANCE:
            raise LaunchCheckError("live key counters differ from authorized settlement")
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "stage": stage,
        "attestation_sha256": actual_sha,
        "live_current_usage_sha256": live_sha,
        "authorization_at_utc": payload["authorized_at_utc"],
        "settled_usage_at_utc": payload["timing"][
            "settlement_current_at_utc"
        ],
        "verified_at_utc": _timestamp_text(current),
        "live_usage_at_utc": _timestamp_text(live_at),
        "key_fingerprint_sha256": live_key["fingerprint"],
        "key_limit_usd": _decimal_text(live_key["limit"]),
        "key_usage_usd": _decimal_text(live_key["usage"]),
        "key_remaining_usd": _decimal_text(live_key["remaining"]),
        "key_expires_at_utc": live_key["expires_at_utc"],
    }
    _validate_verify_receipt_payload(
        receipt, launch=payload, launch_sha=actual_sha,
        live=live, live_sha=live_sha,
    )
    return receipt


def _validate_verify_receipt_payload(
    receipt: dict[str, Any], *, launch: dict[str, Any], launch_sha: str,
    live: dict[str, Any], live_sha: str,
) -> None:
    if set(receipt) != VERIFY_RECEIPT_FIELDS:
        raise LaunchCheckError("launch verification receipt schema is invalid")
    if receipt.get("schema_version") != 1 or receipt.get("status") != "pass":
        raise LaunchCheckError("launch verification receipt header is invalid")
    if receipt.get("stage") != launch.get("stage"):
        raise LaunchCheckError("launch verification receipt stage differs")
    if receipt.get("attestation_sha256") != launch_sha:
        raise LaunchCheckError("launch verification receipt attestation hash differs")
    if receipt.get("live_current_usage_sha256") != live_sha:
        raise LaunchCheckError("launch verification receipt usage hash differs")
    key = _usage_key(live, "receipt live usage")
    settled = launch["settled_key"]
    exact = {
        "authorization_at_utc": launch["authorized_at_utc"],
        "settled_usage_at_utc": launch["timing"]["settlement_current_at_utc"],
        "live_usage_at_utc": live.get("captured_at_utc"),
        "key_fingerprint_sha256": key["fingerprint"],
        "key_limit_usd": _decimal_text(key["limit"]),
        "key_usage_usd": _decimal_text(key["usage"]),
        "key_remaining_usd": _decimal_text(key["remaining"]),
        "key_expires_at_utc": key["expires_at_utc"],
    }
    for field, expected in exact.items():
        if receipt.get(field) != expected:
            raise LaunchCheckError(f"launch verification receipt {field} differs")
    if key["fingerprint"] != settled["key_fingerprint_sha256"]:
        raise LaunchCheckError("receipt key differs from authorized key")
    if key["expires_at_utc"] != settled["expires_at_utc"]:
        raise LaunchCheckError("receipt key expiration differs")
    for key_name, settled_name in (
        ("limit", "limit_usd"), ("usage", "usage_usd"),
        ("remaining", "remaining_usd"),
    ):
        if abs(key[key_name] - _decimal(settled[settled_name], settled_name)) > ARITHMETIC_TOLERANCE:
            raise LaunchCheckError("receipt key counters differ from authorization")
    authorized = _timestamp(receipt["authorization_at_utc"], "receipt authorization")
    settled_at = _timestamp(receipt["settled_usage_at_utc"], "receipt settlement")
    live_at = _timestamp(receipt["live_usage_at_utc"], "receipt live usage")
    verified_at = _timestamp(receipt.get("verified_at_utc"), "receipt verification")
    if not settled_at <= authorized <= live_at <= verified_at:
        raise LaunchCheckError("launch verification receipt times are not ordered")


def validate_verify_receipt(
    *, receipt_path: Path, expected_sha256: str,
    launch_attestation_path: Path, live_current_usage_path: Path,
) -> dict[str, Any]:
    """Strictly validate a persisted receipt and both artifacts it binds."""
    expected_sha256 = _sha256(expected_sha256, "expected verify receipt")
    _validate_private_regular_file(receipt_path, "launch verify receipt")
    _validate_private_regular_file(live_current_usage_path, "receipt live usage")
    receipt, receipt_sha = _load_json(receipt_path, "launch verify receipt")
    if receipt_sha != expected_sha256:
        raise LaunchCheckError("launch verification receipt SHA256 differs")
    launch, launch_sha = _load_json(
        launch_attestation_path, "receipt launch attestation"
    )
    _validate_launch_schema(launch)
    live, live_sha = _load_json(live_current_usage_path, "receipt live usage")
    _validate_verify_receipt_payload(
        receipt, launch=launch, launch_sha=launch_sha,
        live=live, live_sha=live_sha,
    )
    return receipt


def write_json_atomic(
    path: Path, payload: dict[str, Any], *, overwrite: bool = True
) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as target:
            temporary = Path(target.name)
            json.dump(payload, target, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        temporary = None
    except OSError:
        raise LaunchCheckError("unable to publish launch attestation") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--commit", required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--server-log-prefix-sha256", required=True)
    parser.add_argument("--endpoint-version-sha256", required=True)
    parser.add_argument("--endpoint-models-identity-sha256", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--chat-url", required=True)
    parser.add_argument("--model", required=True)


def _context(args: argparse.Namespace) -> LaunchContext:
    return LaunchContext(
        commit=args.commit,
        server_pid=args.server_pid,
        server_log_prefix_sha256=args.server_log_prefix_sha256,
        endpoint_version_sha256=args.endpoint_version_sha256,
        endpoint_models_identity_sha256=args.endpoint_models_identity_sha256,
        base_url=args.base_url,
        chat_url=args.chat_url,
        model=args.model,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--stage", choices=("A", "C"), required=True)
    create.add_argument("--contract-id", required=True)
    create.add_argument("--trace-manifest", type=Path, required=True)
    create.add_argument("--profile", type=Path, required=True)
    create.add_argument("--price-snapshot", type=Path, required=True)
    create.add_argument("--budget-attestation", type=Path, required=True)
    create.add_argument("--canary", type=Path, required=True)
    create.add_argument("--baseline-usage", type=Path, required=True)
    create.add_argument("--settlement-previous", type=Path, required=True)
    create.add_argument("--settlement-current", type=Path, required=True)
    create.add_argument("--stage-budget-gate", type=Path, required=True)
    create.add_argument("--a-dir", type=Path)
    create.add_argument("--a-launch-attestation", type=Path)
    create.add_argument("--output", type=Path, required=True)
    _add_context_args(create)

    verify = commands.add_parser("verify")
    verify.add_argument("--attestation", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--stage", choices=("A", "C"), required=True)
    verify.add_argument("--contract-id", required=True)
    verify.add_argument("--trace-manifest-sha256", required=True)
    verify.add_argument("--profile-sha256", required=True)
    verify.add_argument("--budget-attestation-sha256", required=True)
    verify.add_argument("--trace-manifest", type=Path, required=True)
    verify.add_argument("--profile", type=Path, required=True)
    verify.add_argument("--price-snapshot", type=Path, required=True)
    verify.add_argument("--budget-attestation", type=Path, required=True)
    verify.add_argument("--canary", type=Path, required=True)
    verify.add_argument("--baseline-usage", type=Path, required=True)
    verify.add_argument("--settlement-previous", type=Path, required=True)
    verify.add_argument("--settlement-current", type=Path, required=True)
    verify.add_argument("--stage-budget-gate", type=Path, required=True)
    verify.add_argument("--a-dir", type=Path)
    verify.add_argument("--a-launch-attestation", type=Path)
    verify.add_argument("--live-current-usage", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    _add_context_args(verify)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "create":
            result = create_launch_attestation(
                stage=args.stage,
                contract_id=args.contract_id,
                trace_manifest_path=args.trace_manifest,
                profile_path=args.profile,
                price_snapshot_path=args.price_snapshot,
                budget_attestation_path=args.budget_attestation,
                canary_path=args.canary,
                baseline_usage_path=args.baseline_usage,
                settlement_previous_path=args.settlement_previous,
                settlement_current_path=args.settlement_current,
                stage_budget_gate_path=args.stage_budget_gate,
                context=_context(args),
                a_dir=args.a_dir,
                a_launch_attestation_path=args.a_launch_attestation,
            )
            write_json_atomic(args.output, result)
        else:
            result = verify_launch_attestation(
                attestation_path=args.attestation,
                expected_sha256=args.expected_sha256,
                stage=args.stage,
                contract_id=args.contract_id,
                trace_manifest_sha256=args.trace_manifest_sha256,
                profile_sha256=args.profile_sha256,
                budget_attestation_sha256=args.budget_attestation_sha256,
                context=_context(args),
                trace_manifest_path=args.trace_manifest,
                profile_path=args.profile,
                price_snapshot_path=args.price_snapshot,
                budget_attestation_path=args.budget_attestation,
                canary_path=args.canary,
                baseline_usage_path=args.baseline_usage,
                settlement_previous_path=args.settlement_previous,
                settlement_current_path=args.settlement_current,
                stage_budget_gate_path=args.stage_budget_gate,
                a_dir=args.a_dir,
                a_launch_attestation_path=args.a_launch_attestation,
                live_current_usage_path=args.live_current_usage,
            )
            write_json_atomic(args.output, result, overwrite=False)
    except LaunchCheckError as exc:
        print(f"E12 stage launch check failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
