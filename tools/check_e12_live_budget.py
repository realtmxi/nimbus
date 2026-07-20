#!/usr/bin/env python3
"""Fail closed unless an E12 current-turn manifest fits a dollar budget.

This guard deliberately reads only the text-free materialization manifest.  It
does not open the JSONL trace, and its attestation contains only hashes,
aggregate token counts, pricing inputs, and the resulting Decimal arithmetic.
The E12 trace identity, two-arm count, provider prices, and three-dollar cap
are frozen in code; legacy CLI spellings are accepted only at those exact
values so they cannot weaken the launch guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


PAYLOAD_MODE = "sharegpt_current_turn_retokenized"
CACHE_MODE = "none"
PRICE_MODEL = "qwen/qwen3-32b"
PRICE_PROVIDER = "deepinfra"
TRACE_SHA256 = "e838016a8e55660c565dadb1ad019770f6b88f878d8ca29f165c30887d2cb410"
TRACE_N = 11_604
PROMPT_TOKEN_SUM = 1_289_405
DECODE_TOKEN_SUM = 3_038_796
ARM_COUNT = 2
INPUT_PRICE_PER_MILLION_USD = Decimal("0.08")
OUTPUT_PRICE_PER_MILLION_USD = Decimal("0.28")
REQUEST_PRICE_PER_REQUEST_USD = Decimal("0")
AUTHORIZED_BUDGET_USD = Decimal("3")
PRICE_SNAPSHOT_MAX_AGE = timedelta(hours=24)
MILLION = Decimal(1_000_000)
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
PRICE_SNAPSHOT_FIELDS = frozenset({
    "schema_version",
    "status",
    "captured_at_utc",
    "http_status",
    "model",
    "provider",
    "matching_endpoint_count",
    "context_length",
    "prompt_price_per_token_usd",
    "completion_price_per_token_usd",
    "request_price_per_request_usd",
    "request_price_source",
    "prompt_price_per_million_usd",
    "completion_price_per_million_usd",
})
MATERIALIZER_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "tool_path",
    "tool_sha256",
    "dependency_sha256",
    "command_argv",
    "python_version",
    "transformers_version",
    "input",
    "input_sha256",
    "output",
    "output_sha256",
    "scenario",
    "tokenizer",
    "tokenizer_fingerprint",
    "chat_template_source",
    "payload_mode",
    "cache_mode",
    "semantic_scope",
    "token_count_method",
    "source_fields_deliberately_not_copied",
    "limit",
    "max_decode_tokens",
    "max_context_tokens",
    "overflow_policy",
    "input_rows_n",
    "scenario_rows_n",
    "empty_prompt_rows_n",
    "selected_before_limit_n",
    "selected_n",
    "selected_source_indices_sha256",
    "n",
    "context_overflow_affected_n",
    "context_overflow_events",
    "unique_session_n",
    "source_session_id_missing_n",
    "arrival",
    "actual_prompt_tokens",
    "original_trace_prompt_tokens",
    "actual_minus_original_trace_tokens",
    "decode_cap_affected_n",
    "source_decode_tokens",
    "output_decode_tokens",
    "preservation_checks",
})
DEPENDENCY_FIELDS = frozenset({
    "tools/materialize_token_aligned_trace.py",
    "router/common.py",
})
TOKENIZER_FINGERPRINT_FIELDS = frozenset({
    "class",
    "name_or_path",
    "vocab_size",
    "vocab_sha256",
    "chat_template_sha256",
    "model_max_length",
    "revision",
})
DISTRIBUTION_FIELDS = frozenset({
    "n", "sum", "min", "p50", "p95", "p99", "max",
})
PRESERVATION_FIELDS = frozenset({
    "prompt_text_exact_n",
    "arrived_at_exact_n",
    "decode_tokens_exact_n",
    "decode_source_provenance_n",
    "decode_cap_respected_n",
    "session_id_exact_n",
    "token_metadata_aligned_n",
})
OVERFLOW_EVENT_FIELDS = frozenset({
    "source_request_index",
    "actual_prompt_tokens",
    "decode_tokens",
    "total_tokens",
    "action",
})


class BudgetCheckError(ValueError):
    """A safe, prompt-free budget or manifest validation failure."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _expected_sha256(value: str, field: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise BudgetCheckError(f"{field} must be a 64-character SHA256")
    return value.lower()


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BudgetCheckError(f"{field} must be an integer >= {minimum}")
    return value


def _decimal(value: str | Decimal, field: str, *, positive: bool = False) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise BudgetCheckError(f"{field} must be a finite decimal") from None
    if not result.is_finite():
        raise BudgetCheckError(f"{field} must be a finite decimal")
    if result < 0 or (positive and result == 0):
        comparator = "> 0" if positive else ">= 0"
        raise BudgetCheckError(f"{field} must be {comparator}")
    return result


def _decimal_text(value: Decimal) -> str:
    """Return an exact, non-exponent decimal representation."""
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BudgetCheckError(f"manifest {field} must be an object")
    return value


def _exact_fields(
    value: dict[str, Any], expected: frozenset[str], field: str,
) -> None:
    """Reject unknown/missing fields without reflecting attacker-controlled keys."""
    if set(value) != expected:
        raise BudgetCheckError(
            f"manifest {field} fields do not match the text-free schema v1"
        )


def _safe_string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if (
        not isinstance(value, str)
        or (nonempty and not value)
        or len(value) > 8192
        or "\x00" in value
        or "\r" in value
        or "\n" in value
    ):
        raise BudgetCheckError(f"manifest {field} is not safe text metadata")
    return value


def _finite_number(value: Any, field: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise BudgetCheckError(f"manifest {field} must be a finite number")
    return value


def _distribution(value: Any, field: str, *, expected_n: int) -> dict[str, Any]:
    result = _mapping(value, field)
    _exact_fields(result, DISTRIBUTION_FIELDS, field)
    if _integer(result.get("n"), f"manifest {field}.n") != expected_n:
        raise BudgetCheckError(f"manifest {field} population is inconsistent")
    _finite_number(result.get("sum"), f"{field}.sum")
    ordered = [result.get(key) for key in ("min", "p50", "p95", "p99", "max")]
    if expected_n == 0:
        if any(item is not None for item in ordered):
            raise BudgetCheckError(f"manifest {field} empty distribution is invalid")
        return result
    numeric = [
        float(_finite_number(item, f"{field} distribution"))
        for item in ordered
    ]
    if numeric != sorted(numeric):
        raise BudgetCheckError(f"manifest {field} distribution order is invalid")
    return result


def validate_materializer_manifest_schema(manifest: dict[str, Any]) -> None:
    """Validate the complete, text-free ShareGPT materializer schema.

    Errors name only fixed schema locations and never reflect unknown keys or
    values.  This lets callers safely surface a failure even if an artifact was
    tampered with to contain prompt text.
    """
    if not isinstance(manifest, dict):
        raise BudgetCheckError("manifest root must be an object")
    _exact_fields(manifest, MATERIALIZER_MANIFEST_FIELDS, "root")

    exact = {
        "schema_version": 1,
        "tool_path": "tools/materialize_sharegpt_current_turn_trace.py",
        "scenario": "extreme_burst_1200",
        "payload_mode": PAYLOAD_MODE,
        "cache_mode": CACHE_MODE,
        "semantic_scope": "verbatim_current_user_turn_only",
        "limit": None,
        "max_decode_tokens": 1024,
        "max_context_tokens": 40_960,
        "overflow_policy": "drop",
        "chat_template_source": {"kind": "tokenizer_default"},
        "token_count_method": {
            "api": "tokenizer.apply_chat_template",
            "messages": [{"role": "user", "content": "<prompt_text>"}],
            "tokenize": True,
            "add_generation_prompt": True,
        },
        "source_fields_deliberately_not_copied": [
            "response_text",
            "block_hash_ids",
            "block_size",
        ],
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise BudgetCheckError(
                "manifest fixed materialization contract does not match E12"
            )

    for key in (
        "tool_sha256",
        "input_sha256",
        "output_sha256",
        "selected_source_indices_sha256",
    ):
        value = manifest.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise BudgetCheckError(f"manifest {key} is not a SHA256")

    dependencies = _mapping(manifest.get("dependency_sha256"), "dependency_sha256")
    _exact_fields(dependencies, DEPENDENCY_FIELDS, "dependency_sha256")
    for value in dependencies.values():
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise BudgetCheckError("manifest dependency SHA256 is invalid")

    command_argv = manifest.get("command_argv")
    if (
        not isinstance(command_argv, list)
        or not command_argv
        or len(command_argv) > 64
        or any(not isinstance(item, str) for item in command_argv)
    ):
        raise BudgetCheckError("manifest command_argv is invalid")
    for item in command_argv:
        _safe_string(item, "command_argv item")
    if not command_argv[0].endswith("materialize_sharegpt_current_turn_trace.py"):
        raise BudgetCheckError("manifest command_argv tool identity is invalid")

    for key in (
        "python_version",
        "transformers_version",
        "input",
        "output",
        "tokenizer",
    ):
        _safe_string(manifest.get(key), key)

    tokenizer = _mapping(
        manifest.get("tokenizer_fingerprint"), "tokenizer_fingerprint"
    )
    _exact_fields(tokenizer, TOKENIZER_FINGERPRINT_FIELDS, "tokenizer_fingerprint")
    _safe_string(tokenizer.get("class"), "tokenizer_fingerprint.class")
    _safe_string(
        tokenizer.get("name_or_path"),
        "tokenizer_fingerprint.name_or_path",
        nonempty=False,
    )
    _integer(
        tokenizer.get("vocab_size"),
        "manifest tokenizer_fingerprint.vocab_size",
        minimum=1,
    )
    _integer(
        tokenizer.get("model_max_length"),
        "manifest tokenizer_fingerprint.model_max_length",
        minimum=1,
    )
    for key in ("vocab_sha256", "chat_template_sha256"):
        value = tokenizer.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise BudgetCheckError("manifest tokenizer fingerprint hash is invalid")
    revision = tokenizer.get("revision")
    if revision is not None:
        _safe_string(revision, "tokenizer_fingerprint.revision")

    count_fields = (
        "input_rows_n",
        "scenario_rows_n",
        "empty_prompt_rows_n",
        "selected_before_limit_n",
        "selected_n",
        "n",
        "context_overflow_affected_n",
        "unique_session_n",
        "source_session_id_missing_n",
        "decode_cap_affected_n",
    )
    counts = {
        key: _integer(manifest.get(key), f"manifest {key}") for key in count_fields
    }
    if not (
        counts["input_rows_n"] >= counts["scenario_rows_n"]
        >= counts["selected_before_limit_n"]
        == counts["selected_n"]
        >= counts["n"]
    ):
        raise BudgetCheckError("manifest selection populations are inconsistent")
    if counts["empty_prompt_rows_n"] > counts["scenario_rows_n"]:
        raise BudgetCheckError("manifest empty-prompt population is inconsistent")
    if counts["context_overflow_affected_n"] != (
        counts["selected_n"] - counts["n"]
    ):
        raise BudgetCheckError("manifest overflow population is inconsistent")
    for key in (
        "unique_session_n",
        "source_session_id_missing_n",
        "decode_cap_affected_n",
    ):
        if counts[key] > counts["n"]:
            raise BudgetCheckError("manifest emitted-row population is inconsistent")

    overflow_events = manifest.get("context_overflow_events")
    if not isinstance(overflow_events, list) or len(overflow_events) != min(
        counts["context_overflow_affected_n"], 100
    ):
        raise BudgetCheckError("manifest overflow event population is inconsistent")
    for event in overflow_events:
        event = _mapping(event, "context_overflow_events item")
        _exact_fields(event, OVERFLOW_EVENT_FIELDS, "context_overflow_events item")
        source_index = _integer(
            event.get("source_request_index"),
            "manifest context_overflow_events source index",
        )
        prompt_tokens = _integer(
            event.get("actual_prompt_tokens"),
            "manifest context_overflow_events prompt tokens",
        )
        decode_tokens = _integer(
            event.get("decode_tokens"),
            "manifest context_overflow_events decode tokens",
        )
        total_tokens = _integer(
            event.get("total_tokens"),
            "manifest context_overflow_events total tokens",
        )
        if (
            source_index >= counts["input_rows_n"]
            or total_tokens != prompt_tokens + decode_tokens
        ):
            raise BudgetCheckError("manifest overflow event is inconsistent")
        if event.get("action") != "drop":
            raise BudgetCheckError("manifest overflow event action is invalid")

    arrival = _mapping(manifest.get("arrival"), "arrival")
    _exact_fields(arrival, frozenset({"min", "max", "span_s"}), "arrival")
    if counts["n"] == 0:
        if any(arrival.get(key) is not None for key in ("min", "max", "span_s")):
            raise BudgetCheckError("manifest empty arrival distribution is invalid")
    else:
        arrival_min = _finite_number(arrival.get("min"), "arrival.min")
        arrival_max = _finite_number(arrival.get("max"), "arrival.max")
        arrival_span = _finite_number(arrival.get("span_s"), "arrival.span_s")
        if arrival_min > arrival_max or float(arrival_span) != float(
            arrival_max - arrival_min
        ):
            raise BudgetCheckError("manifest arrival distribution is inconsistent")

    distributions = {
        key: _distribution(manifest.get(key), key, expected_n=counts["n"])
        for key in (
            "actual_prompt_tokens",
            "original_trace_prompt_tokens",
            "actual_minus_original_trace_tokens",
            "source_decode_tokens",
            "output_decode_tokens",
        )
    }
    if (
        distributions["actual_minus_original_trace_tokens"]["sum"]
        != distributions["actual_prompt_tokens"]["sum"]
        - distributions["original_trace_prompt_tokens"]["sum"]
    ):
        raise BudgetCheckError("manifest prompt-token delta sum is inconsistent")

    preservation = _mapping(
        manifest.get("preservation_checks"), "preservation_checks"
    )
    _exact_fields(preservation, PRESERVATION_FIELDS, "preservation_checks")
    preservation_counts = {
        key: _integer(preservation.get(key), f"manifest preservation_checks.{key}")
        for key in PRESERVATION_FIELDS
    }
    for key in (
        "prompt_text_exact_n",
        "arrived_at_exact_n",
        "decode_source_provenance_n",
        "decode_cap_respected_n",
        "session_id_exact_n",
        "token_metadata_aligned_n",
    ):
        if preservation_counts[key] != counts["n"]:
            raise BudgetCheckError("manifest preservation population is incomplete")
    if preservation_counts["decode_tokens_exact_n"] != (
        counts["n"] - counts["decode_cap_affected_n"]
    ):
        raise BudgetCheckError(
            "manifest decode-cap preservation population is inconsistent"
        )


def _utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise BudgetCheckError(f"{field} must be a UTC ISO-8601 Z timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise BudgetCheckError(f"{field} must be a UTC ISO-8601 Z timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise BudgetCheckError(f"{field} must be a UTC ISO-8601 Z timestamp")
    return parsed.astimezone(timezone.utc)


def _utc_now(now: datetime | None) -> datetime:
    current = datetime.now(timezone.utc) if now is None else now
    if current.tzinfo is None or current.utcoffset() is None:
        raise BudgetCheckError("current time must be timezone-aware")
    return current.astimezone(timezone.utc)


def _require_frozen_contract(
    *,
    expected_trace_sha256: str,
    expected_n: int,
    expected_prompt_token_sum: int,
    expected_decode_token_sum: int,
    expected_payload_mode: str,
    expected_cache_mode: str,
    arm_count: int,
    input_price: Decimal,
    output_price: Decimal,
    budget: Decimal,
) -> None:
    frozen = (
        (expected_trace_sha256, TRACE_SHA256, "expected_trace_sha256"),
        (expected_n, TRACE_N, "expected_n"),
        (expected_prompt_token_sum, PROMPT_TOKEN_SUM, "expected_prompt_token_sum"),
        (expected_decode_token_sum, DECODE_TOKEN_SUM, "expected_decode_token_sum"),
        (expected_payload_mode, PAYLOAD_MODE, "expected_payload_mode"),
        (expected_cache_mode, CACHE_MODE, "expected_cache_mode"),
        (arm_count, ARM_COUNT, "arm_count"),
        (input_price, INPUT_PRICE_PER_MILLION_USD, "input_price_per_million_usd"),
        (output_price, OUTPUT_PRICE_PER_MILLION_USD, "output_price_per_million_usd"),
        (budget, AUTHORIZED_BUDGET_USD, "budget_usd"),
    )
    for actual, expected, field in frozen:
        if actual != expected:
            raise BudgetCheckError(f"{field} does not match the frozen E12 contract")


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError:
        raise BudgetCheckError("unable to read manifest") from None
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BudgetCheckError("manifest is not valid UTF-8 JSON") from None
    if not isinstance(parsed, dict):
        raise BudgetCheckError("manifest root must be an object")
    return parsed, _sha256_bytes(raw)


def _load_price_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError:
        raise BudgetCheckError("unable to read price snapshot") from None
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BudgetCheckError("price snapshot is not valid UTF-8 JSON") from None
    if not isinstance(parsed, dict):
        raise BudgetCheckError("price snapshot root must be an object")
    return parsed, _sha256_bytes(raw)


def build_budget_attestation(
    *,
    manifest_path: Path,
    price_snapshot_path: Path,
    expected_manifest_sha256: str,
    expected_trace_sha256: str,
    expected_n: int,
    expected_prompt_token_sum: int,
    expected_decode_token_sum: int,
    expected_payload_mode: str,
    expected_cache_mode: str,
    arm_count: int,
    input_price_per_million_usd: str | Decimal,
    output_price_per_million_usd: str | Decimal,
    budget_usd: str | Decimal,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate the manifest and return a text-free, exact attestation."""
    expected_manifest_sha256 = _expected_sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    expected_trace_sha256 = _expected_sha256(
        expected_trace_sha256, "expected_trace_sha256"
    )
    expected_n = _integer(expected_n, "expected_n", minimum=1)
    expected_prompt_token_sum = _integer(
        expected_prompt_token_sum, "expected_prompt_token_sum"
    )
    expected_decode_token_sum = _integer(
        expected_decode_token_sum, "expected_decode_token_sum"
    )
    arm_count = _integer(arm_count, "arm_count", minimum=1)
    input_price = _decimal(
        input_price_per_million_usd, "input_price_per_million_usd"
    )
    output_price = _decimal(
        output_price_per_million_usd, "output_price_per_million_usd"
    )
    budget = _decimal(budget_usd, "budget_usd", positive=True)
    _require_frozen_contract(
        expected_trace_sha256=expected_trace_sha256,
        expected_n=expected_n,
        expected_prompt_token_sum=expected_prompt_token_sum,
        expected_decode_token_sum=expected_decode_token_sum,
        expected_payload_mode=expected_payload_mode,
        expected_cache_mode=expected_cache_mode,
        arm_count=arm_count,
        input_price=input_price,
        output_price=output_price,
        budget=budget,
    )

    price_snapshot, price_snapshot_sha256 = _load_price_snapshot(
        price_snapshot_path
    )
    if set(price_snapshot) != PRICE_SNAPSHOT_FIELDS:
        raise BudgetCheckError("price snapshot fields do not match schema v1")
    if (
        price_snapshot.get("schema_version") != 1
        or price_snapshot.get("status") != "pass"
        or price_snapshot.get("http_status") != 200
        or price_snapshot.get("model") != PRICE_MODEL
        or price_snapshot.get("provider") != PRICE_PROVIDER
        or price_snapshot.get("matching_endpoint_count") != 1
    ):
        raise BudgetCheckError("price snapshot identity is invalid")
    captured_at = _utc_timestamp(
        price_snapshot.get("captured_at_utc"), "price snapshot captured_at_utc"
    )
    current = _utc_now(now)
    age = current - captured_at
    if age < timedelta(0):
        raise BudgetCheckError("price snapshot timestamp is in the future")
    if age > PRICE_SNAPSHOT_MAX_AGE or captured_at.date() != current.date():
        raise BudgetCheckError("price snapshot is not a fresh same-UTC-day snapshot")
    snapshot_input_price = _decimal(
        price_snapshot.get("prompt_price_per_million_usd"),
        "price snapshot prompt price",
    )
    snapshot_output_price = _decimal(
        price_snapshot.get("completion_price_per_million_usd"),
        "price snapshot completion price",
    )
    if snapshot_input_price != input_price or snapshot_output_price != output_price:
        raise BudgetCheckError("budget prices do not match the verified price snapshot")
    snapshot_input_per_token = _decimal(
        price_snapshot.get("prompt_price_per_token_usd"),
        "price snapshot per-token prompt price",
    )
    snapshot_output_per_token = _decimal(
        price_snapshot.get("completion_price_per_token_usd"),
        "price snapshot per-token completion price",
    )
    snapshot_request_per_request_raw = price_snapshot.get(
        "request_price_per_request_usd"
    )
    if snapshot_request_per_request_raw != "0":
        raise BudgetCheckError(
            "price snapshot per-request price is not canonical frozen zero"
        )
    snapshot_request_per_request = _decimal(
        snapshot_request_per_request_raw,
        "price snapshot per-request price",
    )
    request_price_source = price_snapshot.get("request_price_source")
    if request_price_source not in {
        "absent_not_advertised",
        "explicit_zero",
    }:
        raise BudgetCheckError("price snapshot request-price source is invalid")
    if (
        snapshot_input_per_token * MILLION != input_price
        or snapshot_output_per_token * MILLION != output_price
        or snapshot_request_per_request != REQUEST_PRICE_PER_REQUEST_USD
    ):
        raise BudgetCheckError(
            "price snapshot token/request prices differ from the frozen contract"
        )
    context_length = _integer(
        price_snapshot.get("context_length"), "price snapshot context_length"
    )
    if context_length < 40_960:
        raise BudgetCheckError("price snapshot context length is below 40960")

    manifest, actual_manifest_sha256 = _load_manifest(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise BudgetCheckError("manifest SHA256 does not match the expected value")
    validate_materializer_manifest_schema(manifest)

    output_sha256 = manifest.get("output_sha256")
    if not isinstance(output_sha256, str) or not SHA256_RE.fullmatch(output_sha256):
        raise BudgetCheckError("manifest output_sha256 is invalid")
    if output_sha256.lower() != expected_trace_sha256:
        raise BudgetCheckError("trace SHA256 does not match the expected value")

    n = _integer(manifest.get("n"), "manifest n", minimum=1)
    if n != expected_n:
        raise BudgetCheckError("manifest n does not match the expected value")
    if manifest.get("payload_mode") != expected_payload_mode:
        raise BudgetCheckError("manifest payload_mode does not match current-turn mode")
    if manifest.get("cache_mode") != expected_cache_mode:
        raise BudgetCheckError("manifest cache_mode is not no-cache")

    prompt_stats = _mapping(manifest.get("actual_prompt_tokens"), "actual_prompt_tokens")
    decode_stats = _mapping(manifest.get("output_decode_tokens"), "output_decode_tokens")
    prompt_n = _integer(prompt_stats.get("n"), "actual_prompt_tokens.n")
    decode_n = _integer(decode_stats.get("n"), "output_decode_tokens.n")
    prompt_sum = _integer(prompt_stats.get("sum"), "actual_prompt_tokens.sum")
    decode_sum = _integer(decode_stats.get("sum"), "output_decode_tokens.sum")
    if prompt_n != n or decode_n != n:
        raise BudgetCheckError("aggregate token population does not match manifest n")
    if prompt_sum != expected_prompt_token_sum:
        raise BudgetCheckError("prompt-token sum does not match the expected value")
    if decode_sum != expected_decode_token_sum:
        raise BudgetCheckError("decode-token sum does not match the expected value")

    preservation = _mapping(manifest.get("preservation_checks"), "preservation_checks")
    aligned_n = _integer(
        preservation.get("token_metadata_aligned_n"),
        "preservation_checks.token_metadata_aligned_n",
    )
    if aligned_n != n:
        raise BudgetCheckError("manifest does not attest no-cache token alignment")

    prompt_cost = Decimal(arm_count) * Decimal(prompt_sum) * input_price / MILLION
    decode_cost = Decimal(arm_count) * Decimal(decode_sum) * output_price / MILLION
    request_cost = (
        Decimal(arm_count)
        * Decimal(n)
        * snapshot_request_per_request
    )
    estimated_cost = prompt_cost + decode_cost + request_cost
    if estimated_cost > budget:
        raise BudgetCheckError(
            "estimated full-response cost exceeds the authorized budget: "
            f"{_decimal_text(estimated_cost)} > {_decimal_text(budget)} USD"
        )

    # Whitelist every emitted field.  In particular, do not copy paths,
    # command lines, tokenizer labels, prompt samples, or arbitrary manifest
    # content into this attestation.
    return {
        "schema_version": 1,
        "status": "pass",
        "manifest_sha256": actual_manifest_sha256,
        "price_snapshot_sha256": price_snapshot_sha256,
        "trace_sha256": output_sha256.lower(),
        "trace_n": n,
        "payload_mode": expected_payload_mode,
        "cache_mode": expected_cache_mode,
        "prompt_token_sum": prompt_sum,
        "decode_token_sum": decode_sum,
        "arm_count": arm_count,
        "input_price_per_million_usd": _decimal_text(input_price),
        "output_price_per_million_usd": _decimal_text(output_price),
        "request_price_per_request_usd": _decimal_text(
            snapshot_request_per_request
        ),
        "request_price_source": request_price_source,
        "prompt_cost_usd": _decimal_text(prompt_cost),
        "decode_cost_usd": _decimal_text(decode_cost),
        "estimated_cost_usd": _decimal_text(estimated_cost),
        "budget_usd": _decimal_text(budget),
        "budget_headroom_usd": _decimal_text(budget - estimated_cost),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            json.dump(payload, target, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--price-snapshot", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-trace-sha256", required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--expected-prompt-token-sum", type=int, required=True)
    parser.add_argument("--expected-decode-token-sum", type=int, required=True)
    parser.add_argument("--expected-payload-mode", required=True)
    parser.add_argument("--expected-cache-mode", required=True)
    parser.add_argument("--arm-count", type=int, required=True)
    parser.add_argument("--input-price-per-million-usd", required=True)
    parser.add_argument("--output-price-per-million-usd", required=True)
    parser.add_argument("--budget-usd", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        attestation = build_budget_attestation(
            manifest_path=args.manifest,
            price_snapshot_path=args.price_snapshot,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_trace_sha256=args.expected_trace_sha256,
            expected_n=args.expected_n,
            expected_prompt_token_sum=args.expected_prompt_token_sum,
            expected_decode_token_sum=args.expected_decode_token_sum,
            expected_payload_mode=args.expected_payload_mode,
            expected_cache_mode=args.expected_cache_mode,
            arm_count=args.arm_count,
            input_price_per_million_usd=args.input_price_per_million_usd,
            output_price_per_million_usd=args.output_price_per_million_usd,
            budget_usd=args.budget_usd,
        )
        write_json_atomic(args.output, attestation)
    except BudgetCheckError as exc:
        print(f"budget check failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
