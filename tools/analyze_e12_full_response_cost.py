#!/usr/bin/env python3
"""Reprice the routed cohorts from an audited E12 live pair at full decode caps.

The live experiment deliberately cancelled cloud streams after the first token,
so its per-run cost fields are pending and its settled spend is not a production
full-response estimate.  This analyzer joins the audited routed rows back to the
restricted token-aligned trace and applies the frozen price snapshot to the
trace's capped decode lengths.  It never emits prompt text, request IDs, paths,
or provider-controlled strings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable


SHA256_RE = re.compile(r"[0-9a-f]{64}")
MILLION = Decimal("1000000")
MONEY_QUANTUM = Decimal("0.00000001")
PERCENT_QUANTUM = Decimal("0.000001")

TRACE_N = 11_604
MAX_DECODE_TOKENS = 1_024
MAX_CONTEXT_TOKENS = 40_960
SCENARIO = "extreme_burst_1200"
SCENARIO_START = 1_260_532
SCENARIO_END = 1_261_731
PAYLOAD_MODE = "sharegpt_current_turn_retokenized"
CACHE_MODE = "none"
MODEL = "qwen/qwen3-32b"
PROVIDER = "deepinfra"
INPUT_PRICE = Decimal("0.08")
OUTPUT_PRICE = Decimal("0.28")
REQUEST_PRICE = Decimal("0")
ARM_IDENTITIES = {
    "A": "ttft_pred:cost_cachedisp_old:0",
    "C": "ttft_pred:cost_disp_current:0",
}


class EvidenceError(ValueError):
    """A deliberately text-free validation error."""


@dataclass(frozen=True)
class TraceRow:
    prompt_tokens: int
    uncached_prompt_tokens: int
    decode_tokens: int
    source_request_index: int


@dataclass(frozen=True)
class ArmTotals:
    raw_sha256: str
    route_n: int
    success_n: int
    failure_n: int
    prompt_tokens: int
    decode_tokens: int
    success_prompt_tokens: int
    success_decode_tokens: int
    cloud_ids: frozenset[int]


def _sha256(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise EvidenceError(f"{label} is unreadable") from None
    return digest.hexdigest()


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{field} must be an object")
    return value


def _strict_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{field} must be an integer >= {minimum}")
    return value


def _decimal(value: Any, field: str, *, minimum: Decimal = Decimal("0")) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise EvidenceError(f"{field} must be numeric")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise EvidenceError(f"{field} must be numeric") from None
    if not result.is_finite() or result < minimum:
        raise EvidenceError(f"{field} is outside the accepted range")
    return result


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise EvidenceError(f"{label} is not valid JSON") from None
    return _object(value, label)


def _jsonl(path: Path, label: str) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_n, line in enumerate(handle, 1):
                try:
                    value = json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    raise EvidenceError(f"{label} contains invalid JSON") from None
                yield line_n, _object(value, f"{label} row")
    except OSError:
        raise EvidenceError(f"{label} is unreadable") from None


def _money(value: Decimal) -> str:
    return format(value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP), "f")


def _percent(value: Decimal) -> str:
    return format(value.quantize(PERCENT_QUANTUM, rounding=ROUND_HALF_UP), "f")


def _load_final_audit(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if not SHA256_RE.fullmatch(expected_sha256):
        raise EvidenceError("expected final-audit SHA256 is invalid")
    actual_sha256 = _sha256(path, "final audit")
    if actual_sha256 != expected_sha256:
        raise EvidenceError("final audit does not match the expected SHA256")
    audit = _json_object(path, "final audit")
    if audit.get("schema_version") != 1:
        raise EvidenceError("final audit schema is unsupported")
    if audit.get("analysis") != "E12 live ShareGPT current-turn A/C audit":
        raise EvidenceError("final audit analysis identity is wrong")
    if audit.get("verdict") != "pass" or audit.get("integrity_valid") is not True:
        raise EvidenceError("final audit did not pass")
    if audit.get("text_payload_in_output") is not False:
        raise EvidenceError("final audit text-payload boundary is invalid")
    evidence = _object(audit.get("evidence"), "final audit evidence")
    if _strict_int(evidence.get("trace_n"), "final audit trace_n") != TRACE_N:
        raise EvidenceError("final audit trace cardinality is wrong")
    for field in ("trace_sha256", "trace_manifest_sha256", "price_snapshot_sha256"):
        if not isinstance(evidence.get(field), str) or not SHA256_RE.fullmatch(evidence[field]):
            raise EvidenceError("final audit contains an invalid evidence SHA256")
    stages = _object(audit.get("stages"), "final audit stages")
    if set(stages) != set(ARM_IDENTITIES):
        raise EvidenceError("final audit stage set is wrong")
    return audit, actual_sha256


def _load_trace(
    trace_path: Path,
    manifest_path: Path,
    audit: dict[str, Any],
) -> tuple[list[TraceRow], str, str]:
    evidence = _object(audit["evidence"], "final audit evidence")
    trace_sha256 = _sha256(trace_path, "restricted trace")
    manifest_sha256 = _sha256(manifest_path, "trace manifest")
    if trace_sha256 != evidence["trace_sha256"]:
        raise EvidenceError("restricted trace does not match the final audit")
    if manifest_sha256 != evidence["trace_manifest_sha256"]:
        raise EvidenceError("trace manifest does not match the final audit")

    manifest = _json_object(manifest_path, "trace manifest")
    checks = {
        "schema_version": 1,
        "output_sha256": trace_sha256,
        "n": TRACE_N,
        "scenario": SCENARIO,
        "payload_mode": PAYLOAD_MODE,
        "cache_mode": CACHE_MODE,
        "max_decode_tokens": MAX_DECODE_TOKENS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "overflow_policy": "drop",
    }
    for field, expected in checks.items():
        if manifest.get(field) != expected:
            raise EvidenceError(f"trace manifest {field} is inconsistent")

    sortable: list[tuple[int, int, TraceRow]] = []
    source_indices: set[int] = set()
    for original_index, (_, row) in enumerate(_jsonl(trace_path, "restricted trace")):
        prompt_text = row.get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text:
            raise EvidenceError("restricted trace contains an invalid prompt payload")
        if row.get("payload_mode") != PAYLOAD_MODE or row.get("cache_mode") != CACHE_MODE:
            raise EvidenceError("restricted trace payload/cache identity is wrong")
        arrived_at = _strict_int(row.get("arrived_at"), "trace arrived_at")
        if not SCENARIO_START <= arrived_at <= SCENARIO_END:
            raise EvidenceError("restricted trace row is outside the frozen scenario")
        prompt = _strict_int(row.get("num_prefill_tokens"), "trace prompt tokens")
        uncached = _strict_int(
            row.get("uncached_prompt_tokens"), "trace uncached prompt tokens"
        )
        cached = _strict_int(row.get("num_cached_tokens"), "trace cached tokens")
        decode = _strict_int(row.get("num_decode_tokens"), "trace decode tokens")
        source_index = _strict_int(
            row.get("source_request_index"), "trace source request index"
        )
        if prompt != uncached or cached != 0:
            raise EvidenceError("restricted trace is not exact no-cache input")
        if decode > MAX_DECODE_TOKENS or prompt + decode > MAX_CONTEXT_TOKENS:
            raise EvidenceError("restricted trace exceeds its token contract")
        if source_index in source_indices:
            raise EvidenceError("restricted trace source indices are not unique")
        source_indices.add(source_index)
        sortable.append(
            (
                arrived_at,
                original_index,
                TraceRow(prompt, uncached, decode, source_index),
            )
        )
    sortable.sort(key=lambda item: (item[0], item[1]))
    rows = [item[2] for item in sortable]
    if len(rows) != TRACE_N:
        raise EvidenceError("restricted trace row count is wrong")

    prompt_stats = _object(manifest.get("actual_prompt_tokens"), "prompt stats")
    decode_stats = _object(manifest.get("output_decode_tokens"), "decode stats")
    if _strict_int(prompt_stats.get("sum"), "manifest prompt sum") != sum(
        row.prompt_tokens for row in rows
    ):
        raise EvidenceError("trace manifest prompt sum is inconsistent")
    if _strict_int(decode_stats.get("sum"), "manifest decode sum") != sum(
        row.decode_tokens for row in rows
    ):
        raise EvidenceError("trace manifest decode sum is inconsistent")
    return rows, trace_sha256, manifest_sha256


def _load_price(path: Path, audit: dict[str, Any]) -> tuple[dict[str, Decimal], str]:
    evidence = _object(audit["evidence"], "final audit evidence")
    price_sha256 = _sha256(path, "price snapshot")
    if price_sha256 != evidence["price_snapshot_sha256"]:
        raise EvidenceError("price snapshot does not match the final audit")
    price = _json_object(path, "price snapshot")
    expected_fields = {
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
        "prompt_price_per_million_usd",
        "completion_price_per_million_usd",
        "request_price_per_request_usd",
        "request_price_source",
    }
    if set(price) != expected_fields:
        raise EvidenceError("price snapshot schema is not exact")
    fixed = {
        "schema_version": 1,
        "status": "pass",
        "http_status": 200,
        "model": MODEL,
        "provider": PROVIDER,
        "matching_endpoint_count": 1,
        "context_length": MAX_CONTEXT_TOKENS,
        "request_price_source": "absent_not_advertised",
    }
    for field, expected in fixed.items():
        if price.get(field) != expected:
            raise EvidenceError(f"price snapshot {field} is inconsistent")
    input_price = _decimal(price.get("prompt_price_per_million_usd"), "input price")
    output_price = _decimal(
        price.get("completion_price_per_million_usd"), "output price"
    )
    request_price = _decimal(
        price.get("request_price_per_request_usd"), "request price"
    )
    if (input_price, output_price, request_price) != (
        INPUT_PRICE,
        OUTPUT_PRICE,
        REQUEST_PRICE,
    ):
        raise EvidenceError("price snapshot changed from the frozen contract")
    if _decimal(price.get("prompt_price_per_token_usd"), "input token price") != (
        INPUT_PRICE / MILLION
    ):
        raise EvidenceError("input token price is inconsistent")
    if _decimal(
        price.get("completion_price_per_token_usd"), "output token price"
    ) != (OUTPUT_PRICE / MILLION):
        raise EvidenceError("output token price is inconsistent")
    return {
        "input": input_price,
        "output": output_price,
        "request": request_price,
    }, price_sha256


def _load_arm(
    label: str,
    raw_path: Path,
    trace: list[TraceRow],
    audit: dict[str, Any],
) -> ArmTotals:
    stage = _object(_object(audit["stages"], "final audit stages").get(label), "stage")
    if stage.get("arm") != ARM_IDENTITIES[label]:
        raise EvidenceError(f"stage {label} arm identity is wrong")
    if stage.get("applied_victims_equal_cloud_ids") is not True:
        raise EvidenceError(f"stage {label} did not validate applied victims")
    artifacts = _object(stage.get("artifacts"), f"stage {label} artifacts")
    expected_raw_sha256 = artifacts.get("raw_sha256")
    if not isinstance(expected_raw_sha256, str) or not SHA256_RE.fullmatch(
        expected_raw_sha256
    ):
        raise EvidenceError(f"stage {label} raw SHA256 is invalid")
    raw_sha256 = _sha256(raw_path, f"stage {label} raw")
    if raw_sha256 != expected_raw_sha256:
        raise EvidenceError(f"stage {label} raw does not match the final audit")

    ids: set[int] = set()
    cloud_ids: set[int] = set()
    route_n = success_n = 0
    prompt_tokens = decode_tokens = 0
    success_prompt_tokens = success_decode_tokens = 0
    row_n = 0
    for _, row in _jsonl(raw_path, f"stage {label} raw"):
        row_n += 1
        request_id = _strict_int(row.get("request_id"), f"stage {label} request_id")
        if request_id >= len(trace) or request_id in ids:
            raise EvidenceError(f"stage {label} request IDs are invalid")
        ids.add(request_id)
        endpoint = row.get("endpoint")
        if endpoint not in {"local", "cloud"}:
            raise EvidenceError(f"stage {label} endpoint is invalid")
        if row.get("payload_mode") != PAYLOAD_MODE or row.get("cache_mode") != CACHE_MODE:
            raise EvidenceError(f"stage {label} payload/cache identity is wrong")
        success = row.get("success")
        if not isinstance(success, bool):
            raise EvidenceError(f"stage {label} success flag is invalid")
        scheduler_prompt = _strict_int(
            row.get("scheduler_prompt_tokens"), f"stage {label} scheduler prompt"
        )
        scheduler_uncached = _strict_int(
            row.get("scheduler_uncached_prompt_tokens"),
            f"stage {label} scheduler uncached prompt",
        )
        scheduler_decode = _strict_int(
            row.get("scheduler_decode_tokens"), f"stage {label} scheduler decode"
        )
        source_index = _strict_int(
            row.get("source_request_index"), f"stage {label} source request index"
        )
        expected = trace[request_id]
        if (
            scheduler_prompt != expected.prompt_tokens
            or scheduler_uncached != expected.uncached_prompt_tokens
            or scheduler_decode != expected.decode_tokens
            or source_index != expected.source_request_index
        ):
            raise EvidenceError(f"stage {label} scheduler tokens do not match the trace")
        if endpoint == "cloud":
            cloud_ids.add(request_id)
            route_n += 1
            prompt_tokens += scheduler_uncached
            decode_tokens += scheduler_decode
            if success:
                success_n += 1
                success_prompt_tokens += scheduler_uncached
                success_decode_tokens += scheduler_decode
    if row_n != TRACE_N or ids != set(range(TRACE_N)):
        raise EvidenceError(f"stage {label} raw IDs do not exactly cover the trace")
    failure_n = route_n - success_n
    if route_n == 0:
        raise EvidenceError(f"stage {label} contains no cloud routes")
    expected_counts = {
        "n": TRACE_N,
        "cloud_n": route_n,
        "cloud_success_n": success_n,
        "cloud_failure_n": failure_n,
        "applied_victim_n": route_n,
    }
    for field, expected in expected_counts.items():
        if _strict_int(stage.get(field), f"stage {label} {field}") != expected:
            raise EvidenceError(f"stage {label} {field} does not match the raw rows")
    return ArmTotals(
        raw_sha256=raw_sha256,
        route_n=route_n,
        success_n=success_n,
        failure_n=failure_n,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        success_prompt_tokens=success_prompt_tokens,
        success_decode_tokens=success_decode_tokens,
        cloud_ids=frozenset(cloud_ids),
    )


def _cost(tokens_prompt: int, tokens_decode: int, route_n: int) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    input_cost = Decimal(tokens_prompt) * INPUT_PRICE / MILLION
    output_cost = Decimal(tokens_decode) * OUTPUT_PRICE / MILLION
    request_cost = Decimal(route_n) * REQUEST_PRICE
    return input_cost, output_cost, request_cost, input_cost + output_cost + request_cost


def _arm_result(label: str, arm: ArmTotals) -> dict[str, Any]:
    input_cost, output_cost, request_cost, total_cost = _cost(
        arm.prompt_tokens, arm.decode_tokens, arm.route_n
    )
    s_input, s_output, s_request, s_total = _cost(
        arm.success_prompt_tokens, arm.success_decode_tokens, arm.success_n
    )
    return {
        "arm": ARM_IDENTITIES[label],
        "raw_sha256": arm.raw_sha256,
        "route_n": arm.route_n,
        "success_n": arm.success_n,
        "failure_n": arm.failure_n,
        "prompt_tokens": arm.prompt_tokens,
        "capped_decode_tokens": arm.decode_tokens,
        "input_cost_usd": _money(input_cost),
        "output_cost_usd": _money(output_cost),
        "request_cost_usd": _money(request_cost),
        "modeled_full_cap_cost_usd": _money(total_cost),
        "successful_cloud_rows_only_sensitivity": {
            "route_n": arm.success_n,
            "prompt_tokens": arm.success_prompt_tokens,
            "capped_decode_tokens": arm.success_decode_tokens,
            "input_cost_usd": _money(s_input),
            "output_cost_usd": _money(s_output),
            "request_cost_usd": _money(s_request),
            "modeled_full_cap_cost_usd": _money(s_total),
        },
    }


def analyze(
    *,
    trace_path: Path,
    trace_manifest_path: Path,
    price_snapshot_path: Path,
    final_audit_path: Path,
    expected_final_audit_sha256: str,
    a_raw_path: Path,
    c_raw_path: Path,
) -> dict[str, Any]:
    audit, audit_sha256 = _load_final_audit(
        final_audit_path, expected_final_audit_sha256
    )
    trace, trace_sha256, manifest_sha256 = _load_trace(
        trace_path, trace_manifest_path, audit
    )
    _, price_sha256 = _load_price(price_snapshot_path, audit)
    arms = {
        "A": _load_arm("A", a_raw_path, trace, audit),
        "C": _load_arm("C", c_raw_path, trace, audit),
    }
    a = arms["A"]
    c = arms["C"]
    _, _, _, a_cost = _cost(a.prompt_tokens, a.decode_tokens, a.route_n)
    _, _, _, c_cost = _cost(c.prompt_tokens, c.decode_tokens, c.route_n)
    intersection = len(a.cloud_ids & c.cloud_ids)
    union = len(a.cloud_ids | c.cloud_ids)
    if a_cost < c_cost:
        classification = "a_lower_modeled_cost_on_this_ordered_pair_only"
    elif c_cost < a_cost:
        classification = "c_lower_modeled_cost_on_this_ordered_pair_only"
    else:
        classification = "equal_modeled_cost_on_this_ordered_pair_only"
    result = {
        "schema_version": 1,
        "analysis": "E12 selected-cohort full-response capped-decode counterfactual",
        "integrity_valid": True,
        "text_payload_in_output": False,
        "assumptions": {
            "cost_kind": "modeled_full_response_capped_decode",
            "includes_failed_cloud_routes": True,
            "prompt_tokens": "token_aligned_uncached_trace_tokens",
            "completion_tokens": "trace_oracle_capped_at_1024",
            "early_eos_modeled": False,
            "actual_provider_bill": False,
        },
        "evidence": {
            "tool_sha256": _sha256(Path(__file__).resolve(), "analyzer"),
            "final_audit_sha256": audit_sha256,
            "trace_sha256": trace_sha256,
            "trace_manifest_sha256": manifest_sha256,
            "price_snapshot_sha256": price_sha256,
            "trace_n": len(trace),
        },
        "price": {
            "input_per_million_usd": _money(INPUT_PRICE),
            "output_per_million_usd": _money(OUTPUT_PRICE),
            "request_per_request_usd": _money(REQUEST_PRICE),
        },
        "arms": {label: _arm_result(label, arm) for label, arm in arms.items()},
        "comparison": {
            "c_minus_a_route_n": c.route_n - a.route_n,
            "c_minus_a_prompt_tokens": c.prompt_tokens - a.prompt_tokens,
            "c_minus_a_capped_decode_tokens": c.decode_tokens - a.decode_tokens,
            "c_minus_a_modeled_full_cap_cost_usd": _money(c_cost - a_cost),
            "c_cost_premium_vs_a_pct": _percent(
                (c_cost - a_cost) * Decimal("100") / a_cost
            ),
            "a_cost_saving_vs_c_pct": _percent(
                (c_cost - a_cost) * Decimal("100") / c_cost
            ),
            "pair_modeled_full_cap_cost_usd": _money(a_cost + c_cost),
            "cloud_id_intersection_n": intersection,
            "a_only_n": len(a.cloud_ids - c.cloud_ids),
            "c_only_n": len(c.cloud_ids - a.cloud_ids),
            "cloud_id_union_n": union,
            "cloud_id_jaccard": _percent(
                Decimal(intersection) / Decimal(union)
            ),
            "classification": classification,
        },
    }
    # Defense in depth: this output is a fixed whitelist and must not contain
    # any value derived from prompt text, paths, argv, or arbitrary strings.
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if "prompt_text" in encoded or "source_request_index" in encoded:
        raise EvidenceError("analysis output crossed the text-free boundary")
    return result


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise EvidenceError("analysis output already exists")
    try:
        parent = path.parent.resolve(strict=True)
        fd, temporary_name = tempfile.mkstemp(prefix=".e12_full_cost.", dir=parent)
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                raise EvidenceError("analysis output already exists") from None
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    except EvidenceError:
        raise
    except OSError:
        raise EvidenceError("analysis output could not be written") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--price-snapshot", type=Path, required=True)
    parser.add_argument("--final-audit", type=Path, required=True)
    parser.add_argument("--expected-final-audit-sha256", required=True)
    parser.add_argument("--a-raw", type=Path, required=True)
    parser.add_argument("--c-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = analyze(
            trace_path=args.trace,
            trace_manifest_path=args.trace_manifest,
            price_snapshot_path=args.price_snapshot,
            final_audit_path=args.final_audit,
            expected_final_audit_sha256=args.expected_final_audit_sha256,
            a_raw_path=args.a_raw,
            c_raw_path=args.c_raw,
        )
        payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if args.output is not None:
            _write_new(args.output, payload)
        sys.stdout.buffer.write(payload)
        return 0
    except EvidenceError as exc:
        print(f"full-response cost analysis failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
