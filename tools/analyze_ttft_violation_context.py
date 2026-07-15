#!/usr/bin/env python3
"""Audit why one ``ttft_pred`` arm produced local TTFT violations.

The analysis is deliberately diagnostic.  It reconstructs application-side
dispatch occupancy from completed raw rows, compares that state with the
profile's calibrated support, and associates each violation with the closest
decision logged before dispatch.  A planned peak commitment is *not* actual
KV usage: it is the sum of ``prompt + requested decode`` for locally active
requests.  The engine-reported GPU cache-usage gauge is reported only when the
caller also supplies an explicit run start in the server log's own clock.  Its
semantics are architecture-dependent and are not assumed to be token-KV
pressure.

The current decision schema stores only a snapshot hash/count, not snapshot
IDs, waiting ages, per-request predictions, selector scores, or in-flight
identities.  Consequently this tool cannot replay selector ranking or assign
selector causality; the report says so explicitly.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import heapq
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


class EvidenceError(ValueError):
    """An input artifact is malformed or inconsistent with the other inputs."""


EPSILON_MS = 1e-3
DISPATCH_MATCH_EPSILON_S = 1e-3
SERVER_STAMP_FORMAT = "%m-%d %H:%M:%S"
SERVER_METRIC_RE = re.compile(
    r"(?P<stamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[0-9.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%"
)


def _number(value: Any, field: str, source: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{source}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{source}: {field} must be finite")
    if minimum is not None and result < minimum:
        raise EvidenceError(f"{source}: {field} must be >= {minimum}")
    return result


def _integer(value: Any, field: str, source: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{source}: {field} must be an integer")
    if minimum is not None and value < minimum:
        raise EvidenceError(f"{source}: {field} must be >= {minimum}")
    return value


def _request_id(value: Any, field: str, source: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise EvidenceError(f"{source}: {field} must be a string or integer")
    return value


def _object(mapping: dict[str, Any], key: str, source: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise EvidenceError(f"{source}: missing object {key!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"{path}: cannot read: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{path}: cannot read valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{path}: top-level JSON must be an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_n, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvidenceError(
                        f"{path}:{line_n}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise EvidenceError(f"{path}:{line_n}: row must be an object")
                value["_source_line"] = line_n
                rows.append(value)
    except OSError as exc:
        raise EvidenceError(f"{path}: cannot read: {exc}") from exc
    if not rows:
        raise EvidenceError(f"{path}: JSONL has no non-empty rows")
    return rows


def _artifact_metadata(path: Path) -> dict[str, Any]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"{path}: cannot read bound artifact: {exc}") from exc
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "nonempty_line_n": sum(bool(line.strip()) for line in content.splitlines()),
    }


def _exact_sha256(value: Any, field: str, source: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EvidenceError(f"{source}: {field} must be a lowercase SHA-256 hex digest")
    return value


def _parse_marker_arm(value: Any, source: str) -> tuple[str, str, int, str]:
    if not isinstance(value, str) or value != value.strip():
        raise EvidenceError(f"{source}: marker arm must be an exact string")
    parts = value.split(":")
    if len(parts) != 3 or any(not part for part in parts):
        raise EvidenceError(f"{source}: invalid marker arm {value!r}")
    trigger, selector, seed_text = parts
    if trigger != "ttft_pred":
        raise EvidenceError(f"{source}: marker is not a ttft_pred arm")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise EvidenceError(f"{source}: marker seed is not an integer") from exc
    if str(seed) != seed_text:
        raise EvidenceError(f"{source}: marker seed is not canonical")
    return trigger, selector, seed, value


def _parse_manifest(
    path: Path,
    *,
    marker_fingerprint: str,
    marker_arm: str,
    profile_sha256: str,
    raw_row_n: int,
) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"{path}: cannot read matrix manifest: {exc}") from exc
    fields: dict[str, str] = {}
    arms: list[str] | None = None
    for line_n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if line.startswith("arms="):
            if arms is not None:
                raise EvidenceError(f"{path}:{line_n}: duplicate arms line")
            arms = line[len("arms="):].split()
            continue
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if not key or key in fields:
                raise EvidenceError(f"{path}:{line_n}: duplicate/invalid field {key!r}")
            fields[key] = value
    required = {
        "run_fingerprint", "profile_sha256", "trace_n", "model",
        "prefill_tput", "tpot_ms", "first_token_overhead_ms", "slo_s",
        "guard_ms", "kv_cap", "server_pid", "server_log",
        "server_log_prefix_sha256",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise EvidenceError(f"{path}: manifest missing fields {', '.join(missing)}")
    if fields["run_fingerprint"] != marker_fingerprint:
        raise EvidenceError(f"{path}: manifest/marker run_fingerprint mismatch")
    if fields["profile_sha256"] != profile_sha256:
        raise EvidenceError(f"{path}: manifest profile_sha256 does not match supplied profile")
    _exact_sha256(fields["profile_sha256"], "profile_sha256", str(path))
    _exact_sha256(
        fields["server_log_prefix_sha256"],
        "server_log_prefix_sha256",
        str(path),
    )
    try:
        manifest_trace_n = int(fields["trace_n"])
    except ValueError as exc:
        raise EvidenceError(f"{path}: trace_n must be an integer") from exc
    if manifest_trace_n != raw_row_n:
        raise EvidenceError(
            f"{path}: trace_n={manifest_trace_n} does not match bound raw rows {raw_row_n}"
        )
    if arms is None or arms.count(marker_arm) != 1:
        raise EvidenceError(f"{path}: arms line does not contain this marker arm exactly once")
    fields["arms"] = " ".join(arms)
    return fields


def _load_bound_evidence(
    raw_path: Path,
    decisions_path: Path,
    summary_path: Path,
    marker_path: Path,
    manifest_path: Path,
    profile_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    source = str(marker_path)
    if not marker_path.name.endswith(".complete.json"):
        raise EvidenceError(f"{source}: marker filename must end with .complete.json")
    stem = marker_path.name[:-len(".complete.json")]
    expected_names = {
        raw_path: stem + ".jsonl",
        decisions_path: stem + ".decisions.jsonl",
        summary_path: stem + ".summary.json",
    }
    marker_parent = marker_path.resolve().parent
    for path, expected_name in expected_names.items():
        if path.resolve().parent != marker_parent or path.name != expected_name:
            raise EvidenceError(
                f"{source}: bound artifact must be sibling {expected_name!r}, got {path}"
            )
    if manifest_path.resolve().parent != marker_parent or manifest_path.name != "matrix_manifest.txt":
        raise EvidenceError(f"{source}: manifest must be sibling matrix_manifest.txt")

    actual_artifacts = {
        "raw": _artifact_metadata(raw_path),
        "summary": _artifact_metadata(summary_path),
        "decisions": _artifact_metadata(decisions_path),
    }
    marker = _read_json(marker_path)
    fingerprint = marker.get("run_fingerprint")
    if (not isinstance(fingerprint, str) or not fingerprint.strip()
            or fingerprint != fingerprint.strip()):
        raise EvidenceError(f"{source}: run_fingerprint must be a non-empty exact string")
    trigger, selector, seed, arm_text = _parse_marker_arm(marker.get("arm"), source)
    expected_marker = {
        "schema_version": 1,
        "run_fingerprint": fingerprint,
        "arm": arm_text,
        "artifacts": actual_artifacts,
    }
    if marker != expected_marker:
        raise EvidenceError(
            f"{source}: completion marker does not exactly match supplied artifacts"
        )
    profile_sha256 = _sha256(profile_path)
    manifest = _parse_manifest(
        manifest_path,
        marker_fingerprint=fingerprint,
        marker_arm=arm_text,
        profile_sha256=profile_sha256,
        raw_row_n=actual_artifacts["raw"]["nonempty_line_n"],
    )
    summary = _read_json(summary_path)
    binding = {
        "run_fingerprint": fingerprint,
        "arm": arm_text,
        "trigger": trigger,
        "selector": selector,
        "seed": seed,
        "artifacts": actual_artifacts,
        "profile_sha256": profile_sha256,
    }
    return summary, binding, manifest


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _summary(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {
            "n": 0, "min": None, "p25": None, "p50": None, "p75": None,
            "p90": None, "p95": None, "p99": None, "max": None, "mean": None,
        }
    return {
        "n": len(present),
        "min": min(present),
        "p25": _percentile(present, 0.25),
        "p50": _percentile(present, 0.50),
        "p75": _percentile(present, 0.75),
        "p90": _percentile(present, 0.90),
        "p95": _percentile(present, 0.95),
        "p99": _percentile(present, 0.99),
        "max": max(present),
        "mean": sum(present) / len(present),
    }


def _ratio(value: float | None, denominator: float) -> float | None:
    return None if value is None or denominator == 0 else value / denominator


def _load_profile(path: Path, slo_override: float | None) -> dict[str, Any]:
    source = str(path)
    profile = _read_json(path)
    if profile.get("valid") is not True:
        raise EvidenceError(f"{source}: profile.valid must be true")
    model = profile.get("predictor_model")
    if not isinstance(model, str) or not model:
        raise EvidenceError(f"{source}: predictor_model must be a non-empty string")
    calibration = _object(profile, "predictor_calibration", source)
    profile_config = _object(profile, "profile_config", source)

    target_slo_s = _number(
        calibration.get("target_slo_s"),
        "predictor_calibration.target_slo_s",
        source,
        minimum=0.0,
    )
    if target_slo_s <= 0:
        raise EvidenceError(f"{source}: target SLO must be positive")
    if slo_override is not None:
        slo_s = _number(slo_override, "--slo-s", "arguments", minimum=0.0)
        if slo_s <= 0:
            raise EvidenceError("arguments: --slo-s must be positive")
        if not math.isclose(slo_s, target_slo_s, rel_tol=0.0, abs_tol=1e-9):
            raise EvidenceError(
                "arguments: --slo-s must equal the manifest-bound profile target SLO"
            )
    else:
        slo_s = target_slo_s

    calibrated_tpot_ms = _number(
        calibration.get("recommended_tpot_ms"),
        "predictor_calibration.recommended_tpot_ms",
        source,
        minimum=0.0,
    )
    first_token_overhead_ms = _number(
        calibration.get("recommended_first_token_overhead_ms"),
        "predictor_calibration.recommended_first_token_overhead_ms",
        source,
        minimum=0.0,
    )
    guard_ms = _number(
        calibration.get("recommended_ttft_guard_ms"),
        "predictor_calibration.recommended_ttft_guard_ms",
        source,
        minimum=0.0,
    )
    if guard_ms >= slo_s * 1000.0:
        raise EvidenceError(
            f"{source}: recommended TTFT guard must be smaller than the analyzed SLO"
        )
    tick_ms = _number(
        calibration.get("nimbus_tick_guard_ms"),
        "predictor_calibration.nimbus_tick_guard_ms",
        source,
        minimum=0.0,
    )
    prefill_tput = _number(
        calibration.get("recommended_prefill_tput_tokens_per_s"),
        "predictor_calibration.recommended_prefill_tput_tokens_per_s",
        source,
        minimum=0.0,
    )
    if prefill_tput <= 0:
        raise EvidenceError(f"{source}: calibrated prefill throughput must be positive")
    fixed_decode = _integer(
        profile_config.get("decode_tokens"),
        "profile_config.decode_tokens",
        source,
        minimum=0,
    )
    kv_capacity = _number(
        profile.get("kv_capacity_tokens"),
        "kv_capacity_tokens",
        source,
        minimum=0.0,
    )
    if kv_capacity <= 0:
        raise EvidenceError(f"{source}: kv_capacity_tokens must be positive")

    cells = profile.get("cells")
    if not isinstance(cells, list) or not cells:
        raise EvidenceError(f"{source}: cells must be a non-empty array")
    prompt_points: set[int] = set()
    concurrency_points: set[int] = set()
    peak_values: list[float] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise EvidenceError(f"{source}: cells[{index}] must be an object")
        prompt_points.add(_integer(
            cell.get("prompt_tokens"), f"cells[{index}].prompt_tokens", source,
            minimum=0,
        ))
        concurrency_points.add(_integer(
            cell.get("offered_concurrency"),
            f"cells[{index}].offered_concurrency", source, minimum=1,
        ))
        peak_values.append(_number(
            cell.get("actual_estimated_peak_tokens_max"),
            f"cells[{index}].actual_estimated_peak_tokens_max", source,
            minimum=0.0,
        ))
    if max(peak_values) <= 0:
        raise EvidenceError(f"{source}: calibrated peak commitment must be positive")
    cache_mode_required = profile.get("cache_mode_required")
    if cache_mode_required is not None and not isinstance(cache_mode_required, str):
        raise EvidenceError(f"{source}: cache_mode_required must be a string or null")
    deployment_model = profile.get("model")
    if not isinstance(deployment_model, str) or not deployment_model:
        raise EvidenceError(f"{source}: model must be a non-empty string")
    server_log = profile.get("server_log")
    if not isinstance(server_log, str) or not server_log:
        raise EvidenceError(f"{source}: server_log must be a non-empty string")
    server_pid = _integer(profile.get("server_pid"), "server_pid", source, minimum=1)
    server_log_prefix_bytes = _integer(
        profile.get("server_log_bytes_at_start"),
        "server_log_bytes_at_start",
        source,
        minimum=1,
    )
    server_log_prefix_sha256 = _exact_sha256(
        profile.get("server_log_sha256_at_start"),
        "server_log_sha256_at_start",
        source,
    )
    return {
        "predictor_model": model,
        "slo_s": slo_s,
        "profile_target_slo_s": target_slo_s,
        "calibrated_tpot_ms": calibrated_tpot_ms,
        "first_token_overhead_ms": first_token_overhead_ms,
        "guard_ms": guard_ms,
        "tick_ms": tick_ms,
        "prefill_tput_tokens_per_s": prefill_tput,
        "fixed_decode_tokens": fixed_decode,
        "kv_capacity_tokens": kv_capacity,
        "prompt_points": sorted(prompt_points),
        "prompt_min": min(prompt_points),
        "prompt_max": max(prompt_points),
        "concurrency_points": sorted(concurrency_points),
        "concurrency_max": max(concurrency_points),
        "calibrated_peak_commitment_max_tokens": max(peak_values),
        "calibration_scope": calibration.get("scope"),
        "cache_mode_required": cache_mode_required,
        "deployment_model": deployment_model,
        "server_log": server_log,
        "server_pid": server_pid,
        "server_log_prefix_bytes": server_log_prefix_bytes,
        "server_log_prefix_sha256": server_log_prefix_sha256,
    }


def _manifest_number(manifest: dict[str, str], field: str, source: str) -> float:
    try:
        result = float(manifest[field])
    except (KeyError, ValueError) as exc:
        raise EvidenceError(f"{source}: manifest {field} must be numeric") from exc
    if not math.isfinite(result):
        raise EvidenceError(f"{source}: manifest {field} must be finite")
    return result


def _numbers_match(actual: Any, expected: float, field: str, source: str) -> None:
    value = _number(actual, field, source)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9):
        raise EvidenceError(f"{source}: {field}={value} does not match bound value {expected}")


def _validate_profile_manifest(
    profile: dict[str, Any], manifest: dict[str, str], source: str
) -> None:
    numeric = {
        "prefill_tput": profile["prefill_tput_tokens_per_s"],
        "tpot_ms": profile["calibrated_tpot_ms"],
        "first_token_overhead_ms": profile["first_token_overhead_ms"],
        "slo_s": profile["slo_s"],
        "guard_ms": profile["guard_ms"],
        "kv_cap": profile["kv_capacity_tokens"],
    }
    for field, expected in numeric.items():
        actual = _manifest_number(manifest, field, source)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise EvidenceError(
                f"{source}: manifest {field}={actual} does not match profile {expected}"
            )
    exact = {
        "model": profile["deployment_model"],
        "server_pid": str(profile["server_pid"]),
        "server_log": profile["server_log"],
        "server_log_prefix_sha256": profile["server_log_prefix_sha256"],
    }
    if profile["cache_mode_required"] is not None:
        exact["cache_mode"] = profile["cache_mode_required"]
    for field, expected in exact.items():
        if manifest.get(field) != expected:
            raise EvidenceError(
                f"{source}: manifest {field}={manifest.get(field)!r} does not match "
                f"profile {expected!r}"
            )


def _validate_summary_binding(
    summary: dict[str, Any],
    *,
    profile: dict[str, Any],
    binding: dict[str, Any],
    manifest: dict[str, str],
    raw_rows: Sequence[dict[str, Any]],
    local: Sequence[dict[str, Any]],
    violations: Sequence[dict[str, Any]],
    decision_selector: str,
    source: str,
) -> None:
    if summary.get("policy") != "nimbus":
        raise EvidenceError(f"{source}: summary policy must be 'nimbus'")
    config = _object(summary, "config", source)
    queue = _object(summary, "queue", source)
    if config.get("nimbus_trigger") != binding["trigger"]:
        raise EvidenceError(f"{source}: summary trigger does not match marker")
    if config.get("nimbus_selector") != binding["selector"]:
        raise EvidenceError(f"{source}: summary selector does not match marker")
    if decision_selector != binding["selector"]:
        raise EvidenceError(f"{source}: decision selector does not match marker")
    if queue.get("nimbus_trigger") != binding["trigger"]:
        raise EvidenceError(f"{source}: queue trigger does not match marker")
    if queue.get("nimbus_selector") != binding["selector"]:
        raise EvidenceError(f"{source}: queue selector does not match marker")
    if queue.get("nimbus_prediction_model") != profile["predictor_model"]:
        raise EvidenceError(f"{source}: queue prediction model does not match profile")
    if queue.get("nimbus_prediction_scope") != "waiting_only":
        raise EvidenceError(f"{source}: queue prediction scope must be waiting_only")
    seed = _integer(config.get("seed"), "config.seed", source)
    if seed != binding["seed"]:
        raise EvidenceError(f"{source}: summary seed does not match marker")

    numeric = {
        "config.slo_s": (config.get("slo_s"), profile["slo_s"]),
        "config.ttft_guard_ms": (config.get("ttft_guard_ms"), profile["guard_ms"]),
        "config.prefill_tput": (
            config.get("prefill_tput"), profile["prefill_tput_tokens_per_s"]
        ),
        "config.tpot_ms": (config.get("tpot_ms"), profile["calibrated_tpot_ms"]),
        "config.first_token_overhead_ms": (
            config.get("first_token_overhead_ms"), profile["first_token_overhead_ms"]
        ),
        "config.kv_capacity_tokens": (
            config.get("kv_capacity_tokens"), profile["kv_capacity_tokens"]
        ),
        "slo_s": (summary.get("slo_s"), profile["slo_s"]),
    }
    for field, (actual, expected) in numeric.items():
        _numbers_match(actual, expected, field, source)
    if config.get("local_model") != profile["deployment_model"]:
        raise EvidenceError(f"{source}: config.local_model does not match profile model")
    if manifest["model"] != config["local_model"]:
        raise EvidenceError(f"{source}: summary model does not match manifest model")

    overall = _object(summary, "overall", source)
    local_summary = _object(summary, "local", source)
    cloud_summary = _object(summary, "cloud", source)
    request_n = len(raw_rows)
    local_n = len(local)
    cloud_n = request_n - local_n
    expected_counts = {
        "overall.n": (overall.get("n"), request_n),
        "overall.success": (overall.get("success"), request_n),
        "local.n": (local_summary.get("n"), local_n),
        "local.success": (local_summary.get("success"), local_n),
        "local.slo_measured_n": (local_summary.get("slo_measured_n"), local_n),
        "local.slo_violations": (local_summary.get("slo_violations"), len(violations)),
        "cloud.n": (cloud_summary.get("n"), cloud_n),
        "cloud.success": (cloud_summary.get("success"), cloud_n),
    }
    for field, (actual, expected) in expected_counts.items():
        value = _integer(actual, field, source, minimum=0)
        if value != expected:
            raise EvidenceError(f"{source}: {field}={value}, bound raw requires {expected}")


def _load_raw(
    path: Path,
    slo_s: float,
    cache_mode_required: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_jsonl(path)
    request_ids: set[str | int] = set()
    local: list[dict[str, Any]] = []
    for row in rows:
        line = row.pop("_source_line")
        source = f"{path}:{line}"
        request_id = _request_id(row.get("request_id"), "request_id", source)
        if request_id in request_ids:
            raise EvidenceError(f"{source}: duplicate request_id {request_id!r}")
        request_ids.add(request_id)
        if row.get("success") is not True:
            raise EvidenceError(f"{source}: request is not successful")
        endpoint = row.get("endpoint")
        if endpoint not in {"local", "cloud"}:
            raise EvidenceError(f"{source}: endpoint must be 'local' or 'cloud'")
        arrival_s = _number(
            row.get("relative_arrival_s"), "relative_arrival_s", source, minimum=0.0
        )
        prompt = _integer(
            row.get("scheduler_prompt_tokens"), "scheduler_prompt_tokens", source,
            minimum=0,
        )
        decode = _integer(
            row.get("scheduler_decode_tokens"), "scheduler_decode_tokens", source,
            minimum=0,
        )
        for field, expected in (
            ("prompt_tokens", prompt),
            ("completion_tokens", decode),
        ):
            if row.get(field) is None:
                continue
            actual = _integer(row[field], field, source, minimum=0)
            if actual != expected:
                raise EvidenceError(
                    f"{source}: {field}={actual} does not match scheduler tokens {expected}"
                )
        cache_mode = row.get("cache_mode")
        if (cache_mode_required is not None and cache_mode is not None
                and cache_mode != cache_mode_required):
            raise EvidenceError(
                f"{source}: cache_mode {cache_mode!r} does not match profile requirement "
                f"{cache_mode_required!r}"
            )
        if cache_mode == "none" and row.get("scheduler_uncached_prompt_tokens") is not None:
            uncached = _integer(
                row["scheduler_uncached_prompt_tokens"],
                "scheduler_uncached_prompt_tokens",
                source,
                minimum=0,
            )
            if uncached != prompt:
                raise EvidenceError(
                    f"{source}: no-cache row has scheduler_uncached_prompt_tokens != prompt"
                )
        row["_request_id"] = request_id
        row["_arrival_s"] = arrival_s
        row["_prompt_tokens"] = prompt
        row["_decode_tokens"] = decode
        if endpoint != "local":
            continue
        queue_ms = _number(row.get("queue_delay_ms"), "queue_delay_ms", source, minimum=0.0)
        service_ms = _number(
            row.get("service_ttft_ms"), "service_ttft_ms", source, minimum=0.0
        )
        ttft_ms = _number(row.get("ttft_ms"), "ttft_ms", source, minimum=0.0)
        e2e_ms = _number(row.get("e2e_ms"), "e2e_ms", source, minimum=0.0)
        if not math.isclose(
            ttft_ms, queue_ms + service_ms, rel_tol=0.0, abs_tol=EPSILON_MS
        ):
            raise EvidenceError(
                f"{source}: ttft_ms != queue_delay_ms + service_ttft_ms"
            )
        if e2e_ms + EPSILON_MS < ttft_ms:
            raise EvidenceError(f"{source}: e2e_ms is smaller than ttft_ms")
        tpot = row.get("tpot_ms")
        if tpot is not None:
            tpot = _number(tpot, "tpot_ms", source, minimum=0.0)
        row["_queue_ms"] = queue_ms
        row["_service_ms"] = service_ms
        row["_ttft_ms"] = ttft_ms
        row["_e2e_ms"] = e2e_ms
        row["_tpot_ms"] = tpot
        row["_dispatch_s"] = arrival_s + queue_ms / 1000.0
        row["_first_token_s"] = arrival_s + ttft_ms / 1000.0
        row["_finish_s"] = arrival_s + e2e_ms / 1000.0
        row["_commitment_tokens"] = prompt + decode
        row["_violates"] = ttft_ms > slo_s * 1000.0
        local.append(row)
    if not local:
        raise EvidenceError(f"{path}: arm has no successful local requests")
    return rows, local


def _id_list(value: Any, field: str, source: str) -> list[str | int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvidenceError(f"{source}: {field} must be an array")
    return [_request_id(item, f"{field}[]", source) for item in value]


def _load_decisions(
    path: Path,
    predictor_model: str,
    raw_rows: Sequence[dict[str, Any]],
    max_local_finish_s: float,
) -> tuple[list[dict[str, Any]], str]:
    rows = _read_jsonl(path)
    previous_id = -1
    previous_at = -math.inf
    selectors: set[str] = set()
    applied_ids: list[str | int] = []
    for row in rows:
        line = row.pop("_source_line")
        source = f"{path}:{line}"
        decision_id = _integer(row.get("decision_id"), "decision_id", source, minimum=0)
        if decision_id <= previous_id:
            raise EvidenceError(f"{source}: decision_id must be strictly increasing")
        previous_id = decision_id
        at_s = _number(row.get("at_s"), "at_s", source, minimum=0.0)
        if at_s + 1e-9 < previous_at:
            raise EvidenceError(f"{source}: at_s must be non-decreasing")
        previous_at = at_s
        if row.get("trigger") != "ttft_pred":
            raise EvidenceError(f"{source}: expected trigger 'ttft_pred'")
        selector = row.get("selector")
        if not isinstance(selector, str) or not selector:
            raise EvidenceError(f"{source}: selector must be a non-empty string")
        selectors.add(selector)
        if row.get("prediction_model") != predictor_model:
            raise EvidenceError(
                f"{source}: prediction_model does not match profile predictor_model"
            )
        if row.get("prediction_scope") != "waiting_only":
            raise EvidenceError(f"{source}: prediction_scope must be 'waiting_only'")
        status = row.get("status")
        if not isinstance(status, str) or not status:
            raise EvidenceError(f"{source}: status must be a non-empty string")
        snapshot_hash = row.get("snapshot_hash")
        if not isinstance(snapshot_hash, str) or not snapshot_hash:
            raise EvidenceError(f"{source}: snapshot_hash must be a non-empty string")
        for field in (
            "snapshot_n", "inflight_n", "inflight_decode_n", "inflight_prefill_n",
            "inflight_prefill_tokens",
        ):
            _integer(row.get(field), field, source, minimum=0)
        for field in (
            "waiting_predicted_max_ttft_s", "waiting_post_kick_max_ttft_s",
            "decision_ms",
        ):
            _number(row.get(field), field, source, minimum=0.0)
        row["_proposed_ids"] = _id_list(
            row.get("proposed_victim_ids"), "proposed_victim_ids", source
        )
        row["_applied_ids"] = _id_list(
            row.get("applied_victim_ids"), "applied_victim_ids", source
        )
        applied_ids.extend(row["_applied_ids"])
    if len(selectors) != 1:
        raise EvidenceError(f"{path}: decisions contain multiple selectors {sorted(selectors)}")
    if len(applied_ids) != len(set(applied_ids)):
        raise EvidenceError(f"{path}: a victim request_id was applied more than once")
    cloud_ids = {row["_request_id"] for row in raw_rows if row.get("endpoint") == "cloud"}
    applied_set = set(applied_ids)
    if applied_set != cloud_ids:
        missing = sorted((repr(value) for value in cloud_ids - applied_set))[:5]
        extra = sorted((repr(value) for value in applied_set - cloud_ids))[:5]
        raise EvidenceError(
            f"{path}: applied/cloud request-id mismatch; missing_applied={missing}, "
            f"applied_not_cloud={extra}"
        )
    if rows[0]["at_s"] > max_local_finish_s + 1.0:
        raise EvidenceError(f"{path}: decision timeline does not overlap the raw arm")
    return rows, next(iter(selectors))


def _reconstruct_dispatch_context(local: Sequence[dict[str, Any]]) -> None:
    """Attach application-side active state at each local dispatch.

    Completion rows supply dispatch, first-token, and finish timestamps.  The
    reconstruction is exact for the client-visible HTTP lifecycle, but the
    commitment sum is a planned peak proxy and must not be called actual KV.
    """
    ordered = sorted(local, key=lambda row: (row["_dispatch_s"], repr(row["_request_id"])))
    finish_heap: list[tuple[float, int, int]] = []
    first_heap: list[tuple[float, int, int]] = []
    active_n = 0
    active_commitment = 0
    prefill_n = 0
    prefill_prompt = 0
    serial = 0
    index = 0
    while index < len(ordered):
        dispatch_s = ordered[index]["_dispatch_s"]
        while finish_heap and finish_heap[0][0] <= dispatch_s:
            _, _, commitment = heapq.heappop(finish_heap)
            active_n -= 1
            active_commitment -= commitment
        while first_heap and first_heap[0][0] <= dispatch_s:
            _, _, prompt = heapq.heappop(first_heap)
            prefill_n -= 1
            prefill_prompt -= prompt
        end = index + 1
        while end < len(ordered) and ordered[end]["_dispatch_s"] == dispatch_s:
            end += 1
        group = ordered[index:end]
        for row in group:
            serial += 1
            active_n += 1
            active_commitment += row["_commitment_tokens"]
            heapq.heappush(
                finish_heap,
                (row["_finish_s"], serial, row["_commitment_tokens"]),
            )
            if row["_first_token_s"] > dispatch_s:
                prefill_n += 1
                prefill_prompt += row["_prompt_tokens"]
                heapq.heappush(
                    first_heap,
                    (row["_first_token_s"], serial, row["_prompt_tokens"]),
                )
        for row in group:
            row["_active_n_at_dispatch"] = active_n
            row["_planned_peak_commitment_tokens_at_dispatch"] = active_commitment
            row["_pre_first_token_n_at_dispatch"] = prefill_n
            row["_pre_first_token_prompt_tokens_at_dispatch"] = prefill_prompt
            row["_post_first_token_n_at_dispatch"] = active_n - prefill_n
        index = end


def _temporal_context(
    raw_rows: Sequence[dict[str, Any]],
    local: Sequence[dict[str, Any]],
    violations: Sequence[dict[str, Any]],
    cluster_gap_s: float,
) -> dict[str, Any]:
    cohorts: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        cohorts[row["_arrival_s"]].append(row)
    violation_by_arrival: Counter[float] = Counter(
        row["_arrival_s"] for row in violations
    )
    cohort_rows: list[dict[str, Any]] = []
    all_local_violated_n = 0
    for arrival_s in sorted(violation_by_arrival):
        cohort = cohorts[arrival_s]
        local_cohort = [row for row in cohort if row.get("endpoint") == "local"]
        violation_n = violation_by_arrival[arrival_s]
        all_local_violated = bool(local_cohort) and violation_n == len(local_cohort)
        if all_local_violated:
            all_local_violated_n += violation_n
        cohort_rows.append({
            "arrival_s": arrival_s,
            "violation_n": violation_n,
            "local_n": len(local_cohort),
            "cloud_n": sum(row.get("endpoint") == "cloud" for row in cohort),
            "all_local_requests_violated": all_local_violated,
        })

    clusters: list[list[dict[str, Any]]] = []
    for row in sorted(violations, key=lambda item: item["_arrival_s"]):
        if not clusters or row["_arrival_s"] - clusters[-1][-1]["_arrival_s"] > cluster_gap_s:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    cluster_rows = [
        {
            "start_arrival_s": cluster[0]["_arrival_s"],
            "end_arrival_s": cluster[-1]["_arrival_s"],
            "violation_n": len(cluster),
        }
        for cluster in clusters
    ]
    return {
        "cluster_gap_s": cluster_gap_s,
        "clustering_method": "deterministic_single_link_chaining_by_arrival_gap",
        "interpretation_caveat": (
            "arrival-gap clusters are a descriptive deterministic grouping and do not "
            "by themselves establish common cause or statistical concentration"
        ),
        "violating_arrival_cohort_n": len(cohort_rows),
        "violations_in_all_local_violating_cohorts_n": all_local_violated_n,
        "per_violating_arrival_cohort": cohort_rows,
        "cluster_n": len(cluster_rows),
        "violations_in_multi_violation_clusters_n": sum(
            row["violation_n"] for row in cluster_rows if row["violation_n"] > 1
        ),
        "largest_cluster_violation_n": max(
            (row["violation_n"] for row in cluster_rows), default=0
        ),
        "clusters": cluster_rows,
    }


def _decision_context(
    decisions: Sequence[dict[str, Any]],
    violations: Sequence[dict[str, Any]],
    guard_ms: float,
    slo_s: float,
) -> dict[str, Any]:
    times = [row["at_s"] for row in decisions]
    mapped: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for request in violations:
        index = bisect.bisect_right(
            times, request["_dispatch_s"] + DISPATCH_MATCH_EPSILON_S
        ) - 1
        if index < 0:
            continue
        decision = decisions[index]
        if decision["at_s"] + DISPATCH_MATCH_EPSILON_S < request["_arrival_s"]:
            continue
        mapped.append((request, decision))
    mapped_decisions = [decision for _, decision in mapped]
    post = [row["waiting_post_kick_max_ttft_s"] for row in mapped_decisions]
    nominal_guarded = [value + guard_ms / 1000.0 for value in post]
    return {
        "decision_n": len(decisions),
        "status_counts": dict(sorted(Counter(row.get("status") for row in decisions).items())),
        "overall_predicted_max_ttft_s": _summary(
            row["waiting_predicted_max_ttft_s"] for row in decisions
        ),
        "overall_post_kick_max_ttft_s": _summary(
            row["waiting_post_kick_max_ttft_s"] for row in decisions
        ),
        "violation_closest_pre_dispatch_context": {
            "interpretation": (
                "closest logged decision before dispatch; decision schema does not prove "
                "that the request ID was in that snapshot"
            ),
            "mapped_violation_n": len(mapped),
            "unmapped_violation_n": len(violations) - len(mapped),
            "unique_decision_n": len({row["decision_id"] for row in mapped_decisions}),
            "status_counts": dict(sorted(Counter(
                row.get("status") for row in mapped_decisions
            ).items())),
            "waiting_predicted_max_ttft_s": _summary(
                row["waiting_predicted_max_ttft_s"] for row in mapped_decisions
            ),
            "waiting_post_kick_max_ttft_s": _summary(post),
            "nominal_post_plus_guard_s": _summary(nominal_guarded),
            "nominal_post_plus_guard_within_slo_n": sum(
                value <= slo_s + 1e-9 for value in nominal_guarded
            ),
            "snapshot_n": _summary(row["snapshot_n"] for row in mapped_decisions),
            "inflight_n": _summary(row["inflight_n"] for row in mapped_decisions),
            "inflight_decode_n": _summary(
                row["inflight_decode_n"] for row in mapped_decisions
            ),
            "inflight_prefill_n": _summary(
                row["inflight_prefill_n"] for row in mapped_decisions
            ),
            "inflight_prefill_tokens": _summary(
                row["inflight_prefill_tokens"] for row in mapped_decisions
            ),
        },
    }


def _parse_server_start(value: str) -> dt.datetime:
    try:
        # Prefix a leap-safe fixed year.  Parsing month/day without a year is
        # deprecated and would make 02-29 ambiguous on newer Python releases.
        parsed = dt.datetime.strptime(f"2000-{value}", f"%Y-{SERVER_STAMP_FORMAT}")
    except ValueError as exc:
        raise EvidenceError(
            f"arguments: --server-log-arm-start must match {SERVER_STAMP_FORMAT!r}"
        ) from exc
    return parsed


def _validate_server_log_prefix(
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(expected_bytes)
    except OSError as exc:
        raise EvidenceError(f"{path}: cannot read server-log prefix: {exc}") from exc
    if len(prefix) != expected_bytes:
        raise EvidenceError(
            f"{path}: server log is shorter than profile-bound prefix length {expected_bytes}"
        )
    actual = hashlib.sha256(prefix).hexdigest()
    if actual != expected_sha256:
        raise EvidenceError(
            f"{path}: server-log prefix does not match manifest/profile lifecycle"
        )


def _server_alignment(
    server_log: Path | None,
    server_log_arm_start: str | None,
    violations: Sequence[dict[str, Any]],
    max_gap_s: float,
    *,
    expected_prefix_bytes: int,
    expected_prefix_sha256: str,
) -> dict[str, Any]:
    semantics = {
        "gauge_semantics_unverified": True,
        "gauge_semantics_architecture_dependent": True,
        "gauge_interpretation_caveat": (
            "do not interpret this generic engine-reported GPU cache-usage gauge as "
            "token-KV pressure without a deployment-specific gauge probe"
        ),
    }
    if server_log is None:
        return {
            "supplied": False,
            "available": False,
            "reason": "no server log supplied",
            **semantics,
        }
    _validate_server_log_prefix(
        server_log, expected_prefix_bytes, expected_prefix_sha256
    )
    if server_log_arm_start is None:
        return {
            "supplied": True,
            "available": False,
            "reason": (
                "raw rows have only relative time; supply --server-log-arm-start "
                "in the server log clock to align without guessing timezone/offset"
            ),
            **semantics,
        }
    max_gap_s = _number(
        max_gap_s, "--server-log-max-gap-s", "arguments", minimum=0.0
    )
    arm_start = _parse_server_start(server_log_arm_start)
    samples: list[dict[str, Any]] = []
    try:
        with server_log.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = SERVER_METRIC_RE.search(line)
                if not match:
                    continue
                stamp = _parse_server_start(match.group("stamp"))
                relative_s = (stamp - arm_start).total_seconds()
                samples.append({
                    "sample_index": len(samples),
                    "relative_s": relative_s,
                    "prompt_throughput_tokens_per_s": float(match.group("prompt")),
                    "generation_throughput_tokens_per_s": float(match.group("generation")),
                    "engine_running_n": int(match.group("running")),
                    "engine_waiting_n": int(match.group("waiting")),
                    "engine_reported_gpu_cache_usage_gauge_pct": float(
                        match.group("kv")
                    ),
                })
    except OSError as exc:
        raise EvidenceError(f"{server_log}: cannot read: {exc}") from exc
    if not samples:
        return {
            "supplied": True,
            "available": False,
            "reason": "no supported vLLM periodic metric lines found in server log",
            **semantics,
        }
    samples.sort(key=lambda row: row["relative_s"])
    sample_times = [row["relative_s"] for row in samples]

    def nearest(target_s: float) -> tuple[dict[str, Any], float] | None:
        index = bisect.bisect_left(sample_times, target_s)
        candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(samples)]
        if not candidates:
            return None
        chosen = min(candidates, key=lambda candidate: abs(sample_times[candidate] - target_s))
        gap = abs(sample_times[chosen] - target_s)
        return None if gap > max_gap_s else (samples[chosen], gap)

    def aligned(phase: str) -> dict[str, Any]:
        field = "_dispatch_s" if phase == "dispatch" else "_first_token_s"
        matches = [nearest(row[field]) for row in violations]
        present = [value for value in matches if value is not None]
        metric_rows = [value[0] for value in present]
        return {
            "requested_violation_n": len(violations),
            "aligned_violation_n": len(present),
            "unaligned_violation_n": len(violations) - len(present),
            "unique_metric_sample_n": len({
                row["sample_index"] for row in metric_rows
            }),
            "nearest_sample_gap_s": _summary(value[1] for value in present),
            "engine_reported_gpu_cache_usage_gauge_pct": _summary(
                row["engine_reported_gpu_cache_usage_gauge_pct"]
                for row in metric_rows
            ),
            "engine_reported_gpu_cache_usage_gauge_ge_95_pct_n": sum(
                row["engine_reported_gpu_cache_usage_gauge_pct"] >= 95.0
                for row in metric_rows
            ),
            "engine_reported_gpu_cache_usage_gauge_ge_98_pct_n": sum(
                row["engine_reported_gpu_cache_usage_gauge_pct"] >= 98.0
                for row in metric_rows
            ),
            "engine_running_n": _summary(row["engine_running_n"] for row in metric_rows),
            "engine_waiting_n": _summary(row["engine_waiting_n"] for row in metric_rows),
            "prompt_throughput_tokens_per_s": _summary(
                row["prompt_throughput_tokens_per_s"] for row in metric_rows
            ),
            "generation_throughput_tokens_per_s": _summary(
                row["generation_throughput_tokens_per_s"] for row in metric_rows
            ),
        }

    dispatch = aligned("dispatch")
    first_token = aligned("first_token")
    available = bool(
        dispatch["aligned_violation_n"] or first_token["aligned_violation_n"]
    )
    return {
        "supplied": True,
        "available": available,
        "reason": None if available else "no violation timestamp is within the maximum log gap",
        "server_log_arm_start": server_log_arm_start,
        "server_log_max_gap_s": max_gap_s,
        "parsed_metric_sample_n": len(samples),
        **semantics,
        "timestamp_scope_caveat": (
            "MM-DD timestamps are supported only within one explicitly supplied "
            "calendar-year lifecycle; cross-year alignment is not inferred"
        ),
        "clock_assumption": (
            "caller asserts that --server-log-arm-start is this raw arm's run start "
            "in the server log clock; the artifacts do not independently bind that offset; "
            "metric summaries are per-violation nearest-sample associations and may reuse "
            "one periodic sample for multiple requests"
        ),
        "dispatch": dispatch,
        "first_token": first_token,
    }


def analyze(
    raw_path: Path,
    decisions_path: Path,
    profile_path: Path,
    *,
    summary_path: Path,
    marker_path: Path,
    manifest_path: Path,
    slo_s: float | None = None,
    cluster_gap_s: float = 10.0,
    server_log: Path | None = None,
    server_log_arm_start: str | None = None,
    server_log_max_gap_s: float = 15.0,
) -> dict[str, Any]:
    """Validate and analyze one completed ``ttft_pred`` arm."""
    cluster_gap_s = _number(
        cluster_gap_s, "--cluster-gap-s", "arguments", minimum=0.0
    )
    summary, binding, manifest = _load_bound_evidence(
        raw_path,
        decisions_path,
        summary_path,
        marker_path,
        manifest_path,
        profile_path,
    )
    profile = _load_profile(profile_path, slo_s)
    _validate_profile_manifest(profile, manifest, str(manifest_path))
    raw_rows, local = _load_raw(
        raw_path, profile["slo_s"], profile["cache_mode_required"]
    )
    _reconstruct_dispatch_context(local)
    decisions, selector = _load_decisions(
        decisions_path,
        profile["predictor_model"],
        raw_rows,
        max(row["_finish_s"] for row in local),
    )
    violations = [row for row in local if row["_violates"]]
    _validate_summary_binding(
        summary,
        profile=profile,
        binding=binding,
        manifest=manifest,
        raw_rows=raw_rows,
        local=local,
        violations=violations,
        decision_selector=selector,
        source=str(summary_path),
    )
    cloud_n = sum(row.get("endpoint") == "cloud" for row in raw_rows)
    slo_ms = profile["slo_s"] * 1000.0

    queue_only = sum(
        row["_queue_ms"] > slo_ms and row["_service_ms"] <= slo_ms
        for row in violations
    )
    service_only = sum(
        row["_service_ms"] > slo_ms and row["_queue_ms"] <= slo_ms
        for row in violations
    )
    both_components = sum(
        row["_queue_ms"] > slo_ms and row["_service_ms"] > slo_ms
        for row in violations
    )
    sum_only = sum(
        row["_queue_ms"] <= slo_ms and row["_service_ms"] <= slo_ms
        for row in violations
    )
    calibrated_peak = profile["calibrated_peak_commitment_max_tokens"]
    kv_capacity = profile["kv_capacity_tokens"]
    violation_commitments = [
        row["_planned_peak_commitment_tokens_at_dispatch"] for row in violations
    ]
    all_tpot = _summary(row["_tpot_ms"] for row in local)
    violation_tpot = _summary(row["_tpot_ms"] for row in violations)
    fixed_decode = profile["fixed_decode_tokens"]
    prompt_points = set(profile["prompt_points"])

    result = {
        "schema_version": 1,
        "analysis": "ttft_violation_context",
        "evidence": {
            "run_fingerprint": binding["run_fingerprint"],
            "marker_arm": binding["arm"],
            "raw": {
                "name": raw_path.name,
                **binding["artifacts"]["raw"],
                "row_n": len(raw_rows),
            },
            "decisions": {
                "name": decisions_path.name,
                **binding["artifacts"]["decisions"],
                "row_n": len(decisions),
            },
            "summary": {
                "name": summary_path.name,
                **binding["artifacts"]["summary"],
            },
            "completion_marker": {
                "name": marker_path.name,
                "sha256": _sha256(marker_path),
            },
            "matrix_manifest": {
                "name": manifest_path.name,
                "sha256": _sha256(manifest_path),
            },
            "profile": {
                "name": profile_path.name,
                "sha256": binding["profile_sha256"],
            },
            "server_log": (
                None if server_log is None
                else {"name": server_log.name, "sha256": _sha256(server_log)}
            ),
        },
        "identity": {
            "trigger": binding["trigger"],
            "selector": binding["selector"],
            "seed": binding["seed"],
            "prediction_model": profile["predictor_model"],
            "prediction_scope": "waiting_only",
        },
        "slo_s": profile["slo_s"],
        "counts": {
            "request_n": len(raw_rows),
            "local_n": len(local),
            "cloud_n": cloud_n,
            "local_violation_n": len(violations),
            "local_violation_pct": 100.0 * len(violations) / len(local),
        },
        "queue_vs_service": {
            "violating_ttft_ms": _summary(row["_ttft_ms"] for row in violations),
            "violating_queue_delay_ms": _summary(row["_queue_ms"] for row in violations),
            "violating_service_ttft_ms": _summary(row["_service_ms"] for row in violations),
            "component_exceeds_slo": {
                "queue_delay_n": sum(row["_queue_ms"] > slo_ms for row in violations),
                "service_ttft_n": sum(row["_service_ms"] > slo_ms for row in violations),
            },
            "mutually_exclusive_classification": {
                "queue_component_only_n": queue_only,
                "service_component_only_n": service_only,
                "both_components_n": both_components,
                "neither_component_but_sum_n": sum_only,
            },
            "queue_delay_le_profile_tick_guard_n": sum(
                row["_queue_ms"] <= profile["tick_ms"] for row in violations
            ),
        },
        "dispatch_load_context": {
            "reconstruction_scope": (
                "client-visible active local HTTP requests reconstructed from dispatch/finish; "
                "pre/post-first-token intervals are not engine scheduler phases (requests may "
                "be queued or preempted inside the engine); planned peak commitment sums prompt "
                "+ requested decode and is not actual KV usage"
            ),
            "all_local": {
                "active_n": _summary(row["_active_n_at_dispatch"] for row in local),
                "planned_peak_commitment_tokens": _summary(
                    row["_planned_peak_commitment_tokens_at_dispatch"] for row in local
                ),
                "pre_first_token_n": _summary(
                    row["_pre_first_token_n_at_dispatch"] for row in local
                ),
                "pre_first_token_prompt_tokens": _summary(
                    row["_pre_first_token_prompt_tokens_at_dispatch"] for row in local
                ),
            },
            "violations": {
                "active_n": _summary(
                    row["_active_n_at_dispatch"] for row in violations
                ),
                "planned_peak_commitment_tokens": _summary(violation_commitments),
                "pre_first_token_n": _summary(
                    row["_pre_first_token_n_at_dispatch"] for row in violations
                ),
                "pre_first_token_prompt_tokens": _summary(
                    row["_pre_first_token_prompt_tokens_at_dispatch"]
                    for row in violations
                ),
                "post_first_token_n": _summary(
                    row["_post_first_token_n_at_dispatch"] for row in violations
                ),
            },
            "profile_support": {
                "profile_scope": profile["calibration_scope"],
                "profile_concurrency_points": profile["concurrency_points"],
                "profile_max_concurrency": profile["concurrency_max"],
                "profile_prompt_cell_target_points": profile["prompt_points"],
                "profile_fixed_decode_tokens": fixed_decode,
                "profile_calibrated_peak_commitment_max_tokens": calibrated_peak,
                "declared_kv_capacity_tokens": kv_capacity,
                "violation_active_n_above_profile_max_n": sum(
                    row["_active_n_at_dispatch"] > profile["concurrency_max"]
                    for row in violations
                ),
                "violation_commitment_above_profile_max_n": sum(
                    value > calibrated_peak for value in violation_commitments
                ),
                "violation_commitment_above_declared_kv_capacity_n": sum(
                    value > kv_capacity for value in violation_commitments
                ),
                "all_local_commitment_above_profile_max_n": sum(
                    row["_planned_peak_commitment_tokens_at_dispatch"] > calibrated_peak
                    for row in local
                ),
                "all_local_commitment_above_declared_kv_capacity_n": sum(
                    row["_planned_peak_commitment_tokens_at_dispatch"] > kv_capacity
                    for row in local
                ),
                "violation_commitment_min_over_profile_max_ratio": _ratio(
                    min(violation_commitments) if violation_commitments else None,
                    calibrated_peak,
                ),
                "violation_commitment_max_over_profile_max_ratio": _ratio(
                    max(violation_commitments) if violation_commitments else None,
                    calibrated_peak,
                ),
                "violation_commitment_min_over_kv_capacity_ratio": _ratio(
                    min(violation_commitments) if violation_commitments else None,
                    kv_capacity,
                ),
                "violation_commitment_max_over_kv_capacity_ratio": _ratio(
                    max(violation_commitments) if violation_commitments else None,
                    kv_capacity,
                ),
                "violation_prompt_tokens": _summary(
                    row["_prompt_tokens"] for row in violations
                ),
                "violation_prompt_outside_profile_numeric_range_n": sum(
                    row["_prompt_tokens"] < profile["prompt_min"]
                    or row["_prompt_tokens"] > profile["prompt_max"]
                    for row in violations
                ),
                "violation_prompt_exact_profile_target_point_n": sum(
                    row["_prompt_tokens"] in prompt_points for row in violations
                ),
                "violation_decode_tokens": _summary(
                    row["_decode_tokens"] for row in violations
                ),
                "violation_decode_vs_profile_fixed": {
                    "below_n": sum(
                        row["_decode_tokens"] < fixed_decode for row in violations
                    ),
                    "equal_n": sum(
                        row["_decode_tokens"] == fixed_decode for row in violations
                    ),
                    "above_n": sum(
                        row["_decode_tokens"] > fixed_decode for row in violations
                    ),
                },
            },
        },
        "tpot_comparison": {
            "calibrated_tpot_ms": profile["calibrated_tpot_ms"],
            "arm_local_request_tpot_ms": all_tpot,
            "violation_tpot_ms": violation_tpot,
            "arm_local_request_p50_over_calibrated_ratio": _ratio(
                all_tpot["p50"], profile["calibrated_tpot_ms"]
            ),
            "violation_p50_over_calibrated_ratio": _ratio(
                violation_tpot["p50"], profile["calibrated_tpot_ms"]
            ),
        },
        "temporal_and_cohort_concentration": _temporal_context(
            raw_rows, local, violations, cluster_gap_s
        ),
        "decisions": _decision_context(
            decisions, violations, profile["guard_ms"], profile["slo_s"]
        ),
        "server_alignment": _server_alignment(
            server_log,
            server_log_arm_start,
            violations,
            server_log_max_gap_s,
            expected_prefix_bytes=profile["server_log_prefix_bytes"],
            expected_prefix_sha256=profile["server_log_prefix_sha256"],
        ),
        "selector_replay": {
            "available": False,
            "reason": (
                "decision rows contain snapshot_hash/snapshot_n but not complete snapshot "
                "IDs and ages, per-request predictions, selector scores/order, in-flight "
                "identities/progress, or a bound live-KV sample; selector ranking and causality "
                "cannot be independently replayed"
            ),
        },
        "validation": {
            "completion_marker_exactly_matches_raw_summary_decisions": True,
            "marker_run_fingerprint_matches_matrix_manifest": True,
            "supplied_profile_sha256_matches_matrix_manifest": True,
            "profile_server_lifecycle_matches_matrix_manifest": True,
            "summary_runtime_config_matches_profile_marker_manifest_and_decisions": True,
            "all_requests_successful": True,
            "request_ids_unique": True,
            "ttft_decomposition_exact": True,
            "applied_victims_equal_cloud_rows": True,
            "present_endpoint_usage_token_fields_match_scheduler": True,
            "present_cache_mode_fields_match_profile": True,
        },
    }
    return result


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _range(summary: dict[str, Any], digits: int = 3) -> str:
    if summary["n"] == 0:
        return "—"
    return (
        f"{_fmt(summary['min'], digits)} / {_fmt(summary['p50'], digits)} / "
        f"{_fmt(summary['max'], digits)}"
    )


def render_markdown(result: dict[str, Any]) -> str:
    counts = result["counts"]
    queue = result["queue_vs_service"]
    support = result["dispatch_load_context"]["profile_support"]
    load = result["dispatch_load_context"]["violations"]
    decision = result["decisions"]["violation_closest_pre_dispatch_context"]
    temporal = result["temporal_and_cohort_concentration"]
    lines = [
        "# TTFT violation-context audit",
        "",
        f"Arm: `ttft_pred + {result['identity']['selector']}`; SLO: "
        f"{result['slo_s']:.3f}s. Local violations: **{counts['local_violation_n']} / "
        f"{counts['local_n']} ({counts['local_violation_pct']:.3f}%)**.",
        "",
        "## Queue versus service",
        "",
        "| metric | n or min / p50 / max |",
        "|---|---:|",
        f"| violating total TTFT ms | {_range(queue['violating_ttft_ms'])} |",
        f"| violating queue delay ms | {_range(queue['violating_queue_delay_ms'])} |",
        f"| violating service TTFT ms | {_range(queue['violating_service_ttft_ms'])} |",
        f"| queue component > SLO | {queue['component_exceeds_slo']['queue_delay_n']} |",
        f"| service component > SLO | {queue['component_exceeds_slo']['service_ttft_n']} |",
        f"| neither component > SLO, but sum > SLO | "
        f"{queue['mutually_exclusive_classification']['neither_component_but_sum_n']} |",
        "",
        "## Dispatch load and calibration support",
        "",
        result["dispatch_load_context"]["reconstruction_scope"] + ".",
        "",
        "| metric | observation |",
        "|---|---:|",
        f"| violating active requests, min / p50 / max | {_range(load['active_n'])} |",
        f"| violating planned commitment tokens, min / p50 / max | "
        f"{_range(load['planned_peak_commitment_tokens'], 0)} |",
        f"| profile maximum calibrated commitment | {_fmt(support['profile_calibrated_peak_commitment_max_tokens'], 0)} |",
        f"| declared KV capacity | {_fmt(support['declared_kv_capacity_tokens'], 0)} |",
        f"| violations above profile commitment support | "
        f"{support['violation_commitment_above_profile_max_n']} |",
        f"| violations above declared KV capacity proxy | "
        f"{support['violation_commitment_above_declared_kv_capacity_n']} |",
        f"| decode below / equal / above profile fixed point | "
        f"{support['violation_decode_vs_profile_fixed']['below_n']} / "
        f"{support['violation_decode_vs_profile_fixed']['equal_n']} / "
        f"{support['violation_decode_vs_profile_fixed']['above_n']} |",
        "",
        "## Predictor and timing context",
        "",
        f"Calibrated TPOT: {_fmt(result['tpot_comparison']['calibrated_tpot_ms'])}ms; "
        f"arm-local-request p50 ratio: "
        f"{_fmt(result['tpot_comparison']['arm_local_request_p50_over_calibrated_ratio'])}×; "
        f"violation p50 ratio: {_fmt(result['tpot_comparison']['violation_p50_over_calibrated_ratio'])}×.",
        "",
        f"Closest pre-dispatch decision context mapped {decision['mapped_violation_n']} / "
        f"{counts['local_violation_n']} violations across {decision['unique_decision_n']} decisions. "
        f"Post-kick predicted max min/p50/max: {_range(decision['waiting_post_kick_max_ttft_s'])}s. "
        "This is contextual association, not proof that a request ID was in the hashed snapshot.",
        "",
        f"Temporal clusters (gap ≤ {temporal['cluster_gap_s']:.3f}s): "
        f"{temporal['cluster_n']}; largest {temporal['largest_cluster_violation_n']} violations; "
        f"{temporal['violations_in_all_local_violating_cohorts_n']} violations occurred in "
        "arrival cohorts where every retained local request violated.",
        "This is deterministic single-link arrival-gap grouping; it does not by itself "
        "establish shared cause or statistical concentration.",
        "",
        "## Server telemetry",
        "",
    ]
    server = result["server_alignment"]
    if server["available"]:
        first = server["first_token"]
        lines.extend([
            f"Explicit-clock alignment covered {first['aligned_violation_n']} / "
            f"{first['requested_violation_n']} violation first-token timestamps using "
            f"{first['unique_metric_sample_n']} distinct periodic samples.",
            "",
            "| first-token-nearest server metric | min / p50 / max |",
            "|---|---:|",
            f"| engine-reported GPU cache-usage gauge % | "
            f"{_range(first['engine_reported_gpu_cache_usage_gauge_pct'])} |",
            f"| engine running | {_range(first['engine_running_n'])} |",
            f"| engine waiting | {_range(first['engine_waiting_n'])} |",
            "",
            "This alignment depends on the caller-supplied server-clock start; one "
            "periodic sample may be associated with multiple requests.",
            "Gauge semantics are unverified and architecture-dependent; do not interpret "
            "this as token-KV pressure without a deployment-specific gauge probe.",
        ])
    else:
        lines.append(f"Unavailable: {server['reason']}.")
    lines.extend([
        "",
        "## Selector replay",
        "",
        f"**Unavailable.** {result['selector_replay']['reason']}.",
    ])
    return "\n".join(lines) + "\n"


def render_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="completed arm raw JSONL")
    parser.add_argument(
        "--decisions", type=Path, required=True, help="matching decision JSONL"
    )
    parser.add_argument("--summary", type=Path, required=True, help="matching summary JSON")
    parser.add_argument(
        "--marker", type=Path, required=True, help="matching completion marker JSON"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="sibling matrix_manifest.txt"
    )
    parser.add_argument("--profile", type=Path, required=True, help="bound TTFT profile JSON")
    parser.add_argument(
        "--slo-s", type=float,
        help="optional assertion equal to the bound profile/summary target SLO",
    )
    parser.add_argument("--cluster-gap-s", type=float, default=10.0)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument(
        "--server-log-arm-start",
        help=(
            "run start in the server log clock, format 'MM-DD HH:MM:SS'; "
            "required for server telemetry alignment"
        ),
    )
    parser.add_argument("--server-log-max-gap-s", type=float, default=15.0)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze(
            args.raw,
            args.decisions,
            args.profile,
            summary_path=args.summary,
            marker_path=args.marker,
            manifest_path=args.manifest,
            slo_s=args.slo_s,
            cluster_gap_s=args.cluster_gap_s,
            server_log=args.server_log,
            server_log_arm_start=args.server_log_arm_start,
            server_log_max_gap_s=args.server_log_max_gap_s,
        )
    except EvidenceError as exc:
        print(f"TTFT violation evidence invalid: {exc}", file=sys.stderr)
        return 2
    markdown = render_markdown(result)
    encoded = render_json(result)
    if args.json_out:
        args.json_out.write_text(encoded, encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.write_text(markdown, encoding="utf-8")
    sys.stdout.write(encoded if args.json else markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
