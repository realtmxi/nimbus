#!/usr/bin/env python3
"""Audit the staged E12 ShareGPT-current-turn local L/A/C gate.

The three arms live in independent matrix ``OUT_DIR`` directories, so their
run fingerprints differ.  This analyzer binds each completion marker to its
raw/summary/decision artifacts, verifies that all other experiment inputs are
shared, and reports only aggregate (prompt-text-free) evidence.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED_N = 11604
DEFAULT_TRACE_SHA256 = (
    "e838016a8e55660c565dadb1ad019770f6b88f878d8ca29f165c30887d2cb410"
)
DEFAULT_TRACE_MANIFEST_SHA256 = (
    "698bb94a82d133b0c54aa87d8badd4f181140c352cb29c1e726cfe2b291bf9a8"
)
EXPECTED_ARMS = {
    "L": "anchor:all_local:0",
    "A": "ttft_pred:cost_cachedisp_old:0",
    "C": "ttft_pred:cost_disp_current:0",
}
EXPECTED_IDENTITIES = {
    "L": ("all_local", None, None),
    "A": ("nimbus", "ttft_pred", "cost_cachedisp_old"),
    "C": ("nimbus", "ttft_pred", "cost_disp_current"),
}
ALLOWED_MANIFEST_DIFFERENCES = frozenset({
    "run_fingerprint", "started_at", "arms", "server_log_sha256_at_manifest",
})
REQUIRED_MANIFEST_FIELDS = frozenset({
    "run_fingerprint", "started_at", "commit", "trace_sha256",
    "trace_manifest_sha256", "trace_n", "profile_sha256", "cache_mode",
    "server_pid", "server_log", "server_log_prefix_sha256",
    "server_log_sha256_at_manifest", "endpoint_version_sha256",
    "endpoint_models_identity_sha256", "base_url", "chat_url", "model",
    "scenario", "max_inflight", "kv_cap", "prefill_tput", "tpot_ms",
    "first_token_overhead_ms", "slo_s", "timeout_s", "guard_ms", "tick_ms",
    "temperature", "ignore_eos", "in_price", "out_price",
    "local_ignore_eos", "cloud_ignore_eos", "cloud", "cloud_url",
    "cloud_model", "cloud_api_key_env", "cloud_max_concurrency",
    "cloud_provider", "cloud_no_fallbacks", "cloud_stop_after_first_token",
    "real_cloud_expected_trace_n", "secret_value_recorded", "arm_order_mode",
    "arm_order_seed", "arms",
})
EPSILON = 1e-9
PAYLOAD_MODE = "sharegpt_current_turn_retokenized"
PREDICTION_SCOPE = "waiting_only"
PREDICTION_MODEL = "seq_slots_shared_prefill_lane_v1"


class EvidenceError(ValueError):
    """Artifacts are not structurally valid evidence for this gate."""


@dataclass
class StageAudit:
    label: str
    arm_text: str
    fingerprint: str
    manifest: dict[str, Any]
    request_ids: set[int]
    cloud_ids: set[int]
    local_n: int
    routed_n: int
    success_n: int
    local_violation_n: int
    local_violation_pct: float
    cost_usd: float
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    ttft_p99_ms: float | None
    ttft_max_ms: float | None
    decision_n: int
    applied_victim_n: int
    events_sha256: str
    arm_started_at: datetime
    arm_finished_at: datetime
    matrix_finished_at: datetime


def _integer(value: Any, field: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{source}: {field} must be an integer")
    return value


def _number(value: Any, field: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{source}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{source}: {field} must be finite")
    return result


def _object(value: Any, field: str, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{source}: {field} must be an object")
    return value


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100 * (len(ordered) - 1)))))
    return ordered[index]


def _close(left: float, right: float, *, tolerance: float = EPSILON) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _artifact(path: Path) -> dict[str, Any]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"missing/unreadable artifact {path}: {exc}") from exc
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "nonempty_line_n": sum(bool(line.strip()) for line in content.splitlines()),
    }


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label}: unreadable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label}: top-level JSON must be an object")
    return value


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"{label}: unreadable UTF-8 JSONL: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"{label} line {line_n}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise EvidenceError(f"{label} line {line_n}: row must be an object")
        rows.append(row)
    return rows


def _parse_manifest(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"{path}: unreadable matrix manifest: {exc}") from exc
    result: dict[str, Any] = {}
    single_remainder = {"python", "evidence_scope", "arms"}
    for line_n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        first_key = line.split("=", 1)[0]
        if first_key in single_remainder:
            tokens = [line]
        else:
            tokens = line.split()
        for token in tokens:
            if "=" not in token:
                raise EvidenceError(f"{path} line {line_n}: malformed manifest field")
            key, value = token.split("=", 1)
            if not key or key in result:
                raise EvidenceError(f"{path} line {line_n}: duplicate/empty field {key!r}")
            result[key] = tuple(value.split()) if key == "arms" else value
    missing = sorted(REQUIRED_MANIFEST_FIELDS - result.keys())
    if missing:
        raise EvidenceError(f"{path}: missing manifest fields: {', '.join(missing)}")
    return result


def _parse_iso_z(value: str, source: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as exc:
        raise EvidenceError(
            f"{source}: timestamp must be exact UTC ISO second form"
        ) from exc
    return parsed.replace(tzinfo=timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_matrix_events(
    path: Path, label: str, expected_arm: str, expected_fingerprint: str,
) -> tuple[str, datetime, datetime, datetime]:
    source = f"{label} matrix_events.log"
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"{source}: unreadable: {exc}") from exc
    lines = text.splitlines()
    if len(lines) != 3 or any(not line for line in lines):
        raise EvidenceError(
            f"{source}: expected exactly three nonempty start/finish/matrix-finish lines"
        )
    start_match = re.fullmatch(
        r"arm_started_at=(\S+) arm=(\S+) command=(.+)", lines[0]
    )
    finish_match = re.fullmatch(
        r"arm_finished_at=(\S+) arm=(\S+)", lines[1]
    )
    matrix_match = re.fullmatch(
        r"matrix_finished_at=(\S+) run_fingerprint=(\S+)", lines[2]
    )
    if not start_match or not finish_match or not matrix_match:
        raise EvidenceError(f"{source}: lines do not match the runner event schema")
    if start_match.group(2) != expected_arm or finish_match.group(2) != expected_arm:
        raise EvidenceError(f"{source}: start/finish arm does not match stage arm")
    if matrix_match.group(2) != expected_fingerprint:
        raise EvidenceError(
            f"{source}: matrix-finish fingerprint does not match stage manifest"
        )
    started_at = _parse_iso_z(start_match.group(1), source)
    arm_finished_at = _parse_iso_z(finish_match.group(1), source)
    matrix_finished_at = _parse_iso_z(matrix_match.group(1), source)
    if not started_at <= arm_finished_at <= matrix_finished_at:
        raise EvidenceError(
            f"{source}: require start <= arm finish <= matrix finish"
        )
    return (
        hashlib.sha256(raw).hexdigest(),
        started_at,
        arm_finished_at,
        matrix_finished_at,
    )


def _manifest_number(manifest: dict[str, Any], key: str, source: str) -> float:
    try:
        value = float(manifest[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"{source}: manifest {key} is not numeric") from exc
    if not math.isfinite(value):
        raise EvidenceError(f"{source}: manifest {key} is not finite")
    return value


def _manifest_int(manifest: dict[str, Any], key: str, source: str) -> int:
    value = _manifest_number(manifest, key, source)
    if not value.is_integer():
        raise EvidenceError(f"{source}: manifest {key} is not an integer")
    return int(value)


def recompute_run_fingerprint(manifest: dict[str, Any]) -> str:
    """Reproduce ``run_ttft_selector_matrix.sh``'s exact argv hash.

    Values remain strings because the shell passes the precise textual values
    later persisted in ``matrix_manifest.txt``.  Numeric normalization here
    would make, for example, ``5`` and ``5.0`` hash differently from runner.
    """
    ordered_keys = (
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
    try:
        argv = [manifest[key] for key in ordered_keys]
        arms = manifest["arms"]
    except KeyError as exc:
        raise EvidenceError(
            f"matrix manifest lacks fingerprint input {exc.args[0]!r}"
        ) from exc
    if not all(isinstance(value, str) for value in argv):
        raise EvidenceError("matrix manifest fingerprint inputs must be strings")
    if not isinstance(arms, tuple) or not all(isinstance(arm, str) for arm in arms):
        raise EvidenceError("matrix manifest arms must be an exact string tuple")
    payload = json.dumps(
        argv + list(arms), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_stage_manifest(
    manifest: dict[str, Any], label: str, expected_n: int,
    expected_trace_sha256: str | None, expected_trace_manifest_sha256: str | None,
    expected_profile_sha256: str | None,
) -> None:
    source = f"{label} matrix_manifest.txt"
    if manifest["arms"] != (EXPECTED_ARMS[label],):
        raise EvidenceError(
            f"{source}: arms={manifest['arms']!r}, expected {(EXPECTED_ARMS[label],)!r}"
        )
    if manifest["arm_order_mode"] != "explicit":
        raise EvidenceError(f"{source}: arm_order_mode must be explicit")
    recomputed_fingerprint = recompute_run_fingerprint(manifest)
    if manifest["run_fingerprint"] != recomputed_fingerprint:
        raise EvidenceError(
            f"{source}: run_fingerprint does not match exact runner inputs"
        )
    if _manifest_int(manifest, "trace_n", source) != expected_n:
        raise EvidenceError(f"{source}: trace_n does not match expected N")
    if expected_trace_sha256 and manifest["trace_sha256"] != expected_trace_sha256:
        raise EvidenceError(f"{source}: trace_sha256 does not match frozen trace")
    if (expected_trace_manifest_sha256 and
            manifest["trace_manifest_sha256"] != expected_trace_manifest_sha256):
        raise EvidenceError(f"{source}: trace_manifest_sha256 does not match frozen trace")
    if expected_profile_sha256 and manifest["profile_sha256"] != expected_profile_sha256:
        raise EvidenceError(f"{source}: profile_sha256 does not match expected profile")
    exact = {
        "cache_mode": "none", "cloud": "null", "cloud_no_fallbacks": "0",
        "cloud_stop_after_first_token": "0", "secret_value_recorded": "false",
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise EvidenceError(f"{source}: {key}={manifest.get(key)!r}, expected {expected!r}")


def _validate_common_manifests(stages: dict[str, StageAudit]) -> None:
    baseline = stages["L"].manifest
    baseline_keys = set(baseline) - ALLOWED_MANIFEST_DIFFERENCES
    for label in ("A", "C"):
        candidate = stages[label].manifest
        candidate_keys = set(candidate) - ALLOWED_MANIFEST_DIFFERENCES
        if candidate_keys != baseline_keys:
            raise EvidenceError(f"{label} manifest fields differ from L")
        for key in sorted(baseline_keys):
            if candidate[key] != baseline[key]:
                raise EvidenceError(
                    f"{label} manifest {key}={candidate[key]!r} differs from L {baseline[key]!r}"
                )


def _validate_config(
    config: dict[str, Any], manifest: dict[str, Any], label: str,
    trigger: str | None, selector: str | None,
) -> None:
    source = f"{label} summary config"
    numeric = {
        "prefill_tput": "prefill_tput", "tpot_ms": "tpot_ms",
        "first_token_overhead_ms": "first_token_overhead_ms", "slo_s": "slo_s",
        "timeout_s": "timeout_s", "ttft_guard_ms": "guard_ms",
        "nimbus_tick_ms": "tick_ms", "temperature": "temperature",
        "kv_capacity_tokens": "kv_cap", "in_price": "in_price",
        "out_price": "out_price", "max_inflight": "max_inflight",
    }
    for config_key, manifest_key in numeric.items():
        actual = _number(config.get(config_key), config_key, source)
        expected = _manifest_number(manifest, manifest_key, source)
        if not _close(actual, expected):
            raise EvidenceError(f"{source}: {config_key} differs from manifest")
    exact = {
        "scenario": manifest["scenario"], "seed": 0, "time_scale": 1.0,
        "max_tokens_override": None, "kv_hysteresis_fraction": 0.05,
        "ignore_eos": manifest["ignore_eos"] == "1",
        "local_ignore_eos": manifest["local_ignore_eos"] == "1",
        "cloud_ignore_eos": manifest["cloud_ignore_eos"] == "1",
        "local_model": manifest["model"], "local_url": manifest["chat_url"],
        "cloud": "null", "cloud_url": None,
        "cloud_model": manifest["model"], "cloud_api_key_env": None,
        "cloud_max_concurrency": _manifest_int(manifest, "cloud_max_concurrency", source),
        "cloud_provider_order": None, "cloud_no_fallbacks": False,
        "cloud_stop_after_first_token": False,
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise EvidenceError(f"{source}: {key}={config.get(key)!r}, expected {expected!r}")
    if label in ("A", "C"):
        if config.get("nimbus_trigger") != trigger or config.get("nimbus_selector") != selector:
            raise EvidenceError(f"{source}: trigger/selector does not match exact arm")


def _token_exact(row: dict[str, Any], source: str) -> None:
    prompt = _integer(row.get("prompt_tokens"), "prompt_tokens", source)
    decode = _integer(row.get("completion_tokens"), "completion_tokens", source)
    scheduler_prompt = _integer(
        row.get("scheduler_prompt_tokens"), "scheduler_prompt_tokens", source
    )
    scheduler_uncached = _integer(
        row.get("scheduler_uncached_prompt_tokens"),
        "scheduler_uncached_prompt_tokens", source,
    )
    scheduler_decode = _integer(
        row.get("scheduler_decode_tokens"), "scheduler_decode_tokens", source
    )
    for field, value in (
        ("prompt_tokens", prompt), ("completion_tokens", decode),
        ("scheduler_prompt_tokens", scheduler_prompt),
        ("scheduler_uncached_prompt_tokens", scheduler_uncached),
        ("scheduler_decode_tokens", scheduler_decode),
    ):
        if value < 0:
            raise EvidenceError(f"{source}: {field} must be nonnegative")
    if prompt != scheduler_prompt or scheduler_uncached != scheduler_prompt:
        raise EvidenceError(f"{source}: no-cache prompt tokens are not exact")
    if decode != scheduler_decode:
        raise EvidenceError(f"{source}: decode tokens are not exact")


def _audit_stage(
    directory: Path, label: str, expected_n: int,
    expected_trace_sha256: str | None, expected_trace_manifest_sha256: str | None,
    expected_profile_sha256: str | None,
) -> StageAudit:
    if not directory.is_dir():
        raise EvidenceError(f"{label}: stage directory does not exist: {directory}")
    markers = sorted(directory.glob("*.complete.json"))
    if len(markers) != 1:
        raise EvidenceError(f"{label}: expected exactly one completion marker, found {len(markers)}")
    marker_path = markers[0]
    suffix = ".complete.json"
    stem = marker_path.name[:-len(suffix)]
    paths = {
        "raw": directory / f"{stem}.jsonl",
        "summary": directory / f"{stem}.summary.json",
        "decisions": directory / f"{stem}.decisions.jsonl",
    }
    artifacts = {name: _artifact(path) for name, path in paths.items()}
    marker = _json_object(marker_path, f"{label} completion marker")
    manifest = _parse_manifest(directory / "matrix_manifest.txt")
    _validate_stage_manifest(
        manifest, label, expected_n, expected_trace_sha256,
        expected_trace_manifest_sha256, expected_profile_sha256,
    )
    (
        events_sha256, arm_started_at, arm_finished_at, matrix_finished_at,
    ) = _parse_matrix_events(
        directory / "matrix_events.log", label, EXPECTED_ARMS[label],
        str(manifest["run_fingerprint"]),
    )
    expected_marker = {
        "schema_version": 2,
        "run_fingerprint": manifest["run_fingerprint"],
        "arm": EXPECTED_ARMS[label],
        "artifacts": artifacts,
    }
    if marker != expected_marker:
        raise EvidenceError(f"{label}: completion marker does not exactly bind current artifacts")
    if artifacts["raw"]["nonempty_line_n"] != expected_n:
        raise EvidenceError(f"{label}: raw line count does not equal expected N")
    if label == "L" and artifacts["decisions"]["nonempty_line_n"] != 0:
        raise EvidenceError("L: all-local decision log must be empty")
    if label != "L" and artifacts["decisions"]["nonempty_line_n"] <= 0:
        raise EvidenceError(f"{label}: Nimbus decision log must be nonempty")

    summary = _json_object(paths["summary"], f"{label} summary")
    raw_rows = _jsonl(paths["raw"], f"{label} raw")
    decisions = _jsonl(paths["decisions"], f"{label} decisions")
    policy, trigger, selector = EXPECTED_IDENTITIES[label]
    if summary.get("policy") != policy:
        raise EvidenceError(f"{label}: summary policy does not match exact arm")
    config = _object(summary.get("config"), "config", f"{label} summary")
    _validate_config(config, manifest, label, trigger, selector)
    slo_s = _number(summary.get("slo_s"), "slo_s", f"{label} summary")
    if not _close(slo_s, _number(config.get("slo_s"), "config.slo_s", label)):
        raise EvidenceError(f"{label}: summary/config SLO mismatch")

    ids: list[int] = []
    local_rows: list[dict[str, Any]] = []
    cloud_rows: list[dict[str, Any]] = []
    local_violations = 0
    cost_usd = 0.0
    local_ttfts: list[float] = []
    for line_n, row in enumerate(raw_rows, 1):
        source = f"{label} raw line {line_n}"
        request_id = _integer(row.get("request_id"), "request_id", source)
        if not 0 <= request_id < expected_n:
            raise EvidenceError(f"{source}: request_id outside [0, N)")
        ids.append(request_id)
        endpoint = row.get("endpoint")
        if endpoint not in {"local", "cloud"}:
            raise EvidenceError(f"{source}: unsupported endpoint {endpoint!r}")
        if row.get("model") != manifest["model"]:
            raise EvidenceError(f"{source}: model does not match matrix model")
        if row.get("payload_mode") != PAYLOAD_MODE:
            raise EvidenceError(
                f"{source}: payload_mode must be {PAYLOAD_MODE!r}"
            )
        if row.get("cache_mode") != "none":
            raise EvidenceError(f"{source}: cache_mode must be none")
        success = row.get("success")
        if not isinstance(success, bool):
            raise EvidenceError(f"{source}: success must be boolean")
        if success is not True:
            raise EvidenceError(f"{source}: completed local gate rows must succeed")
        _token_exact(row, source)
        if endpoint == "local":
            local_rows.append(row)
            if row.get("routed_only", False) is not False:
                raise EvidenceError(f"{source}: local row is routed_only")
            if _number(row.get("cost_usd"), "cost_usd", source) != 0.0:
                raise EvidenceError(f"{source}: local cost must be zero")
            ttft = row.get("ttft_ms")
            if success and ttft is not None:
                ttft_value = _number(ttft, "ttft_ms", source)
                if ttft_value < 0:
                    raise EvidenceError(f"{source}: TTFT is negative")
                local_ttfts.append(ttft_value)
            if (not success or ttft is None or
                    (isinstance(ttft, (int, float)) and float(ttft) > slo_s * 1000)):
                local_violations += 1
        else:
            cloud_rows.append(row)
            if row.get("routed_only") is not True or success is not True:
                raise EvidenceError(f"{source}: NullCloud row must be successful routed_only")
            if row.get("ttft_ms") is not None:
                raise EvidenceError(f"{source}: NullCloud row must not claim TTFT")
            row_cost = _number(row.get("cost_usd"), "cost_usd", source)
            expected_cost = (
                _integer(row["prompt_tokens"], "prompt_tokens", source)
                * _manifest_number(manifest, "in_price", source)
                + _integer(row["completion_tokens"], "completion_tokens", source)
                * _manifest_number(manifest, "out_price", source)
            ) / 1e6
            if not _close(row_cost, expected_cost, tolerance=1e-12):
                raise EvidenceError(f"{source}: NullCloud cost does not match token prices")
            cost_usd += row_cost
    request_id_set = set(ids)
    if len(ids) != expected_n or len(request_id_set) != expected_n:
        raise EvidenceError(f"{label}: request IDs are not unique for all N rows")
    if request_id_set != set(range(expected_n)):
        raise EvidenceError(f"{label}: request IDs are not exactly 0..N-1")
    if label == "L" and cloud_rows:
        raise EvidenceError("L: all-local anchor contains cloud rows")
    if label == "L" and any(row.get("success") is not True for row in local_rows):
        raise EvidenceError("L: all-local anchor must have N successful local rows")

    applied: list[int] = []
    previous_at_s = -math.inf
    known_statuses = {
        "applied", "no_op", "stale_retry", "stale_future_arrival",
        # Current runner emits this when progress remains stale after its
        # bounded retries; it may apply the latest exact proposed set.
        "applied_stale_bounded",
    }
    for line_n, decision in enumerate(decisions, 1):
        source = f"{label} decisions line {line_n}"
        if _integer(decision.get("decision_id"), "decision_id", source) != line_n:
            raise EvidenceError(f"{source}: decision IDs must be consecutive")
        if decision.get("trigger") != trigger or decision.get("selector") != selector:
            raise EvidenceError(f"{source}: trigger/selector does not match exact arm")
        status = decision.get("status")
        if status not in known_statuses:
            raise EvidenceError(f"{source}: unknown status {status!r}")
        at_s = _number(decision.get("at_s"), "at_s", source)
        decision_ms = _number(decision.get("decision_ms"), "decision_ms", source)
        if at_s < 0 or at_s < previous_at_s:
            raise EvidenceError(f"{source}: at_s must be nonnegative and monotone")
        if decision_ms < 0:
            raise EvidenceError(f"{source}: decision_ms must be nonnegative")
        previous_at_s = at_s

        if decision.get("prediction_scope") != PREDICTION_SCOPE:
            raise EvidenceError(f"{source}: unexpected prediction_scope")
        if decision.get("prediction_model") != PREDICTION_MODEL:
            raise EvidenceError(f"{source}: unexpected prediction_model")
        predicted = _number(
            decision.get("waiting_predicted_max_ttft_s"),
            "waiting_predicted_max_ttft_s", source,
        )
        post_kick = _number(
            decision.get("waiting_post_kick_max_ttft_s"),
            "waiting_post_kick_max_ttft_s", source,
        )
        if predicted < 0 or post_kick < 0 or post_kick > predicted + EPSILON:
            raise EvidenceError(f"{source}: invalid predicted TTFT pre/post values")

        proposed = decision.get("proposed_victim_ids")
        if not isinstance(proposed, list):
            raise EvidenceError(f"{source}: proposed_victim_ids must be a list")
        proposed_ids: list[int] = []
        for victim in proposed:
            request_id = _integer(victim, "proposed_victim_id", source)
            if request_id not in request_id_set:
                raise EvidenceError(f"{source}: proposed victim is outside raw request IDs")
            proposed_ids.append(request_id)
        if len(proposed_ids) != len(set(proposed_ids)):
            raise EvidenceError(f"{source}: proposed_victim_ids contains duplicates")

        victims = decision.get("applied_victim_ids", [])
        if not isinstance(victims, list):
            raise EvidenceError(f"{source}: applied_victim_ids must be a list")
        applied_this_row: list[int] = []
        for victim in victims:
            request_id = _integer(victim, "applied_victim_id", source)
            if request_id not in request_id_set:
                raise EvidenceError(f"{source}: applied victim is outside raw request IDs")
            applied_this_row.append(request_id)
        if len(applied_this_row) != len(set(applied_this_row)):
            raise EvidenceError(f"{source}: applied_victim_ids contains duplicates")
        if status in {"applied", "applied_stale_bounded"}:
            if applied_this_row != proposed_ids:
                raise EvidenceError(
                    f"{source}: applied status requires one exact proposed/applied list"
                )
            if status == "applied" and not applied_this_row:
                raise EvidenceError(
                    f"{source}: applied status requires nonempty applied victims"
                )
            if applied_this_row:
                deadline_s = slo_s - _number(
                    config.get("ttft_guard_ms"), "config.ttft_guard_ms", source
                ) / 1000.0
                if predicted <= deadline_s + EPSILON:
                    raise EvidenceError(
                        f"{source}: applied victims require predicted TTFT above trigger limit"
                    )
        elif applied_this_row:
            raise EvidenceError(
                f"{source}: only applied status may carry applied victims"
            )
        if status == "no_op":
            if proposed_ids:
                raise EvidenceError(f"{source}: no_op status may not propose victims")
            if not _close(predicted, post_kick):
                raise EvidenceError(f"{source}: no_op status changed predicted TTFT")
        applied.extend(applied_this_row)
    if len(applied) != len(set(applied)):
        raise EvidenceError(f"{label}: a victim was applied more than once")
    cloud_ids = {int(row["request_id"]) for row in cloud_rows}
    if label != "L" and set(applied) != cloud_ids:
        raise EvidenceError(f"{label}: applied victim IDs do not equal NullCloud IDs")

    overall = _object(summary.get("overall"), "overall", f"{label} summary")
    local = _object(summary.get("local"), "local", f"{label} summary")
    cloud = _object(summary.get("cloud"), "cloud", f"{label} summary")
    alignment = _object(
        summary.get("token_alignment"), "token_alignment", f"{label} summary"
    )
    expected_counts = {
        "overall.n": (overall, "n", expected_n),
        "overall.success": (overall, "success", sum(bool(r["success"]) for r in raw_rows)),
        "local.n": (local, "n", len(local_rows)),
        "local.success": (local, "success", sum(bool(r["success"]) for r in local_rows)),
        "local.slo_violations": (local, "slo_violations", local_violations),
        "local.slo_measured_n": (local, "slo_measured_n", len(local_rows)),
        "cloud.n": (cloud, "n", len(cloud_rows)),
        "cloud.success": (cloud, "success", len(cloud_rows)),
    }
    for field, (obj, key, expected) in expected_counts.items():
        if _integer(obj.get(key), field, f"{label} summary") != expected:
            raise EvidenceError(f"{label}: summary {field} does not match raw")
    local_success_n = sum(bool(row["success"]) for row in local_rows)
    alignment_expected = {
        "local_success_n": local_success_n, "measured_n": local_success_n,
        "missing_prompt_usage_n": 0, "prompt_exact_n": local_success_n,
        "decode_measured_n": local_success_n, "missing_completion_usage_n": 0,
        "decode_cap_hit_n": local_success_n,
    }
    for key, expected in alignment_expected.items():
        if _integer(alignment.get(key), f"token_alignment.{key}", label) != expected:
            raise EvidenceError(f"{label}: summary token alignment is not exact")
    summary_cost = _number(overall.get("cost_usd"), "overall.cost_usd", label)
    if not _close(summary_cost, cost_usd):
        raise EvidenceError(f"{label}: summary cost does not match raw")

    percentiles = {
        "ttft_p50_ms": _percentile(local_ttfts, 50),
        "ttft_p95_ms": _percentile(local_ttfts, 95),
        "ttft_p99_ms": _percentile(local_ttfts, 99),
    }
    for key, expected in percentiles.items():
        actual = local.get(key)
        if expected is None:
            if actual is not None:
                raise EvidenceError(f"{label}: summary {key} should be null")
        elif not _close(_number(actual, key, label), expected):
            raise EvidenceError(f"{label}: summary {key} does not match raw")
    pct = 100.0 * local_violations / max(len(local_rows), 1)
    if not _close(_number(local.get("slo_violation_pct"), "local.slo_violation_pct", label), pct):
        raise EvidenceError(f"{label}: summary local violation percentage mismatch")

    return StageAudit(
        label=label, arm_text=EXPECTED_ARMS[label],
        fingerprint=str(manifest["run_fingerprint"]), manifest=manifest,
        request_ids=request_id_set, cloud_ids=cloud_ids, local_n=len(local_rows),
        routed_n=len(cloud_rows), success_n=sum(bool(r["success"]) for r in raw_rows),
        local_violation_n=local_violations, local_violation_pct=pct,
        cost_usd=cost_usd, ttft_p50_ms=percentiles["ttft_p50_ms"],
        ttft_p95_ms=percentiles["ttft_p95_ms"],
        ttft_p99_ms=percentiles["ttft_p99_ms"],
        ttft_max_ms=max(local_ttfts) if local_ttfts else None,
        decision_n=len(decisions), applied_victim_n=len(applied),
        events_sha256=events_sha256, arm_started_at=arm_started_at,
        arm_finished_at=arm_finished_at, matrix_finished_at=matrix_finished_at,
    )


def _arm_output(stage: StageAudit, expected_n: int) -> dict[str, Any]:
    return {
        "arm": stage.arm_text, "n": expected_n, "success_n": stage.success_n,
        "local_n": stage.local_n, "routed_n": stage.routed_n,
        "routed_pct": 100.0 * stage.routed_n / expected_n,
        "local_violation_n": stage.local_violation_n,
        "local_violation_pct": stage.local_violation_pct,
        "estimated_cloud_cost_usd": stage.cost_usd,
        "local_ttft_ms": {
            "p50": stage.ttft_p50_ms, "p95": stage.ttft_p95_ms,
            "p99": stage.ttft_p99_ms, "max": stage.ttft_max_ms,
        },
        "decision_n": stage.decision_n,
        "applied_victim_n": stage.applied_victim_n,
        "local_token_exact": True,
    }


def analyze(
    l_dir: str | Path, a_dir: str | Path, c_dir: str | Path, *,
    expected_n: int = DEFAULT_EXPECTED_N,
    expected_trace_sha256: str | None = DEFAULT_TRACE_SHA256,
    expected_trace_manifest_sha256: str | None = DEFAULT_TRACE_MANIFEST_SHA256,
    expected_profile_sha256: str | None = None,
    min_cooldown_s: float = 20.0,
) -> dict[str, Any]:
    if expected_n <= 0:
        raise EvidenceError("expected_n must be positive")
    if (isinstance(min_cooldown_s, bool)
            or not isinstance(min_cooldown_s, (int, float))
            or not math.isfinite(float(min_cooldown_s))
            or float(min_cooldown_s) < 0):
        raise EvidenceError("min_cooldown_s must be finite and nonnegative")
    min_cooldown_s = float(min_cooldown_s)
    directories = {"L": Path(l_dir), "A": Path(a_dir), "C": Path(c_dir)}
    stages = {
        label: _audit_stage(
            directories[label], label, expected_n, expected_trace_sha256,
            expected_trace_manifest_sha256, expected_profile_sha256,
        )
        for label in ("L", "A", "C")
    }
    _validate_common_manifests(stages)
    if len({frozenset(stage.request_ids) for stage in stages.values()}) != 1:
        raise EvidenceError("L/A/C raw request-ID sets differ")
    l_to_a_cooldown_s = (
        stages["A"].arm_started_at - stages["L"].matrix_finished_at
    ).total_seconds()
    a_to_c_cooldown_s = (
        stages["C"].arm_started_at - stages["A"].matrix_finished_at
    ).total_seconds()
    if l_to_a_cooldown_s + EPSILON < min_cooldown_s:
        raise EvidenceError(
            f"stage order/cooldown invalid: A start is {l_to_a_cooldown_s}s "
            f"after L matrix finish, require >= {min_cooldown_s}s"
        )
    if a_to_c_cooldown_s + EPSILON < min_cooldown_s:
        raise EvidenceError(
            f"stage order/cooldown invalid: C start is {a_to_c_cooldown_s}s "
            f"after A matrix finish, require >= {min_cooldown_s}s"
        )

    a_ids, c_ids = stages["A"].cloud_ids, stages["C"].cloud_ids
    intersection_n = len(a_ids & c_ids)
    union_n = len(a_ids | c_ids)
    jaccard = intersection_n / union_n if union_n else 1.0
    pressure = stages["L"].local_violation_n > 0
    a_safe = stages["A"].local_violation_n == 0
    c_safe = stages["C"].local_violation_n == 0
    gate_pass = pressure and a_safe and c_safe

    def delta(left: float | int | None, right: float | int | None) -> float | int | None:
        return None if left is None or right is None else left - right

    manifest = stages["L"].manifest
    result = {
        "schema_version": 1,
        "analysis": "E12 ShareGPT current-turn local L/A/C gate",
        "integrity_valid": True,
        "text_payload_in_output": False,
        "evidence": {
            "n": expected_n,
            "trace_sha256": manifest["trace_sha256"],
            "trace_manifest_sha256": manifest["trace_manifest_sha256"],
            "profile_sha256": manifest["profile_sha256"],
            "commit": manifest["commit"],
            "server_pid": _manifest_int(manifest, "server_pid", "L manifest"),
            "server_log_prefix_sha256": manifest["server_log_prefix_sha256"],
            "stage_fingerprints": {
                label: stages[label].fingerprint for label in ("L", "A", "C")
            },
            "completion_markers_valid": True,
            "request_id_sets_equal": True,
            "common_manifest_fields_equal": True,
            "stage_order": ["L", "A", "C"],
            "min_cooldown_s": min_cooldown_s,
            "stage_timing": {
                label: {
                    "arm_started_at": _iso_z(stages[label].arm_started_at),
                    "arm_finished_at": _iso_z(stages[label].arm_finished_at),
                    "matrix_finished_at": _iso_z(
                        stages[label].matrix_finished_at
                    ),
                    "matrix_events_sha256": stages[label].events_sha256,
                }
                for label in ("L", "A", "C")
            },
            "cooldowns_s": {
                "l_matrix_finish_to_a_start": l_to_a_cooldown_s,
                "a_matrix_finish_to_c_start": a_to_c_cooldown_s,
            },
        },
        "arms": {label: _arm_output(stages[label], expected_n) for label in ("L", "A", "C")},
        "victim_overlap": {
            "intersection_n": intersection_n, "union_n": union_n,
            "a_only_n": len(a_ids - c_ids), "c_only_n": len(c_ids - a_ids),
            "jaccard": jaccard,
        },
        "deltas_c_minus_a": {
            "routed_n": stages["C"].routed_n - stages["A"].routed_n,
            "routed_pct_points": 100.0 * (
                stages["C"].routed_n - stages["A"].routed_n
            ) / expected_n,
            "estimated_cloud_cost_usd": stages["C"].cost_usd - stages["A"].cost_usd,
            "local_ttft_p50_ms": delta(stages["C"].ttft_p50_ms, stages["A"].ttft_p50_ms),
            "local_ttft_p95_ms": delta(stages["C"].ttft_p95_ms, stages["A"].ttft_p95_ms),
            "local_ttft_p99_ms": delta(stages["C"].ttft_p99_ms, stages["A"].ttft_p99_ms),
        },
        "gates": {
            "l_pressure_observed": {
                "pass": pressure,
                "local_violation_n": stages["L"].local_violation_n,
            },
            "a_retained_local_safety": {
                "pass": a_safe,
                "local_violation_n": stages["A"].local_violation_n,
            },
            "c_retained_local_safety": {
                "pass": c_safe,
                "local_violation_n": stages["C"].local_violation_n,
            },
        },
        "gate_pass": gate_pass,
        "verdict": "pass" if gate_pass else "fail",
    }
    return result


def render_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# E12 current-turn local gate", "",
        f"- Verdict: **{result['verdict'].upper()}**",
        f"- Trace: `{result['evidence']['trace_sha256']}`",
        f"- Profile: `{result['evidence']['profile_sha256']}`", "",
        "| Arm | Local | Routed | Local 5s violations | Cost (USD) | TTFT p50/p95/p99 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("L", "A", "C"):
        arm = result["arms"][label]
        ttft = arm["local_ttft_ms"]
        lines.append(
            f"| {label} | {arm['local_n']} | {arm['routed_n']} | "
            f"{arm['local_violation_n']} | {arm['estimated_cloud_cost_usd']:.9f} | "
            f"{ttft['p50']}/{ttft['p95']}/{ttft['p99']} |"
        )
    overlap = result["victim_overlap"]
    evidence = result["evidence"]
    timing = evidence["stage_timing"]
    lines.extend([
        "", "## Stage timing", "",
        "| Stage | Arm start | Arm finish | Matrix finish | Events SHA-256 |",
        "|---|---|---|---|---|",
        *[
            f"| {label} | {timing[label]['arm_started_at']} | "
            f"{timing[label]['arm_finished_at']} | "
            f"{timing[label]['matrix_finished_at']} | "
            f"`{timing[label]['matrix_events_sha256']}` |"
            for label in ("L", "A", "C")
        ],
        "",
        f"- L matrix finish to A start: "
        f"{evidence['cooldowns_s']['l_matrix_finish_to_a_start']} s",
        f"- A matrix finish to C start: "
        f"{evidence['cooldowns_s']['a_matrix_finish_to_c_start']} s",
        "", "## A/C aggregate comparison", "",
        f"- Victim intersection/union: {overlap['intersection_n']}/{overlap['union_n']}",
        f"- Jaccard: {overlap['jaccard']:.6f}",
        f"- C minus A routed rows: {result['deltas_c_minus_a']['routed_n']}", "",
        "No prompt text or request-level victim IDs are included.", "",
    ])
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as output:
        temp = Path(output.name)
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--l-dir", type=Path, required=True)
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--c-dir", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=DEFAULT_EXPECTED_N)
    parser.add_argument("--expected-trace-sha256", default=DEFAULT_TRACE_SHA256)
    parser.add_argument(
        "--expected-trace-manifest-sha256", default=DEFAULT_TRACE_MANIFEST_SHA256,
    )
    parser.add_argument("--expected-profile-sha256")
    parser.add_argument("--min-cooldown-s", type=float, default=20.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = analyze(
            args.l_dir, args.a_dir, args.c_dir,
            expected_n=args.expected_n,
            expected_trace_sha256=args.expected_trace_sha256,
            expected_trace_manifest_sha256=args.expected_trace_manifest_sha256,
            expected_profile_sha256=args.expected_profile_sha256,
            min_cooldown_s=args.min_cooldown_s,
        )
        rendered = render_json(result)
        if args.json_out:
            _atomic_write(args.json_out, rendered)
        else:
            sys.stdout.write(rendered)
        if args.markdown_out:
            _atomic_write(args.markdown_out, render_markdown(result))
        return 0 if result["gate_pass"] else 1
    except EvidenceError as exc:
        print(f"current-turn gate evidence invalid: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
