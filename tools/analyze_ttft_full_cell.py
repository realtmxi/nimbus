#!/usr/bin/env python3
"""Validate and score the pre-registered five-arm TTFT full-cell gate.

The input is one matrix directory containing exactly five completion markers
and their stem-derived raw/summary/decision artifacts.  The statistical unit
is the complete run/arm; this tool never treats requests as independent
samples and performs no request-level bootstrap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


EXPECTED_ARMS = frozenset(("A", "B", "C", "K", "L"))
IDENTITIES = {
    ("nimbus", "ttft_pred", "cost_cachedisp_old"): "A",
    ("nimbus", "ttft_pred", "newest"): "B",
    ("nimbus", "ttft_pred", "cost_disp_current"): "C",
    ("nimbus", "kv_gap", "cost_disp_current"): "K",
    ("all_local", None, None): "L",
}
ARM_CONFIG_IDENTITIES = {
    "A": ("ttft_pred", "cost_cachedisp_old"),
    "B": ("ttft_pred", "newest"),
    "C": ("ttft_pred", "cost_disp_current"),
    "K": ("kv_gap", "cost_disp_current"),
    # The all-local anchor still records argparse's frozen Nimbus defaults.
    "L": ("kv_gap", "cost_disp_current"),
}
ARM_DESCRIPTIONS = {
    "A": "ttft_pred + cost_cachedisp_old (candidate)",
    "B": "ttft_pred + newest",
    "C": "ttft_pred + cost_disp_current",
    "K": "kv_gap + cost_disp_current (shipped-v3 anchor)",
    "L": "all_local (pressure anchor)",
}
PREDICTION_SCOPE = "waiting_only"
PREDICTION_MODEL = "seq_slots_shared_prefill_lane_v1"
FROZEN_ARM_ORDER = (
    "ttft_pred:newest:0",
    "kv_gap:cost_disp_current:0",
    "ttft_pred:cost_disp_current:0",
    "ttft_pred:cost_cachedisp_old:0",
    "anchor:all_local:0",
)
EPSILON = 1e-9
DEFAULT_EXPECTED_N = 11605
DEFAULT_RUN_FINGERPRINT = (
    "7036a1c8cecc923b5bee514bbe7f4b7e999ae4ec029265ff9bec3589f60c6c8d"
)
DEFAULT_CONFIG_CONTRACT = {
    "scenario": "extreme_burst_1200",
    "seed": 0,
    "time_scale": 1.0,
    "max_inflight": 128,
    "max_tokens_override": None,
    "temperature": 0.0,
    "ignore_eos": True,
    "prefill_tput": 3255.004414021381,
    "tpot_ms": 152.31521785505774,
    "first_token_overhead_ms": 440.4170340755918,
    "slo_s": 5.0,
    "ttft_guard_ms": 1712.0,
    "nimbus_tick_ms": 250.0,
    "kv_capacity_tokens": 112656.0,
    "kv_hysteresis_fraction": 0.05,
    "cloud": "null",
    "cloud_max_concurrency": 32,
    "local_url": "http://127.0.0.1:8010/v1/chat/completions",
    "local_model": "qwen3-32b",
    "in_price": 0.15,
    "out_price": 1.20,
}
DEFAULT_MANIFEST_CONTRACT = {
    "commit": "c6de62a3c84e5f6e92df2ef60d00866fb5b79e42",
    "trace_sha256": (
        "465ef070d2a4a399ad41142b9e40bd9c505599d05af2f9d4dd56f9eb02024c52"
    ),
    "trace_manifest_sha256": (
        "c5621d3e45f7b1e2ee49485f7948a267dde65d29b34a377247061f7cccc72f7a"
    ),
    "profile_sha256": (
        "ab703ddc11bc63a2e1a3ca5e7b2367dc8232eb37f623a8e625a306021278c819"
    ),
}


class EvidenceError(ValueError):
    """The matrix is not structurally valid evidence."""


@dataclass(frozen=True)
class ArmMetrics:
    arm: str
    description: str
    source: str
    policy: str
    trigger: str | None
    selector: str | None
    seed: int
    n: int
    success_n: int
    routed_n: int
    local_n: int
    local_violation_n: int
    local_violation_pct: float
    pessimistic_violation_n: int
    pessimistic_combined_pct: float
    cost_usd: float
    local_ttft_p50_s: float | None
    local_ttft_p95_s: float | None
    local_ttft_p99_s: float | None
    slo_s: float
    ttft_guard_s: float
    post_kick_limit_s: float | None
    max_post_kick_ttft_s: float | None
    token_exact: bool
    prediction_scope: str | None
    prediction_model: str | None
    kv_read_failures: int | None
    decision_n: int
    applied_victim_n: int


@dataclass(frozen=True)
class MarkerEvidence:
    marker_path: Path
    raw_path: Path
    summary_path: Path
    decisions_path: Path
    arm: str
    arm_text: str
    seed: int
    run_fingerprint: str
    artifacts: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class JsonlAudit:
    decision_n: int
    applied_victim_n: int
    max_predicted_ttft_s: float | None
    max_post_kick_ttft_s: float | None


def _object(mapping: dict[str, Any], key: str, source: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise EvidenceError(f"{source}: missing object {key!r}")
    return value


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


def _optional_ms_to_s(value: Any, field: str, source: str) -> float | None:
    return None if value is None else _number(value, field, source) / 1000.0


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=EPSILON)


def _contract_equal(actual: Any, expected: Any) -> bool:
    """Exact JSON-value equality, allowing int/float representation parity."""
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and float(actual) == float(expected)
        )
    return type(actual) is type(expected) and actual == expected


def _validate_config_contract(
    config: dict[str, Any], arm: str, source: str,
    expected_config: dict[str, Any],
) -> None:
    for field, expected in expected_config.items():
        actual = config.get(field)
        if not _contract_equal(actual, expected):
            raise EvidenceError(
                f"{source}: config.{field}={actual!r} does not match frozen "
                f"full-cell value {expected!r}"
            )
    expected_trigger, expected_selector = ARM_CONFIG_IDENTITIES[arm]
    for field, expected in (
        ("nimbus_trigger", expected_trigger),
        ("nimbus_selector", expected_selector),
    ):
        actual = config.get(field)
        if actual != expected:
            raise EvidenceError(
                f"{source}: config.{field}={actual!r}, expected {expected!r} "
                f"for arm {arm}"
            )


def _jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"{label} is unreadable UTF-8 JSONL: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(
                f"{label} line {line_n} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise EvidenceError(f"{label} line {line_n} must be a JSON object")
        rows.append(row)
    return rows


def _request_id(value: Any, field: str, source: str, expected_n: int) -> int:
    request_id = _integer(value, field, source)
    if not 0 <= request_id < expected_n:
        raise EvidenceError(
            f"{source}: {field}={request_id} is outside [0, {expected_n})"
        )
    return request_id


def _request_id_list(
    value: Any, field: str, source: str, expected_n: int,
) -> list[int]:
    if not isinstance(value, list):
        raise EvidenceError(f"{source}: {field} must be a list")
    ids = [
        _request_id(item, f"{field}[{index}]", source, expected_n)
        for index, item in enumerate(value)
    ]
    if len(ids) != len(set(ids)):
        raise EvidenceError(f"{source}: {field} contains duplicate request IDs")
    return ids


def _require_queue_counter(
    queue: dict[str, Any], field: str, expected: int, source: str,
) -> None:
    actual = _integer(queue.get(field), f"queue.{field}", source)
    if actual != expected:
        raise EvidenceError(
            f"{source}: queue.{field}={actual}, but JSONL evidence requires {expected}"
        )


def _audit_jsonl(
    *,
    raw_path: Path,
    decisions_path: Path,
    arm: str,
    trigger: str | None,
    selector: str | None,
    expected_n: int,
    local_n: int,
    routed_n: int,
    expected_local_violation_n: int,
    expected_pessimistic_violation_n: int,
    expected_cost_usd: float,
    queue: dict[str, Any],
    post_kick_limit_s: float | None,
    expected_model: str,
    slo_s: float,
    in_price_mtok: float,
    out_price_mtok: float,
) -> JsonlAudit:
    """Independently validate request rows and the recorded decision lifecycle.

    This deliberately does not replay selector ranks: the current evidence
    schema records only snapshot count/hash, not snapshot IDs, waiting ages, or
    the full in-flight predictor state needed for an independent replay.
    """
    source = raw_path.name
    raw_rows = _jsonl_objects(raw_path, source)
    if len(raw_rows) != expected_n:
        raise EvidenceError(
            f"{source}: parsed {len(raw_rows)} raw rows, expected {expected_n}"
        )

    raw_ids: list[int] = []
    local_ids: set[int] = set()
    cloud_ids: set[int] = set()
    local_violation_n = 0
    raw_cost_usd = 0.0
    for line_n, row in enumerate(raw_rows, 1):
        row_source = f"{source} line {line_n}"
        request_id = _request_id(
            row.get("request_id"), "request_id", row_source, expected_n
        )
        raw_ids.append(request_id)
        if row.get("success") is not True:
            raise EvidenceError(f"{row_source}: success must be true")
        if row.get("error") is not None or row.get("error_type") is not None:
            raise EvidenceError(f"{row_source}: successful row carries an error")
        if row.get("cache_mode") != "none":
            raise EvidenceError(f"{row_source}: cache_mode must be 'none'")
        if row.get("model") != expected_model:
            raise EvidenceError(
                f"{row_source}: model={row.get('model')!r}, expected {expected_model!r}"
            )

        prompt = _integer(row.get("prompt_tokens"), "prompt_tokens", row_source)
        decode = _integer(
            row.get("completion_tokens"), "completion_tokens", row_source
        )
        scheduler_prompt = _integer(
            row.get("scheduler_prompt_tokens"),
            "scheduler_prompt_tokens",
            row_source,
        )
        scheduler_uncached = _integer(
            row.get("scheduler_uncached_prompt_tokens"),
            "scheduler_uncached_prompt_tokens",
            row_source,
        )
        scheduler_decode = _integer(
            row.get("scheduler_decode_tokens"),
            "scheduler_decode_tokens",
            row_source,
        )
        if prompt != scheduler_prompt or scheduler_uncached != scheduler_prompt:
            raise EvidenceError(
                f"{row_source}: no-cache prompt/scheduler token accounting is not exact"
            )
        if decode != scheduler_decode:
            raise EvidenceError(
                f"{row_source}: completion/scheduler decode accounting is not exact"
            )

        endpoint = row.get("endpoint")
        if endpoint == "local":
            if row.get("routed_only", False) is not False:
                raise EvidenceError(f"{row_source}: local routed_only must be false")
            local_cost = _number(row.get("cost_usd"), "cost_usd", row_source)
            if local_cost != 0.0:
                raise EvidenceError(f"{row_source}: local cost_usd must be zero")
            ttft_value = row.get("ttft_ms")
            if ttft_value is None:
                local_violation_n += 1
            else:
                ttft_ms = _number(ttft_value, "ttft_ms", row_source)
                if ttft_ms < 0:
                    raise EvidenceError(f"{row_source}: ttft_ms must be non-negative")
                if ttft_ms > slo_s * 1000.0:
                    local_violation_n += 1
            local_ids.add(request_id)
        elif endpoint == "cloud":
            if row.get("routed_only") is not True:
                raise EvidenceError(f"{row_source}: cloud row must be routed_only")
            actual_cost = _number(row.get("cost_usd"), "cost_usd", row_source)
            expected_row_cost = (
                prompt * in_price_mtok + decode * out_price_mtok
            ) / 1e6
            if not math.isclose(
                actual_cost, expected_row_cost, rel_tol=0.0, abs_tol=1e-12
            ):
                raise EvidenceError(
                    f"{row_source}: cloud cost_usd={actual_cost} does not match "
                    f"token-price cost {expected_row_cost}"
                )
            raw_cost_usd += actual_cost
            cloud_ids.add(request_id)
        else:
            raise EvidenceError(
                f"{row_source}: endpoint must be 'local' or 'cloud', got {endpoint!r}"
            )

    if len(raw_ids) != len(set(raw_ids)):
        raise EvidenceError(f"{source}: raw request IDs are not unique")
    expected_ids = set(range(expected_n))
    if set(raw_ids) != expected_ids:
        missing = sorted(expected_ids - set(raw_ids))[:5]
        extra = sorted(set(raw_ids) - expected_ids)[:5]
        raise EvidenceError(
            f"{source}: raw request IDs are not exactly 0..{expected_n - 1}; "
            f"missing={missing}, extra={extra}"
        )
    if len(local_ids) != local_n or len(cloud_ids) != routed_n:
        raise EvidenceError(
            f"{source}: raw local/cloud counts {len(local_ids)}/{len(cloud_ids)} "
            f"do not match summary {local_n}/{routed_n}"
        )
    if local_violation_n != expected_local_violation_n:
        raise EvidenceError(
            f"{source}: raw local 5s violations={local_violation_n}, "
            f"summary={expected_local_violation_n}"
        )
    pessimistic_n = local_violation_n + len(cloud_ids)
    if pessimistic_n != expected_pessimistic_violation_n:
        raise EvidenceError(
            f"{source}: raw pessimistic violations={pessimistic_n}, "
            f"summary={expected_pessimistic_violation_n}"
        )
    if not _close(raw_cost_usd, expected_cost_usd):
        raise EvidenceError(
            f"{source}: raw cloud cost={raw_cost_usd} does not match "
            f"summary cost={expected_cost_usd}"
        )

    decision_source = decisions_path.name
    decisions = _jsonl_objects(decisions_path, decision_source)
    if arm == "L":
        if decisions:
            raise EvidenceError(
                f"{decision_source}: all-local anchor decision log must be empty"
            )
        if cloud_ids:
            raise EvidenceError(f"{source}: all-local anchor contains cloud rows")
        return JsonlAudit(0, 0, None, None)
    if not decisions:
        raise EvidenceError(f"{decision_source}: Nimbus decision log is empty")
    assert trigger is not None and selector is not None

    statuses = {
        "no_op", "applied", "applied_stale_bounded",
        "stale_retry", "stale_future_arrival",
    }
    stale_statuses = {"stale_retry", "stale_future_arrival"}
    applied_ids: set[int] = set()
    previous_at_s = -math.inf
    proposed_round_n = 0
    applied_round_n = 0
    stale_n = 0
    decision_ms_values: list[float] = []
    predicted_values: list[float] = []
    post_values: list[float] = []
    for index, row in enumerate(decisions, 1):
        row_source = f"{decision_source} line {index}"
        decision_id = _integer(row.get("decision_id"), "decision_id", row_source)
        if decision_id != index:
            raise EvidenceError(
                f"{row_source}: decision_id={decision_id}, expected consecutive {index}"
            )
        at_s = _number(row.get("at_s"), "at_s", row_source)
        if at_s < 0 or at_s < previous_at_s:
            raise EvidenceError(f"{row_source}: at_s is negative or non-monotone")
        previous_at_s = at_s
        if row.get("trigger") != trigger or row.get("selector") != selector:
            raise EvidenceError(
                f"{row_source}: trigger/selector does not match arm {arm}"
            )
        status = row.get("status")
        if status not in statuses:
            raise EvidenceError(f"{row_source}: unknown decision status {status!r}")
        proposed = _request_id_list(
            row.get("proposed_victim_ids"),
            "proposed_victim_ids",
            row_source,
            expected_n,
        )
        if status in stale_statuses:
            applied_value = row.get("applied_victim_ids", [])
        else:
            if "applied_victim_ids" not in row:
                raise EvidenceError(
                    f"{row_source}: final decision lacks applied_victim_ids"
                )
            applied_value = row["applied_victim_ids"]
        applied = _request_id_list(
            applied_value, "applied_victim_ids", row_source, expected_n
        )
        if not set(applied).issubset(proposed):
            raise EvidenceError(f"{row_source}: applied victims were not proposed")
        if status in stale_statuses:
            stale_n += 1
            if applied:
                raise EvidenceError(f"{row_source}: stale decision applied victims")
        elif status == "no_op":
            if proposed or applied:
                raise EvidenceError(f"{row_source}: no_op decision has victims")
        elif status == "applied":
            if not applied or applied != proposed:
                raise EvidenceError(
                    f"{row_source}: applied status requires one exact proposed/applied set"
                )
        else:  # applied_stale_bounded may validly carry an empty latest set.
            if applied != proposed:
                raise EvidenceError(
                    f"{row_source}: applied_stale_bounded proposed/applied differ"
                )
        if proposed:
            proposed_round_n += 1
        if applied:
            applied_round_n += 1
        for request_id in applied:
            if request_id in applied_ids:
                raise EvidenceError(
                    f"{row_source}: request {request_id} was applied more than once"
                )
            applied_ids.add(request_id)

        decision_ms = _number(row.get("decision_ms"), "decision_ms", row_source)
        if decision_ms < 0:
            raise EvidenceError(f"{row_source}: decision_ms must be non-negative")
        decision_ms_values.append(decision_ms)

        if arm in ("A", "B", "C"):
            predicted = _number(
                row.get("waiting_predicted_max_ttft_s"),
                "waiting_predicted_max_ttft_s",
                row_source,
            )
            post = _number(
                row.get("waiting_post_kick_max_ttft_s"),
                "waiting_post_kick_max_ttft_s",
                row_source,
            )
            if predicted < 0 or post < 0 or post > predicted + EPSILON:
                raise EvidenceError(
                    f"{row_source}: invalid pre/post predicted TTFT ordering"
                )
            assert post_kick_limit_s is not None
            if applied and predicted <= post_kick_limit_s + EPSILON:
                raise EvidenceError(
                    f"{row_source}: applied decision requires pre-kick TTFT > "
                    f"{post_kick_limit_s}"
                )
            if not proposed and not _close(predicted, post):
                raise EvidenceError(
                    f"{row_source}: no-proposal decision changed predicted TTFT"
                )
            # A post-kick bound breach is an algorithm outcome.  It is retained
            # and scored by the frozen gate below rather than rejected as bad
            # evidence here.
            predicted_values.append(predicted)
            post_values.append(post)

    if applied_ids != cloud_ids:
        raise EvidenceError(
            f"{decision_source}: union(applied victims) does not equal raw cloud IDs; "
            f"applied_only={sorted(applied_ids - cloud_ids)[:5]}, "
            f"cloud_only={sorted(cloud_ids - applied_ids)[:5]}"
        )

    decision_n = len(decisions)
    _require_queue_counter(queue, "nimbus_ticks", decision_n, decision_source)
    _require_queue_counter(
        queue, "nimbus_decision_calls", decision_n, decision_source
    )
    _require_queue_counter(
        queue, "nimbus_kick_rounds", proposed_round_n, decision_source
    )
    _require_queue_counter(
        queue, "nimbus_applied_kick_rounds", applied_round_n, decision_source
    )
    _require_queue_counter(
        queue, "nimbus_stale_decisions", stale_n, decision_source
    )
    _require_queue_counter(
        queue, "nimbus_kicked", len(applied_ids), decision_source
    )
    mean_ms = sum(decision_ms_values) / decision_n
    max_ms = max(decision_ms_values)
    for field, expected in (
        ("nimbus_decision_mean_ms", mean_ms),
        ("nimbus_decision_max_ms", max_ms),
    ):
        actual = _number(queue.get(field), f"queue.{field}", decision_source)
        if not _close(actual, expected):
            raise EvidenceError(
                f"{decision_source}: queue.{field}={actual} does not match "
                f"decision JSONL value {expected}"
            )

    max_predicted = max(predicted_values) if predicted_values else None
    max_post = max(post_values) if post_values else None
    if arm in ("A", "B", "C"):
        assert max_predicted is not None and max_post is not None
        for field, expected in (
            ("nimbus_max_waiting_predicted_ttft_s", max_predicted),
            ("nimbus_max_waiting_post_kick_ttft_s", max_post),
        ):
            actual = _number(queue.get(field), f"queue.{field}", decision_source)
            if not _close(actual, expected):
                raise EvidenceError(
                    f"{decision_source}: queue.{field}={actual} does not match "
                    f"decision JSONL maximum {expected}"
                )
    return JsonlAudit(decision_n, len(applied_ids), max_predicted, max_post)


def _artifact(path: Path) -> dict[str, Any]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"missing/unreadable marker artifact {path.name}: {exc}") from exc
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "nonempty_line_n": sum(bool(line.strip()) for line in content.splitlines()),
    }


def _parse_marker_arm(arm_text: Any, source: str) -> tuple[str, int]:
    if not isinstance(arm_text, str):
        raise EvidenceError(f"{source}: marker arm must be a string")
    if arm_text == "anchor:all_local:0":
        return "L", 0
    parts = arm_text.split(":")
    if len(parts) != 3 or any(not part for part in parts):
        raise EvidenceError(f"{source}: invalid marker arm {arm_text!r}")
    trigger, selector, seed_text = parts
    try:
        if seed_text != seed_text.strip():
            raise ValueError
        seed = int(seed_text)
    except ValueError as exc:
        raise EvidenceError(f"{source}: marker arm seed is not an integer") from exc
    arm = IDENTITIES.get(("nimbus", trigger, selector))
    if arm is None:
        raise EvidenceError(f"{source}: unexpected marker arm {arm_text!r}")
    return arm, seed


def _load_marker(path: Path, expected_n: int) -> MarkerEvidence:
    source = path.name
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{source}: cannot read valid completion marker: {exc}") from exc
    if not isinstance(marker, dict):
        raise EvidenceError(f"{source}: completion marker must be an object")
    fingerprint = marker.get("run_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or not fingerprint.strip()
        or fingerprint != fingerprint.strip()
    ):
        raise EvidenceError(f"{source}: run_fingerprint must be a nonempty exact string")
    arm_text = marker.get("arm")
    arm, seed = _parse_marker_arm(arm_text, source)

    suffix = ".complete.json"
    if not path.name.endswith(suffix):
        raise EvidenceError(f"{source}: invalid completion-marker filename")
    stem = path.name[:-len(suffix)]
    artifact_paths = {
        "raw": path.with_name(stem + ".jsonl"),
        "summary": path.with_name(stem + ".summary.json"),
        "decisions": path.with_name(stem + ".decisions.jsonl"),
    }
    artifacts = {name: _artifact(value) for name, value in artifact_paths.items()}
    expected_marker = {
        "schema_version": 1,
        "run_fingerprint": fingerprint,
        "arm": arm_text,
        "artifacts": artifacts,
    }
    if marker != expected_marker:
        raise EvidenceError(
            f"{source}: completion marker does not exactly match current artifacts"
        )
    if artifacts["raw"]["nonempty_line_n"] != expected_n:
        raise EvidenceError(
            f"{source}: raw artifact has {artifacts['raw']['nonempty_line_n']} "
            f"rows, expected {expected_n}"
        )
    decision_n = artifacts["decisions"]["nonempty_line_n"]
    if (arm == "L" and decision_n != 0) or (arm != "L" and decision_n <= 0):
        raise EvidenceError(f"{source}: invalid decision-log line count for arm {arm}")
    return MarkerEvidence(
        marker_path=path,
        raw_path=artifact_paths["raw"],
        summary_path=artifact_paths["summary"],
        decisions_path=artifact_paths["decisions"],
        arm=arm,
        arm_text=arm_text,
        seed=seed,
        run_fingerprint=fingerprint,
        artifacts=artifacts,
    )


def _validate_manifest(
    path: Path,
    fingerprint: str,
    expected_n: int,
    expected_manifest: dict[str, str],
) -> None:
    manifest_path = path / "matrix_manifest.txt"
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"matrix_manifest.txt is missing or unreadable: {exc}") from exc
    fingerprint_lines = [line for line in lines if line.startswith("run_fingerprint=")]
    expected = f"run_fingerprint={fingerprint}"
    if fingerprint_lines != [expected]:
        raise EvidenceError(
            "matrix_manifest.txt must contain exactly one exact shared run_fingerprint line"
        )
    arm_lines = [line for line in lines if line.startswith("arms=")]
    manifest_arms = tuple(arm_lines[0][len("arms="):].split()) if len(arm_lines) == 1 else ()
    if len(arm_lines) != 1 or manifest_arms != FROZEN_ARM_ORDER:
        raise EvidenceError("matrix_manifest.txt arms line does not match frozen arm order")
    exact_lines = {
        "commit=": f"commit={expected_manifest['commit']}",
        "trace_sha256=": (
            f"trace_sha256={expected_manifest['trace_sha256']} "
            f"trace_manifest_sha256={expected_manifest['trace_manifest_sha256']} "
            f"trace_n={expected_n}"
        ),
        "profile_sha256=": (
            f"profile_sha256={expected_manifest['profile_sha256']}"
        ),
    }
    for prefix, expected_line in exact_lines.items():
        matches = [line for line in lines if line.startswith(prefix)]
        if matches != [expected_line]:
            raise EvidenceError(
                f"matrix_manifest.txt frozen contract mismatch for {prefix[:-1]}: "
                f"expected exactly {expected_line!r}"
            )


def _identity(summary: dict[str, Any], source: str) -> tuple[str, str, str | None, str | None]:
    policy = summary.get("policy")
    if policy == "all_local":
        return "L", "all_local", None, None
    if policy != "nimbus":
        raise EvidenceError(f"{source}: unsupported policy {policy!r}")
    config = _object(summary, "config", source)
    queue = _object(summary, "queue", source)
    trigger = config.get("nimbus_trigger")
    selector = config.get("nimbus_selector")
    if queue.get("nimbus_trigger") != trigger:
        raise EvidenceError(f"{source}: config/queue trigger mismatch")
    if queue.get("nimbus_selector") != selector:
        raise EvidenceError(f"{source}: config/queue selector mismatch")
    arm = IDENTITIES.get(("nimbus", trigger, selector))
    if arm is None:
        raise EvidenceError(
            f"{source}: unexpected Nimbus arm trigger={trigger!r}, selector={selector!r}"
        )
    return arm, "nimbus", str(trigger), str(selector)


def _validate_token_alignment(
    alignment: dict[str, Any], local_n: int, source: str
) -> None:
    expected = {
        "local_success_n": local_n,
        "measured_n": local_n,
        "missing_prompt_usage_n": 0,
        "prompt_exact_n": local_n,
        "decode_measured_n": local_n,
        "missing_completion_usage_n": 0,
        "decode_cap_hit_n": local_n,
    }
    mismatches = []
    for field, wanted in expected.items():
        actual = _integer(alignment.get(field), f"token_alignment.{field}", source)
        if actual != wanted:
            mismatches.append(f"{field}={actual} (expected {wanted})")
    if mismatches:
        raise EvidenceError(
            f"{source}: token alignment is not exact for every local success: "
            + ", ".join(mismatches)
        )


def load_arm(
    path: Path,
    expected_n: int,
    *,
    raw_path: Path,
    decisions_path: Path,
    expected_config: dict[str, Any],
) -> ArmMetrics:
    source = path.name
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{source}: cannot read valid JSON: {exc}") from exc
    if not isinstance(summary, dict):
        raise EvidenceError(f"{source}: top-level JSON must be an object")

    arm, policy, trigger, selector = _identity(summary, source)
    overall = _object(summary, "overall", source)
    local = _object(summary, "local", source)
    cloud = _object(summary, "cloud", source)
    pessimistic = _object(summary, "pessimistic_combined", source)
    alignment = _object(summary, "token_alignment", source)
    config = _object(summary, "config", source)
    queue = _object(summary, "queue", source) if policy == "nimbus" else summary.get("queue", {})
    if not isinstance(queue, dict):
        raise EvidenceError(f"{source}: queue must be an object")
    seed = _integer(config.get("seed"), "config.seed", source)
    _validate_config_contract(config, arm, source, expected_config)

    n = _integer(overall.get("n"), "overall.n", source)
    success_n = _integer(overall.get("success"), "overall.success", source)
    local_n = _integer(local.get("n"), "local.n", source)
    local_success = _integer(local.get("success"), "local.success", source)
    routed_n = _integer(cloud.get("n"), "cloud.n", source)
    cloud_success = _integer(cloud.get("success"), "cloud.success", source)
    if n != expected_n:
        raise EvidenceError(f"{source}: overall.n={n} != expected {expected_n}")
    if success_n != n:
        raise EvidenceError(f"{source}: overall success {success_n}/{n}")
    if local_n + routed_n != n:
        raise EvidenceError(f"{source}: local.n + cloud.n != overall.n")
    if local_success != local_n or cloud_success != routed_n:
        raise EvidenceError(f"{source}: side-level n/success mismatch")
    if arm == "L" and routed_n != 0:
        raise EvidenceError(f"{source}: all_local anchor routed {routed_n} requests")

    local_violations = _integer(local.get("slo_violations"), "local.slo_violations", source)
    local_slo_n = _integer(local.get("slo_measured_n"), "local.slo_measured_n", source)
    if local_slo_n != local_n:
        raise EvidenceError(f"{source}: local SLO denominator {local_slo_n} != local.n {local_n}")
    if not 0 <= local_violations <= local_n:
        raise EvidenceError(f"{source}: invalid local violation count")
    local_pct = _number(local.get("slo_violation_pct"), "local.slo_violation_pct", source)
    expected_local_pct = 100.0 * local_violations / max(local_n, 1)
    if not _close(local_pct, expected_local_pct):
        raise EvidenceError(f"{source}: inconsistent local violation percentage")

    pessimistic_n = _integer(
        pessimistic.get("slo_violations"), "pessimistic_combined.slo_violations", source
    )
    pessimistic_total = _integer(pessimistic.get("slo_n"), "pessimistic_combined.slo_n", source)
    pessimistic_pct = _number(
        pessimistic.get("slo_violation_pct"), "pessimistic_combined.slo_violation_pct", source
    )
    if pessimistic_total != n or pessimistic_n != local_violations + routed_n:
        raise EvidenceError(f"{source}: inconsistent pessimistic-combined counts")
    if not _close(pessimistic_pct, 100.0 * pessimistic_n / n):
        raise EvidenceError(f"{source}: inconsistent pessimistic-combined percentage")
    cost_usd = _number(pessimistic.get("cost_usd"), "pessimistic_combined.cost_usd", source)
    if cost_usd < 0:
        raise EvidenceError(f"{source}: cost must be non-negative")
    overall_cost = _number(overall.get("cost_usd"), "overall.cost_usd", source)
    if not _close(overall_cost, cost_usd):
        raise EvidenceError(f"{source}: overall/pessimistic cost mismatch")

    _validate_token_alignment(alignment, local_n, source)
    p50 = _optional_ms_to_s(local.get("ttft_p50_ms"), "local.ttft_p50_ms", source)
    p95 = _optional_ms_to_s(local.get("ttft_p95_ms"), "local.ttft_p95_ms", source)
    p99 = _optional_ms_to_s(local.get("ttft_p99_ms"), "local.ttft_p99_ms", source)
    if local_n and any(value is None for value in (p50, p95, p99)):
        raise EvidenceError(f"{source}: missing local TTFT percentile")

    slo_s = _number(summary.get("slo_s", config.get("slo_s")), "slo_s", source)
    config_slo = _number(config.get("slo_s"), "config.slo_s", source)
    if not _close(slo_s, config_slo) or slo_s <= 0:
        raise EvidenceError(f"{source}: invalid or inconsistent SLO")
    guard_s = _number(config.get("ttft_guard_ms"), "config.ttft_guard_ms", source) / 1000.0
    if guard_s < 0 or guard_s >= slo_s:
        raise EvidenceError(f"{source}: invalid TTFT guard")

    prediction_scope = None
    prediction_model = None
    post_kick_limit = None
    max_post_kick = None
    kv_failures = None
    if arm in ("A", "B", "C"):
        prediction_scope = queue.get("nimbus_prediction_scope")
        prediction_model = queue.get("nimbus_prediction_model")
        if prediction_scope != PREDICTION_SCOPE or prediction_model != PREDICTION_MODEL:
            raise EvidenceError(
                f"{source}: invalid waiting-only/shared-prefill prediction telemetry"
            )
        max_post_kick = _number(
            queue.get("nimbus_max_waiting_post_kick_ttft_s"),
            "queue.nimbus_max_waiting_post_kick_ttft_s", source,
        )
        post_kick_limit = slo_s - guard_s
    elif arm == "K":
        kv_failures = _integer(queue.get("kv_read_failures"), "queue.kv_read_failures", source)
        if kv_failures != 0:
            raise EvidenceError(f"{source}: K kv_read_failures={kv_failures}, expected 0")

    audit = _audit_jsonl(
        raw_path=raw_path,
        decisions_path=decisions_path,
        arm=arm,
        trigger=trigger,
        selector=selector,
        expected_n=expected_n,
        local_n=local_n,
        routed_n=routed_n,
        expected_local_violation_n=local_violations,
        expected_pessimistic_violation_n=pessimistic_n,
        expected_cost_usd=cost_usd,
        queue=queue,
        post_kick_limit_s=post_kick_limit,
        expected_model=str(expected_config["local_model"]),
        slo_s=slo_s,
        in_price_mtok=float(expected_config["in_price"]),
        out_price_mtok=float(expected_config["out_price"]),
    )

    return ArmMetrics(
        arm=arm, description=ARM_DESCRIPTIONS[arm], source=source,
        policy=policy, trigger=trigger, selector=selector, seed=seed, n=n,
        success_n=success_n, routed_n=routed_n, local_n=local_n,
        local_violation_n=local_violations, local_violation_pct=local_pct,
        pessimistic_violation_n=pessimistic_n,
        pessimistic_combined_pct=pessimistic_pct, cost_usd=cost_usd,
        local_ttft_p50_s=p50, local_ttft_p95_s=p95, local_ttft_p99_s=p99,
        slo_s=slo_s, ttft_guard_s=guard_s, post_kick_limit_s=post_kick_limit,
        max_post_kick_ttft_s=max_post_kick, token_exact=True,
        prediction_scope=prediction_scope, prediction_model=prediction_model,
        kv_read_failures=kv_failures,
        decision_n=audit.decision_n,
        applied_victim_n=audit.applied_victim_n,
    )


def analyze(
    matrix_dir: str | Path,
    *,
    expected_n: int = DEFAULT_EXPECTED_N,
    equivalence_band_n: int = 363,
    ab_min_advantage_pp: float = 5.0,
    a_cost_limit_ratio: float = 1.05,
    expected_fingerprint: str = DEFAULT_RUN_FINGERPRINT,
    expected_config: dict[str, Any] | None = None,
    expected_manifest: dict[str, str] | None = None,
) -> dict[str, Any]:
    path = Path(matrix_dir)
    config_contract = dict(
        DEFAULT_CONFIG_CONTRACT if expected_config is None else expected_config
    )
    manifest_contract = dict(
        DEFAULT_MANIFEST_CONTRACT if expected_manifest is None else expected_manifest
    )
    if expected_n <= 0:
        raise EvidenceError("expected_n must be positive")
    if equivalence_band_n < 0:
        raise EvidenceError("equivalence_band_n must be non-negative")
    if not path.is_dir():
        raise EvidenceError(f"matrix directory does not exist: {path}")
    marker_paths = sorted(path.glob("*.complete.json"), key=lambda item: item.name)
    if len(marker_paths) != 5:
        raise EvidenceError(
            f"expected exactly 5 completion markers, found {len(marker_paths)}"
        )
    markers = [_load_marker(marker_path, expected_n) for marker_path in marker_paths]
    nonzero_seeds = [marker.arm_text for marker in markers if marker.seed != 0]
    if nonzero_seeds:
        raise EvidenceError(
            "all five pre-registered arms require seed==0: " + ", ".join(nonzero_seeds)
        )
    fingerprints = {marker.run_fingerprint for marker in markers}
    if len(fingerprints) != 1:
        raise EvidenceError("completion markers do not share one run_fingerprint")
    fingerprint = next(iter(fingerprints))
    if not expected_fingerprint or expected_fingerprint != expected_fingerprint.strip():
        raise EvidenceError("expected_fingerprint must be a nonempty exact string")
    if fingerprint != expected_fingerprint:
        raise EvidenceError(
            f"shared run_fingerprint {fingerprint!r} != expected {expected_fingerprint!r}"
        )
    required_config_fields = set(DEFAULT_CONFIG_CONTRACT)
    if set(config_contract) != required_config_fields:
        raise EvidenceError(
            "expected_config must contain exactly the frozen full-cell config fields"
        )
    required_manifest_fields = set(DEFAULT_MANIFEST_CONTRACT)
    if set(manifest_contract) != required_manifest_fields:
        raise EvidenceError(
            "expected_manifest must contain exactly the frozen manifest fields"
        )
    _validate_manifest(path, fingerprint, expected_n, manifest_contract)

    summaries = sorted(path.glob("*.summary.json"), key=lambda item: item.name)
    if len(summaries) != 5:
        raise EvidenceError(f"expected exactly 5 summary JSONs, found {len(summaries)}")
    marker_summaries = {marker.summary_path.resolve() for marker in markers}
    discovered_summaries = {summary.resolve() for summary in summaries}
    if marker_summaries != discovered_summaries:
        raise EvidenceError("summary JSON set does not exactly match completion markers")

    arms: dict[str, ArmMetrics] = {}
    markers_by_summary = {marker.summary_path.resolve(): marker for marker in markers}
    for summary_path in summaries:
        marker = markers_by_summary[summary_path.resolve()]
        arm = load_arm(
            summary_path,
            expected_n,
            raw_path=marker.raw_path,
            decisions_path=marker.decisions_path,
            expected_config=config_contract,
        )
        if marker.arm != arm.arm or marker.seed != arm.seed:
            raise EvidenceError(
                f"{marker.marker_path.name}: marker arm {marker.arm_text!r} "
                f"does not match summary identity/seed ({arm.arm}, seed={arm.seed})"
            )
        if arm.arm in arms:
            raise EvidenceError(f"duplicate arm {arm.arm}: {arms[arm.arm].source}, {arm.source}")
        arms[arm.arm] = arm
    missing = sorted(EXPECTED_ARMS - arms.keys())
    if missing:
        raise EvidenceError(f"missing required arms: {', '.join(missing)}")
    slo_values = {arm.slo_s for arm in arms.values()}
    if len(slo_values) != 1:
        raise EvidenceError("arms do not share one SLO")

    a, b, c, k, local_anchor = (arms[key] for key in ("A", "B", "C", "K", "L"))
    pressure_pass = (
        local_anchor.local_violation_pct + EPSILON >= 90.0
        and local_anchor.local_ttft_p50_s is not None
        and local_anchor.local_ttft_p50_s > local_anchor.slo_s + EPSILON
    )
    selectors_safe = all(arms[key].local_violation_n == 0 for key in ("A", "B", "C"))
    post_kick_safe = all(
        arms[key].max_post_kick_ttft_s is not None
        and arms[key].post_kick_limit_s is not None
        and arms[key].max_post_kick_ttft_s
        <= arms[key].post_kick_limit_s + EPSILON
        for key in ("A", "B", "C")
    )
    ac_delta_n = a.pessimistic_violation_n - c.pessimistic_violation_n
    ac_delta_pp = 100.0 * ac_delta_n / expected_n
    equivalence_band_pp = 100.0 * equivalence_band_n / expected_n
    if ac_delta_n > equivalence_band_n:
        ac_classification = "C_beats_A_beyond_band_reject"
        ac_pass = False
    elif ac_delta_n < -equivalence_band_n:
        ac_classification = "A_one_pass_advantage_beyond_band"
        ac_pass = True
    else:
        ac_classification = "route_equivalent_within_band"
        ac_pass = True
    cost_pass = a.cost_usd <= a_cost_limit_ratio * c.cost_usd + EPSILON
    ab_advantage = b.pessimistic_combined_pct - a.pessimistic_combined_pct
    ab_pass = ab_advantage + EPSILON >= ab_min_advantage_pp
    pareto_no_worse_local = a.local_violation_pct <= k.local_violation_pct + EPSILON
    pareto_no_worse_combined = a.pessimistic_combined_pct <= k.pessimistic_combined_pct + EPSILON
    pareto_strict = (
        a.local_violation_pct < k.local_violation_pct - EPSILON
        or a.pessimistic_combined_pct < k.pessimistic_combined_pct - EPSILON
    )
    pareto_pass = pareto_no_worse_local and pareto_no_worse_combined and pareto_strict

    gates = {
        "pressure_anchor": {
            "pass": pressure_pass,
            "requirement": "L local violation >=90% and local TTFT p50 > SLO",
            "local_violation_pct": local_anchor.local_violation_pct,
            "local_ttft_p50_s": local_anchor.local_ttft_p50_s,
            "slo_s": local_anchor.slo_s,
        },
        "abc_local_safety": {
            "pass": selectors_safe,
            "requirement": "A/B/C each have zero measured local violations",
            "violations": {key: arms[key].local_violation_n for key in ("A", "B", "C")},
        },
        "abc_post_kick_bound": {
            "pass": post_kick_safe,
            "requirement": "A/B/C max recorded post-kick prediction <= SLO-guard (3.288 s)",
            "max_post_kick_ttft_s": {
                key: arms[key].max_post_kick_ttft_s for key in ("A", "B", "C")
            },
            "limits_s": {
                key: arms[key].post_kick_limit_s for key in ("A", "B", "C")
            },
        },
        "a_vs_c_equivalence": {
            "pass": ac_pass,
            "requirement": (
                "C must not beat A beyond the frozen 16/512 screen band, "
                "encoded as an inclusive whole-row bound"
            ),
            "a_minus_c_n": ac_delta_n,
            "a_minus_c_pp": ac_delta_pp,
            "band_n": equivalence_band_n,
            "band_pp": equivalence_band_pp,
            "classification": ac_classification,
        },
        "a_cost_vs_c": {
            "pass": cost_pass,
            "requirement": f"A cost <= {a_cost_limit_ratio:.3f} * C cost",
            "a_cost_usd": a.cost_usd,
            "c_cost_usd": c.cost_usd,
            "ratio": (a.cost_usd / c.cost_usd if c.cost_usd else (0.0 if not a.cost_usd else None)),
        },
        "a_vs_b_signal": {
            "pass": ab_pass,
            "requirement": f"A beats B by >= {ab_min_advantage_pp:.3f} pp pessimistic combined",
            "b_minus_a_pp": ab_advantage,
        },
        "a_pareto_vs_k": {
            "pass": pareto_pass,
            "requirement": "A no worse than K on local and combined violation %, strict on >=1",
            "no_worse_local": pareto_no_worse_local,
            "no_worse_combined": pareto_no_worse_combined,
            "strict_on_at_least_one": pareto_strict,
        },
    }
    heldout_pass = all(gate["pass"] for gate in gates.values())
    return {
        "schema_version": 1,
        "analysis": "ttft_full_cell_preregistered_gate",
        "statistical_unit": "one complete arm/run",
        "per_request_bootstrap": False,
        "expected_request_n": expected_n,
        "run_fingerprint": fingerprint,
        "completion_markers_valid": True,
        "integrity_valid": True,
        "arms": {key: asdict(arms[key]) for key in ("A", "B", "C", "K", "L")},
        "gates": gates,
        "failed_gates": [name for name, gate in gates.items() if not gate["pass"]],
        "default_flip_authorized": False,
        "default_flip_blockers": [
            "cache-aware trace evidence",
            "estimated (not oracle) decode evidence",
            "real-cloud latency evidence",
            "additional sweeps/oracle evidence",
        ],
        "heldout_gate_pass": heldout_pass,
    }


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# TTFT full-cell held-out gate",
        "",
        f"Integrity: **PASS** — five exact completion markers/artifact hashes, "
        f"one manifest-bound fingerprint, N={result['expected_request_n']} successful "
        "requests per arm, exact raw token usage, and decision/application JSONL semantics.",
        "",
        "Statistical unit: **one complete arm/run**. No request-level bootstrap or resampling is used.",
        "",
        "| arm | treatment | routed / local | local viol % | pessimistic combined % | cost $ | local TTFT p50 / p95 / p99 s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("A", "B", "C", "K", "L"):
        arm = result["arms"][key]
        ttft = " / ".join(_fmt(arm[name]) for name in (
            "local_ttft_p50_s", "local_ttft_p95_s", "local_ttft_p99_s"
        ))
        lines.append(
            f"| {key} | {arm['description']} | {arm['routed_n']} / {arm['local_n']} | "
            f"{_fmt(arm['local_violation_pct'])} | {_fmt(arm['pessimistic_combined_pct'])} | "
            f"{_fmt(arm['cost_usd'], 6)} | {ttft} |"
        )
    lines.extend(["", "## Frozen gates", "", "| gate | result | requirement / observation |", "|---|---:|---|"])
    for name, gate in result["gates"].items():
        observation = ""
        if name == "a_vs_c_equivalence":
            observation = (
                f"; A-C={gate['a_minus_c_n']:+d} rows "
                f"({gate['a_minus_c_pp']:+.4f} pp), band=±{gate['band_n']} rows "
                f"(±{gate['band_pp']:.4f} pp), {gate['classification']}"
            )
        elif name == "a_vs_b_signal":
            observation = f"; B-A={gate['b_minus_a_pp']:+.3f} pp"
        lines.append(f"| `{name}` | **{'PASS' if gate['pass'] else 'FAIL'}** | {gate['requirement']}{observation} |")
    lines.extend([
        "",
        "`default_flip_authorized = false` — this gate does not supply the missing cache-aware, decode-estimator, real-cloud, and sweep/oracle evidence.",
        "",
        f"`heldout_gate_pass = {str(result['heldout_gate_pass']).lower()}`",
    ])
    return "\n".join(lines) + "\n"


def render_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    parser.add_argument("--json-out", type=Path, help="also write JSON to this path")
    parser.add_argument("--markdown-out", type=Path, help="also write Markdown to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze(
            args.matrix_dir,
        )
    except EvidenceError as exc:
        print(f"full-cell evidence invalid: {exc}", file=sys.stderr)
        return 2
    markdown = render_markdown(result)
    encoded = render_json(result)
    if args.json_out:
        args.json_out.write_text(encoded, encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.write_text(markdown, encoding="utf-8")
    sys.stdout.write(encoded if args.json else markdown)
    return 0 if result["heldout_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
