"""Arm parsing and completion-marker validation for the TTFT matrix runner.

This module intentionally uses only the Python standard library.  The shell
runner calls it both before launching an arm and when creating/resuming
evidence, so the audited all-local anchor cannot accidentally be passed to
``router.run`` as a Nimbus trigger or selector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from router.nimbus import NIMBUS_SELECTORS, NIMBUS_TRIGGERS


ANCHOR_ARM = "anchor:all_local:0"
SYNTHETIC_TRACE_TOOL_PATH = "tools/materialize_token_aligned_trace.py"
TRACE_SCENARIO_DEPENDENCY_PATH = "router/common.py"
SHAREGPT_CURRENT_TURN_PAYLOAD_MODE = "sharegpt_current_turn_retokenized"
SHAREGPT_CURRENT_TURN_TOOL_PATH = (
    "tools/materialize_sharegpt_current_turn_trace.py"
)


class MatrixEvidenceError(ValueError):
    """The arm or its persisted evidence is not audit-valid."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trace_materializer_manifest(
    manifest: dict[str, Any],
    checkout_root: Path | str,
) -> None:
    """Bind a trace manifest to its materializer(s) in this checkout.

    Historical token-aligned synthetic manifests predate ``tool_path`` and
    continue to use their original single-tool hash check.  A verbatim
    current-turn trace is a second, explicit payload mode: its manifest must
    bind both the new frontend and the old materializer module from which it
    imports the shared chat-count/fingerprint implementation.
    """
    if not isinstance(manifest, dict):
        raise MatrixEvidenceError("trace manifest is not a JSON object")
    # The shell preflight passes ``os.curdir`` (a string), while Python callers
    # commonly pass a Path.  Normalize both before resolving repo-relative
    # materializer paths.
    checkout_root = Path(checkout_root).resolve()
    synthetic_sha = _file_sha256(checkout_root / SYNTHETIC_TRACE_TOOL_PATH)

    if manifest.get("payload_mode") != SHAREGPT_CURRENT_TURN_PAYLOAD_MODE:
        if manifest.get("tool_sha256") != synthetic_sha:
            raise MatrixEvidenceError(
                "trace was not materialized by the current checkout tool"
            )
        return

    if manifest.get("tool_path") != SHAREGPT_CURRENT_TURN_TOOL_PATH:
        raise MatrixEvidenceError(
            "current-turn trace manifest tool_path is not the audited repo path"
        )
    current_turn_sha = _file_sha256(
        checkout_root / SHAREGPT_CURRENT_TURN_TOOL_PATH
    )
    if manifest.get("tool_sha256") != current_turn_sha:
        raise MatrixEvidenceError(
            "current-turn trace was not materialized by the current checkout tool"
        )
    dependencies = manifest.get("dependency_sha256")
    if not isinstance(dependencies, dict):
        raise MatrixEvidenceError(
            "current-turn trace manifest lacks dependency_sha256"
        )
    if dependencies.get(SYNTHETIC_TRACE_TOOL_PATH) != synthetic_sha:
        raise MatrixEvidenceError(
            "current-turn trace materializer dependency differs from checkout: "
            f"{SYNTHETIC_TRACE_TOOL_PATH}"
        )
    common_sha = _file_sha256(checkout_root / TRACE_SCENARIO_DEPENDENCY_PATH)
    if dependencies.get(TRACE_SCENARIO_DEPENDENCY_PATH) != common_sha:
        raise MatrixEvidenceError(
            "current-turn trace materializer dependency differs from checkout: "
            f"{TRACE_SCENARIO_DEPENDENCY_PATH}"
        )


@dataclass(frozen=True)
class MatrixArm:
    text: str
    kind: str
    policy: str
    trigger: str | None
    selector: str | None
    seed: int

    @property
    def policy_cli(self) -> tuple[str, ...]:
        args = ["--policy", self.policy]
        if self.kind == "nimbus":
            assert self.trigger is not None and self.selector is not None
            args.extend(("--nimbus-trigger", self.trigger,
                         "--nimbus-selector", self.selector))
        return tuple(args)


@dataclass(frozen=True)
class MatrixExpectations:
    trace_n: int
    prefill_tput: float
    tpot_ms: float
    first_token_overhead_ms: float
    ttft_guard_ms: float
    in_price: float
    out_price: float
    slo_s: float
    nimbus_tick_ms: float
    max_inflight: int
    temperature: float
    ignore_eos: bool
    kv_capacity_tokens: float
    model: str
    chat_url: str
    scenario: str
    cloud: str = "null"
    cloud_url: str | None = None
    cloud_model: str | None = None
    cloud_api_key_env: str | None = None
    cloud_max_concurrency: int = 32
    cloud_provider_order: tuple[str, ...] = ()
    cloud_no_fallbacks: bool = False
    cloud_stop_after_first_token: bool = False
    local_ignore_eos: bool | None = None
    cloud_ignore_eos: bool | None = None

    @property
    def effective_local_ignore_eos(self) -> bool:
        return (
            self.ignore_eos
            if self.local_ignore_eos is None
            else self.local_ignore_eos
        )

    @property
    def effective_cloud_ignore_eos(self) -> bool:
        return (
            self.ignore_eos
            if self.cloud_ignore_eos is None
            else self.cloud_ignore_eos
        )


def parse_arm(text: str) -> MatrixArm:
    parts = text.split(":")
    if len(parts) != 3 or any(not part for part in parts):
        raise MatrixEvidenceError(
            f"invalid arm {text!r}; expected trigger:selector:seed"
        )
    trigger, selector, seed_text = parts
    reserved = trigger in {"anchor", "all_local"} or selector in {
        "anchor", "all_local"
    }
    if reserved:
        if text != ANCHOR_ARM:
            raise MatrixEvidenceError(
                f"unsupported anchor arm {text!r}; only {ANCHOR_ARM!r} is valid"
            )
        return MatrixArm(text, "anchor", "all_local", None, None, 0)

    try:
        if seed_text != seed_text.strip():
            raise ValueError
        seed = int(seed_text)
    except ValueError as exc:
        raise MatrixEvidenceError(f"arm seed is not an integer: {seed_text!r}") from exc
    if trigger not in NIMBUS_TRIGGERS:
        raise MatrixEvidenceError(
            f"unsupported Nimbus trigger {trigger!r}; choose from {NIMBUS_TRIGGERS}"
        )
    if selector not in NIMBUS_SELECTORS:
        raise MatrixEvidenceError(
            f"unsupported Nimbus selector {selector!r}; choose from {NIMBUS_SELECTORS}"
        )
    return MatrixArm(text, "nimbus", "nimbus", trigger, selector, seed)


def _artifact(path: Path) -> tuple[bytes, dict[str, Any]]:
    content = path.read_bytes()
    return content, {
        "sha256": hashlib.sha256(content).hexdigest(),
        "nonempty_line_n": sum(bool(line.strip()) for line in content.splitlines()),
    }


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise MatrixEvidenceError(f"{label}={actual!r} != {expected!r}")


def _require_float(config: dict[str, Any], key: str, expected: float) -> None:
    try:
        actual = float(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise MatrixEvidenceError(f"summary config lacks numeric {key}") from exc
    if not math.isfinite(actual) or actual != expected:
        raise MatrixEvidenceError(f"summary config {key}={actual!r} != {expected!r}")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _parse_jsonl(content: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_n, raw_line in enumerate(content.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MatrixEvidenceError(
                f"{label} line {line_n} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise MatrixEvidenceError(f"{label} line {line_n} is not an object")
        rows.append(row)
    return rows


def _validate_real_cloud_rows(
    rows: list[dict[str, Any]], summary: dict[str, Any],
    decision_rows: list[dict[str, Any]], expected: MatrixExpectations,
) -> None:
    """Validate the row-level contract for an observed TTFT-cancel run.

    Endpoint failures remain valid experimental observations: ``summarize``
    counts them as TTFT-SLO violations.  Successful cloud rows, however, must
    prove that a first token was observed and the stream was deliberately
    aborted rather than silently treated as a completed response.
    """
    request_ids = [row.get("request_id") for row in rows]
    try:
        unique_ids = set(request_ids)
    except TypeError as exc:
        raise MatrixEvidenceError("raw request_id is not hashable") from exc
    if any(request_id is None for request_id in request_ids):
        raise MatrixEvidenceError("raw row lacks request_id")
    if len(unique_ids) != len(request_ids):
        raise MatrixEvidenceError("raw request_id values are not unique")

    endpoints = {row.get("endpoint") for row in rows}
    if not endpoints <= {"local", "cloud"}:
        raise MatrixEvidenceError(f"raw contains unsupported endpoints: {endpoints}")
    local_rows = [row for row in rows if row.get("endpoint") == "local"]
    cloud_rows = [row for row in rows if row.get("endpoint") == "cloud"]
    if not cloud_rows:
        raise MatrixEvidenceError("real-cloud Nimbus arm routed no cloud rows")
    for index, row in enumerate(rows, 1):
        if not isinstance(row.get("success"), bool):
            raise MatrixEvidenceError(f"raw row {index} success is not boolean")

    for index, row in enumerate(local_rows, 1):
        if not row.get("success"):
            continue
        for scheduler_key, measured_key in (
            ("scheduler_prompt_tokens", "prompt_tokens"),
            ("scheduler_decode_tokens", "completion_tokens"),
        ):
            scheduler_value = row.get(scheduler_key)
            measured_value = row.get(measured_key)
            if not isinstance(scheduler_value, int) or isinstance(
                scheduler_value, bool
            ):
                raise MatrixEvidenceError(
                    f"local success row {index} lacks integer {scheduler_key}"
                )
            if not isinstance(measured_value, int) or isinstance(
                measured_value, bool
            ):
                raise MatrixEvidenceError(
                    f"local success row {index} lacks integer {measured_key}"
                )
            if scheduler_value != measured_value:
                raise MatrixEvidenceError(
                    f"local success row {index} {measured_key} "
                    f"{measured_value} != {scheduler_key} {scheduler_value}"
                )

    for index, row in enumerate(cloud_rows, 1):
        if row.get("routed_only"):
            raise MatrixEvidenceError(
                f"cloud row {index} is routed_only in a real-cloud run"
            )
        if not row.get("success"):
            continue
        ttft_ms = row.get("ttft_ms")
        if not _finite_number(ttft_ms) or float(ttft_ms) < 0:
            raise MatrixEvidenceError(
                f"cloud success row {index} lacks a finite nonnegative TTFT"
            )
        latency_parts: list[float] = []
        for key in (
            "pre_route_queue_ms", "cloud_gate_wait_ms", "service_ttft_ms",
        ):
            value = row.get(key)
            if not _finite_number(value) or float(value) < 0:
                raise MatrixEvidenceError(
                    f"cloud success row {index} lacks finite nonnegative {key}"
                )
            latency_parts.append(float(value))
        component_sum = sum(latency_parts)
        if not math.isclose(
            float(ttft_ms), component_sum, rel_tol=1e-9, abs_tol=1e-6,
        ):
            raise MatrixEvidenceError(
                f"cloud success row {index} ttft_ms {ttft_ms} != latency "
                f"component sum {component_sum}"
            )
        if row.get("response_completed") is not False:
            raise MatrixEvidenceError(
                f"cloud success row {index} response_completed="
                f"{row.get('response_completed')!r} != False"
            )
        if row.get("stream_abort_requested") is not True:
            raise MatrixEvidenceError(
                f"cloud success row {index} stream_abort_requested="
                f"{row.get('stream_abort_requested')!r} != True"
            )
        _require_equal(
            row.get("probe_mode"), "ttft_cancel",
            f"cloud success row {index} probe_mode",
        )

    applied_ids: list[Any] = []
    for index, decision in enumerate(decision_rows, 1):
        # Stale/discarded decision records intentionally have no applied list;
        # only records that reached the apply phase carry this field.
        victims = decision.get("applied_victim_ids", [])
        if not isinstance(victims, list):
            raise MatrixEvidenceError(
                f"decision row {index} applied_victim_ids is not a list"
            )
        applied_ids.extend(victims)
    try:
        applied_set = set(applied_ids)
        cloud_id_set = {row["request_id"] for row in cloud_rows}
    except TypeError as exc:
        raise MatrixEvidenceError("applied/cloud request_id is not hashable") from exc
    if len(applied_ids) != len(applied_set):
        raise MatrixEvidenceError("applied_victim_ids contain duplicate request IDs")
    if len(applied_ids) != len(cloud_rows) or applied_set != cloud_id_set:
        missing = sorted(cloud_id_set - applied_set, key=repr)
        unexpected = sorted(applied_set - cloud_id_set, key=repr)
        raise MatrixEvidenceError(
            "applied_victim_ids do not match raw cloud request IDs: "
            f"applied_n={len(applied_ids)} cloud_n={len(cloud_rows)} "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    overall = summary.get("overall", {})
    local = summary.get("local", {})
    cloud = summary.get("cloud", {})
    _require_equal(overall.get("slo_measured_n"), expected.trace_n,
                   "overall measured SLO row count")
    _require_equal(cloud.get("routed_only"), 0, "cloud routed_only count")
    _require_equal(local.get("n"), len(local_rows), "local raw/summary row count")
    _require_equal(cloud.get("n"), len(cloud_rows), "cloud raw/summary row count")
    _require_equal(
        overall.get("success"), sum(bool(row.get("success")) for row in rows),
        "overall raw/summary success count",
    )
    _require_equal(
        local.get("success"),
        sum(bool(row.get("success")) for row in local_rows),
        "local raw/summary success count",
    )
    _require_equal(
        cloud.get("success"),
        sum(bool(row.get("success")) for row in cloud_rows),
        "cloud raw/summary success count",
    )


def _validate_summary(
    summary: dict[str, Any], arm: MatrixArm, expected: MatrixExpectations,
) -> None:
    expected_policy = "all_local" if arm.kind == "anchor" else "nimbus"
    _require_equal(summary.get("policy"), expected_policy, "summary policy")

    overall = summary.get("overall", {})
    _require_equal(overall.get("n"), expected.trace_n, "overall row count")
    if expected.cloud != "real":
        _require_equal(overall.get("success"), overall.get("n"),
                       "overall success")

    alignment = summary.get("token_alignment", {})
    measured_n = alignment.get("measured_n")
    if not isinstance(measured_n, int) or measured_n <= 0:
        raise MatrixEvidenceError("no measured local rows for token-alignment audit")
    _require_equal(alignment.get("missing_prompt_usage_n"), 0,
                   "missing prompt usage")
    _require_equal(alignment.get("prompt_exact_n"), measured_n,
                   "prompt exact count")
    _require_equal(alignment.get("decode_measured_n"), measured_n,
                   "decode measured count")
    _require_equal(alignment.get("missing_completion_usage_n"), 0,
                   "missing completion usage")
    _require_equal(alignment.get("decode_cap_hit_n"), measured_n,
                   "decode cap-hit count")

    if arm.kind == "anchor":
        # Every trace row is local in this anchor, so partial endpoint usage is
        # not acceptable evidence even when all measured rows happen to align.
        _require_equal(measured_n, expected.trace_n, "anchor measured row count")
        _require_equal(alignment.get("local_success_n"), expected.trace_n,
                       "anchor local-success count")
        local = summary.get("local", {})
        cloud = summary.get("cloud", {})
        _require_equal(local.get("n"), expected.trace_n, "anchor local row count")
        _require_equal(local.get("success"), expected.trace_n,
                       "anchor local success")
        _require_equal(cloud.get("n"), 0, "anchor cloud row count")

    config = summary.get("config", {})
    expected_float = {
        "prefill_tput": expected.prefill_tput,
        "tpot_ms": expected.tpot_ms,
        "first_token_overhead_ms": expected.first_token_overhead_ms,
        "ttft_guard_ms": expected.ttft_guard_ms,
        "in_price": expected.in_price,
        "out_price": expected.out_price,
        "slo_s": expected.slo_s,
        "nimbus_tick_ms": expected.nimbus_tick_ms,
        "temperature": expected.temperature,
        "kv_capacity_tokens": expected.kv_capacity_tokens,
        "kv_hysteresis_fraction": 0.05,
        "time_scale": 1.0,
    }
    for key, value in expected_float.items():
        _require_float(config, key, value)
    expected_exact = {
        "scenario": expected.scenario,
        "seed": arm.seed,
        "max_inflight": expected.max_inflight,
        "ignore_eos": expected.ignore_eos,
        "local_model": expected.model,
        "local_url": expected.chat_url,
        "local_ignore_eos": expected.effective_local_ignore_eos,
        "cloud_ignore_eos": expected.effective_cloud_ignore_eos,
        "cloud": expected.cloud,
        "cloud_url": expected.cloud_url,
        "cloud_model": expected.cloud_model or expected.model,
        "cloud_api_key_env": expected.cloud_api_key_env,
        "cloud_max_concurrency": expected.cloud_max_concurrency,
        "cloud_provider_order": (
            list(expected.cloud_provider_order)
            if expected.cloud_provider_order else None
        ),
        "cloud_no_fallbacks": expected.cloud_no_fallbacks,
        "cloud_stop_after_first_token": expected.cloud_stop_after_first_token,
        "max_tokens_override": None,
    }
    for key, value in expected_exact.items():
        _require_equal(config.get(key), value, f"summary config {key}")

    if arm.kind == "nimbus":
        _require_equal(config.get("nimbus_trigger"), arm.trigger,
                       "summary config nimbus_trigger")
        _require_equal(config.get("nimbus_selector"), arm.selector,
                       "summary config nimbus_selector")
        queue = summary.get("queue", {})
        _require_equal(queue.get("nimbus_trigger"), arm.trigger,
                       "queue nimbus_trigger")
        _require_equal(queue.get("nimbus_selector"), arm.selector,
                       "queue nimbus_selector")
        if arm.trigger == "kv_gap":
            _require_equal(queue.get("kv_read_failures"), 0,
                           "queue kv_read_failures")


def validate_or_write_marker(
    *, raw_path: Path, summary_path: Path, decisions_path: Path,
    marker_path: Path, fingerprint: str, arm_text: str,
    expected: MatrixExpectations, write_marker: bool = False,
) -> dict[str, Any]:
    """Validate semantic evidence and either verify or atomically write marker."""
    arm = parse_arm(arm_text)
    contents_and_artifacts = {
        name: _artifact(path)
        for name, path in (
            ("raw", raw_path),
            ("summary", summary_path),
            ("decisions", decisions_path),
        )
    }
    artifacts = {
        name: value[1] for name, value in contents_and_artifacts.items()
    }
    if artifacts["raw"]["nonempty_line_n"] != expected.trace_n:
        raise MatrixEvidenceError("raw artifact line count does not equal trace n")
    raw_rows = _parse_jsonl(contents_and_artifacts["raw"][0], "raw")
    decision_rows = _parse_jsonl(
        contents_and_artifacts["decisions"][0], "decisions"
    )
    decision_n = artifacts["decisions"]["nonempty_line_n"]
    if arm.kind == "anchor":
        if decision_n != 0:
            raise MatrixEvidenceError("all-local anchor decision log is not empty")
    elif decision_n <= 0:
        raise MatrixEvidenceError("Nimbus decision log is empty")

    try:
        summary = json.loads(contents_and_artifacts["summary"][0])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MatrixEvidenceError(f"summary is not valid JSON: {exc}") from exc
    if not isinstance(summary, dict):
        raise MatrixEvidenceError("summary JSON is not an object")
    _validate_summary(summary, arm, expected)
    if expected.cloud == "real":
        _validate_real_cloud_rows(raw_rows, summary, decision_rows, expected)

    marker = {
        "schema_version": 2,
        "run_fingerprint": fingerprint,
        "arm": arm_text,
        "artifacts": artifacts,
    }
    if write_marker:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=marker_path.parent,
            prefix=".complete.", suffix=".tmp", delete=False,
        ) as output:
            tmp = Path(output.name)
            json.dump(marker, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp, marker_path)
        return marker

    try:
        persisted = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MatrixEvidenceError(f"completion marker is unreadable: {exc}") from exc
    if persisted != marker:
        raise MatrixEvidenceError(
            "completion marker does not match this run or its current artifacts"
        )
    return persisted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    describe = sub.add_parser("describe-arm")
    describe.add_argument("arm")
    policy_cli = sub.add_parser("policy-cli")
    policy_cli.add_argument("arm")

    validate = sub.add_parser("validate")
    validate.add_argument("--raw", type=Path, required=True)
    validate.add_argument("--summary", type=Path, required=True)
    validate.add_argument("--decisions", type=Path, required=True)
    validate.add_argument("--marker", type=Path, required=True)
    validate.add_argument("--fingerprint", required=True)
    validate.add_argument("--arm", required=True)
    validate.add_argument("--trace-n", type=int, required=True)
    validate.add_argument("--prefill-tput", type=float, required=True)
    validate.add_argument("--tpot-ms", type=float, required=True)
    validate.add_argument("--first-token-overhead-ms", type=float, required=True)
    validate.add_argument("--ttft-guard-ms", type=float, required=True)
    validate.add_argument("--in-price", type=float, required=True)
    validate.add_argument("--out-price", type=float, required=True)
    validate.add_argument("--slo-s", type=float, required=True)
    validate.add_argument("--nimbus-tick-ms", type=float, required=True)
    validate.add_argument("--max-inflight", type=int, required=True)
    validate.add_argument("--temperature", type=float, required=True)
    validate.add_argument("--ignore-eos", choices=("0", "1"), required=True)
    validate.add_argument("--kv-capacity-tokens", type=float, required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--chat-url", required=True)
    validate.add_argument("--scenario", required=True)
    validate.add_argument("--cloud", choices=("null", "real"), default="null")
    validate.add_argument("--cloud-url")
    validate.add_argument("--cloud-model")
    validate.add_argument("--cloud-api-key-env")
    validate.add_argument("--cloud-max-concurrency", type=int, default=32)
    validate.add_argument("--cloud-provider", action="append", default=[])
    validate.add_argument("--cloud-no-fallbacks", choices=("0", "1"), default="0")
    validate.add_argument(
        "--cloud-stop-after-first-token", choices=("0", "1"), default="0",
    )
    validate.add_argument("--local-ignore-eos", choices=("0", "1"))
    validate.add_argument("--cloud-ignore-eos", choices=("0", "1"))
    validate.add_argument("--write-marker", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "describe-arm":
            arm = parse_arm(args.arm)
            stem_trigger, stem_selector, stem_seed = arm.text.split(":", 2)
            print(arm.kind, arm.policy, stem_trigger, stem_selector, stem_seed)
            return 0
        if args.command == "policy-cli":
            arm = parse_arm(args.arm)
            print("\n".join(arm.policy_cli))
            return 0
        expected = MatrixExpectations(
            trace_n=args.trace_n,
            prefill_tput=args.prefill_tput,
            tpot_ms=args.tpot_ms,
            first_token_overhead_ms=args.first_token_overhead_ms,
            ttft_guard_ms=args.ttft_guard_ms,
            in_price=args.in_price,
            out_price=args.out_price,
            slo_s=args.slo_s,
            nimbus_tick_ms=args.nimbus_tick_ms,
            max_inflight=args.max_inflight,
            temperature=args.temperature,
            ignore_eos=args.ignore_eos == "1",
            kv_capacity_tokens=args.kv_capacity_tokens,
            model=args.model,
            chat_url=args.chat_url,
            scenario=args.scenario,
            cloud=args.cloud,
            cloud_url=args.cloud_url,
            cloud_model=args.cloud_model,
            cloud_api_key_env=args.cloud_api_key_env,
            cloud_max_concurrency=args.cloud_max_concurrency,
            cloud_provider_order=tuple(args.cloud_provider),
            cloud_no_fallbacks=args.cloud_no_fallbacks == "1",
            cloud_stop_after_first_token=(
                args.cloud_stop_after_first_token == "1"
            ),
            local_ignore_eos=(
                None if args.local_ignore_eos is None
                else args.local_ignore_eos == "1"
            ),
            cloud_ignore_eos=(
                None if args.cloud_ignore_eos is None
                else args.cloud_ignore_eos == "1"
            ),
        )
        validate_or_write_marker(
            raw_path=args.raw,
            summary_path=args.summary,
            decisions_path=args.decisions,
            marker_path=args.marker,
            fingerprint=args.fingerprint,
            arm_text=args.arm,
            expected=expected,
            write_marker=args.write_marker,
        )
        return 0
    except (MatrixEvidenceError, OSError) as exc:
        print(f"matrix evidence invalid: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
