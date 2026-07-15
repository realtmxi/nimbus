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


class MatrixEvidenceError(ValueError):
    """The arm or its persisted evidence is not audit-valid."""


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


def _validate_summary(
    summary: dict[str, Any], arm: MatrixArm, expected: MatrixExpectations,
) -> None:
    expected_policy = "all_local" if arm.kind == "anchor" else "nimbus"
    _require_equal(summary.get("policy"), expected_policy, "summary policy")

    overall = summary.get("overall", {})
    _require_equal(overall.get("success"), overall.get("n"), "overall success")
    _require_equal(overall.get("n"), expected.trace_n, "overall row count")

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
        "cloud": "null",
        "cloud_max_concurrency": 32,
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

    marker = {
        "schema_version": 1,
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
