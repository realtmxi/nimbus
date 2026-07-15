#!/usr/bin/env python3
"""Analyze block-level repeatability of the TTFT selector campaign.

The statistical unit is one complete block (one run of every deterministic
selector), never an individual request.  Inputs may be block directories,
shell-expanded directory globs, or a campaign directory whose immediate child
directories contain ``*.summary.json`` files.

Selector aliases used in the report:

* A = ``cost_cachedisp_old``
* B = ``newest``
* C = ``cost_disp_current``
* R = ``waiting_random`` (optional, at most one seed per block)

By default, any failed request, inexact token accounting, or measured local
SLO violation makes the command fail with exit status 2.  The last condition
can be relaxed for diagnostic output with ``--allow-local-violations``; such
output is explicitly marked as invalid evidence.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SELECTOR_TO_ARM = {
    "cost_cachedisp_old": "A",
    "newest": "B",
    "cost_disp_current": "C",
    "waiting_random": "R",
}
ARM_ORDER = ("A", "B", "C", "R")
REQUIRED_ARMS = frozenset(("A", "B", "C"))

# Every paired metric is minimized.  This is intentionally explicit so a
# negative A-X delta has one stable interpretation throughout the report.
COMPARED_METRICS = (
    "routed_n",
    "pessimistic_violation_pct",
    "cost_usd",
    "local_ttft_p50_s",
    "local_ttft_p95_s",
    "local_ttft_p99_s",
)


class EvidenceError(ValueError):
    """Input discovery or evidence validation failed."""


@dataclass(frozen=True)
class ArmMetrics:
    arm: str
    selector: str
    trigger: str
    seed: int
    n: int
    success_n: int
    routed_n: int
    local_n: int
    local_violation_n: int
    pessimistic_violation_n: int
    pessimistic_violation_pct: float
    cost_usd: float
    local_ttft_p50_s: float | None
    local_ttft_p95_s: float | None
    local_ttft_p99_s: float | None
    token_measured_n: int
    prompt_exact_n: int
    decode_exact_n: int
    n_equals_success: bool
    token_exact: bool
    zero_local_violations: bool
    source: str


@dataclass(frozen=True)
class BlockMetrics:
    block: str
    arms: dict[str, ArmMetrics]


def _int_field(value: Any, field: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{source}: {field} must be an integer")
    return value


def _float_field(value: Any, field: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{source}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{source}: {field} must be finite")
    return result


def _optional_ms_to_s(value: Any, field: str, source: str) -> float | None:
    if value is None:
        return None
    return _float_field(value, field, source) / 1000.0


def _nested(mapping: dict[str, Any], key: str, source: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise EvidenceError(f"{source}: missing object {key!r}")
    return value


def _identity(summary: dict[str, Any], source: str) -> tuple[str, str, int]:
    config = _nested(summary, "config", source)
    queue = summary.get("queue")
    queue = queue if isinstance(queue, dict) else {}
    selector = config.get("nimbus_selector", queue.get("nimbus_selector"))
    trigger = config.get("nimbus_trigger", queue.get("nimbus_trigger"))
    if selector not in SELECTOR_TO_ARM:
        raise EvidenceError(f"{source}: unsupported selector {selector!r}")
    if trigger != "ttft_pred":
        raise EvidenceError(f"{source}: expected trigger 'ttft_pred', got {trigger!r}")
    if (queue.get("nimbus_selector") is not None
            and queue.get("nimbus_selector") != selector):
        raise EvidenceError(f"{source}: config/queue selector mismatch")
    if (queue.get("nimbus_trigger") is not None
            and queue.get("nimbus_trigger") != trigger):
        raise EvidenceError(f"{source}: config/queue trigger mismatch")
    seed = _int_field(config.get("seed"), "config.seed", source)
    return str(selector), str(trigger), seed


def load_arm(path: Path, block: str) -> ArmMetrics:
    """Load and structurally validate one router summary."""
    source = f"{block}/{path.name}"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{source}: cannot read valid JSON: {exc}") from exc
    if not isinstance(summary, dict):
        raise EvidenceError(f"{source}: top-level JSON must be an object")
    if summary.get("policy") != "nimbus":
        raise EvidenceError(f"{source}: policy must be 'nimbus'")

    selector, trigger, seed = _identity(summary, source)
    overall = _nested(summary, "overall", source)
    local = _nested(summary, "local", source)
    cloud = _nested(summary, "cloud", source)
    pessimistic = _nested(summary, "pessimistic_combined", source)
    alignment = _nested(summary, "token_alignment", source)

    n = _int_field(overall.get("n"), "overall.n", source)
    success_n = _int_field(overall.get("success"), "overall.success", source)
    local_n = _int_field(local.get("n"), "local.n", source)
    local_success_n = _int_field(local.get("success"), "local.success", source)
    routed_n = _int_field(cloud.get("n"), "cloud.n", source)
    cloud_success_n = _int_field(cloud.get("success"), "cloud.success", source)
    local_violation_n = _int_field(
        local.get("slo_violations"), "local.slo_violations", source
    )
    pessimistic_n = _int_field(
        pessimistic.get("slo_violations"),
        "pessimistic_combined.slo_violations",
        source,
    )
    pessimistic_total = _int_field(
        pessimistic.get("slo_n"), "pessimistic_combined.slo_n", source
    )
    pessimistic_pct = _float_field(
        pessimistic.get("slo_violation_pct"),
        "pessimistic_combined.slo_violation_pct",
        source,
    )
    cost_value = pessimistic.get("cost_usd", overall.get("cost_usd"))
    cost_usd = _float_field(cost_value, "pessimistic_combined.cost_usd", source)

    measured_n = _int_field(
        alignment.get("measured_n"), "token_alignment.measured_n", source
    )
    local_alignment_n = _int_field(
        alignment.get("local_success_n"),
        "token_alignment.local_success_n",
        source,
    )
    missing_prompt_n = _int_field(
        alignment.get("missing_prompt_usage_n"),
        "token_alignment.missing_prompt_usage_n",
        source,
    )
    prompt_exact_n = _int_field(
        alignment.get("prompt_exact_n"), "token_alignment.prompt_exact_n", source
    )
    decode_measured_n = _int_field(
        alignment.get("decode_measured_n"),
        "token_alignment.decode_measured_n",
        source,
    )
    missing_decode_n = _int_field(
        alignment.get("missing_completion_usage_n"),
        "token_alignment.missing_completion_usage_n",
        source,
    )
    decode_exact_n = _int_field(
        alignment.get("decode_cap_hit_n"),
        "token_alignment.decode_cap_hit_n",
        source,
    )

    if n <= 0:
        raise EvidenceError(f"{source}: overall.n must be positive")
    if success_n != n:
        raise EvidenceError(f"{source}: overall success {success_n}/{n}")
    if local_n + routed_n != n:
        raise EvidenceError(
            f"{source}: local.n + cloud.n ({local_n} + {routed_n}) != {n}"
        )
    if local_success_n != local_n or cloud_success_n != routed_n:
        raise EvidenceError(f"{source}: side-level n/success mismatch")
    if pessimistic_total != n:
        raise EvidenceError(
            f"{source}: pessimistic denominator {pessimistic_total} != {n}"
        )
    if pessimistic_n != local_violation_n + routed_n:
        raise EvidenceError(
            f"{source}: pessimistic violations do not equal local violations + routed"
        )
    expected_pct = 100.0 * pessimistic_n / n
    if not math.isclose(pessimistic_pct, expected_pct, rel_tol=0.0, abs_tol=1e-9):
        raise EvidenceError(
            f"{source}: pessimistic percentage {pessimistic_pct} != {expected_pct}"
        )

    token_exact = (
        local_n > 0
        and local_alignment_n == local_n
        and measured_n == local_n
        and missing_prompt_n == 0
        and prompt_exact_n == measured_n
        and decode_measured_n == local_n
        and missing_decode_n == 0
        and decode_exact_n == decode_measured_n
    )
    if not token_exact:
        raise EvidenceError(
            f"{source}: prompt/decode token alignment is not exact for all local rows"
        )

    return ArmMetrics(
        arm=SELECTOR_TO_ARM[selector],
        selector=selector,
        trigger=trigger,
        seed=seed,
        n=n,
        success_n=success_n,
        routed_n=routed_n,
        local_n=local_n,
        local_violation_n=local_violation_n,
        pessimistic_violation_n=pessimistic_n,
        pessimistic_violation_pct=pessimistic_pct,
        cost_usd=cost_usd,
        local_ttft_p50_s=_optional_ms_to_s(
            local.get("ttft_p50_ms"), "local.ttft_p50_ms", source
        ),
        local_ttft_p95_s=_optional_ms_to_s(
            local.get("ttft_p95_ms"), "local.ttft_p95_ms", source
        ),
        local_ttft_p99_s=_optional_ms_to_s(
            local.get("ttft_p99_ms"), "local.ttft_p99_ms", source
        ),
        token_measured_n=measured_n,
        prompt_exact_n=prompt_exact_n,
        decode_exact_n=decode_exact_n,
        n_equals_success=True,
        token_exact=True,
        zero_local_violations=local_violation_n == 0,
        source=path.name,
    )


def discover_blocks(inputs: Sequence[str | Path]) -> list[Path]:
    """Resolve explicit blocks, shell/literal globs, and campaign roots."""
    candidates: list[Path] = []
    for raw_value in inputs:
        raw = str(raw_value)
        matches = sorted(glob.glob(raw)) if glob.has_magic(raw) else [raw]
        if not matches:
            raise EvidenceError(f"input pattern matched no directories: {Path(raw).name}")
        for match in matches:
            path = Path(match)
            if not path.is_dir():
                raise EvidenceError(f"input is not a directory: {path.name}")
            direct = sorted(path.glob("*.summary.json"))
            children = sorted(
                child for child in path.iterdir()
                if child.is_dir() and any(child.glob("*.summary.json"))
            )
            if direct and children:
                raise EvidenceError(
                    f"ambiguous input {path.name}: summaries exist both directly and in children"
                )
            if direct:
                candidates.append(path)
            elif children:
                candidates.extend(children)
            else:
                raise EvidenceError(f"no *.summary.json files under {path.name}")

    unique: dict[Path, Path] = {}
    for path in candidates:
        unique[path.resolve()] = path
    blocks = sorted(unique.values(), key=lambda path: path.name)
    names = [path.name for path in blocks]
    if len(names) != len(set(names)):
        raise EvidenceError("block directory basenames must be unique")
    return blocks


def load_block(path: Path) -> BlockMetrics:
    block = path.name
    arms: dict[str, ArmMetrics] = {}
    for summary_path in sorted(path.glob("*.summary.json"), key=lambda p: p.name):
        arm = load_arm(summary_path, block)
        if arm.arm in arms:
            raise EvidenceError(
                f"{block}: duplicate arm {arm.arm} ({arm.selector}); "
                "waiting_random supports at most one seed per block"
            )
        arms[arm.arm] = arm
    missing = sorted(REQUIRED_ARMS - arms.keys())
    if missing:
        raise EvidenceError(f"{block}: missing required arms {', '.join(missing)}")
    n_values = {arm.n for arm in arms.values()}
    if len(n_values) != 1:
        raise EvidenceError(f"{block}: arms do not have a common request count")
    ordered = {key: arms[key] for key in ARM_ORDER if key in arms}
    return BlockMetrics(block=block, arms=ordered)


def _metric(arm: ArmMetrics, name: str) -> float | None:
    value = getattr(arm, name)
    return None if value is None else float(value)


def _describe(values: Iterable[float]) -> dict[str, float | int]:
    rows = list(values)
    if not rows:
        raise EvidenceError("cannot summarize an empty metric")
    return {
        "n_blocks": len(rows),
        "mean": statistics.fmean(rows),
        "median": statistics.median(rows),
        "min": min(rows),
        "max": max(rows),
    }


def _winner(delta: float, other: str) -> str:
    if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "tie"
    return "A" if delta < 0.0 else other


def build_analysis(
    block_paths: Sequence[Path], *, require_zero_local_violations: bool = True
) -> dict[str, Any]:
    blocks = [load_block(path) for path in block_paths]
    if not blocks:
        raise EvidenceError("at least one block is required")
    all_n = {arm.n for block in blocks for arm in block.arms.values()}
    if len(all_n) != 1:
        raise EvidenceError("all blocks must use the same request count")

    unsafe = [
        f"{block.block}/{arm.arm}={arm.local_violation_n}"
        for block in blocks
        for arm in block.arms.values()
        if not arm.zero_local_violations
    ]
    if unsafe and require_zero_local_violations:
        raise EvidenceError(
            "measured local SLO violations (use --allow-local-violations only "
            "for diagnostic output): " + ", ".join(unsafe)
        )

    paired: list[dict[str, Any]] = []
    for block in blocks:
        a = block.arms["A"]
        for other in ("B", "C"):
            comparator = block.arms[other]
            deltas: dict[str, float | None] = {}
            winners: dict[str, str | None] = {}
            for metric_name in COMPARED_METRICS:
                a_value = _metric(a, metric_name)
                other_value = _metric(comparator, metric_name)
                delta = (
                    None if a_value is None or other_value is None
                    else a_value - other_value
                )
                deltas[metric_name] = delta
                winners[metric_name] = (
                    None if delta is None else _winner(delta, other)
                )
            paired.append({
                "block": block.block,
                "comparison": f"A-{other}",
                "delta_a_minus_other": deltas,
                "winner": winners,
            })

    aggregate_by_arm: dict[str, dict[str, dict[str, float | int]]] = {}
    for arm_key in ARM_ORDER:
        present = [
            block.arms[arm_key] for block in blocks if arm_key in block.arms
        ]
        if not present:
            continue
        metric_rows: dict[str, dict[str, float | int]] = {}
        for metric_name in COMPARED_METRICS:
            values = [
                value for arm in present
                if (value := _metric(arm, metric_name)) is not None
            ]
            if values:
                metric_rows[metric_name] = _describe(values)
        aggregate_by_arm[arm_key] = metric_rows

    paired_aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    for comparison, other in (("A-B", "B"), ("A-C", "C")):
        comparisons = [row for row in paired if row["comparison"] == comparison]
        metric_rows = {}
        for metric_name in COMPARED_METRICS:
            values = [
                row["delta_a_minus_other"][metric_name]
                for row in comparisons
                if row["delta_a_minus_other"][metric_name] is not None
            ]
            if not values:
                continue
            description = _describe(values)
            winners = [row["winner"][metric_name] for row in comparisons]
            description["wins"] = {
                "A": winners.count("A"),
                "tie": winners.count("tie"),
                other: winners.count(other),
            }
            metric_rows[metric_name] = description
        paired_aggregate[comparison] = metric_rows

    block_dicts = []
    for block in blocks:
        block_dicts.append({
            "block": block.block,
            "arms": {
                key: asdict(arm) for key, arm in block.arms.items()
            },
        })

    arm_count = sum(len(block.arms) for block in blocks)
    all_zero = not unsafe
    return {
        "schema_version": 1,
        "statistical_unit": "block",
        "per_request_resampling": False,
        "metric_direction": "lower_is_better",
        "delta_definition": "A minus comparator",
        "selector_aliases": {
            arm: selector for selector, arm in SELECTOR_TO_ARM.items()
        },
        "validation": {
            "block_count": len(blocks),
            "arm_count": arm_count,
            "common_request_n": next(iter(all_n)),
            "all_n_equal_success": True,
            "all_token_exact": True,
            "zero_local_violations": all_zero,
            "require_zero_local_violations": require_zero_local_violations,
            "evidence_valid": all_zero,
            "unsafe_arms": unsafe,
        },
        "blocks": block_dicts,
        "paired_deltas": paired,
        "aggregate_by_arm": aggregate_by_arm,
        "paired_aggregate": paired_aggregate,
    }


def analyze(
    inputs: Sequence[str | Path], *, require_zero_local_violations: bool = True
) -> dict[str, Any]:
    return build_analysis(
        discover_blocks(inputs),
        require_zero_local_violations=require_zero_local_violations,
    )


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _delta_cell(delta: float | None, winner: str | None, digits: int) -> str:
    if delta is None:
        return "—"
    return f"{delta:+.{digits}f} ({winner})"


def render_markdown(analysis: dict[str, Any]) -> str:
    validation = analysis["validation"]
    status = "PASS" if validation["evidence_valid"] else "INVALID (diagnostic only)"
    lines = [
        "# TTFT selector repeatability analysis",
        "",
        f"Validation: **{status}** — {validation['block_count']} blocks, "
        f"{validation['arm_count']} arms, N={validation['common_request_n']} per arm; "
        f"n=success={str(validation['all_n_equal_success']).lower()}, "
        f"token-exact={str(validation['all_token_exact']).lower()}, "
        f"zero-local-violations={str(validation['zero_local_violations']).lower()}.",
        "",
        "Statistical unit: **one complete block/run**. No per-request resampling "
        "or bootstrap is used. All compared metrics are lower-is-better. "
        "Delta is A minus comparator, so a negative delta favors A.",
        "",
        "Aliases: A=`cost_cachedisp_old`, B=`newest`, "
        "C=`cost_disp_current`, R=`waiting_random`.",
        "",
        "## Per-block metrics",
        "",
        "| block | arm | seed | routed / local | local viol | pessimistic % | cost $ | local TTFT p50 / p95 / p99 s | token exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for block in analysis["blocks"]:
        for arm_key in ARM_ORDER:
            if arm_key not in block["arms"]:
                continue
            arm = block["arms"][arm_key]
            token = (
                f"{arm['prompt_exact_n']}/{arm['token_measured_n']} prompt; "
                f"{arm['decode_exact_n']}/{arm['token_measured_n']} decode"
            )
            ttft = " / ".join(
                _fmt(arm[name], 3)
                for name in (
                    "local_ttft_p50_s",
                    "local_ttft_p95_s",
                    "local_ttft_p99_s",
                )
            )
            lines.append(
                f"| {block['block']} | {arm_key} | {arm['seed']} | "
                f"{arm['routed_n']} / {arm['local_n']} | "
                f"{arm['local_violation_n']} | "
                f"{_fmt(arm['pessimistic_violation_pct'], 3)} | "
                f"{_fmt(arm['cost_usd'], 6)} | {ttft} | {token} |"
            )

    lines.extend([
        "",
        "## Paired deltas within each block",
        "",
        "| block | pair | Δ routed | Δ pessimistic pp | Δ cost $ | Δ local p50 s | Δ local p95 s | Δ local p99 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in analysis["paired_deltas"]:
        delta = row["delta_a_minus_other"]
        winner = row["winner"]
        lines.append(
            f"| {row['block']} | {row['comparison']} | "
            f"{_delta_cell(delta['routed_n'], winner['routed_n'], 0)} | "
            f"{_delta_cell(delta['pessimistic_violation_pct'], winner['pessimistic_violation_pct'], 3)} | "
            f"{_delta_cell(delta['cost_usd'], winner['cost_usd'], 6)} | "
            f"{_delta_cell(delta['local_ttft_p50_s'], winner['local_ttft_p50_s'], 3)} | "
            f"{_delta_cell(delta['local_ttft_p95_s'], winner['local_ttft_p95_s'], 3)} | "
            f"{_delta_cell(delta['local_ttft_p99_s'], winner['local_ttft_p99_s'], 3)} |"
        )

    lines.extend([
        "",
        "## Whole-run distribution by arm",
        "",
        "Each row summarizes whole-block observations, not requests.",
        "",
        "| arm | metric | blocks | mean | median | range [min, max] |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for arm_key in ARM_ORDER:
        for metric_name, stats in analysis["aggregate_by_arm"].get(arm_key, {}).items():
            digits = 6 if metric_name == "cost_usd" else 3
            lines.append(
                f"| {arm_key} | {metric_name} | {stats['n_blocks']} | "
                f"{_fmt(stats['mean'], digits)} | {_fmt(stats['median'], digits)} | "
                f"[{_fmt(stats['min'], digits)}, {_fmt(stats['max'], digits)}] |"
            )

    lines.extend([
        "",
        "## Paired win directions",
        "",
        "| pair | metric | blocks | Δ mean | Δ median | Δ range [min, max] | wins (A / tie / other) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for comparison in ("A-B", "A-C"):
        other = comparison[-1]
        for metric_name, stats in analysis["paired_aggregate"][comparison].items():
            digits = 6 if metric_name == "cost_usd" else 3
            wins = stats["wins"]
            lines.append(
                f"| {comparison} | {metric_name} | {stats['n_blocks']} | "
                f"{_fmt(stats['mean'], digits)} | {_fmt(stats['median'], digits)} | "
                f"[{_fmt(stats['min'], digits)}, {_fmt(stats['max'], digits)}] | "
                f"{wins['A']} / {wins['tie']} / {wins[other]} |"
            )
    return "\n".join(lines) + "\n"


def render_json(analysis: dict[str, Any]) -> str:
    return json.dumps(analysis, indent=2, sort_keys=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "blocks",
        nargs="+",
        help=(
            "block directories, directory globs, or campaign roots whose "
            "immediate children are blocks"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit reproducible JSON instead of Markdown",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="also write reproducible JSON to this file",
    )
    parser.add_argument(
        "--allow-local-violations",
        action="store_true",
        help=(
            "diagnostic only: do not exit 2 on local violations; the report "
            "will still set evidence_valid=false"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        analysis = analyze(
            args.blocks,
            require_zero_local_violations=not args.allow_local_violations,
        )
    except EvidenceError as exc:
        parser.exit(2, f"error: {exc}\n")
    json_text = render_json(analysis)
    if args.json_out is not None:
        args.json_out.write_text(json_text, encoding="utf-8")
    sys.stdout.write(json_text if args.json else render_markdown(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
