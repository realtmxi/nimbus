#!/usr/bin/env python3
"""Print a comparable Markdown table from router ``*.summary.json`` files.

The primary SLO column is the pessimistic combined bound: every cloud-routed
request is counted as a violation.  Older summaries that predate the explicit
field are reconstructed from local violations plus the cloud route count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def load_row(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    local = summary.get("local", {})
    cloud = summary.get("cloud", {})
    overall = summary.get("overall", {})
    queue = summary.get("queue", {})
    config = summary.get("config", {})
    pessimistic = summary.get("pessimistic_combined")
    if pessimistic is None:
        total = int(overall.get("n") or 0)
        violations = int(local.get("slo_violations") or 0) + int(cloud.get("n") or 0)
        pessimistic = {
            "slo_violation_pct": 100.0 * violations / max(total, 1),
            "cost_usd": overall.get("cost_usd"),
        }
    return {
        "run": path.stem.removesuffix(".summary"),
        "trigger": config.get("nimbus_trigger", queue.get("nimbus_trigger", "—")),
        "selector": config.get("nimbus_selector", queue.get("nimbus_selector", "—")),
        "seed": config.get("seed", "—"),
        "outsourced_pct": 100.0 * float(summary.get("actual_fraction") or 0.0),
        "local_p50_s": (
            float(local["ttft_p50_ms"]) / 1000.0
            if local.get("ttft_p50_ms") is not None else None
        ),
        "local_violation_pct": local.get("slo_violation_pct"),
        "pessimistic_pct": pessimistic.get("slo_violation_pct"),
        "cost_usd": pessimistic.get("cost_usd", overall.get("cost_usd")),
        "peak_waiting": queue.get("peak_waiting"),
        "decision_mean_ms": queue.get("nimbus_decision_mean_ms"),
        "decision_max_ms": queue.get("nimbus_decision_max_ms"),
        "stale": queue.get("nimbus_stale_decisions"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="emit normalized JSON instead")
    args = parser.parse_args()
    rows = [load_row(path) for path in args.summaries]
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print("| run | trigger | selector | seed | out % | local p50 s | local viol % | pessimistic % | cost $ | peak q | decision ms mean/max | stale |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        decision = f"{_num(row['decision_mean_ms'])}/{_num(row['decision_max_ms'])}"
        print(
            f"| {row['run']} | {row['trigger']} | {row['selector']} | {row['seed']} "
            f"| {_num(row['outsourced_pct'])} | {_num(row['local_p50_s'], 3)} "
            f"| {_num(row['local_violation_pct'])} | {_num(row['pessimistic_pct'])} "
            f"| {_num(row['cost_usd'], 4)} | {row['peak_waiting'] if row['peak_waiting'] is not None else '—'} "
            f"| {decision} | {row['stale'] if row['stale'] is not None else '—'} |"
        )


if __name__ == "__main__":
    main()
