#!/usr/bin/env python3
"""Analyze and plot online engine sweep results.

Input is the ``engine_summary.csv`` written by ``experiments/run_engine_sweep.py``.
The script always prints a compact table and an iso-SLO cost table. Plotting is
optional and only imports matplotlib when needed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


LABELS = {
    "nimbus": "Nimbus",
    "cachedisp_oracle": "CacheDisp fixed",
    "random": "Random",
    "all_local": "All local",
    "all_cloud": "All cloud",
}

COLORS = {
    "nimbus": "#111111",
    "cachedisp_oracle": "#2ca02c",
    "random": "#7f7f7f",
    "all_local": "#1f77b4",
    "all_cloud": "#9467bd",
}


class MissingPlotDependency(RuntimeError):
    """Raised when plotting dependencies are not installed."""


def _require_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise MissingPlotDependency(
            "matplotlib is required for plotting. Install project deps with "
            "`python3 -m pip install -r requirements.txt`, or pass --no-plot."
        ) from exc
    return plt


def parse_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_fraction(value):
    if value == "self":
        return "self"
    return parse_float(value)


def load_engine_summary(path: Path) -> list[dict]:
    """Load normalized rows from an engine_summary.csv file or sweep dir."""
    path = Path(path)
    if path.is_dir():
        path = path / "engine_summary.csv"
    if not path.exists():
        return []

    rows: list[dict] = []
    with open(path) as f:
        for raw in csv.DictReader(f):
            row = {
                "policy": raw.get("policy", ""),
                "weight": raw.get("weight", "-"),
                "fraction": parse_fraction(raw.get("fraction")),
                "slo_s": parse_float(raw.get("slo_s")),
                "total": int(parse_float(raw.get("total"))),
                "outsourced": int(parse_float(raw.get("outsourced"))),
                "outsource_pct": parse_float(raw.get("outsource_pct")),
                "total_cost_usd": parse_float(raw.get("total_cost_usd")),
                "remote_cached_input_tokens": int(
                    parse_float(raw.get("remote_cached_input_tokens"))
                ),
                "remote_cache_hit_pct": parse_float(raw.get("remote_cache_hit_pct")),
                "metrics_read_failures": int(parse_float(raw.get("metrics_read_failures"))),
                "serving_engine": raw.get("serving_engine") or "unknown",
                "tpot_profile": raw.get("tpot_profile") or "-",
                "tpot_profile_points": int(parse_float(raw.get("tpot_profile_points"))),
                "tpot_ms_min": parse_float(raw.get("tpot_ms_min")),
                "tpot_ms_mean": parse_float(raw.get("tpot_ms_mean")),
                "tpot_ms_max": parse_float(raw.get("tpot_ms_max")),
                "slo_violation_pct": parse_float(raw.get("slo_violation_pct")),
                "ttft_p50_ms": parse_float(raw.get("ttft_p50_ms")),
                "ttft_p99_ms": parse_float(raw.get("ttft_p99_ms")),
            }
            row["label"] = label_for_row(row)
            rows.append(row)
    return rows


def label_for_row(row: dict) -> str:
    base = LABELS.get(row.get("policy"), row.get("policy") or "unknown")
    weight = row.get("weight")
    if row.get("policy") == "nimbus" and weight and weight != "-":
        return f"{base} ({weight})"
    return base


def best_slo_rows(
    rows: list[dict],
    ttft_slo_ms: float,
    max_violation_pct: float,
) -> dict[str, dict | None]:
    """Return the lowest-cost row per label that satisfies the SLO gates."""
    labels = sorted({row["label"] for row in rows})
    result: dict[str, dict | None] = {label: None for label in labels}
    for label in labels:
        candidates = [
            row for row in rows
            if row["label"] == label
            and row["ttft_p99_ms"] <= ttft_slo_ms
            and row["slo_violation_pct"] <= max_violation_pct
        ]
        if candidates:
            result[label] = min(
                candidates,
                key=lambda row: (row["total_cost_usd"], row["outsource_pct"]),
            )
    return result


def print_run_table(rows: list[dict]) -> None:
    print("\n=== Engine Sweep Runs ===")
    print(
        f"{'Policy':<20} {'Frac':>8} {'Out%':>8} {'Cost($)':>10} "
        f"{'Viol%':>8} {'P99(ms)':>10} {'RemoteHit%':>11}"
    )
    print("-" * 84)
    for row in sorted(rows, key=lambda r: (r["label"], str(r["fraction"]))):
        fraction = row["fraction"]
        frac_s = "self" if fraction == "self" else f"{fraction:.0%}"
        print(
            f"{row['label']:<20} {frac_s:>8} {row['outsource_pct']:>7.1%} "
            f"{row['total_cost_usd']:>10.4f} {row['slo_violation_pct']:>7.1%} "
            f"{row['ttft_p99_ms']:>10.1f} {row['remote_cache_hit_pct']:>10.1%}"
        )


def print_iso_slo_table(
    rows: list[dict],
    ttft_slo_ms: float,
    max_violation_pct: float,
) -> None:
    best = best_slo_rows(rows, ttft_slo_ms, max_violation_pct)
    print(
        f"\n=== Lowest Cost Meeting SLO "
        f"(p99 <= {ttft_slo_ms:.0f}ms, violations <= {max_violation_pct:.1%}) ==="
    )
    print(f"{'Policy':<20} {'Frac':>8} {'Cost($)':>10} {'Out%':>8} {'P99(ms)':>10}")
    print("-" * 62)
    for label, row in best.items():
        if row is None:
            print(f"{label:<20} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>10}")
            continue
        fraction = row["fraction"]
        frac_s = "self" if fraction == "self" else f"{fraction:.0%}"
        print(
            f"{label:<20} {frac_s:>8} {row['total_cost_usd']:>10.4f} "
            f"{row['outsource_pct']:>7.1%} {row['ttft_p99_ms']:>10.1f}"
        )


def _marker_label(row: dict) -> str:
    fraction = row["fraction"]
    if fraction == "self":
        return "self"
    return f"{fraction:.0%}"


def plot_engine_sweep(rows: list[dict], output: Path) -> None:
    plt = _require_pyplot()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = sorted({row["label"] for row in rows})

    for label in labels:
        data = [row for row in rows if row["label"] == label]
        data.sort(key=lambda row: row["total_cost_usd"])
        color = COLORS.get(data[0]["policy"], "#333333")
        costs = [row["total_cost_usd"] for row in data]
        p99s = [max(row["ttft_p99_ms"], 1.0) for row in data]
        violations = [row["slo_violation_pct"] for row in data]
        ax1.plot(costs, p99s, "o-", label=label, color=color, lw=2, ms=6)
        ax2.plot(costs, violations, "o-", label=label, color=color, lw=2, ms=6)
        for row in data:
            ax1.annotate(
                _marker_label(row),
                (row["total_cost_usd"], max(row["ttft_p99_ms"], 1.0)),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    ax1.set_xlabel("Cloud API cost (USD)")
    ax1.set_ylabel("TTFT p99 (ms)")
    ax1.set_yscale("log")
    ax1.set_title("Iso-SLO Cost Curve")
    ax1.grid(alpha=0.3, which="both")

    ax2.set_xlabel("Cloud API cost (USD)")
    ax2.set_ylabel("SLO violation rate")
    ax2.set_ylim(bottom=0, top=max(0.01, max((r["slo_violation_pct"] for r in rows), default=0) * 1.15))
    ax2.set_title("Violation Rate vs Cost")
    ax2.grid(alpha=0.3)

    handles, labels_ = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=min(4, len(labels)))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    plt.savefig(output, dpi=150)
    plt.savefig(output.with_suffix(".pdf"))
    print(f"\nSaved {output}")


def write_normalized_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "label", "policy", "weight", "fraction", "outsource_pct", "total_cost_usd",
        "slo_violation_pct", "ttft_p50_ms", "ttft_p99_ms",
        "remote_cached_input_tokens", "remote_cache_hit_pct", "metrics_read_failures",
        "serving_engine", "tpot_profile", "tpot_profile_points",
        "tpot_ms_min", "tpot_ms_mean", "tpot_ms_max",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in cols})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze online engine sweep results")
    parser.add_argument("sweep_dir", type=Path, help="Sweep directory or engine_summary.csv")
    parser.add_argument("--output", type=Path, default=None, help="Plot output PNG")
    parser.add_argument("--normalized-csv", type=Path, default=None)
    parser.add_argument("--ttft-slo-ms", type=float, default=5000.0)
    parser.add_argument("--max-violation-pct", type=float, default=0.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rows = load_engine_summary(args.sweep_dir)
    if not rows:
        print(f"No engine_summary.csv rows found under {args.sweep_dir}")
        return

    print_run_table(rows)
    print_iso_slo_table(rows, args.ttft_slo_ms, args.max_violation_pct)

    normalized_csv = args.normalized_csv
    if normalized_csv is None:
        base = args.sweep_dir if args.sweep_dir.is_dir() else args.sweep_dir.parent
        normalized_csv = base / "engine_sweep_normalized.csv"
    write_normalized_csv(rows, normalized_csv)
    print(f"\nSaved {normalized_csv}")

    if args.no_plot:
        return

    output = args.output
    if output is None:
        base = args.sweep_dir if args.sweep_dir.is_dir() else args.sweep_dir.parent
        output = base / "engine_sweep_plot.png"
    try:
        plot_engine_sweep(rows, output)
    except MissingPlotDependency as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
