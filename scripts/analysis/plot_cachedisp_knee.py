#!/usr/bin/env python3
"""Plot Cache Displacement vs FLOP-based outsourcing knee sweep results.

Reads strategy_summary.csv from each fraction subdir and produces:
  1. Two-panel knee plot: TTFT p50 vs fraction (top), cache hit vs fraction (bottom)
  2. TTFT p99 vs fraction
  3. Dollar efficiency table (min fraction to meet TTFT SLO)

Usage:
    python plot_cachedisp_knee.py logs/cachedisp_knee1
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

STRATEGY_COLORS = {
    "flop_based": "#d62728",      # red
    "cache_disp": "#2ca02c",      # green
    "session_aware": "#1f77b4",   # blue
    "oracle_size": "#9467bd",     # purple
}
STRATEGY_LABELS = {
    "flop_based": "FLOP-based (Nimbus v1)",
    "cache_disp": "Cache Displacement (ours)",
    "session_aware": "Session-aware",
    "oracle_size": "Oracle (size)",
}


def load_sweep(sweep_root: Path):
    """Return {strategy: [(fraction, ttft_p50, ttft_p99, cache_hit, success_pct), ...]}"""
    results = {}
    for run_dir in sorted(sweep_root.glob("strategies_*")):
        summary = run_dir / "strategy_summary.csv"
        cfg = run_dir / "config.json"
        if not summary.exists() or not cfg.exists():
            continue
        config = json.loads(cfg.read_text())
        frac = config["fraction"]

        with open(summary) as f:
            for row in csv.DictReader(f):
                strat = row["strategy"]
                ttft_p50 = float(row.get("ttft_p50_ms") or row.get("ttft_p50") or 0)
                ttft_p99 = float(row.get("ttft_p99_ms") or row.get("ttft_p99") or 0)
                success = float(row.get("success_rate_pct") or row.get("success_pct") or 0)
                # Cache hit rate: load from metrics.csv if available
                cache_hit = load_mean_cache_hit(run_dir / strat / "metrics.csv")
                results.setdefault(strat, []).append(
                    (frac, ttft_p50, ttft_p99, cache_hit, success)
                )
    for strat in results:
        results[strat].sort()
    return results


def load_mean_cache_hit(metrics_csv: Path) -> float:
    if not metrics_csv.exists():
        return float("nan")
    vals = []
    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            v = row.get("cache_hit_rate", "")
            try:
                f_ = float(v)
                if f_ > 0:
                    vals.append(f_)
            except (ValueError, TypeError):
                continue
    return sum(vals) / len(vals) if vals else float("nan")


def plot_knee(results, output: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for strat in ["flop_based", "cache_disp", "session_aware", "oracle_size"]:
        if strat not in results:
            continue
        data = results[strat]
        fracs = [d[0] for d in data]
        p50s = [d[1] for d in data]
        hits = [d[3] for d in data]
        ax1.plot(fracs, p50s, "o-",
                 color=STRATEGY_COLORS[strat],
                 label=STRATEGY_LABELS[strat], lw=2, ms=8)
        ax2.plot(fracs, hits, "s-",
                 color=STRATEGY_COLORS[strat],
                 label=STRATEGY_LABELS[strat], lw=2, ms=8)

    ax1.set_ylabel("TTFT p50 (ms)")
    ax1.set_yscale("log")
    ax1.set_title("Knee: TTFT vs Outsource Fraction")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="upper right")

    ax2.set_ylabel("Cache Hit Rate (mean)")
    ax2.set_xlabel("Outsource Fraction")
    ax2.set_title("Mechanism: Cache Preservation")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.savefig(output.with_suffix(".pdf"))
    print(f"Saved {output}")


def print_dollar_efficiency(results, ttft_slo_ms: float = 5000):
    print(f"\n=== Dollar Efficiency (TTFT p99 SLO = {ttft_slo_ms}ms) ===")
    print(f"{'Strategy':<25} {'Min fraction':>15} {'API cost ratio':>18}")
    print("-" * 60)
    thresholds = {}
    for strat, data in results.items():
        meet_slo = [f for f, p50, p99, h, s in data if p99 <= ttft_slo_ms]
        if meet_slo:
            thresholds[strat] = min(meet_slo)
        else:
            thresholds[strat] = None
    base = thresholds.get("flop_based")
    for strat, thr in thresholds.items():
        if thr is None:
            print(f"{STRATEGY_LABELS.get(strat, strat):<25} {'N/A':>15} {'N/A':>18}")
        else:
            ratio = thr / base if base else 1.0
            savings = (1 - ratio) * 100 if ratio < 1 else 0
            print(f"{STRATEGY_LABELS.get(strat, strat):<25} {thr:>14.0%} "
                  f"{ratio:>14.2f}x  (-{savings:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    results = load_sweep(args.sweep_dir)
    if not results:
        print(f"No results in {args.sweep_dir}")
        return

    print(f"Found {len(results)} strategies:")
    for s, d in results.items():
        print(f"  {s}: {len(d)} fractions")

    output = args.output or (args.sweep_dir / "knee_plot.png")
    plot_knee(results, output)
    print_dollar_efficiency(results, 5000)
    print_dollar_efficiency(results, 1000)


if __name__ == "__main__":
    main()
