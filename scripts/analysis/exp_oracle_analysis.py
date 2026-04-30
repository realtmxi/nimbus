#!/usr/bin/env python3
"""Analyze why Oracle (size-based) performs worse than CacheDisp at low outsource fractions.

Oracle selects the largest-prefill requests for outsourcing, which should seem
optimal with perfect future knowledge. However, large-prefill requests often
have short decode phases. CacheDisp selects by prefill * decode product, which
removes requests that occupy the decode batch longest. This script quantifies
the gap by comparing selection overlap, outsourced request characteristics,
total cache displacement removed, and resulting TTFT / success metrics.

Usage:
    python scripts/analysis/exp_oracle_analysis.py [LOG_DIR]
    python scripts/analysis/exp_oracle_analysis.py logs/cachedisp_knee1

Output:
    - Prints analysis tables to stdout
    - Saves JSON results to docs/images/oracle_analysis.json
    - Saves scatter plot to docs/images/oracle_analysis.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_PAIR = ("cache_disp", "oracle_size")
FRACTIONS_OF_INTEREST = [0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50]
STRATEGY_COLORS = {
    "cache_disp": "#2ca02c",    # green
    "oracle_size": "#9467bd",   # purple
}
STRATEGY_LABELS = {
    "cache_disp": "CacheDisp (ours)",
    "oracle_size": "Oracle (size)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_run_dirs(sweep_root: Path) -> dict[float, Path]:
    """Map outsource fraction -> run directory path."""
    fraction_dirs = {}
    for run_dir in sorted(sweep_root.glob("strategies_*")):
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        config = json.loads(cfg_path.read_text())
        frac = config["fraction"]
        fraction_dirs[frac] = run_dir
    return fraction_dirs


def load_requests(run_dir: Path, strategy: str) -> list[dict]:
    """Load requests.csv for a strategy, returning list of row dicts."""
    csv_path = run_dir / strategy / "requests.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "idx": int(row["idx"]),
                "prefill_tokens": int(row["prefill_tokens"]),
                "decode_tokens": int(row["decode_tokens"]),
                "outsourced": row["outsourced"] == "True",
                "success": row["success"] == "True",
                "ttft_ms": float(row["ttft_ms"]) if row["ttft_ms"] else None,
                "latency_ms": float(row["latency_ms"]) if row["latency_ms"] else None,
            })
    return rows


def load_summary(run_dir: Path) -> dict[str, dict]:
    """Load strategy_summary.csv, keyed by label."""
    summary_path = run_dir / "strategy_summary.csv"
    result = {}
    if not summary_path.exists():
        return result
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            result[row["label"]] = row
    return result


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def outsourced_indices(rows: list[dict]) -> set[int]:
    """Return set of idx values for outsourced requests."""
    return {r["idx"] for r in rows if r["outsourced"]}


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a list."""
    if not values:
        return float("nan")
    arr = sorted(values)
    k = (len(arr) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(arr) - 1)
    weight = k - lo
    return arr[lo] * (1 - weight) + arr[hi] * weight


def distribution_stats(values: list[float]) -> dict:
    """Return mean, median, p90, min, max for a list of numeric values."""
    if not values:
        return {"mean": 0, "median": 0, "p90": 0, "min": 0, "max": 0, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# Analysis 1: Selection overlap
# ---------------------------------------------------------------------------

def analyze_overlap(fraction_dirs: dict[float, Path]) -> list[dict]:
    """Compute overlap between oracle and cachedisp outsourced sets at each fraction."""
    results = []
    for frac in sorted(fraction_dirs.keys()):
        run_dir = fraction_dirs[frac]
        cd_rows = load_requests(run_dir, "cache_disp")
        os_rows = load_requests(run_dir, "oracle_size")
        if not cd_rows or not os_rows:
            continue

        cd_set = outsourced_indices(cd_rows)
        os_set = outsourced_indices(os_rows)

        n_cd = len(cd_set)
        n_os = len(os_set)
        intersection = cd_set & os_set

        row = {
            "fraction": frac,
            "n_cachedisp": n_cd,
            "n_oracle": n_os,
            "overlap": len(intersection),
            "jaccard": jaccard(cd_set, os_set),
            "overlap_pct_of_cd": len(intersection) / n_cd * 100 if n_cd else 0,
            "overlap_pct_of_os": len(intersection) / n_os * 100 if n_os else 0,
        }
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Analysis 2: Outsourced request characteristics at a single fraction
# ---------------------------------------------------------------------------

def analyze_characteristics(run_dir: Path) -> dict:
    """Compare outsourced-request distributions for oracle vs cachedisp."""
    output = {}
    for strat in STRATEGY_PAIR:
        rows = load_requests(run_dir, strat)
        outsourced = [r for r in rows if r["outsourced"]]
        kept = [r for r in rows if not r["outsourced"]]

        prefill_out = [r["prefill_tokens"] for r in outsourced]
        decode_out = [r["decode_tokens"] for r in outsourced]
        disp_out = [r["prefill_tokens"] * r["decode_tokens"] for r in outsourced]

        prefill_kept = [r["prefill_tokens"] for r in kept]
        decode_kept = [r["decode_tokens"] for r in kept]
        disp_kept = [r["prefill_tokens"] * r["decode_tokens"] for r in kept]

        output[strat] = {
            "outsourced": {
                "n": len(outsourced),
                "prefill": distribution_stats(prefill_out),
                "decode": distribution_stats(decode_out),
                "cache_disp": distribution_stats(disp_out),
                "total_cache_disp": int(sum(disp_out)),
                "total_prefill": int(sum(prefill_out)),
            },
            "kept": {
                "n": len(kept),
                "prefill": distribution_stats(prefill_kept),
                "decode": distribution_stats(decode_kept),
                "cache_disp": distribution_stats(disp_kept),
            },
        }
    return output


# ---------------------------------------------------------------------------
# Analysis 3: Cross-fraction system impact
# ---------------------------------------------------------------------------

def analyze_system_impact(fraction_dirs: dict[float, Path]) -> list[dict]:
    """Compare TTFT, success rate, and total cache displacement at each fraction."""
    results = []
    for frac in sorted(fraction_dirs.keys()):
        run_dir = fraction_dirs[frac]
        summary = load_summary(run_dir)
        row = {"fraction": frac}

        for strat in STRATEGY_PAIR:
            if strat not in summary:
                continue
            s = summary[strat]
            prefix = strat

            row[f"{prefix}_ttft_p50"] = float(s.get("ttft_p50", 0))
            row[f"{prefix}_ttft_p99"] = float(s.get("ttft_p99", 0))
            row[f"{prefix}_success_rate"] = float(s.get("local_success_rate", 0))

            # Compute total cache displacement outsourced.
            req_rows = load_requests(run_dir, strat)
            outsourced = [r for r in req_rows if r["outsourced"]]
            total_disp = sum(
                r["prefill_tokens"] * r["decode_tokens"] for r in outsourced
            )
            row[f"{prefix}_total_disp_outsourced"] = total_disp
            row[f"{prefix}_n_outsourced"] = len(outsourced)

        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Scatter plot: outsourced requests (prefill vs decode)
# ---------------------------------------------------------------------------

def plot_scatter(run_dir: Path, output_path: Path, fraction: float) -> None:
    """Scatter plot of outsourced requests: prefill vs decode, colored by strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect data for both strategies to determine axis limits.
    strat_data = {}
    all_prefill = []
    all_decode = []
    for strat in STRATEGY_PAIR:
        rows = load_requests(run_dir, strat)
        outsourced = [r for r in rows if r["outsourced"]]
        prefill = [r["prefill_tokens"] for r in outsourced]
        decode = [r["decode_tokens"] for r in outsourced]
        strat_data[strat] = (prefill, decode)
        all_prefill.extend(prefill)
        all_decode.extend(decode)

    # Clip axis at p99 to avoid extreme outliers dominating the view.
    x_max = float(np.percentile(all_prefill, 99.5)) * 1.1
    y_max = float(np.percentile(all_decode, 99.5)) * 1.1

    # Left panel: scatter of outsourced requests.
    ax = axes[0]
    for strat in STRATEGY_PAIR:
        prefill, decode = strat_data[strat]
        ax.scatter(
            prefill, decode,
            alpha=0.25, s=12,
            color=STRATEGY_COLORS[strat],
            label=STRATEGY_LABELS[strat],
        )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Prefill tokens", fontsize=12)
    ax.set_ylabel("Decode tokens", fontsize=12)
    ax.set_title(
        f"Outsourced requests at {fraction:.0%} fraction",
        fontsize=13,
    )
    ax.legend(fontsize=10)

    # Draw iso-displacement lines (prefill * decode = constant).
    x_range = np.linspace(10, x_max, 300)
    for iso_val, label in [(500_000, "500K"), (1_000_000, "1M"), (2_000_000, "2M")]:
        y_iso = iso_val / x_range
        mask = y_iso <= y_max
        if mask.any():
            ax.plot(
                x_range[mask], y_iso[mask],
                "--", color="gray", alpha=0.5, linewidth=0.8,
            )
            # Place label near the middle of the visible curve.
            vis_indices = np.where(mask)[0]
            label_idx = vis_indices[len(vis_indices) // 2]
            ax.text(
                x_range[label_idx], y_iso[label_idx] * 1.05,
                f"disp={label}",
                fontsize=7, color="gray", ha="center",
            )

    # Right panel: CDF of cache displacement (prefill * decode) for outsourced.
    ax2 = axes[1]
    for strat in STRATEGY_PAIR:
        rows = load_requests(run_dir, strat)
        outsourced = [r for r in rows if r["outsourced"]]
        disp = sorted([r["prefill_tokens"] * r["decode_tokens"] for r in outsourced])
        cdf = np.arange(1, len(disp) + 1) / len(disp)
        ax2.plot(
            disp, cdf,
            color=STRATEGY_COLORS[strat],
            label=STRATEGY_LABELS[strat],
            linewidth=2,
        )

    ax2.set_xlabel("Cache displacement (prefill * decode)", fontsize=12)
    ax2.set_ylabel("CDF", fontsize=12)
    ax2.set_title(
        f"CDF of per-request cache displacement outsourced ({fraction:.0%})",
        fontsize=13,
    )
    ax2.legend(fontsize=10)
    ax2.set_xscale("log")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved scatter plot to {output_path}")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_separator(char: str = "=", width: int = 80) -> None:
    print(char * width)


def print_section(title: str) -> None:
    print()
    print_separator()
    print(f"  {title}")
    print_separator()


def fmt(val, width: int = 12) -> str:
    """Format a numeric value for table display."""
    if isinstance(val, float):
        if abs(val) >= 1000:
            return f"{val:>{width},.0f}"
        return f"{val:>{width}.3f}"
    if isinstance(val, int):
        return f"{val:>{width},}"
    return f"{str(val):>{width}}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze oracle vs cachedisp outsourcing strategies."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="logs/cachedisp_knee1",
        help="Root directory containing strategies_* subdirectories.",
    )
    parser.add_argument(
        "--focus-fraction",
        type=float,
        default=0.15,
        help="Primary fraction to analyze in detail (default: 0.15).",
    )
    args = parser.parse_args()

    sweep_root = Path(args.log_dir)
    if not sweep_root.exists():
        print(f"[ERROR] Directory not found: {sweep_root}", file=sys.stderr)
        sys.exit(1)

    fraction_dirs = discover_run_dirs(sweep_root)
    if not fraction_dirs:
        print(f"[ERROR] No strategy run dirs found in {sweep_root}", file=sys.stderr)
        sys.exit(1)

    focus = args.focus_fraction
    if focus not in fraction_dirs:
        # Find closest available fraction.
        available = sorted(fraction_dirs.keys())
        focus = min(available, key=lambda x: abs(x - args.focus_fraction))
        print(f"[WARN] Requested fraction {args.focus_fraction} not found; "
              f"using closest: {focus}")

    all_results = {}

    # ------------------------------------------------------------------
    # 1. Selection overlap
    # ------------------------------------------------------------------
    print_section("1. Selection Overlap: Oracle vs CacheDisp")
    overlap_data = analyze_overlap(fraction_dirs)
    all_results["overlap"] = overlap_data

    header = (
        f"{'Frac':>6}  {'N_CD':>7}  {'N_OS':>7}  {'Overlap':>7}  "
        f"{'Jaccard':>8}  {'%ofCD':>7}  {'%ofOS':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in overlap_data:
        print(
            f"{row['fraction']:>6.0%}  "
            f"{row['n_cachedisp']:>7,}  "
            f"{row['n_oracle']:>7,}  "
            f"{row['overlap']:>7,}  "
            f"{row['jaccard']:>8.3f}  "
            f"{row['overlap_pct_of_cd']:>6.1f}%  "
            f"{row['overlap_pct_of_os']:>6.1f}%"
        )

    # ------------------------------------------------------------------
    # 2. Outsourced request characteristics at focus fraction
    # ------------------------------------------------------------------
    print_section(
        f"2. Outsourced Request Characteristics at {focus:.0%} Fraction"
    )
    focus_dir = fraction_dirs[focus]
    chars = analyze_characteristics(focus_dir)
    all_results["characteristics"] = chars

    for strat in STRATEGY_PAIR:
        label = STRATEGY_LABELS.get(strat, strat)
        info = chars[strat]
        print(f"\n  --- {label} (n_outsourced={info['outsourced']['n']}) ---")

        for subset_name in ("outsourced", "kept"):
            subset = info[subset_name]
            print(f"\n    [{subset_name.upper()} requests, n={subset['n']}]")
            print(
                f"    {'Metric':>20}  {'Mean':>10}  {'Median':>10}  "
                f"{'P90':>10}  {'Min':>10}  {'Max':>10}"
            )
            print(f"    {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
            for metric_name in ("prefill", "decode", "cache_disp"):
                stats = subset[metric_name]
                print(
                    f"    {metric_name:>20}  "
                    f"{stats['mean']:>10,.0f}  "
                    f"{stats['median']:>10,.0f}  "
                    f"{stats['p90']:>10,.0f}  "
                    f"{stats['min']:>10,.0f}  "
                    f"{stats['max']:>10,.0f}"
                )

    # ------------------------------------------------------------------
    # 3. Key mechanism: cache displacement removed
    # ------------------------------------------------------------------
    print_section("3. Cache Displacement Removed (the Key Mechanism)")

    cd_out = chars["cache_disp"]["outsourced"]
    os_out = chars["oracle_size"]["outsourced"]

    print(f"\n  CacheDisp outsources {cd_out['n']} requests:")
    print(f"    Total cache displacement removed:  {cd_out['total_cache_disp']:>15,}")
    print(f"    Total prefill removed:             {cd_out['total_prefill']:>15,}")
    print(f"    Mean prefill:                      {cd_out['prefill']['mean']:>15,.0f}")
    print(f"    Mean decode:                       {cd_out['decode']['mean']:>15,.0f}")
    print(f"    Mean cache displacement:           {cd_out['cache_disp']['mean']:>15,.0f}")

    print(f"\n  Oracle outsources {os_out['n']} requests:")
    print(f"    Total cache displacement removed:  {os_out['total_cache_disp']:>15,}")
    print(f"    Total prefill removed:             {os_out['total_prefill']:>15,}")
    print(f"    Mean prefill:                      {os_out['prefill']['mean']:>15,.0f}")
    print(f"    Mean decode:                       {os_out['decode']['mean']:>15,.0f}")
    print(f"    Mean cache displacement:           {os_out['cache_disp']['mean']:>15,.0f}")

    disp_ratio = (
        cd_out["total_cache_disp"] / os_out["total_cache_disp"]
        if os_out["total_cache_disp"] > 0
        else float("inf")
    )
    prefill_ratio = (
        os_out["total_prefill"] / cd_out["total_prefill"]
        if cd_out["total_prefill"] > 0
        else float("inf")
    )

    print(f"\n  CacheDisp removes {disp_ratio:.2f}x more cache displacement")
    print(f"  Oracle removes {prefill_ratio:.2f}x more total prefill tokens")
    print(
        f"  But Oracle's outsourced requests have "
        f"{os_out['decode']['mean']:.0f} mean decode "
        f"vs CacheDisp's {cd_out['decode']['mean']:.0f} "
        f"({cd_out['decode']['mean'] / os_out['decode']['mean']:.1f}x longer)"
    )

    # ------------------------------------------------------------------
    # 4. Cross-fraction system impact
    # ------------------------------------------------------------------
    print_section("4. TTFT and System Impact Across Fractions")
    impact_data = analyze_system_impact(fraction_dirs)
    all_results["system_impact"] = impact_data

    header = (
        f"{'Frac':>6}  "
        f"{'CD_TTFT50':>11}  {'OS_TTFT50':>11}  "
        f"{'CD_TTFT99':>11}  {'OS_TTFT99':>11}  "
        f"{'CD_Succ%':>9}  {'OS_Succ%':>9}  "
        f"{'CD_Disp':>14}  {'OS_Disp':>14}  {'Ratio':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in impact_data:
        cd_disp = row.get("cache_disp_total_disp_outsourced", 0)
        os_disp = row.get("oracle_size_total_disp_outsourced", 0)
        ratio = cd_disp / os_disp if os_disp > 0 else float("inf")
        print(
            f"{row['fraction']:>6.0%}  "
            f"{row.get('cache_disp_ttft_p50', 0):>11,.1f}  "
            f"{row.get('oracle_size_ttft_p50', 0):>11,.1f}  "
            f"{row.get('cache_disp_ttft_p99', 0):>11,.1f}  "
            f"{row.get('oracle_size_ttft_p99', 0):>11,.1f}  "
            f"{row.get('cache_disp_success_rate', 0):>8.1%}  "
            f"{row.get('oracle_size_success_rate', 0):>8.1%}  "
            f"{cd_disp:>14,}  "
            f"{os_disp:>14,}  "
            f"{ratio:>5.2f}x"
        )

    # ------------------------------------------------------------------
    # 5. Summary: why oracle is worse
    # ------------------------------------------------------------------
    print_section("5. Summary: Why Oracle (Size-Based) Loses to CacheDisp")

    focus_impact = next(
        (r for r in impact_data if r["fraction"] == focus), None
    )
    if focus_impact:
        cd_ttft = focus_impact.get("cache_disp_ttft_p50", 0)
        os_ttft = focus_impact.get("oracle_size_ttft_p50", 0)
        cd_succ = focus_impact.get("cache_disp_success_rate", 0)
        os_succ = focus_impact.get("oracle_size_success_rate", 0)

        print(f"""
  At {focus:.0%} outsource fraction:

  Oracle selects the {os_out['n']} requests with the LARGEST prefill tokens.
  CacheDisp selects the {cd_out['n']} requests with the LARGEST prefill*decode product.

  The critical difference:
    Oracle's outsourced requests have mean decode = {os_out['decode']['mean']:.0f} tokens
    CacheDisp's outsourced requests have mean decode = {cd_out['decode']['mean']:.0f} tokens

  Oracle picks large-prefill requests that often have SHORT decode phases.
  These requests occupy the KV cache briefly during decode, so evicting them
  frees memory only momentarily. They do not reduce sustained batch occupancy.

  CacheDisp picks requests with high prefill*decode product -- requests that
  would occupy the decode batch for a LONG time with a LARGE KV footprint.
  Removing these frees sustained cache capacity, allowing more concurrent
  requests to proceed without queuing.

  Quantitative gap:
    Cache displacement removed: CacheDisp {cd_out['total_cache_disp']:,} vs Oracle {os_out['total_cache_disp']:,} ({disp_ratio:.2f}x)
    TTFT p50:                   CacheDisp {cd_ttft:,.0f} ms vs Oracle {os_ttft:,.0f} ms ({os_ttft/cd_ttft:.1f}x worse)
    Success rate:               CacheDisp {cd_succ:.1%} vs Oracle {os_succ:.1%}

  The lesson: under memory pressure, the right outsourcing metric is not
  "what is expensive to prefill" but "what will occupy the most capacity
  for the longest time." Cache displacement (prefill * decode) captures
  the area under the memory-time curve that a request consumes.
""")

    # ------------------------------------------------------------------
    # Save JSON output
    # ------------------------------------------------------------------
    output_dir = Path("docs/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "oracle_analysis.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"[INFO] Saved JSON results to {json_path}")

    # ------------------------------------------------------------------
    # Generate figure
    # ------------------------------------------------------------------
    png_path = output_dir / "oracle_analysis.png"
    plot_scatter(focus_dir, png_path, focus)


if __name__ == "__main__":
    main()
