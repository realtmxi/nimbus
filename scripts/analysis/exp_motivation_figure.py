#!/usr/bin/env python3
"""Generate motivation figure: memory capacity is the bottleneck, not compute.

Three-panel publication-quality figure:
  (A) Memory fills before compute saturates (KV throughput data).
  (B) Memory pressure causes cliff failure (cliff experiment data).
  (C) KV cache saturation timeline (knee sweep metrics.csv at f00).

Data sources:
  - docs/images/kv_throughput/kv_throughput_7B_mixed.json
  - docs/images/cliff/cliff_coderforge_7b.json
  - logs/cachedisp_knee1/strategies_20260405_022449_f00/cache_disp/metrics.csv

Output:
  - docs/images/motivation_memory_bottleneck.png
  - docs/images/motivation_memory_bottleneck.pdf
"""

import csv
import json
import sys
from pathlib import Path

# Allow imports from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KV_THROUGHPUT_PATH = (
    PROJECT_ROOT / "docs" / "images" / "kv_throughput"
    / "kv_throughput_7B_mixed.json"
)
CLIFF_PATH = (
    PROJECT_ROOT / "docs" / "images" / "cliff" / "cliff_coderforge_7b.json"
)
METRICS_CSV_PATH = (
    PROJECT_ROOT / "logs" / "cachedisp_knee1"
    / "strategies_20260405_022449_f00" / "cache_disp" / "metrics.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "docs" / "images"


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLOR_BASELINE = "#d62728"  # red
COLOR_SELECTIVE = "#2ca02c"  # green
COLOR_BF16 = "#1f77b4"  # blue
COLOR_QUEUE = "#ff7f0e"  # orange
COLOR_UTIL = "#7f7f7f"  # gray
COLOR_OOM = "#d62728"  # red (same as baseline)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 1.0,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kv_throughput() -> dict:
    """Load KV throughput benchmark data."""
    with open(KV_THROUGHPUT_PATH) as f:
        return json.load(f)


def load_cliff() -> dict:
    """Load cliff experiment data."""
    with open(CLIFF_PATH) as f:
        return json.load(f)


def load_metrics_csv() -> dict[str, list]:
    """Load metrics CSV into column arrays."""
    columns: dict[str, list] = {
        "elapsed_s": [],
        "token_usage": [],
        "max_total_tokens": [],
        "num_running": [],
        "num_queued": [],
        "num_used_tokens": [],
        "gen_throughput": [],
    }
    with open(METRICS_CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in columns:
                columns[key].append(float(row[key]))
    return columns


# ---------------------------------------------------------------------------
# Panel A: Memory fills before compute saturates
# ---------------------------------------------------------------------------

def plot_panel_a(ax: plt.Axes, kv_data: dict) -> None:
    """Plot KV cache memory scaling vs GPU VRAM capacity."""
    meta = kv_data["metadata"]
    available_gb = meta["available_for_kv_gb"]
    vram_gb = meta["gpu_vram_gb"]

    scaling = kv_data["memory_scaling"]
    agents = []
    bf16_gb = []
    selective_gb = []
    oom_agents = []

    for entry in scaling:
        n = entry["n_agents"]
        if entry["oom"]:
            oom_agents.append(n)
            continue
        agents.append(n)
        bf16_gb.append(entry["fp16_alloc_mb"] / 1024.0)
        sel_mb = entry["format_results"]["selective_int8"]["tensor_mb"]
        selective_gb.append(sel_mb / 1024.0)

    max_conc = kv_data["max_concurrency"]

    # Extrapolate BF16 memory linearly to show where OOM hits.
    if len(agents) >= 2:
        slope = (bf16_gb[-1] - bf16_gb[0]) / (agents[-1] - agents[0])
        intercept = bf16_gb[0] - slope * agents[0]
        oom_n = max_conc["bf16"]
        # Extend agents range to show OOM point.
        ext_agents = agents + [oom_n]
        ext_bf16 = bf16_gb + [slope * oom_n + intercept]
        sel_slope = (
            (selective_gb[-1] - selective_gb[0])
            / (agents[-1] - agents[0])
        )
        sel_intercept = selective_gb[0] - sel_slope * agents[0]
        sel_ext_n = max_conc["selective_int8"]
        ext_sel_agents = agents + [sel_ext_n]
        ext_sel_gb = selective_gb + [
            sel_slope * sel_ext_n + sel_intercept
        ]
    else:
        ext_agents = agents
        ext_bf16 = bf16_gb
        ext_sel_agents = agents
        ext_sel_gb = selective_gb

    # Plot BF16 memory scaling.
    ax.plot(
        ext_agents, ext_bf16, "o-",
        color=COLOR_BF16, linewidth=2.0, markersize=7,
        label="BF16 KV cache", zorder=5,
    )
    # Plot selective INT8 memory scaling.
    ax.plot(
        ext_sel_agents, ext_sel_gb, "s-",
        color=COLOR_SELECTIVE, linewidth=2.0, markersize=7,
        label="Selective INT8", zorder=5,
    )

    # OOM markers at max concurrency.
    ax.plot(
        max_conc["bf16"],
        slope * max_conc["bf16"] + intercept,
        "X", color=COLOR_OOM, markersize=14, zorder=10,
        markeredgecolor="black", markeredgewidth=0.8,
    )
    ax.plot(
        max_conc["selective_int8"],
        sel_slope * max_conc["selective_int8"] + sel_intercept,
        "X", color=COLOR_OOM, markersize=14, zorder=10,
        markeredgecolor="black", markeredgewidth=0.8,
    )

    # GPU VRAM capacity line.
    ax.axhline(
        y=available_gb, color=COLOR_UTIL, linestyle="--",
        linewidth=1.5, alpha=0.7,
    )
    ax.text(
        1, available_gb + 1.5,
        f"Available for KV: {available_gb:.1f} GB",
        fontsize=8, color=COLOR_UTIL, va="bottom",
    )

    # Annotate OOM points.
    ax.annotate(
        f"BF16 OOM\n(n={max_conc['bf16']})",
        xy=(max_conc["bf16"], slope * max_conc["bf16"] + intercept),
        xytext=(max_conc["bf16"] - 8, available_gb - 15),
        fontsize=8, color=COLOR_OOM, ha="center",
        arrowprops=dict(arrowstyle="->", color=COLOR_OOM, lw=1.2),
    )
    ax.annotate(
        f"Selective OOM\n(n={max_conc['selective_int8']})",
        xy=(
            max_conc["selective_int8"],
            sel_slope * max_conc["selective_int8"] + sel_intercept,
        ),
        xytext=(max_conc["selective_int8"] - 2, available_gb - 25),
        fontsize=8, color=COLOR_SELECTIVE, ha="center",
        arrowprops=dict(arrowstyle="->", color=COLOR_SELECTIVE, lw=1.2),
    )

    # Shade the "headroom" region between BF16 and selective.
    shared_n = min(len(ext_agents), len(ext_sel_agents))
    fill_agents = ext_agents[:shared_n]
    fill_bf16 = ext_bf16[:shared_n]
    fill_sel = ext_sel_gb[:shared_n]
    ax.fill_between(
        fill_agents, fill_sel, fill_bf16,
        alpha=0.12, color=COLOR_SELECTIVE,
        label="Memory saved",
    )

    ax.set_xlabel("Number of concurrent agents")
    ax.set_ylabel("KV cache memory (GB)")
    ax.set_title("(a) Memory fills before compute saturates", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(0, max(max_conc["selective_int8"] + 5, 65))
    ax.set_ylim(0, available_gb + 10)
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# Panel B: Memory pressure causes cliff failure
# ---------------------------------------------------------------------------

def plot_panel_b(ax: plt.Axes, cliff_data: dict) -> None:
    """Plot success rate cliff: baseline vs selective."""
    baseline = cliff_data["baseline"]
    selective = cliff_data["selective"]

    b_conc = [p["conc"] for p in baseline]
    b_rate = [p["success_rate"] for p in baseline]
    s_conc = [p["conc"] for p in selective]
    s_rate = [p["success_rate"] for p in selective]

    ax.plot(
        b_conc, b_rate, "o-",
        color=COLOR_BASELINE, linewidth=2.5, markersize=9,
        label="Baseline (BF16)", zorder=5,
    )
    ax.plot(
        s_conc, s_rate, "s-",
        color=COLOR_SELECTIVE, linewidth=2.5, markersize=9,
        label="Selective (INT8)", zorder=5,
    )

    # 95% SLA line.
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(b_conc[0] - 0.5, 96.5, "95% SLA", fontsize=8, color="gray")

    # Annotate the cliff.
    ax.annotate(
        "KV cache OOM\n(38.5%)",
        xy=(8, 38.5),
        xytext=(18, 55),
        fontsize=9, color=COLOR_BASELINE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_BASELINE, lw=1.5),
    )

    # Annotate selective holding.
    ax.annotate(
        "6x more sessions",
        xy=(48, 98.5),
        xytext=(30, 80),
        fontsize=9, color=COLOR_SELECTIVE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_SELECTIVE, lw=1.5),
    )

    ax.set_xlabel("Concurrent sessions")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("(b) Memory pressure causes cliff failure", fontsize=11)
    ax.set_ylim(-5, 110)
    ax.set_xlim(0, 55)
    ax.legend(loc="center right", frameon=True)
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# Panel C: KV cache saturation timeline
# ---------------------------------------------------------------------------

def plot_panel_c(ax: plt.Axes, metrics: dict[str, list]) -> None:
    """Plot KV utilization and queue depth over time."""
    elapsed = np.array(metrics["elapsed_s"])
    token_usage = np.array(metrics["token_usage"])
    num_queued = np.array(metrics["num_queued"])

    # Convert token_usage fraction to percentage.
    kv_util_pct = token_usage * 100.0

    # Focus on the saturation window: ramp-up through peak queue.
    # The main saturation event is approximately 400-700s where utilization
    # climbs from ~7% to ~87% and queue depth explodes to ~3000.
    window_mask = (elapsed >= 400) & (elapsed <= 700)
    t = elapsed[window_mask]
    util = kv_util_pct[window_mask]
    queued = num_queued[window_mask]

    # Downsample for cleaner plotting if too many points.
    if len(t) > 500:
        step = max(1, len(t) // 500)
        t = t[::step]
        util = util[::step]
        queued = queued[::step]

    # Left axis: KV utilization.
    ln1 = ax.plot(
        t, util, "-",
        color=COLOR_BF16, linewidth=1.8, alpha=0.9,
        label="KV utilization (%)",
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("KV cache utilization (%)", color=COLOR_BF16)
    ax.tick_params(axis="y", labelcolor=COLOR_BF16)
    ax.set_ylim(-5, 110)

    # Right axis: queue depth.
    ax2 = ax.twinx()
    ln2 = ax2.plot(
        t, queued, "-",
        color=COLOR_QUEUE, linewidth=1.8, alpha=0.9,
        label="Queue depth",
    )
    ax2.set_ylabel("Queue depth", color=COLOR_QUEUE)
    ax2.tick_params(axis="y", labelcolor=COLOR_QUEUE)

    # Shade the saturation region (where queue > 0).
    queue_nonzero = queued > 0
    if queue_nonzero.any():
        first_q = np.argmax(queue_nonzero)
        ax.axvspan(
            t[first_q], t[-1],
            alpha=0.08, color=COLOR_OOM,
            label="_nolegend_",
        )
        # Find where sustained queue begins and annotate.
        onset_time = t[first_q]
        onset_util = util[first_q]
        ax.annotate(
            f"Queuing begins\n(util ~{onset_util:.0f}%)",
            xy=(onset_time, onset_util),
            xytext=(onset_time + 50, onset_util + 25),
            fontsize=8, color=COLOR_OOM, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR_OOM, lw=1.2),
        )

    # Annotate peak queue depth.
    peak_q_idx = np.argmax(queued)
    if queued[peak_q_idx] > 0:
        ax2.annotate(
            f"Peak: {queued[peak_q_idx]:.0f}",
            xy=(t[peak_q_idx], queued[peak_q_idx]),
            xytext=(t[peak_q_idx] - 40, queued[peak_q_idx] * 0.75),
            fontsize=8, color=COLOR_QUEUE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR_QUEUE, lw=1.2),
        )

    # Combined legend.
    lns = ln1 + ln2
    labs = [ln.get_label() for ln in lns]
    ax.legend(lns, labs, loc="upper left", frameon=True)

    ax.set_title("(c) KV cache saturation timeline", fontsize=11)
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(
    kv_data: dict,
    cliff_data: dict,
    metrics: dict[str, list],
) -> None:
    """Print key statistics to stdout."""
    print("=" * 60)
    print("Motivation Figure -- Key Statistics")
    print("=" * 60)

    # Panel A stats.
    meta = kv_data["metadata"]
    max_conc = kv_data["max_concurrency"]
    print(
        f"\n[Panel A] Memory Scaling (Qwen2.5-7B, "
        f"{meta['gpu_vram_gb']:.1f} GB VRAM)"
    )
    print(f"  Available for KV cache: {meta['available_for_kv_gb']:.1f} GB")
    print(f"  BF16 max agents:        {max_conc['bf16']}")
    print(f"  Selective INT8 max:      {max_conc['selective_int8']}")
    print(
        f"  Capacity improvement:   "
        f"{max_conc['selective_int8'] / max_conc['bf16']:.2f}x"
    )

    scaling = [e for e in kv_data["memory_scaling"] if not e["oom"]]
    if scaling:
        last = scaling[-1]
        savings = last["format_results"]["selective_int8"][
            "savings_tensor_pct"
        ]
        print(
            f"  Memory savings at n={last['n_agents']}: "
            f"{savings:.1f}%"
        )

    # Panel B stats.
    baseline = cliff_data["baseline"]
    selective = cliff_data["selective"]
    print(f"\n[Panel B] Cliff Experiment (7B, coderforge)")
    print(f"  Baseline at conc=4:     {baseline[0]['success_rate']}%")
    print(f"  Baseline at conc=8:     {baseline[1]['success_rate']}%")
    print(f"  Selective at conc=48:   {selective[-1]['success_rate']}%")
    b_max = max(
        (p["conc"] for p in baseline if p["success_rate"] >= 95.0),
        default=0,
    )
    s_max = max(
        (p["conc"] for p in selective if p["success_rate"] >= 95.0),
        default=0,
    )
    if b_max > 0 and s_max > 0:
        print(
            f"  Sustainable conc (>95%): baseline={b_max}, "
            f"selective={s_max} ({s_max / b_max:.0f}x)"
        )

    # Panel C stats.
    token_usage = np.array(metrics["token_usage"])
    num_queued = np.array(metrics["num_queued"])
    elapsed = np.array(metrics["elapsed_s"])
    peak_util = token_usage.max() * 100.0
    peak_queue = num_queued.max()

    queue_mask = num_queued > 0
    if queue_mask.any():
        queue_onset_idx = np.argmax(queue_mask)
        queue_onset_util = token_usage[queue_onset_idx] * 100.0
        queue_onset_time = elapsed[queue_onset_idx]
    else:
        queue_onset_util = 0.0
        queue_onset_time = 0.0

    print(f"\n[Panel C] KV Saturation Timeline (f00, 0% outsource)")
    print(f"  Peak KV utilization:    {peak_util:.1f}%")
    print(f"  Peak queue depth:       {peak_queue:.0f}")
    print(
        f"  Queue onset at:         "
        f"{queue_onset_util:.1f}% util, t={queue_onset_time:.0f}s"
    )

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate the three-panel motivation figure."""
    # Load data.
    kv_data = load_kv_throughput()
    cliff_data = load_cliff()
    metrics = load_metrics_csv()

    # Print statistics.
    print_statistics(kv_data, cliff_data, metrics)

    # Create figure.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    plot_panel_a(axes[0], kv_data)
    plot_panel_b(axes[1], cliff_data)
    plot_panel_c(axes[2], metrics)

    fig.suptitle(
        "Memory Capacity Is the Bottleneck in Local LLM Deployment",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.tight_layout()

    # Save outputs.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "motivation_memory_bottleneck.png"
    pdf_path = OUTPUT_DIR / "motivation_memory_bottleneck.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigure saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


if __name__ == "__main__":
    main()
