#!/usr/bin/env python3
"""Compare knapsack (dp_scaled) vs greedy sorting (fractional) for outsourcing decisions.

Answers Juncheng's question: "Do you need knapsack, or can you just sort
by value/weight ratio?"

Methodology:
  1. Load real trace data and simulate queue snapshots of varying sizes.
  2. For each snapshot, build knapsack items (weight = cache displacement,
     value = API cost) and run both dp_scaled and fractional solvers.
  3. Compare single-shot decisions (budget = total_weight - 1) and iterative
     outsourcing (outsource one at a time until a target fraction).
  4. Report agreement rate, value gap, and disagreement magnitude.

Usage:
  python scripts/analysis/exp_knapsack_vs_sorting.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

# Allow imports from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nimbus.knapsack import KnapsackSolver


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACE_PATH = Path(__file__).resolve().parents[2] / (
    "data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl"
)
OUTPUT_PATH = Path(__file__).resolve().parents[2] / (
    "docs/images/knapsack_vs_sorting.json"
)

INPUT_PRICE_PER_MILLION = 1.25   # USD / M input tokens
OUTPUT_PRICE_PER_MILLION = 10.0  # USD / M output tokens

# Queue sizes to sweep.
QUEUE_SIZES = [10, 20, 50, 100, 150, 200]

# Number of queue snapshots per queue size (single-shot experiment).
SNAPSHOTS_PER_SIZE = 20

# Number of queue snapshots for the iterative experiment.  Iterative is
# O(n * k * DP_cost), so we use fewer snapshots for large queue sizes.
ITERATIVE_SNAPSHOTS_PER_SIZE = 5

# Burst window (seconds) for collecting queue snapshots.
BURST_WINDOW_SEC = 30

# Iterative outsourcing: outsource until this fraction of total weight
# has been removed.
ITERATIVE_TARGET_FRACTIONS = [0.1, 0.25, 0.5]

# Maximum iterative steps per snapshot.  Caps runtime for large queues.
MAX_ITERATIVE_STEPS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trace(path: Path, max_lines: int = 50000) -> list[dict]:
    """Load trace records from JSONL file.

    Args:
        path: Path to the JSONL trace file.
        max_lines: Maximum number of lines to read.

    Returns:
        List of trace records with num_prefill_tokens, num_decode_tokens,
        session_id, and arrived_at fields.
    """
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            rec = json.loads(line)
            records.append(rec)
    return records


def build_item(req_id: str, prefill_tokens: int, decode_tokens: int) -> dict:
    """Build a knapsack item from a trace record.

    Weight = prefill_tokens * decode_tokens (proxy for cache displacement).
    Value  = API cost in milli-dollars (int-safe for DP).

    Args:
        req_id: Unique request identifier.
        prefill_tokens: Number of prefill tokens.
        decode_tokens: Number of decode tokens.

    Returns:
        Dict with id, weight, and value keys.
    """
    weight = max(1, prefill_tokens * decode_tokens)
    api_cost = (
        (prefill_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
        + (decode_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    )
    value = max(1, int(api_cost * 1_000_000))  # micro-dollars, int-safe
    return {"id": req_id, "weight": weight, "value": value}


def extract_queue_snapshots(
    records: list[dict],
    queue_size: int,
    num_snapshots: int,
    burst_window: float,
) -> list[list[dict]]:
    """Extract queue snapshots from trace data.

    Strategy: walk through the trace sorted by arrival time.  For each
    snapshot, find a burst window where at least ``queue_size`` requests
    arrive.  If the window is not dense enough, fall back to taking a
    contiguous slice of ``queue_size`` requests starting from a random
    offset.

    Args:
        records: Sorted trace records.
        queue_size: Desired number of requests per snapshot.
        num_snapshots: How many snapshots to generate.
        burst_window: Time window (seconds) for burst detection.

    Returns:
        List of snapshots, each a list of trace records.
    """
    snapshots: list[list[dict]] = []
    n = len(records)
    if n < queue_size:
        return snapshots

    # Deterministic step through the trace to pick starting points.
    step = max(1, (n - queue_size) // max(1, num_snapshots))

    for snap_idx in range(num_snapshots):
        start = snap_idx * step
        if start + queue_size > n:
            break

        # Try burst window first.
        t0 = records[start]["arrived_at"]
        burst = [
            r for r in records[start : start + queue_size * 3]
            if r["arrived_at"] - t0 <= burst_window
        ]
        if len(burst) >= queue_size:
            snapshots.append(burst[:queue_size])
        else:
            # Fallback: contiguous slice.
            snapshots.append(records[start : start + queue_size])

    return snapshots


def compute_value_kept(items: list[dict], keep_ids: set[str]) -> int:
    """Sum of values for items in keep_ids."""
    return sum(it["value"] for it in items if it["id"] in keep_ids)


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Single-shot comparison
# ---------------------------------------------------------------------------

def run_single_shot(snapshots: list[list[dict]], queue_size: int) -> dict:
    """Compare dp_scaled vs fractional on single-shot outsourcing.

    For each snapshot, set budget = total_weight - 1 (force outsourcing
    at least one request) and compare.

    Args:
        snapshots: List of queue snapshots.
        queue_size: Queue size label.

    Returns:
        Summary dict with agreement and value gap statistics.
    """
    dp_solver = KnapsackSolver(strategy="dp_scaled")
    greedy_solver = KnapsackSolver(strategy="fractional")

    exact_agreements = 0
    jaccard_scores: list[float] = []
    value_gaps: list[float] = []
    dp_times: list[float] = []
    greedy_times: list[float] = []
    disagreement_cases: list[dict] = []

    for snap_idx, snapshot in enumerate(snapshots):
        items = [
            build_item(
                req_id=f"q{queue_size}_s{snap_idx}_r{i}",
                prefill_tokens=r["num_prefill_tokens"],
                decode_tokens=r["num_decode_tokens"],
            )
            for i, r in enumerate(snapshot)
        ]

        total_weight = sum(it["weight"] for it in items)
        budget = max(1, total_weight - 1)

        # DP scaled.
        t0 = time.perf_counter()
        dp_keep, dp_outsource = dp_solver.solve(items, budget)
        dp_elapsed = time.perf_counter() - t0
        dp_times.append(dp_elapsed)

        # Greedy fractional.
        t0 = time.perf_counter()
        gr_keep, gr_outsource = greedy_solver.solve(items, budget)
        gr_elapsed = time.perf_counter() - t0
        greedy_times.append(gr_elapsed)

        # Compare outsource sets.
        dp_out_set = set(dp_outsource)
        gr_out_set = set(gr_outsource)

        if dp_out_set == gr_out_set:
            exact_agreements += 1

        jac = jaccard(dp_out_set, gr_out_set)
        jaccard_scores.append(jac)

        dp_val = compute_value_kept(items, dp_keep)
        gr_val = compute_value_kept(items, gr_keep)
        total_val = sum(it["value"] for it in items)
        gap_pct = (
            abs(dp_val - gr_val) / max(1, total_val) * 100
        )
        value_gaps.append(gap_pct)

        if dp_out_set != gr_out_set:
            disagreement_cases.append({
                "snapshot": snap_idx,
                "queue_size": queue_size,
                "dp_outsource_count": len(dp_outsource),
                "greedy_outsource_count": len(gr_outsource),
                "dp_value_kept": dp_val,
                "greedy_value_kept": gr_val,
                "value_gap_pct": round(gap_pct, 4),
                "jaccard": round(jac, 4),
                "dp_only": sorted(dp_out_set - gr_out_set),
                "greedy_only": sorted(gr_out_set - dp_out_set),
            })

    n = len(snapshots)
    return {
        "queue_size": queue_size,
        "num_snapshots": n,
        "exact_agreement_rate": round(exact_agreements / max(1, n), 4),
        "mean_jaccard": round(statistics.mean(jaccard_scores), 4) if jaccard_scores else 0.0,
        "mean_value_gap_pct": round(statistics.mean(value_gaps), 4) if value_gaps else 0.0,
        "max_value_gap_pct": round(max(value_gaps), 4) if value_gaps else 0.0,
        "p95_value_gap_pct": round(
            sorted(value_gaps)[int(0.95 * len(value_gaps))] if value_gaps else 0.0, 4
        ),
        "dp_mean_ms": round(statistics.mean(dp_times) * 1000, 3),
        "greedy_mean_ms": round(statistics.mean(greedy_times) * 1000, 3),
        "num_disagreements": len(disagreement_cases),
        "disagreement_cases": disagreement_cases[:5],  # Keep top 5 for JSON.
    }


# ---------------------------------------------------------------------------
# Iterative outsourcing comparison
# ---------------------------------------------------------------------------

def run_iterative(
    snapshots: list[list[dict]],
    queue_size: int,
    target_fraction: float,
) -> dict:
    """Compare dp_scaled vs fractional under iterative outsourcing.

    At each iteration, outsource the single request selected by the solver
    (budget = total_weight - 1), remove it, and repeat until the cumulative
    outsourced weight reaches target_fraction of the original total weight.

    Args:
        snapshots: List of queue snapshots.
        queue_size: Queue size label.
        target_fraction: Fraction of total weight to outsource.

    Returns:
        Summary dict.
    """
    dp_solver = KnapsackSolver(strategy="dp_scaled")
    greedy_solver = KnapsackSolver(strategy="fractional")

    per_iteration_agreement: list[float] = []
    total_value_gaps: list[float] = []
    total_iterations: list[int] = []

    for snap_idx, snapshot in enumerate(snapshots):
        items_orig = [
            build_item(
                req_id=f"q{queue_size}_s{snap_idx}_r{i}",
                prefill_tokens=r["num_prefill_tokens"],
                decode_tokens=r["num_decode_tokens"],
            )
            for i, r in enumerate(snapshot)
        ]

        total_weight_orig = sum(it["weight"] for it in items_orig)
        target_removed = total_weight_orig * target_fraction

        # Run both solvers independently, tracking per-step picks.
        dp_items = list(items_orig)
        gr_items = list(items_orig)
        dp_removed_weight = 0
        gr_removed_weight = 0
        dp_removed_ids: list[str] = []
        gr_removed_ids: list[str] = []

        agree_steps = 0
        iteration = 0
        max_iter = min(len(items_orig), MAX_ITERATIVE_STEPS)

        while iteration < max_iter:
            if dp_removed_weight >= target_removed and gr_removed_weight >= target_removed:
                break

            dp_pick = None
            gr_pick = None

            # DP step.
            if dp_removed_weight < target_removed and len(dp_items) > 1:
                total_w = sum(it["weight"] for it in dp_items)
                budget = max(1, total_w - 1)
                _, dp_outsource = dp_solver.solve(dp_items, budget)
                if dp_outsource:
                    dp_pick = dp_outsource[0]
                    picked_item = next(
                        it for it in dp_items if it["id"] == dp_pick
                    )
                    dp_removed_weight += picked_item["weight"]
                    dp_removed_ids.append(dp_pick)
                    dp_items = [it for it in dp_items if it["id"] != dp_pick]

            # Greedy step.
            if gr_removed_weight < target_removed and len(gr_items) > 1:
                total_w = sum(it["weight"] for it in gr_items)
                budget = max(1, total_w - 1)
                _, gr_outsource = greedy_solver.solve(gr_items, budget)
                if gr_outsource:
                    gr_pick = gr_outsource[0]
                    picked_item = next(
                        it for it in gr_items if it["id"] == gr_pick
                    )
                    gr_removed_weight += picked_item["weight"]
                    gr_removed_ids.append(gr_pick)
                    gr_items = [it for it in gr_items if it["id"] != gr_pick]

            if dp_pick is not None and gr_pick is not None:
                if dp_pick == gr_pick:
                    agree_steps += 1

            if dp_pick is None and gr_pick is None:
                break

            iteration += 1

        total_iterations.append(iteration)

        if iteration > 0:
            per_iteration_agreement.append(agree_steps / iteration)

        # Compare total value kept.
        dp_kept_val = sum(it["value"] for it in dp_items)
        gr_kept_val = sum(it["value"] for it in gr_items)
        total_val = sum(it["value"] for it in items_orig)
        gap = abs(dp_kept_val - gr_kept_val) / max(1, total_val) * 100
        total_value_gaps.append(gap)

    n = len(snapshots)
    return {
        "queue_size": queue_size,
        "target_fraction": target_fraction,
        "num_snapshots": n,
        "per_step_agreement_rate": round(
            statistics.mean(per_iteration_agreement), 4
        ) if per_iteration_agreement else 0.0,
        "mean_value_gap_pct": round(
            statistics.mean(total_value_gaps), 4
        ) if total_value_gaps else 0.0,
        "max_value_gap_pct": round(
            max(total_value_gaps), 4
        ) if total_value_gaps else 0.0,
        "mean_iterations": round(
            statistics.mean(total_iterations), 1
        ) if total_iterations else 0,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_single_shot_table(results: list[dict]) -> None:
    """Print a formatted table of single-shot comparison results."""
    header = (
        f"{'QueueSize':>9} | {'Agree%':>7} | {'Jaccard':>7} | "
        f"{'MeanGap%':>8} | {'MaxGap%':>8} | {'P95Gap%':>8} | "
        f"{'DP(ms)':>7} | {'Greedy(ms)':>10} | {'Disagree':>8}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("SINGLE-SHOT COMPARISON: dp_scaled vs fractional")
    print("(budget = total_weight - 1, forcing at least 1 outsource)")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['queue_size']:>9} | "
            f"{r['exact_agreement_rate']*100:>6.1f}% | "
            f"{r['mean_jaccard']:>7.4f} | "
            f"{r['mean_value_gap_pct']:>8.4f} | "
            f"{r['max_value_gap_pct']:>8.4f} | "
            f"{r['p95_value_gap_pct']:>8.4f} | "
            f"{r['dp_mean_ms']:>7.3f} | "
            f"{r['greedy_mean_ms']:>10.3f} | "
            f"{r['num_disagreements']:>8}"
        )
    print(sep)


def print_iterative_table(results: list[dict]) -> None:
    """Print a formatted table of iterative comparison results."""
    header = (
        f"{'QueueSize':>9} | {'Target%':>7} | {'StepAgree%':>10} | "
        f"{'MeanGap%':>8} | {'MaxGap%':>8} | {'MeanIter':>8}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("ITERATIVE COMPARISON: outsource one at a time")
    print("(each step: budget = total_weight - 1, pick lowest-value outsource)")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['queue_size']:>9} | "
            f"{r['target_fraction']*100:>6.0f}% | "
            f"{r['per_step_agreement_rate']*100:>9.1f}% | "
            f"{r['mean_value_gap_pct']:>8.4f} | "
            f"{r['max_value_gap_pct']:>8.4f} | "
            f"{r['mean_iterations']:>8.1f}"
        )
    print(sep)


def print_verdict(single_results: list[dict], iterative_results: list[dict]) -> None:
    """Print a summary verdict."""
    all_single_gaps = [r["mean_value_gap_pct"] for r in single_results]
    all_single_agree = [r["exact_agreement_rate"] for r in single_results]
    all_iter_gaps = [r["mean_value_gap_pct"] for r in iterative_results]
    all_iter_agree = [r["per_step_agreement_rate"] for r in iterative_results]

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(
        f"  Single-shot: mean agreement = "
        f"{statistics.mean(all_single_agree)*100:.1f}%, "
        f"mean value gap = {statistics.mean(all_single_gaps):.4f}%"
    )
    print(
        f"  Iterative:   mean step agreement = "
        f"{statistics.mean(all_iter_agree)*100:.1f}%, "
        f"mean value gap = {statistics.mean(all_iter_gaps):.4f}%"
    )

    avg_gap = statistics.mean(all_single_gaps + all_iter_gaps)
    if avg_gap < 0.01:
        print("  => Greedy sorting is EQUIVALENT to knapsack (gap < 0.01%).")
        print("     Recommendation: use fractional (greedy) for simplicity.")
    elif avg_gap < 0.1:
        print("  => Greedy sorting is NEAR-EQUIVALENT (gap < 0.1%).")
        print("     Recommendation: fractional is sufficient for production.")
    elif avg_gap < 1.0:
        print("  => Minor differences exist (gap < 1%).")
        print("     Recommendation: knapsack has marginal benefit; profile before choosing.")
    else:
        print("  => Knapsack provides meaningful benefit (gap >= 1%).")
        print("     Recommendation: keep dp_scaled for value optimization.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full comparison experiment."""
    print("Loading trace data...", flush=True)
    records = load_trace(TRACE_PATH, max_lines=50000)
    print(f"  Loaded {len(records)} records from {TRACE_PATH.name}", flush=True)

    # Sort by arrival time for snapshot extraction.
    records.sort(key=lambda r: r["arrived_at"])

    # Print trace statistics.
    prefills = [r["num_prefill_tokens"] for r in records]
    decodes = [r["num_decode_tokens"] for r in records]
    print(
        f"  Prefill tokens: min={min(prefills)}, "
        f"median={sorted(prefills)[len(prefills)//2]}, "
        f"max={max(prefills)}",
        flush=True,
    )
    print(
        f"  Decode  tokens: min={min(decodes)}, "
        f"median={sorted(decodes)[len(decodes)//2]}, "
        f"max={max(decodes)}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Single-shot experiment
    # -----------------------------------------------------------------------
    print("\nRunning single-shot comparison...", flush=True)
    single_results = []
    for qs in QUEUE_SIZES:
        snapshots = extract_queue_snapshots(
            records, qs, SNAPSHOTS_PER_SIZE, BURST_WINDOW_SEC
        )
        if not snapshots:
            print(f"  Skipping queue_size={qs}: not enough data", flush=True)
            continue
        result = run_single_shot(snapshots, qs)
        single_results.append(result)
        print(
            f"  queue_size={qs:>3}: "
            f"agree={result['exact_agreement_rate']*100:.0f}%, "
            f"gap={result['mean_value_gap_pct']:.4f}%, "
            f"dp={result['dp_mean_ms']:.2f}ms, "
            f"greedy={result['greedy_mean_ms']:.3f}ms",
            flush=True,
        )

    print_single_shot_table(single_results)

    # -----------------------------------------------------------------------
    # Iterative experiment
    # -----------------------------------------------------------------------
    print("\nRunning iterative comparison...", flush=True)
    iterative_results = []
    for qs in QUEUE_SIZES:
        snapshots = extract_queue_snapshots(
            records, qs, ITERATIVE_SNAPSHOTS_PER_SIZE, BURST_WINDOW_SEC
        )
        if not snapshots:
            continue
        for frac in ITERATIVE_TARGET_FRACTIONS:
            result = run_iterative(snapshots, qs, frac)
            iterative_results.append(result)
            print(
                f"  queue_size={qs:>3}, target={frac*100:.0f}%: "
                f"step_agree={result['per_step_agreement_rate']*100:.0f}%, "
                f"gap={result['mean_value_gap_pct']:.4f}%, "
                f"iters={result['mean_iterations']:.1f}",
                flush=True,
            )

    print_iterative_table(iterative_results)

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print_verdict(single_results, iterative_results)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "experiment": "knapsack_vs_sorting",
        "description": (
            "Compare dp_scaled (0/1 knapsack) vs fractional (greedy sorting "
            "by value/weight) for outsourcing decisions."
        ),
        "trace": str(TRACE_PATH),
        "pricing": {
            "input_price_per_million": INPUT_PRICE_PER_MILLION,
            "output_price_per_million": OUTPUT_PRICE_PER_MILLION,
        },
        "single_shot": single_results,
        "iterative": iterative_results,
    }

    # Strip disagreement details from JSON to keep file size reasonable.
    for r in output["single_shot"]:
        r["disagreement_cases"] = r.get("disagreement_cases", [])[:3]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
