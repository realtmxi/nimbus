# Experiment: Knapsack vs Greedy Sorting

## Motivation

Nimbus uses a 0/1 knapsack solver to decide which request to outsource from
the waiting queue. The knapsack maximizes the total value (API cost saved)
of locally-kept requests subject to a weight budget (cache displacement).

A natural question: **is the knapsack solver necessary, or would simply
sorting by value/weight ratio and picking greedily give the same result?**

Greedy sorting is O(n log n) and trivial to implement. Knapsack DP is
O(n * W) and more complex. If they produce identical decisions, we should
use the simpler one.

## Background: How Nimbus Uses Knapsack

When a TTFT violation is detected, Nimbus runs an iterative loop:

```
while TTFT_violation_exists():
    items = [knapsack_item(r) for r in waiting_queue]
    budget = total_weight - 1        # force at least 1 outsource
    keep_set, outsource_set = knapsack.solve(items, budget)
    outsource(outsource_set[0])      # kick 1 request
    remove from queue
    recheck violations
```

Each request becomes a knapsack item:
- **weight** = `prefill_tokens * decode_tokens` (cache displacement)
- **value** = API cost if outsourced (what we save by keeping it local)

The budget is set to `total_weight - 1` to guarantee at least one request
is outsourced per iteration.

## Experiment Design

### Two Solvers (already implemented in `routing/outsourcing/knapsack.py`)

| Solver | Strategy | How it works |
|--------|----------|-------------|
| **dp_scaled** | 0/1 Knapsack DP | Finds globally optimal subset within budget via dynamic programming. Scales weights to keep DP table tractable. |
| **fractional** | Greedy sorting | Sorts items by value/weight ratio (descending), greedily fills budget from top. This is the classical fractional knapsack / greedy approach. |

### Input: Queue Snapshots from Real Trace

From the ShareGPT+BurstGPT trace (200K requests), we extract queue snapshots
that simulate "what the waiting queue looks like at a given moment":

- Walk through the trace sorted by arrival time
- For each snapshot, take a contiguous window of `queue_size` requests
- Sweep queue sizes: 10, 20, 50, 100, 150, 200
- 20 snapshots per queue size

### Test 1: Single-Shot Decision

For each snapshot, both solvers make one decision with `budget = total_weight - 1`:

```
dp_keep, dp_outsource = dp_scaled.solve(items, budget)
gr_keep, gr_outsource = fractional.solve(items, budget)
```

We measure:
- **Exact agreement rate**: do both solvers outsource the same request?
- **Jaccard similarity**: overlap of outsource sets
- **Value gap**: |dp_value_kept - greedy_value_kept| / total_value
- **Solver latency**: wall-clock time per solve

### Test 2: Iterative Outsourcing

Simulates the actual Nimbus loop where requests are outsourced one at a time:

```
for each solver independently:
    while removed_weight < target_fraction * original_total_weight:
        solve with budget = total_weight - 1
        remove the outsourced request from the item list
        repeat
```

Target fractions swept: 10%, 25%, 50%.

We measure:
- **Per-step agreement rate**: at each iteration, did both pick the same request?
- **Cumulative value gap**: after all iterations, how much more value did DP retain?

### Why the Two Tests Matter

Single-shot tells us: "for one decision, does it matter?"
Iterative tells us: "when decisions compound, does the gap grow?"

If knapsack is only marginally better in single-shot but the gap compounds
in iterative mode, it is still important for Nimbus (which runs iteratively).

## Results

### Single-Shot

| Queue Size | Agreement | Mean Value Gap | Max Value Gap | DP Time | Greedy Time |
|-----------|-----------|---------------|--------------|---------|-------------|
| 10 | 0% | 11.8% | 20.7% | 9ms | 0.008ms |
| 20 | 0% | 7.3% | 12.7% | 20ms | 0.019ms |
| 50 | 0% | 3.7% | 8.0% | 48ms | 0.03ms |
| 100 | 0% | 1.8% | 4.0% | 105ms | 0.06ms |
| 150 | 0% | 0.8% | 1.4% | 150ms | 0.08ms |
| 200 | 5% | 0.4% | 0.8% | 201ms | 0.11ms |

### Iterative (selected)

| Queue Size | Target | Per-Step Agreement | Mean Value Gap |
|-----------|--------|-------------------|---------------|
| 10 | 25% | 0% | 21.4% |
| 20 | 25% | 0% | 33.4% |
| 50 | 25% | 0% | 24.8% |
| 50 | 50% | 0% | 39.1% |
| 100 | 25% | 0% | 20.3% |
| 200 | 25% | 0.8% | 3.8% |

## Analysis

### Why do they almost never agree?

Because LLM request workloads have **heterogeneous weight-to-value ratios**.
A request with large `prefill * decode` (high weight) might have low API cost
(low value), and vice versa. The greedy sorts by ratio and picks items that
"look efficient" individually, but misses combinations that pack better
globally.

Classic example:
```
Item A: weight=1000, value=50  (ratio=0.05)  <- large displacement, cheap
Item B: weight=100,  value=40  (ratio=0.40)  <- small displacement, medium
Item C: weight=80,   value=35  (ratio=0.44)  <- small displacement, medium
Budget = 1179

Greedy: keep C+B (value=75), outsource A
DP:     keep A+C (value=85), outsource B
DP wins by 13%.
```

Greedy is tricked by A's low ratio, but A is actually the best single item
to keep because it fills most of the budget alone.

### Why does the gap grow in iterative mode?

Each iteration, greedy makes a slightly suboptimal choice. The "wrong" request
stays in the queue, distorting future decisions. Over 10-20 iterations, these
errors compound, leading to 20-45% cumulative value gap.

### Greedy is 1000x faster

DP: 10-200ms per solve. Greedy: 0.01-0.1ms. For Nimbus's online decision loop
(which runs at each violation check, potentially multiple times per second),
DP latency is acceptable but not negligible.

## Conclusion

**Knapsack (dp_scaled) is justified.** The value gap is meaningful:
- 4-12% in single-shot (small queues)
- 20-45% in iterative mode

Greedy sorting is **not equivalent** to knapsack for this workload.
The heterogeneity of LLM request characteristics (prefill/decode distributions
are heavy-tailed and weakly correlated) creates exactly the conditions where
0/1 knapsack outperforms greedy.

**Recommendation**: Keep dp_scaled for production. The 10-200ms solve time
is acceptable for Nimbus's violation-check cadence.

## Artifacts

- Script: `scripts/analysis/exp_knapsack_vs_sorting.py`
- Data: `docs/images/knapsack_vs_sorting.json`
