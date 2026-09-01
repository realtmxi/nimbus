# Nimbus: Cache-Displacement-Aware Outsourcing for Hybrid LLM Inference

Nimbus is a hybrid LLM inference research system that combines local GPU
deployment with cloud APIs. Its current design separates **when/how much** to
outsource (a predicted-TTFT trigger) from **which requests** to outsource (a
cost-aware resource-residence selector).

**Historical motivation result:** at a fixed 15% outsource fraction, the early
cache-displacement prototype achieved 11.7x lower TTFT p50 than FLOP-based
outsourcing on one ShareGPT+BurstGPT / Qwen2.5-7B setup. This is not the current
audited headline or evidence for the shipped trigger.

Start with the self-contained current algorithm and evidence summary:
[`docs/nimbus_algorithm_and_results_2026-07.zh-CN.md`](docs/nimbus_algorithm_and_results_2026-07.zh-CN.md)
(Chinese).
The executable experiment harness is `router/`; the historical shipped-v3 KV
baseline is preserved in
[`docs/notion_algorithm_design_v3.md`](docs/notion_algorithm_design_v3.md).
The top-level `nimbus/` package is the legacy FLOP/online-knapsack prototype;
it is retained only for historical analysis and is not imported by the router.

**Algorithm status (2026-07-21):** after selecting `--policy nimbus`, the
default trigger/selector combination
remains `kv_gap + cost_disp_current`, but the completed dense-32B no-cache full
cell falsified it as a sufficient TTFT-safety trigger on that workload (94.7%
of retained-local requests violated 5 s). The experimental direction is
`ttft_pred + cost_cachedisp_old`: old V2 survives as an ordering signal, not as
a token-second capacity. In the completed real-cloud A→C pair, old/current
selectors had 11/3 overall 5 s violations and zero retained-local violations;
the comparison remains unresolved because it is one fixed order and the
difference is below the preregistered 1 pp threshold. Support-envelope fallback
and an online decode estimator are still missing, so no default has changed.
See the current summary above or
[`docs/v3_experiments_2026-07.md`](docs/v3_experiments_2026-07.md) for the
frozen evidence trail.
中文时间线、每次实验目的和结果总账见
[`docs/nimbus_experiment_ledger_2026-07.zh-CN.md`](docs/nimbus_experiment_ledger_2026-07.zh-CN.md)。

## Repository Structure

```
router/                          Current Nimbus v3 runtime
  run.py                         External queue, dispatcher, live KV monitor
  nimbus.py                      KV-gap + cost/displacement shedding policy
  common.py                      Trace replay, endpoint, billing, summaries

nimbus/                          Legacy v0 online-knapsack prototype
  decision.py                    OutsourcingEngine: iterative knapsack loop
  knapsack.py                    KnapsackSolver: dp_scaled, fractional, dp, random
  violation_detection.py         TTFT SLO violation detector
  candidate_selection.py         Candidate request filtering
  cost_calculator.py             API cost model
  flop_calculator.py             Compute cost (Nimbus v0)
  profiled_flop_calculator.py    Profiled compute cost
  request.py                     OutsourcingRequestInfo dataclass
  request_tracker.py             Outsourced request tracking
  queue.py                       WaitingQueueInterface
  adapters.py                    SGLangWaitingQueueAdapter

experiments/                     End-to-end trace replay against live SGLang
  run_offload_strategies.py      9 outsourcing strategies + replay infra
  metrics_collector.py           Time-series KV/queue/throughput metrics
  run_cachedisp_knee.sh          Knee sweep across outsource fractions
  run_cachedisp_repeats.sh       Multi-seed repeats for error bars

scripts/analysis/                Offline/historical analysis (no GPU required)
  exp_knapsack_vs_sorting.py     Legacy knapsack vs greedy comparison
  exp_oracle_analysis.py         Why size-based oracle is suboptimal
  exp_motivation_figure.py       Memory-bottleneck motivation figure
  plot_cachedisp_knee.py         Knee sweep plot

docs/
  nimbus_algorithm_and_results_2026-07.zh-CN.md Current algorithm/evidence entry point
  notion_algorithm_design_v3.md Historical shipped-v3 KV specification
  nimbus_v2_pitch.md             Paper pitch (Cache Displacement story)
  tcpo_oracle_design.md          Trace-Clairvoyant Pressure Oracle design
  exp_knapsack_vs_sorting_design.md

scripts/download_data.sh         Fetch trace data from gpu1
data/                            Local trace files (gitignored)
```

## Historical fixed-fraction experiment strategies

`experiments/run_offload_strategies.py` implements 11 strategies behind a unified
`OffloadStrategy` interface, organized in three tiers. These belong to the
legacy fixed-fraction experiment path and are **not** the current `router/`
trigger/selector policy.

### Intuitive baselines (oblivious / extremes)

| Strategy | CLI name | Description |
|----------|----------|-------------|
| AllLocalStrategy | `all_local` | No outsourcing (cost lower bound, latency upper bound) |
| AllCloudStrategy | `all_cloud` | Outsource everything (cost upper bound, no local pressure) |
| FIFOStrategy | `fifo` | Outsource the first N% requests by arrival order |
| RandomRequestStrategy | `random_request` | Outsource each request i.i.d. with probability `fraction` |

### System-state baselines (use system signals, not request features)

| Strategy | CLI name | Description |
|----------|----------|-------------|
| PressureGatedStrategy | `pressure_gated` | Outsource only when KV pressure exceeds threshold |
| SessionAwareStrategy | `session_aware` | Outsource entire sessions to preserve prefix continuity |
| GatedSessionAwareStrategy | `gated_session_aware` | Pressure gate + session-sticky decisions |

### Feature-aware baselines (use per-request features)

| Strategy | CLI name | Description |
|----------|----------|-------------|
| SizeOutsourceLongStrategy | `size_long` | Outsource largest-prefill requests |
| SizeOutsourceShortStrategy | `size_short` | Outsource smallest-prefill requests (worst case) |
| FlopBasedStrategy | `flop_based` | Weight = `prefill_flops + 0.6 * decode_flops` (Nimbus v0) |

### Historical CacheDisp prototype

| Strategy | CLI name | Description |
|----------|----------|-------------|
| **CacheDispStrategy** | `cache_disp` | Weight = `prefill_tokens * decode_tokens` (memory-time product) |

## Quick Start

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download trace data
```bash
bash scripts/download_data.sh           # Primary trace (~491 MB)
bash scripts/download_data.sh --all     # All traces (~20 GB)
```
See `data/README.md` for details on available datasets.

### Run Offline Analysis (no GPU needed)
```bash
# Knapsack vs greedy sorting comparison
python scripts/analysis/exp_knapsack_vs_sorting.py

# Why size-based oracle is suboptimal
python scripts/analysis/exp_oracle_analysis.py

# Motivation: memory is the bottleneck
python scripts/analysis/exp_motivation_figure.py
```

### Run End-to-End Knee Sweep (requires SGLang server)
```bash
# Start SGLang on a GPU box, then:
python experiments/run_offload_strategies.py \
    --sglang-url http://localhost:8200 \
    --mode knee \
    --fractions 0.0 0.15 0.20 0.25 0.30 0.35 0.50 \
    --strategies all_local all_cloud fifo random_request \
                 flop_based cache_disp session_aware size_long \
    --output-dir logs/cachedisp_knee
```

Note: `all_local` and `all_cloud` ignore the `fraction` argument and run
identically across all fractions; including them once at any fraction is
sufficient for cost/latency reference points.

## Data

Trace files are not committed (multi-GB). Use `scripts/download_data.sh` to
fetch them from the out-of-band configured `$JSCRATCH/workloads/`. See
`data/README.md` for schema and dataset details.

- ShareGPT+BurstGPT: 200K requests with `block_hash_ids` for prefix overlap
- RouteWise traces: long-context rednote agent + production freeinference logs
