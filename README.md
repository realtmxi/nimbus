# Nimbus: Cache-Displacement-Aware Outsourcing for Hybrid LLM Inference

Nimbus is a hybrid LLM inference system that combines local GPU deployment with
serverless cloud APIs. When burst traffic exceeds local capacity, Nimbus
intelligently selects which requests to outsource to cloud APIs based on their
**cache displacement** — the product of KV memory footprint and residence time
— rather than compute cost (FLOPs).

**Key result**: At 15% outsource fraction, Nimbus achieves **11.7x lower TTFT
p50** than FLOP-based outsourcing on the ShareGPT+BurstGPT trace
(Qwen2.5-7B on RTX PRO 6000 Blackwell).

## Repository Structure

```
nimbus/                          Core algorithm (Python module)
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

scripts/analysis/                Offline analysis (no GPU required)
  exp_knapsack_vs_sorting.py     Knapsack vs greedy comparison
  exp_oracle_analysis.py         Why size-based oracle is suboptimal
  exp_motivation_figure.py       Memory-bottleneck motivation figure
  plot_cachedisp_knee.py         Knee sweep plot

docs/
  nimbus_v2_pitch.md             Paper pitch (Cache Displacement story)
  tcpo_oracle_design.md          Trace-Clairvoyant Pressure Oracle design
  exp_knapsack_vs_sorting_design.md

scripts/download_data.sh         Fetch trace data from gpu1
data/                            Local trace files (gitignored)
```

## Outsourcing Strategies (Baselines + Ours)

`experiments/run_offload_strategies.py` implements 11 strategies behind a unified
`OffloadStrategy` interface, organized in three tiers:

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

### Ours

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
fetch them from `/scratch/murphy/workloads/` on gpu1. See `data/README.md` for
schema and dataset details.

- ShareGPT+BurstGPT: 200K requests with `block_hash_ids` for prefix overlap
- RouteWise traces: long-context rednote agent + production freeinference logs

## Citation

```
@inproceedings{nimbus2026,
  title  = {Nimbus: Cache-Displacement-Aware Outsourcing for Hybrid LLM Inference},
  author = {Murphy and Yiyan Zhai and Yiyu Liu and Juncheng Yang},
  booktitle = {EuroSys},
  year = {2026}
}
```
