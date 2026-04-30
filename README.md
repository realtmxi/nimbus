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

`experiments/run_offload_strategies.py` implements 9 strategies behind a unified
`OffloadStrategy` interface:

| Strategy | Description | Type |
|----------|-------------|------|
| RandomRequestStrategy | Outsource each request with probability `fraction` | Baseline |
| PressureGatedStrategy | Outsource only when KV pressure exceeds threshold | Baseline |
| SessionAwareStrategy | Outsource entire sessions (preserve prefix continuity) | Baseline |
| SizeOutsourceLongStrategy | Outsource largest-prefill requests | Baseline |
| SizeOutsourceShortStrategy | Outsource smallest-prefill requests (worst case) | Baseline |
| FlopBasedStrategy | Weight = `prefill_flops + 0.6 * decode_flops` | Nimbus v0 |
| **CacheDispStrategy** | Weight = `prefill_tokens * decode_tokens` | **Nimbus v1** |
| GatedSessionAwareStrategy | Pressure gate + session-sticky decisions | Hybrid baseline |

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
    --strategies flop_based cache_disp session_aware oracle_size \
    --output-dir logs/cachedisp_knee
```

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
