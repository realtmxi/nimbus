# Nimbus online engine - corrected unique-prompt validation (Qwen3-32B)

**Date:** 2026-06-11  
**Branch:** `murphy/hybrid-routing`  
**Figure:** `docs/images/frontier_32b_fix_unique.png`  
**Raw summary:** `logs/sweep_32b_fix_unique/engine_summary.csv`

## Why this rerun exists

The earlier synthetic `sized` prompt mode generated every prompt as repeated
`hello` tokens. That made the workload a degenerate prefix-cache benchmark and
could leak cache state across policy runs. This rerun isolates synthetic prompt
content by adding a deterministic run/request salt while preserving the same
token-count metadata used by the scheduler.

The rerun also adds local KV reservation in real serving mode. vLLM metrics can
lag behind requests the harness has already admitted, so controller decisions now
use `max(observed_engine_kv, locally_reserved_prompt_tokens)`.

## Setup

- **Local serving:** real vLLM, Qwen3-32B, 32k-capable launch on gpu1.
- **Workload:** 80-request synthetic burst, unique `sized` prompts, prompt cap
  6000 tokens, fixed seed across policies.
- **Cloud:** simulated at 900 ms +/- 30%.
- **Controller parameters:** `slo_s=5`, `prefill_tput=2000`,
  `max_inflight=8`, `local_kv_tokens=90064`.
- **SLO:** TTFT p99 <= 5 s and 0% per-request SLO violations.

## Results

| policy | fraction | out% | cost$ | viol% | p50 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|
| **nimbus (adaptive, v2)** | self | 75.00 | **0.0717** | **0.00** | 2796.6 | **4955.0** |
| all_cloud | 1.00 | 100.00 | 0.0878 | 0.00 | 877.8 | 1164.4 |
| all_local | 0.00 | 0.00 | 0.0000 | 93.75 | 68737.0 | 153463.1 |
| cachedisp_oracle | 0.25 | 18.75 | 0.0295 | 75.00 | 32602.0 | 104895.1 |
| flop_oracle | 0.25 | 18.75 | 0.0295 | 75.00 | 37215.4 | 95492.5 |
| random | 0.25 | 30.00 | 0.0210 | 65.00 | 21998.8 | 105518.4 |
| cachedisp_oracle | 0.50 | 48.75 | 0.0607 | 41.25 | 14871.7 | 43061.7 |
| flop_oracle | 0.50 | 48.75 | 0.0587 | 36.25 | 14038.1 | 35428.3 |
| random | 0.50 | 51.25 | 0.0418 | 38.75 | 9317.1 | 60692.2 |
| cachedisp_oracle | 0.75 | 78.75 | 0.0828 | 1.25 | 898.8 | 5140.9 |
| flop_oracle | 0.75 | 76.25 | 0.0781 | 1.25 | 904.8 | 6409.8 |
| random | 0.75 | 80.00 | 0.0673 | 12.50 | 870.9 | 17190.4 |

## Takeaways

- Nimbus is the only non-all-cloud policy that reaches 0% SLO violations in this
  sweep.
- Nimbus meets the p99 SLO at `$0.0717`, about 82% of all-cloud cost.
- Fixed-fraction baselines still miss 0% violations even around 75-80%
  outsourced.
- The older "one third of all-cloud cost" headline should not be reused for this
  corrected unique-prompt experiment.

## Remaining evidence gap

This sweep validates the online adaptive controller against fixed-fraction
baselines under real vLLM, but it does not prove the CacheDisp-vs-FLOP weight
claim. Synthetic caps still make CacheDisp and FLOP selections too similar. That
claim needs a real trace with independently varying prompt and decode lengths.
