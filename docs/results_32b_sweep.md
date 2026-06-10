# Nimbus online engine — first real-GPU validation (Qwen3-32B)

**Date:** 2026-06-10 · **Branch:** `murphy/hybrid-routing` · **Figure:** `docs/images/frontier_32b.png`

## Setup
- **Local serving:** Qwen3-32B on RTX PRO 6000 Blackwell (gpu1), real vLLM v0.19.0
  (Triton attention backend + `--enforce-eager` for Blackwell; `--max-model-len 8192`,
  `--gpu-memory-utilization 0.9`). Reported KV capacity **90,064 tokens**.
- **Workload:** 200-request synthetic burst, `sized` prompts capped at 6000 tokens,
  decode ∈ {64,256,512,1024}, prompt/decode independent. Same fixed seed across all
  policies → identical workload.
- **Decision signals:** real — `kv_avail` read live from vLLM `/metrics`
  (`metrics_read_failures=0`). **Cloud:** simulated (900 ms ± 30 %, $0.15 / $1.20 per
  Mtok in/out). **SLO:** per-request TTFT < 5 s.
- Harness: `experiments/run_engine.py` (nimbus + baselines through one loop + shared
  SimCloud cost meter + iso-accounting).

## Results (19 runs)

| policy | frac | out% | **viol%** | cost$ | p50 ms | p99 ms |
|---|---|---|---|---|---|---|
| all_local | 0.0 | 0.0 | **50.5** | 0.0000 | 5360 | 14653 |
| all_cloud | 1.0 | 100 | **0.0** | 0.2285 | 885 | 1169 |
| **nimbus (adaptive, v2)** | self | 26.0 | **0.0** | **0.0717** | **206** | 4844 |
| cachedisp_oracle | 0.30 | 31.0 | 11.5 | 0.1125 | 743 | 8486 |
| flop_oracle | 0.30 | 34.5 | 9.0 | 0.1189 | 737 | 6929 |
| random | 0.30 | 35.0 | 9.5 | 0.0792 | 753 | 7408 |
| fifo | 0.30 | 30.0 | 38.5 | 0.0635 | 2546 | 11331 |

(cachedisp/flop/fifo/random also run at 0.15/0.20/0.25 — see `logs/sweep_32b/engine_summary.csv`.)

## Headline (validated)
**Nimbus is the only policy besides all-cloud to reach 0 % SLO violations, and it does
so at $0.072 — 31 % of all-cloud's $0.229.** No fixed-fraction baseline reaches 0 %
violations even at 30 %+ outsource. Adaptive online outsourcing driven by real KV
pressure Pareto-dominates fixed-fraction selection; the low p50 (206 ms) shows the
controller keeps the kept-local requests fast while shedding only the overflow.

Directional invariants all hold: all_local collapses (50 % viol), all_cloud is
violation-free but 3× the cost, nimbus sits in between but on the efficient frontier.

## Honest gaps (next iterations)
1. **CacheDisp vs FLOP weight is inconclusive here** (near-tied across fractions). The
   8 k context cap truncates the long-input heavy tail that separates them. The *weight*
   claim (the paper's core) needs a long-context / real-trace workload — next.
2. **Weight constants not calibrated:** `prefill_tput=50000`, `tpot=30 ms` are defaults,
   not measured on this 32B. They size per-request weights and the deadline trigger.
   Calibrate via `experiments/profile_serving_tpot.py`.
3. **Cloud is simulated.** Relative comparison is fair (all policies share it), but
   absolute cost/cloud-TTFT need `--cloud real`.
4. **Budget-units open question** (token·s vs tokens) still unresolved; sweep suggests
   nimbus's self-selected 26 % is efficient (0 viol at low cost), not over-shedding.

---

## Iteration 2 — long-context probe (32k ctx, heavy-tail prompts ≤28k, n=40)

Relaunched 32B at `--max-model-len 32768` to admit the long-input tail, then ran
cachedisp vs flop at matched fraction (`logs/weight_32b/`). **Two findings, both
important:**

### (a) Synthetic cannot test the CacheDisp-vs-FLOP weight claim
`cachedisp_oracle@0.25` and `flop_oracle@0.25` came out **byte-identical** (22.5 % out,
$0.0458, p99 1112 ms) — the prompt-length cap collapses the heavy tail to a single value
(28000), so both oracles select the same requests. **The weight claim (paper C2/C3) is
NOT validated by synthetic data; it requires a real trace** (ShareGPT/BurstGPT/Rednote)
with continuous, independent (prompt, decode) variation. Recommend
`scripts/download_data.sh` + re-run, or the offline selection analysis
(`scripts/analysis/exp_oracle_analysis.py`) on real token counts.

### (b) 🚩 Real bug: nimbus underperforms all_local on near-uniform large-prompt load
On this workload all_local is fine (0 % viol, service p50 323 ms) — 40 huge requests
don't saturate when admission backpressure paces them. But **nimbus gets 70 % violations,
service p50 13.8 s**, while *outsourcing 25 %*. Request-level diagnosis:

| | nimbus local-kept (n=30) | all_local (n=40) |
|---|---|---|
| queue_delay (harness) | med 7 ms (max 28 ms) | med 6 ms (**max 2807 ms**) |
| service TTFT (vLLM)   | **med 13 842 ms** | med 323 ms |

Root cause — **two control gates fight**: (1) nimbus's knapsack sheds the overflow, so
the waiting queue looks short → (2) the admission backpressure loop then admits all 30
kept requests almost immediately (no pacing, queue_delay ~7 ms) → vLLM internally queues
them (only ~3 of these 28k-prompt requests fit in 90 k-token KV) → 13.8 s service. In
all_local the queue stays long, so backpressure *paces* dispatch (queue_delay up to 2.8 s)
and vLLM is never overloaded. Nimbus over-triggers because the **token·s budget mis-sizes
capacity for huge requests** (a single 28k/512 request's weight ≈ 445k token·s > the whole
~405k budget), so `Σweight > budget` fires even when the system has real slack.

**Implications for Murphy (design decisions, not for me to finalize unilaterally):**
- This is concrete evidence for the **budget-in-tokens** reformulation (footprint =
  prompt_tokens vs KV capacity), with token·s used only as shed-priority — it would bound
  shedding to physical KV and stop the spurious over-trigger.
- The **knapsack ↔ admission-backpressure relationship must be specified**: today they
  double-control admission and conflict. Either the engine should own pacing, or shedding
  should feed the admission rate rather than just removing items.
- The realistic-workload result (8k mixed, §above) is unaffected and remains strong;
  the pathology is specific to near-uniform very-large-prompt load.
