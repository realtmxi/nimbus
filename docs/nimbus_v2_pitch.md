# Nimbus v2: Cache Displacement for Hybrid LLM Inference

**Target**: EuroSys
**Authors**: Murphy, Yiyan Zhai, Yiyu Liu, Juncheng Yang
**Date**: April 2026

---

## One-Sentence Thesis

Hybrid LLM inference systems outsource the wrong requests because they optimize for
compute (FLOPs) when the real bottleneck is cache displacement (memory × time).

---

## Problem

Local LLM deployment is cheap but limited. When burst traffic exceeds local capacity,
some requests must be outsourced to cloud APIs. The question: **which requests to outsource?**

Existing systems (including our Nimbus v1) use **FLOP cost** as the outsourcing weight:
outsource the most compute-expensive requests first to free the most GPU cycles.

**This is wrong.** In prefix-caching LLM serving, the dominant cost is not compute but
**cache displacement** — how much a request degrades the system's ability to serve future
requests from cache.

---

## Key Insight: Cache Miss >> Batch Penalty (152×)

We measured two penalties on Qwen2.5-7B (RTX PRO 6000 Blackwell):

| Penalty | Magnitude | What it means |
|---|---|---|
| **Cache miss** (prefix cache evicted) | **251× TTFT** | Losing cache is catastrophic |
| **Batch contention** (larger batch) | **1.65× TPOT** | Sharing GPU is mild |
| **Ratio** | **152×** | Cache preservation is 152× more important than compute saving |

So the right outsourcing objective is: **preserve the prefix cache**, not save compute.

---

## Bistability: Why Cache Collapse Is a Cliff, Not a Slope

Prefix-caching LLM serving exhibits bistability — two stable states with a sharp,
hysteretic phase transition:

- **Healthy**: high cache hit → fast service → short queue → low memory pressure →
  cache preserved → high cache hit (self-reinforcing)
- **Saturated**: cache evicted → slow service → queue buildup → memory pressure →
  more eviction → cache gone (self-reinforcing)

**Experimental evidence** (Qwen2.5-7B, ShareGPT+BurstGPT trace, 28K requests):

| Metric | Value |
|---|---|
| Phase transition | 251× TTFT degradation |
| Hysteresis gap | 470× at same outsource fraction |
| Policy invariance | All routing policies equivalent under saturation |
| Session coherence | 37× better than oracle request-level admission |

This is not gradual degradation. It is a feedback-driven cliff. Once you fall off,
you cannot climb back without drastic action.

**Classical parallel**: Denning's thrashing (1968) — same feedback loop between
working set and page faults in virtual memory systems.

---

## Cache Displacement = prefill × decode

A request r occupies KV_memory(r) ∝ prefill_tokens in GPU memory for the duration
of its decode phase = decode_tokens steps. Total resource occupancy:

```
cache_displacement(r) = ∫₀ᵈᵉᶜᵒᵈᵉ KV_memory(r) dt
                      = KV_per_token × prefill_tokens × decode_tokens
```

This is the **memory-time product** — the integral of resource occupancy over time.
Multiplication falls out of the integral, not an arbitrary design choice.

**Classical parallels**: Denning's page residence time (1968), TCP bandwidth-delay
product, job scheduling area-under-curve.

---

## FLOP-Based Picks the Wrong Requests

FLOP weight is dominated by prefill² (quadratic attention). It outsources
long-input/short-output requests (big compute, small cache displacement).

Cache displacement outsources high memory×time requests (large KV held for long decode).

On real coding agent traces (Rednote, 10K requests):

| Metric | Value |
|---|---|
| I/O correlation | 0.071 (near zero — long input ≠ long output) |
| Top-10% selection overlap | **11%** (89% different requests!) |
| Decode relief at 5% outsource | CacheDisp **6.3×** more than FLOP |

The two strategies are nearly **orthogonal** on production workloads.

### Why multiply, not add/max/other? — 6-Way Ablation

| Weight Function | CacheDisp Removed | Rank |
|---|---|---|
| **prefill × decode (ours)** | **262M** | **#1** |
| decode only | 201M | #2 |
| prefill² + decode² | 140M | #3 |
| prefill + decode | 139M | #4 |
| FLOPs (Nimbus v1) | 132M | #5 |
| prefill only | 117M | #6 |

Multiply captures both dimensions (memory footprint × time held). Additive methods
are dominated by whichever dimension is larger, losing the other.

---

## End-to-End Results

**Setup**: Qwen2.5-7B-Instruct, RTX PRO 6000 Blackwell 96GB, ShareGPT+BurstGPT
trace (28K requests), live SGLang server, trace replay at 5× speedup.

### TTFT p50 (ms) by Outsource Fraction

| Fraction | Session-aware | FLOP-based | **CacheDisp** | Oracle (size) | **CD/FLOP** |
|---|---|---|---|---|---|
| 0% | 143,395 | 103,736 | 99,395 | 106,101 | 1.0× |
| **15%** | 35,344 | **20,572** | **1,756** | 48,466 | **11.7×** |
| 20% | 19,372 | 466 | **277** | 1,474 | 1.7× |
| 25% | 5,324 | 161 | **108** | 225 | 1.5× |
| 30% | 416 | 96 | 72 | 103 | 1.3× |
| 50% | 68 | 58 | 58 | 59 | 1.0× |

**Hero number**: At 15% outsource, CacheDisp achieves 1.8s TTFT while FLOP is still
at 20.6s — **11.7× improvement** from selecting the right requests to outsource.

### Mechanism Evidence

At the critical 15% fraction:

| Metric | CacheDisp | FLOP-based | Session-aware |
|---|---|---|---|
| KV utilization p90 | **12.1%** | 19.5% | 27.1% |
| Max queue depth | **18** | 1,482 | 2,683 |

CacheDisp keeps memory pressure low enough to preserve the prefix cache.
FLOP-based still triggers queue buildup (1,482 peak) → cache eviction → bistability.

### Surprise: Oracle (prefill size) Is Worst at 15%

Oracle outsources the largest KV requests (long prefill, but short decode).
This frees memory momentarily but doesn't reduce decode batch occupancy time.
CacheDisp outsources the highest memory×time product — freeing both dimensions.

### Dollar Efficiency

To meet TTFT p99 < 5s SLO:
- FLOP-based needs ~30% outsource
- CacheDisp needs ~25% outsource
- **CacheDisp saves ~17% API cost** at equivalent SLO

---

## What Changed in the Code

One line in `routing/outsourcing/decision.py`:

```python
# Before (Nimbus v1 — FLOP-based):
weight = int(prefill_flops + 0.6 * decode_flops)

# After (Nimbus v2 — Cache Displacement):
weight = int(req.prefill_tokens * req.estimated_decode_tokens)
```

The rest of the Nimbus system (TTFT predictor, iterative knapsack, prefix cache
awareness) remains unchanged.

### Output Token Estimation Is Not a Problem

At request arrival, decode_tokens is unknown. But ranking robustness analysis shows:

| Estimation noise | Spearman ρ with true ranking | Top-10% overlap |
|---|---|---|
| 1.5× noise | 0.999 | 91.6% |
| 3× noise | 0.992 | 80.4% |
| 5× noise | 0.984 | 76.6% |
| **FLOP-based** | **0.932** | **53.1%** |

Even 5× noisy estimates produce better ranking than exact FLOPs.
In practice, use `max_tokens` (client-specified) or session historical mean.

---

## Paper Contributions

| # | Contribution | Type | Key Number |
|---|---|---|---|
| C1 | Cache miss >> batch penalty (152×) | Measurement | 152× ratio |
| C2 | Cache Displacement metric (memory-time product) | Theory + Ablation | #1 in 6-way ablation |
| C3 | FLOP-based picks wrong requests | Key Finding | 89% different, 11.7× E2E gap |
| C4 | Bistability in prefix-caching serving | Formalization | 251× TTFT, 470× hysteresis |
| C5 | Nimbus v2 system | System | 11.7× at 15%, 17% cost saving |
| C6 | FreeInference + production traces | Infrastructure | 55K LOC, 3 traces |

---

## Related Work Positioning

- **Mooncake, Preble, SGLang Router**: Cache-aware routing — decide WHERE to send.
  We decide WHAT to outsource. Orthogonal.
- **vLLM, SGLang, FlashAttention**: Serving engines — improve local throughput.
  We decide when local is not enough.
- **Metastable failures (OSDI'22)**: General framework. We show a specific instance
  in LLM serving and exploit it for outsourcing.
- **Splitwise, DistServe**: Prefill/decode disaggregation. Orthogonal axis.

---

## Experiment Status

- [x] Bistability measurement (251×, 470× hysteresis) — Done
- [x] 6-way weight function ablation — Done
- [x] DI reversal on 3 traces — Done
- [x] E2E knee sweep (4 strategies × 7 fractions) — Done
- [x] 3-panel mechanism figure (TTFT + KV util + queue depth) — Done
- [x] Output estimation robustness analysis — Done
- [ ] Near-knee repeat runs for error bars — Running (~9h)
- [ ] Rednote trace validation (optional, expected gap larger)
- [ ] Nimbus v2 integration test (knapsack + TTFT predictor)

---

## Key Figures

- `docs/images/cachedisp_knee_7b_3panel.pdf` — Hero figure (TTFT + mechanism)
- `docs/images/cachedisp_knee_7b.pdf` — 2-panel (TTFT p50 + p99)

## Data

- Knee sweep results: gpu1 `logs/cachedisp_knee1/`
- Repeat runs: gpu1 `logs/cachedisp_repeats/` (running)
- Bistability data: gpu1 `logs/phase2_7b_rednote/`
