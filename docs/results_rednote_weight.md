# Weight comparison on REAL Rednote trace (Qwen3-32B) — honest negative finding

**Date:** 2026-06-11 · **Branch:** `murphy/hybrid-routing`
**Figure:** `docs/images/rednote_weight_32b.png` · **Data:** `logs/rednote_weight/engine_summary.csv`

## Why this run
Synthetic prompts cannot separate CacheDisp from FLOP (caps collapse the heavy
tail). This run uses the **real Rednote coding-agent trace**
(Rednote production logs, ~15 GB; staged on the GPU host), the workload the
pitch cites for the strongest weight numbers. Slice: 80 real requests, real prompt
text, prefill 2k–14.5k (med 7148), **decode med 29 / max 4072**, uniform burst
arrivals, real vLLM Qwen3-32B @ 32k ctx, KV cap 90,064, sim cloud, SLO 5s.

## Offline mechanism check (full trace, 17,793 reqs) — claim looks supported
- corr(prefill, decode) = 0.06 (independent), as the pitch says.
- CacheDisp vs FLOP **top-10% selection overlap = 34.7%** (65% different requests).
- CacheDisp frees **1.8× more decode tokens** than FLOP at top-10%.

## End-to-end result (the slice) — claim does NOT hold here

| policy | frac | out% | viol% | cost$ | p99 ms |
|---|---:|---:|---:|---:|---:|
| all_local | 0.0 | 0 | 80.0 | 0.0000 | 106695 |
| all_cloud | 1.0 | 100 | 0.0 | 0.1069 | 1164 |
| **nimbus v2 (adaptive)** | self | 73.8 | **1.2** | 0.0904 | 5443 |
| cachedisp (v1) | 0.5 | 48.8 | 41.2 | 0.0598 | 22290 |
| flop | 0.5 | 48.8 | 40.0 | 0.0725 | 44266 |
| cachedisp (v1) | 0.7 | 70.0 | 13.8 | 0.0831 | 18536 |
| **flop** | 0.7 | 70.0 | **5.0** | 0.0920 | 6007 |
| cachedisp (v1) | 0.8 | 81.2 | 3.8 | 0.0932 | 5538 |
| **flop** | 0.8 | 81.2 | **0.0** | 0.0990 | 1164 |

**At matched fraction, FLOP beats CacheDisp on this slice (0.7: 5.0% vs 13.8% viol;
0.8: 0.0% vs 3.8%).** This is the OPPOSITE of paper claim C2/C3 on this workload.

## Why the offline and end-to-end disagree (root cause)
CacheDisp weight = prefill × **decode**. This slice has **decode median 29** —
near-constant tiny outputs — so the decode factor carries almost no discriminative
signal and CacheDisp collapses toward ~prefill. Under this prefill-bound regime,
FLOP (∝ prefill²) targets the giant-prefill requests that dominate TTFT more
sharply, so it sheds the right ones first. **The CacheDisp advantage requires
meaningful decode-length variance** (long, varied outputs holding KV for many
steps). The offline "1.8× more decode freed" was computed over the full 17k-request
trace; the 80-request burst slice does not preserve that decode spread.

## Honest status of the weight claim (C2/C3)
- **NOT validated; partially contradicted** on a real short-output slice.
- It is plausibly real on **long-decode** workloads (where memory×time genuinely
  separates from prefill²) — but that must be **shown**, not assumed. Next: build a
  slice with high decode variance (filter Rednote/ShareGPT for decode ≫ 0 spread),
  or run the multi-turn long-generation regime.

---

# UPDATE (same day): hypothesis CONFIRMED on a decode-variance slice

Built the test the hypothesis demands: real ShareGPT slice with genuine decode
variance (n=80, prefill med 633 / max 5687, decode med 286 / max 778, **decode
cv = 0.54**; cachedisp-vs-flop top-25% selection overlap 65% → separable).
Same server/params as above. Data: `logs/decodevar_weight/engine_summary.csv`.

| policy | frac | out% | viol% | cost$ | p99 ms |
|---|---:|---:|---:|---:|---:|
| all_local | 0.0 | 0 | 87.5 | 0.0000 | 95916 |
| all_cloud | 1.0 | 100 | 0.0 | 0.0388 | 1164 |
| **nimbus v2 (adaptive)** | self | 66.2 | **0.0** | **0.0255** | 4975 |
| **cachedisp (v1)** | 0.5 | 48.8 | **6.2** | 0.0258 | **8027** |
| flop | 0.5 | 48.8 | **38.8** | 0.0224 | 23836 |
| cachedisp (v1) | 0.25 | 23.8 | 63.7 | 0.0141 | 49361 |
| flop | 0.25 | 23.8 | 65.0 | 0.0127 | 60537 |

**When decode varies, CacheDisp beats FLOP decisively at matched fraction:
6.2% vs 38.8% violations (6.3× lower), p99 8.0s vs 23.8s (3× lower).**

## Final story (regime-conditional — stronger than the original blanket claim)

| workload regime | winner (fixed frac) | evidence |
|---|---|---|
| decode-length VARIES (chat/ShareGPT) | **CacheDisp** (6.3× lower viol) | this table |
| decode ~constant & tiny (agent/Rednote) | FLOP | table above |
| **both** | **adaptive nimbus v2** (0% and 1.2% viol; only 0-viol non-all-cloud policy on every workload tested) | both tables |

The v2 weight `prompt × (prefill_time + decode_time)` interpolates the two regimes
(its prefill-time term covers the Rednote case; its decode term covers ShareGPT),
which is why adaptive nimbus wins on both. **Recommended paper framing: lead with
residence-time (v2) + adaptive control; present prefill×decode (v1) as the
special case valid when decode varies, with the regime analysis as a contribution
rather than a caveat.**

Figure: `docs/images/nimbus_regime_comparison.png` (two-panel, both real workloads).

Caveats unchanged: n=80 single-seed (needs repeats/error bars), `prefill_tput=2000`
uncalibrated (affects nimbus's absolute outsource rate, not the fixed-frac
comparisons), cloud simulated.

## What DOES hold
- **Adaptive online control (nimbus) is robust**: it self-selects 73.8% outsource
  and lands at the 5s SLO (1.2% viol, $0.090 ≈ 85% of all-cloud) WITHOUT a tuned
  fraction knob, on a workload where the fixed-fraction baselines need 80% to get
  near 0%. The adaptive-vs-fixed story is the solid contribution; the
  weight-function story is workload-dependent and currently unproven.
- Caveat: `prefill_tput=2000` is uncalibrated and inflates absolute outsource rates;
  it does not affect the fixed-fraction cachedisp-vs-flop comparison (same for both).
