# TCPO: Trace-Clairvoyant Pressure Oracle for Nimbus

## 1. Why We Need an Offline Oracle

Nimbus v1 uses `prefill_tokens * decode_tokens` (cache displacement) as the
knapsack weight to decide which requests to outsource. This achieves 11.7x TTFT
improvement over FLOP-based at 15% outsource fraction.

But a natural question remains: **how close is CacheDisp to the best possible
outsourcing decision?** Without an offline optimal baseline, we cannot answer this.

### What existing work does

Recent papers (Jaillet et al. 2025, Ao et al. 2025, Sorted-F 2025) formulate
offline optimal scheduling for LLM inference with KV cache constraints. However,
they all assume a **single local cluster** and optimize batch scheduling order.
None considers **outsourcing to an external API** as a decision variable.

### What we need

An offline oracle that:
- Has full future knowledge (all request arrivals, token counts, prefix overlaps)
- Makes the **same type of decision** as Nimbus: per-request local vs remote
- Does **not** change the local serving stack (no re-scheduling, no cache policy changes)
- Scales to real production traces (28K requests)
- Can be validated against real SGLang replay

## 2. Key Insight: Bistability Simplifies the Problem

Our bistability finding means the system has only two states:
- **Healthy**: TTFT ~ 100ms, prefix cache preserved, short queues
- **Collapsed**: TTFT ~ 100,000ms, prefix cache evicted, queue explosion

There is no smooth degradation between them. This means:

> **We do not need to predict exact TTFT. We only need to ensure the system
> never crosses the cliff threshold.**

This reduces the oracle from "predict latency for every possible assignment"
(intractable) to "keep memory pressure below threshold at every time point"
(a classical optimization problem).

## 3. Problem Formulation

### The Temporal Knapsack Problem (TKP)

Our problem maps directly to the classical **Temporal Knapsack Problem**
(Bartholdi 1980, Caprara et al. 2013), also equivalent to **Unsplittable Flow
on a Path** (Grandoni et al. STOC 2022, which admits a PTAS).

### Setup

Given a trace of N requests, each request i has:
- `arrival_i`: when it arrives
- `b_i`: block_hash_ids (its KV cache block sequence)
- `N_i = |b_i|`: total blocks
- `decode_tokens_i`: number of output tokens
- `c_i`: API cost if outsourced

Given system parameters:
- `C_safe`: KV cache pressure threshold (below cliff)
- `T_prefill`: prefill throughput (tokens/sec, profiled)
- `TPOT_healthy`: time per output token in healthy regime (profiled)

### Decision Variables

```
Y = (y_1, y_2, ..., y_N)
y_i in {0, 1}
y_i = 1 means request i is outsourced to API
y_i = 0 means request i is served locally
```

### Per-Request Pressure Profile

For each locally-served request i (y_i = 0), we compute:

**Marginal memory footprint** (depends on Y due to prefix sharing):
```
h_i(Y) = prefix cache hit length (blocks already cached from prior requests)
m_i(Y) = N_i - h_i(Y)          (new blocks this request inserts)
```

**Timing** (from profiled throughput, NOT from collapsed runs):
```
s_i = arrival_i                                         (pressure starts)
e_i = arrival_i + m_i * block_size / T_prefill          (prefill done)
      + decode_tokens_i * TPOT_healthy                  (decode done, pressure ends)
```

**Pressure contribution to time bucket k**:
```
q_{ik}(Y) = m_i(Y)    if bucket k overlaps [s_i, e_i)
          = 0          otherwise
```

### Objective and Constraints

```
minimize    sum_i  c_i * y_i                       (total API cost)

subject to  for all time buckets k:
            sum_i  q_{ik}(Y) * (1 - y_i)  <=  C_safe
                                                   (stay below cliff)

            y_i in {0, 1}
```

In words: find the minimum-cost set of requests to outsource such that the
local GPU's KV cache pressure never exceeds the safe threshold at any point
in time.

### Why `q_{ik}` depends on Y (prefix coupling)

If request A is outsourced, its blocks are never inserted into local cache.
Subsequent requests B, C in the same session lose their prefix hit:
- `h_B` decreases (fewer cached blocks to match)
- `m_B` increases (more new blocks to insert)
- `q_{Bk}` grows (larger pressure rectangle)

This coupling is **intra-session only** (verified: block_hash_ids in our
traces use per-session namespacing, so cross-session overlap = 0).

## 4. Algorithm: Fixed-Point TCPO

Because `q_{ik}` depends on Y, we cannot solve the ILP in one shot.
We use fixed-point iteration:

```
Algorithm: Fixed-Point TCPO

Input:  trace, T_prefill, TPOT_healthy, C_safe
Output: outsourcing assignment Y*

1. Initialize Y^(0) = 0  (all requests local)

2. For t = 0, 1, 2, ..., T_max:

   a. REPLAY: Run logical radix tracker with assignment Y^(t)
      - Process requests in arrival order
      - Maintain LRU block cache (same policy as SGLang RadixAttention)
      - For outsourced requests (y_i = 1): skip, do not insert blocks
      - For local requests (y_i = 0):
          * Match prefix -> compute h_i, m_i
          * Insert new blocks at prefill_done time (not arrival)
          * Record (s_i, e_i, m_i)

   b. BUILD: Construct sparse pressure profile
      - For each local request, record active time interval and m_i
      - Index by time bucket for constraint building

   c. SOLVE: ILP with PuLP/CBC
      - min  sum c_i * y_i
      - s.t. per-bucket pressure constraints (sparse)
      - Extract new assignment Y^(t+1)

   d. CHECK: If Y^(t+1) == Y^(t) or change < 1%, stop.

3. Return Y^(T)
```

### Convergence

This is a best-response dynamic. Empirically, we expect convergence in 2-5
iterations because:
- Outsourcing more requests only increases m_i for same-session requests
- The dependency graph is per-session (disjoint across sessions)
- Sessions are small (2-5 requests typically)

Safety: cap at T_max = 5 iterations, take the last assignment.

### Consistency Check (do we even need fixed-point?)

Before committing to iteration, measure: run the tracker under all-local (Y=0)
and under CacheDisp-15% assignment. Compare m_i values for remaining local
requests. If delta_m < 5% on average, one-shot TCPO is sufficient.

## 5. Computing `m_i`: The Logical Radix Tracker

This is the core computation. It is NOT a simulator -- it only tracks
which blocks are in cache and which are not.

```python
class BlockRadixCache:
    """Logical LRU block cache mimicking SGLang RadixAttention."""

    def __init__(self, capacity_blocks):
        self.capacity = capacity_blocks
        self.cache = {}  # block_hash -> last_access_time

    def match_and_insert(self, block_hash_ids, insert_time):
        """Match prefix, insert new blocks at insert_time.

        Returns (h_i, m_i): prefix hit length and marginal new blocks.
        """
        # (a) Count consecutive prefix hits
        h_i = 0
        for bh in block_hash_ids:
            if bh in self.cache:
                self.cache[bh] = insert_time  # LRU touch
                h_i += 1
            else:
                break

        # (b) Insert new blocks (evict LRU if full)
        new_blocks = block_hash_ids[h_i:]
        for bh in new_blocks:
            if len(self.cache) >= self.capacity:
                lru_key = min(self.cache, key=self.cache.get)
                del self.cache[lru_key]
            self.cache[bh] = insert_time

        m_i = len(new_blocks)
        return h_i, m_i
```

~30 lines of Python. No GPU, no timing simulation, no batch modeling.

### Important detail: insert at prefill_done, not arrival

In SGLang, KV blocks become reusable in the radix tree only AFTER prefill
completes. So:

```
insert_time = arrival_time + m_i * block_size / T_prefill
```

NOT `arrival_time`. Using arrival would overestimate prefix hits for
requests that arrive close together.

## 6. Calibration

### TPOT_healthy

Source: knee sweep experiment, healthy-regime fraction (e.g., CacheDisp at 25%).
Extract from `metrics.csv`:
```
TPOT_healthy = 1 / mean(gen_throughput) over stable window
```
Do NOT use f00 (collapsed) or batch=1 profiling.

### C_safe

Source: knee sweep experiment.
- For each fraction, record peak `token_usage` from `metrics.csv`
- For each fraction, check if TTFT p50 < 1 second (healthy)
- `C_safe = 0.9 * max(peak_token_usage where fraction is healthy)`

This is a system parameter calibrated from real experiments,
not a number from a collapsed trace.

## 7. Validation

### Layer 1: Lower Bound (LB)

From the all-local pressure profile, compute the minimum pressure
that must be removed:
```
LB = sum over all buckets k:  max(0, pressure_k - C_safe)
```
Any strategy must outsource at least this much memory-time.

### Layer 2: Real Replay (primary validation)

Feed TCPO's assignment Y* into `run_offload_strategies.py` as a new strategy:
```python
class TCPOStrategy(OffloadStrategy):
    def should_outsource(self, req, kv_pressure):
        return self.precomputed_Y[req.idx] == 1
```
Run on real SGLang. Measure TTFT, cost, outsource fraction.
Compare against CacheDisp at the same fraction.

### Layer 3: Counterfactual Replay Oracle (CRO)

On 2-3 sampled overload windows, do local search over outsource subsets
using real SGLang replay as evaluator. Verify TCPO's choices are close
to the replay-search optimum (Jaccard > 0.7).

## 8. What We Claim (and Don't Claim)

### We claim

TCPO is a **routing upper bound under a replay-calibrated KV-pressure model**.
It is optimal within the reduced model (bistability-justified capacity
constraint + prefix-aware marginal pressure).

### We do not claim

TCPO is the global optimum of the real system. The reduced model ignores:
- Batch contention (mild, ~1.65x TPOT, captured in TPOT_healthy)
- Queue dynamics beyond the cliff (we only model cliff vs not-cliff)
- Cross-session prefix sharing (zero in current traces)

### Paper story

> CacheDisp (online heuristic) is the **scalar relaxation** of the pressure
> profile that TCPO (offline oracle) operates on. CacheDisp uses
> `prefill_tokens * decode_tokens` as a single-number summary of each
> request's pressure rectangle. TCPO uses the full time-expanded profile.
> If the gap between them is small, CacheDisp is near-optimal.

## 9. Related Work

| Paper | What they do | Difference from TCPO |
|-------|-------------|---------------------|
| Jaillet et al. 2025 | IP for single-cluster LLM scheduling, 40-60 requests | No outsourcing, no prefix awareness, doesn't scale |
| Ao et al. WAIT 2025 | Fluid model for single-cluster throughput | No outsourcing, no prefix awareness |
| Sorted-F 2025 | NP-hardness + constant approx for scheduling | Single cluster, latency objective |
| INFERCEPT ICML'24 | (unused_mem * unused_time) for tool-call KV | Different context (tool pauses, not outsourcing) |
| Hybrid LLM ICLR'24 | Quality-driven local vs cloud routing | No resource modeling (KV, memory, queues) |
| TKP (Bartholdi'80, Caprara'13) | Classical temporal knapsack | We add prefix-aware m_i(Y) coupling |
| UFP PTAS (Grandoni STOC'22) | Admits (1+eps) approximation | Theoretical tool we build on |

TCPO is the **first offline optimal formulation for hybrid LLM inference
with outsourcing as a first-class decision variable and prefix-cache-aware
pressure modeling**.
