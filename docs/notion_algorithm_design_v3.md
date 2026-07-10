**English** | [简体中文](notion_algorithm_design_v3.zh-CN.md)

# Nimbus Algorithm Design (v3, KV-bound)

**Scope assumption (stated up front):** the binding local resource is the KV
cache. Experiments use workloads where KV saturates first (e.g. long-prompt
production slices). Compute/slot-bound overload is out of scope for this
design and is discussed under limitations.

---

## Part 1 — Four quantities of a single request

Running example used throughout: **a request arrives with a 2,000-token
prompt, and we expect it to generate a 300-token response.**

### ① footprint — how much GPU memory it will occupy (unit: tokens)

```
footprint(r) = prompt tokens + expected output tokens = 2000 + 300 = 2300 tokens
```

Why addition: the KV cache stores state per token. The 2,000 prompt tokens
occupy their slots as soon as the request enters; the 300 generated tokens
occupy slots one by one as they are produced. At peak the request holds 2,300
token-slots.

**This quantity answers: "does it fit?"**

Where the numbers come from: prompt length is directly countable; output
length is unknown online and must be estimated (a known weakness, discussed
separately in the paper; experiments use the trace's true value first, clearly
labeled as an oracle).

**Prefix caching note:** this base version assumes no prefix-cache hit. In
prefix-aware mode, replace prompt tokens with the request's **uncached /
marginal** KV tokens (blocks it would actually add); a remote (provider-side)
cache hit affects only `cost`, never local capacity.

### ② residence — how long that memory stays occupied (unit: seconds)

```
residence(r) = prompt processing time + generation time
             = 2000 / 20000  +  300 × 0.0095
             = 0.1 s + 2.85 s ≈ 3 s
```

Two phases: first the prompt is processed (the GPU handles roughly 20,000
tokens/s — a constant calibrated by offline profiling), then tokens are
generated one at a time (about 9.5 ms per token — **measured on our
machine**; note this per-token time depends on how many requests run
concurrently, which is where the MTP/speculative-decoding thread plugs in).

**This quantity answers: "for how long?"**

### ③ displacement — how much × how long (unit: token·seconds)

```
displacement(r) = footprint × residence = 2300 × 3 ≈ 6,900 token·seconds
```

Intuition: a request's "damage" to the cache is not just how much it occupies
but for how long. Occupying 2,300 tokens for 3 seconds and occupying 230
tokens for 30 seconds do comparable damage. **This is the V2 formula — the
core idea of the paper.** It is NOT used to decide whether things fit (that is
footprint's job); it is used only for **ordering: whom to kick first**.

(Strictly, the true quantity is the integral `∫ KV_tokens(t) dt`; we use
peak-reservation × residence as a conservative online proxy for it.)

### ④ cost — what outsourcing it would charge us (unit: dollars)

```
cost(r) = 2000 × ($0.15 / 1M) + 300 × ($1.20 / 1M) ≈ $0.0007
```

Where the numbers come from: the cloud API's price sheet (e.g. $0.15 per
million input tokens, $1.20 per million output tokens).

---

## Part 2 — Two quantities of the system

**K_headroom — how much GPU memory we may still commit (unit: tokens).** Not
an estimate: the serving engine (vLLM) exposes a metrics endpoint that we poll
every 0.25 s for the real KV usage. Example: total capacity 216,512 tokens,
requests currently running inside the engine occupy 166,000 → K_headroom =
50,512. (If a safety-capped capacity `K_safe` — e.g. 90 % of total — is used,
then `K_headroom = K_safe − current_KV_usage`; by default `K_headroom =
K_avail`, the raw free amount.)

**remaining_decode of an in-flight request (unit: tokens).** A running request
keeps growing — each generated token takes one more KV slot. Its remaining
growth is `expected output tokens − tokens generated so far`. For local vLLM
requests, the implementation enables `stream_options.continuous_usage_stats`
and consumes the engine's exact cumulative completion-token count on every
stream update. It deliberately does not equate content chunks with tokens:
under MTP, one content chunk can contain several accepted tokens. This
committed future growth must be counted as pressure. If the endpoint does not
honor continuous usage, the request fails explicitly rather than silently
falling back to biased chunk counting.

**The queue** — requests that have not yet been sent into the GPU. It lives
in OUR hands (an external queue in front of the engine), so we know each
waiting request's identity and size. (This is necessary: engines only expose
queue *counts*, never the identity of internally queued requests — selective
outsourcing needs a name list.)

---

## Part 3 — The algorithm (three steps)

Scenario used below: **30 requests are waiting, footprints summing to 69,000
tokens; in-flight requests still have 8,000 tokens of remaining output between
them; K_headroom = 50,512.**

### Step 1 — Should we kick? (trigger)

```
G = [ Σ_waiting footprint(r)  +  Σ_inflight remaining_decode(r)  −  K_headroom ]₊

  = [ 69,000 + 8,000 − 50,512 ]₊ = 26,488 tokens        ([x]₊ means max(0, x))
```

G > 0 means: **under the current KV headroom, this waiting set cannot all be
kept local safely** — the committed demand (waiting footprints plus the
running requests' remaining growth) exceeds what is free by 26,488 tokens'
worth. If G ≤ 0, do nothing. **Under no pressure the algorithm performs no
shedding and preserves all-local routing behavior** (verified experimentally:
with free capacity, zero kicks and matching TTFT). Nimbus still reads KV and
generation progress to establish that condition.

The in-flight term is part of the MAIN formula, not an optional refinement —
omitting it systematically underestimates pressure, because K_headroom is a
*current* reading that does not yet include the running requests' future
growth.

### Step 2 — How much to kick?

Kick until the kicked requests' footprints reach the release target:

```
release_target = G + h · K_headroom          (h = 0.05)
               = 26,488 + 0.05 × 50,512 ≈ 29,014 tokens
```

Not just back to the alarm line but leaving an extra 5 % of the current
headroom — like draining water to slightly *below* the line instead of exactly
to it, to avoid re-triggering on the very next arrival.

### Step 3 — Whom to kick? (the ordering — the core)

Compute a value-for-money score for every waiting request:

```
score(r) = cost(r) / displacement(r)
         = dollars charged ÷ cache·time released
```

**Kick in ascending order of score** — each dollar spent buys back the most
"cache·time". Comparing two requests:

|              | request A    | request B                          |
| ------------ | ------------ | ---------------------------------- |
| footprint    | 2,300 tokens | 2,300 tokens                       |
| residence    | 3 s          | 30 s (very long generation)        |
| displacement | 6,900        | 69,000                             |
| cost         | $0.0007      | $0.003                             |
| **score**    | 1.0×10⁻⁷     | **0.4×10⁻⁷ ← kick this one first** |

B costs 4× more to outsource, but its cache-time damage is 10× larger —
kicking it is the better deal. **This is where displacement acts, and the
only place it acts** — it decides the order, never the fit.

Kick in this order until the release target (29,014 tokens in the scenario) is covered. Kicked
requests go to the cloud API and we pay their cost.

(When the MTP/batch coupling is enabled: each kick shrinks the running batch,
which changes everyone's per-token time and hence residence — so re-estimate
and re-rank after each kick. Under a static approximation, one ranking pass
suffices.)

**Online vs offline, stated precisely:** online, the algorithm is a **density
greedy over the cover form** ("release ≥ release_target tokens, cheapest
cache·time first") — no knapsack solver runs online. The static cover problem
("release ≥ G tokens at minimum total cost") has an exact DP solution. That DP
is **planned as an offline evaluation oracle**, but is not implemented in the
current repository.

---

## Part 4 — Why the three quantities must never be mixed

| quantity     | unit    | answers exactly one question                                 |
| ------------ | ------- | ------------------------------------------------------------ |
| footprint    | tokens  | does it fit (compared against K_headroom / default K_avail — **same unit, comparable**) |
| displacement | token·s | whom to kick first (**ordering only, never compared to capacity** — there is no such thing as a "token·seconds capacity". Using it as the knapsack weight was exactly the earlier failure: a single 48k-prompt request "weighed" 768k token·s, more than the entire 450k budget, while the GPU could physically hold it with room to spare — causing massive over-shedding, 70% violations vs 0% for all-local) |
| cost         | dollars | what kicking it charges (the numerator of the score)         |

---

## Part 5 — Two honest footnotes (for the paper; they do not change the algorithm)

1. Output length is unknown online. Quantities ①③④ all depend on an estimate
   of it. Experiments must include an "oracle length vs estimated length"
   comparison.
2. Kicked requests can violate the latency SLO too (measured cloud
   time-to-first-token: median ≈ 10 s on our channel — the cloud is not a
   fast escape hatch). Total accounting must count their violations as well,
   not just the local side's.

---

## Appendix — What changed vs. the previous Algorithm Design

| Before                                                       | Now                                                          | Why                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| budget = `total_weight − min_weight` (forces ≥1 kick per round) | budget/gap = real capacity: `G = [Σ_waiting footprint + Σ_inflight remaining_decode − K_headroom]₊`, K_headroom defaults to live K_avail from the engine | the old budget was not a capacity, just a "kick at least one" trick; the knapsack degenerated under it |
| knapsack **weight = token·seconds** (V2 displacement)        | **weight = tokens** (footprint); displacement moved to the ordering position | token·seconds has no capacity to compare against (48k-prompt counterexample above); tokens and K_headroom share a unit |
| solve 0/1 DP online, kick one per round                      | greedy by `cost/displacement` until `release_target` is covered; exact cover-form DP planned as an offline oracle | under the old budget the DP already degenerated to density selection; greedy is O(n log n), online-affordable, while the planned DP will serve as an evaluation upper bound |
| gap counts the waiting queue only                            | gap also counts in-flight requests' remaining output growth  | running requests keep consuming KV; ignoring them underestimates pressure |
| (implicit) violations counted on the local side              | violations counted on BOTH sides (kicked requests too)       | measured cloud latency exceeds the SLO itself; outsourcing buys protection for the *rest* of the queue, not for the kicked request |

The V2 open question — "if weight is token·seconds, what is the budget?" — is
resolved by dissolution: weight and budget both live in tokens now; the
token·seconds formula (V2) is intact as the ordering signal, which is the
paper's core claim.