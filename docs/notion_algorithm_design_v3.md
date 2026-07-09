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

### ④ cost — what outsourcing it would charge us (unit: dollars)

```
cost(r) = 2000 × ($0.15 / 1M) + 300 × ($1.20 / 1M) ≈ $0.0007
```

Where the numbers come from: the cloud API's price sheet (e.g. $0.15 per
million input tokens, $1.20 per million output tokens).

---

## Part 2 — Two quantities of the system

**K_avail — how much GPU memory is free right now (unit: tokens).** Not an
estimate: the serving engine (vLLM) exposes a metrics endpoint that we poll
every 0.25 s for the real reading. Example: total capacity 216,512 tokens,
requests currently running inside the engine occupy 166,000 → K_avail =
50,512.

**The queue** — requests that have not yet been sent into the GPU. It lives
in OUR hands (an external queue in front of the engine), so we know each
waiting request's identity and size. (This is necessary: engines only expose
queue *counts*, never the identity of internally queued requests — selective
outsourcing needs a name list.)

---

## Part 3 — The algorithm (three steps)

Scenario used below: **30 requests are waiting; their footprints sum to
69,000 tokens; K_avail = 50,512.**

### Step 1 — Should we kick? (trigger)

```
gap G = total footprint of the queue − K_avail = 69,000 − 50,512 = 18,488 tokens
```

G > 0 means: even if the GPU finished everything it is running, this queue
would not fit — it exceeds capacity by 18,488 tokens' worth. If G ≤ 0, do
nothing. **Under no pressure the algorithm has zero overhead and behaves
exactly like all-local** (verified experimentally: with free capacity, zero
kicks, TTFT identical to the all-local baseline).

Refinement: the running requests keep growing (each generated token takes one
more slot), so G should also count the in-flight requests' *remaining*
expected output — otherwise the gap is systematically underestimated.

### Step 2 — How much to kick?

Kick until the kicked requests' footprints sum to ≥ G, then a little extra
margin (e.g. 5%), so the very next arrival does not immediately re-trigger —
like draining water to slightly *below* the alarm line instead of exactly to
it, to avoid oscillation.

### Step 3 — Whom to kick? (the ordering — the core)

Compute a value-for-money score for every waiting request:

```
score(r) = cost(r) / displacement(r)
         = dollars charged ÷ cache·time released
```

**Kick in ascending order of score** — each dollar spent buys back the most
"cache·time". Comparing two requests:

|                | request A | request B |
|----------------|-----------|-----------|
| footprint      | 2,300 tokens | 2,300 tokens |
| residence      | 3 s       | 30 s (very long generation) |
| displacement   | 6,900     | 69,000 |
| cost           | $0.0007   | $0.003 |
| **score**      | 1.0×10⁻⁷  | **0.4×10⁻⁷ ← kick this one first** |

B costs 4× more to outsource, but its cache-time damage is 10× larger —
kicking it is the better deal. **This is where displacement acts, and the
only place it acts** — it decides the order, never the fit.

Kick in this order until the 18,488-token gap (plus margin) is covered. Kicked
requests go to the cloud API and we pay their cost.

(When the MTP/batch coupling is enabled: each kick shrinks the running batch,
which changes everyone's per-token time and hence residence — so re-estimate
and re-rank after each kick. Under a static approximation, one ranking pass
suffices.)

**Offline reference:** the same selection problem — "release ≥ G tokens at
minimum total cost" — has an exact solution by dynamic programming. We do NOT
run it online; it serves as the offline upper bound in evaluation, to measure
how far the greedy order is from optimal.

---

## Part 4 — Why the three quantities must never be mixed

| quantity     | unit     | answers exactly one question |
|--------------|----------|------------------------------|
| footprint    | tokens   | does it fit (compared against K_avail — **same unit, comparable**) |
| displacement | token·s  | whom to kick first (**ordering only, never compared to capacity** — there is no such thing as a "token·seconds capacity". Using it as the knapsack weight was exactly the earlier failure: a single 48k-prompt request "weighed" 768k token·s, more than the entire 450k budget, while the GPU could physically hold it with room to spare — causing massive over-shedding, 70% violations vs 0% for all-local) |
| cost         | dollars  | what kicking it charges (the numerator of the score) |

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

| Before | Now | Why |
|---|---|---|
| budget = `total_weight − min_weight` (forces ≥1 kick per round) | budget/gap = real capacity: `G = Σ footprint − K_avail`, K_avail read live from the engine | the old budget was not a capacity, just a "kick at least one" trick; the knapsack degenerated under it |
| knapsack **weight = token·seconds** (V2 displacement) | **weight = tokens** (footprint); displacement moved to the ordering position | token·seconds has no capacity to compare against (48k-prompt counterexample above); tokens and K_avail share a unit |
| solve 0/1 DP online, kick one per round | greedy by `cost/displacement` until the gap is covered; DP kept as offline oracle | under the old budget the DP already degenerated to density selection; greedy is O(n log n), online-affordable, and the DP now serves evaluation as an upper bound |
| gap counts the waiting queue only | gap also counts in-flight requests' remaining output growth | running requests keep consuming KV; ignoring them underestimates pressure |
| (implicit) violations counted on the local side | violations counted on BOTH sides (kicked requests too) | measured cloud latency exceeds the SLO itself; outsourcing buys protection for the *rest* of the queue, not for the kicked request |

The V2 open question — "if weight is token·seconds, what is the budget?" — is
resolved by dissolution: weight and budget both live in tokens now; the
token·seconds formula (V2) is intact as the ordering signal, which is the
paper's core claim.
