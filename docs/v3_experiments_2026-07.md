# Nimbus v3 — experiment record & handoff (July 2026)

Self-contained status document: what was built, what was measured, what broke our
assumptions, and the exact queue of next experiments. Written so a person or agent
(e.g. Codex) can continue without access to prior conversations. Historical
shipped-v3 (`kv_gap`) design baseline:
[`notion_algorithm_design_v3.md`](notion_algorithm_design_v3.md); the current
July-14 TTFT experiment contract and results are authoritative in Sections
5b–5d below. Framework usage: [`../router/README.md`](../router/README.md).

**Path convention** (per [`../AGENTS.md`](../AGENTS.md), machine names / usernames /
absolute scratch paths are never committed): `$MSCRATCH` = the project owner's
scratch dir on the GPU box; `$JSCRATCH` = the scratch dir of the teammate who
owns the workloads/serving setup. Actual values are configured out-of-band.

---

## 1. TL;DR

1. The `router/` framework (external FIFO + work-conserving dispatcher + policies)
   remains validated against the trusted open-loop harness (parity 1.003; queue
   neutrality 110 vs 111 ms). **75/75** unit tests pass without network/GPU.
2. The July-14 rerun explicitly uses **predicted TTFT violation as the trigger**;
   KV is neither its objective nor its trigger. The repository default remains
   the shipped `kv_gap + cost_disp_current` baseline until a separate design
   decision changes it. The original v2 weight
   `prompt × (uncached_prompt/prefill_tput + decode×TPOT)` is preserved exactly
   in `classic_cachedisp_token_s`; experimental selector
   `cost_cachedisp_old` greedily orders victims by `cloud_cost / old_weight`.
   This is not the historical 0/1 DP.
3. The July 12–13 headline runs are now **exploratory only, not selector proof**.
   A July 14 audit found that the scheduler used cumulative trace-token metadata
   while HTTP often sent only a tiny current-turn prompt; simultaneous arrivals
   were also decided row-by-row, often leaving the selector one candidate. The
   recorded latencies are real for those payloads, but claims that displacement
   caused the win are not identified.
4. The independent gauge finding remains valid: hybrid Qwen3.6-35B-A3B exposes
   a per-sequence state pool (~41 concurrent), while dense Qwen3-32B exposes
   token KV. That explains why a KV-gauge trigger is architecture-dependent; it
   is evidence against treating KV as a universal overload trigger and says
   nothing against the original v2 weight as a victim-ordering signal.
5. That dense-32B evidence path is now complete through one exploratory
   512-request matrix. All three selectors had **0 measured local violations**;
   old-v2 ordering outsourced 257/512 (50.20% pessimistic combined), current
   displacement 266/512 (51.95%), and `newest` 326/512 (63.67%). This is a
   strong signal that selection matters, but one ordered pass is not a paper
   result. The leg has `cached_tokens=0`, so it does not validate the cached
   term, provider cache pricing, or prefix-cache displacement.

---

## 2. What is built and verified

| Artifact | What it is |
|---|---|
| `router/run.py` | Single entry point: external FIFO, work-conserving dispatcher, KV monitor, inflight-KV tracker, CLI |
| `router/common.py` | Trace loading (BurstGPT windows byte-identical to the trusted `vllm/run.py`; `--scenario full` for arbitrary traces), payloads, SSE client, NullCloud sink, billing, summaries |
| `router/nimbus.py` | Shipped KV-gap baseline plus orthogonal `kv_gap` / `ttft_pred` triggers and selector ablations |
| `router/test_*.py` | 75 unit tests, no network/GPU (`python3 -m unittest router.test_common router.test_run router.test_nimbus`) |
| `tools/kv_gauge_probe.py` | Live probe that established the Section-4 finding (stdlib only) |
| `tools/analyze_eb1200.py` | Timeline reconstruction that flagged the anomaly from a result JSONL |
| `tools/materialize_token_aligned_trace.py` | Atomic, tokenizer-fingerprinted no-cache trace materializer |
| `tools/profile_ttft_batch.py` | Same-payload warmup/profile with a held-out shared-prefill-lane calibration artifact |
| `experiments/run_ttft_selector_matrix.sh` | Server/profile/trace-bound multi-arm runner with per-arm completion markers |

Validation ladder (all on the GPU box, Qwen3.6-35B-A3B + MTP(5), details in
[`../router/README.md`](../router/README.md)):

| Check | Result |
|---|---|
| Parity anchor: open-loop all_local ≡ trusted `vllm/run.py` | paired TTFT p50 ratio **1.003** |
| Queue neutrality: framework all_local ≈ open-loop | p50 **110 vs 111 ms**, p99 233 vs 223 ms |
| Pressure pacing: queueing client-side, engine never flooded | queue_delay p50 83.8 s while service TTFT p50 90 ms, 756/756 success |
| `random` end-to-end | 29.4% actual vs 30% target; billing recomputation exact; local $0 |
| nimbus neutrality under no pressure | 0 kicks, p50 113 vs 112 ms |

Server recipe (launch script `start_vllm_qwen36.sh` in `$MSCRATCH`): vLLM v0.19.0,
`--max-num-seqs 128`, MTP `num_speculative_tokens=5`, `--gpu-memory-utilization 0.95`,
`--attention-backend TRITON_ATTN` (Blackwell sm_120 PTX workaround),
`--limit-mm-per-prompt '{"image":0,"video":0}'` (VL model), port 8010.

---

## 3. Experiment record

> **Evidence-status warning (2026-07-14).** Sections 3.1–3.3 and 5a preserve
> historical numbers for diagnosis, but direct ShareGPT replay did not align
> scheduler tokens with endpoint tokens and did not batch simultaneous arrivals.
> Do not cite them as proof that one displacement selector beats another. Only
> a token-aligned rerun passing the per-arm usage audit can restore that claim.

### 3.1 Pre-v3 selection signal (old max-displacement policy, `extreme_burst_1200`)

Three-way on the same server, matched shed fraction:

| Policy | Outsourced | Local TTFT p50 | Local SLO viol |
|---|---|---|---|
| all_local | 0% | 324.8 s | 97.9% |
| nimbus (pre-v3) | 25.6% | **12.4 s** | 94.8% |
| random @ 0.256 (matched) | 25.6% | 108.7 s | 97.0% |

Takeaway: *which* requests you shed matters (nimbus ≫ random at the same
fraction), but the pre-v3 trigger under-shed badly. This motivated v3.

### 3.2 v3 on a KV-heavy slice ("rednote", n=80 long prompts, `--scenario full --time-scale 0.05`)

Production long-prompt slice (prefill ~3–7k tokens), 80 arrivals at 1/s replayed
20×-compressed. Slice: `$MSCRATCH/nimbus/data/rednote_slice.jsonl`.

| | all_local | v3 nimbus |
|---|---|---|
| Outsourced | 0% | 56.3% (self-selected) |
| Local TTFT p50 / p95 | 10.74 s / 26.7 s | **2.98 s / 7.7 s** |
| Local SLO viol (5 s) | 67.5% (54/80) | **22.9%** (8/35) |
| peak_inflight | 79 | 34 |
| Cost | $0 | $0.072 |

Honest caveat: with kicked requests *counted as violations* (real cloud TTFT p50
≈ 10 s per the teammate's 14,537-row OpenRouter measurement set for qwen3-32b,
`$JSCRATCH/initial_result/`), nimbus scores (8+45)/80 = 66.3% vs all_local 67.5%
— a wash. Aggressive shedding does not automatically win the *combined* metric
when the cloud is slow; this is precisely the cost/SLO-frontier question
(Section 7, item 5).

### 3.3 v3 on the hardest cell (`extreme_burst_1200`, n=11,605) — headline + audit

Command (leg 1 of `$MSCRATCH/v3_accept.sh`):

```bash
python -m router.run --data $DATA --scenario extreme_burst_1200 --policy nimbus \
  --local-url http://127.0.0.1:8010/v1/chat/completions --local-model Qwen3.6-35B-A3B \
  --max-inflight 128 --kv-capacity-tokens 216512 \
  --out-dir $OUT --output v3_nimbus_eb1200.jsonl
```

| Metric | all_local (prior) | v3 nimbus |
|---|---|---|
| Outsourced | 0% | **33.9%** (3,931/11,605) |
| Local TTFT p50 / p95 / p99 | 324.8 s / — / — | **319.8 ms** / 1.61 s / 2.32 s |
| Local SLO viol | 97.9% | **0.0%** (0/7,674) |
| TPOT p50 | — | 21.4 ms |
| queue_delay p50 / max | — | 6.0 ms / 14.3 ms |
| peak_inflight | 128 | **59** |
| Cost | $0 | $1.79 |
| Telemetry | — | ticks 11,642, kick_rounds 3,935, kv_read_failures 0 |

**Audit that led to Section 4** (reproduce: `tools/analyze_eb1200.py <jsonl>`;
raw results: `$MSCRATCH/router_v3/v3_nimbus_eb1200.jsonl`):

- Kicks spread uniformly t = 13 s … 1,198 s (of 1,199 s), 20–55% of arrivals per
  10 s bucket — steady-state shedding, not burst response.
- At kick moments: inflight p50 43 (max 55); **KV commitment upper bound p50 =
  20,841 tokens, max = 45,840 — never above 21% of the 216,512-token capacity.**
  Token-KV pressure did not exist at any kick.
- Scheduler metadata *appeared* selective: kicked prompt p50 = 644 vs kept p50
  = 28 (footprint p50 920 vs 289). The July-14 payload audit showed these were
  controller-side cumulative counts, not necessarily the tokens sent to the
  engine, so this line is not valid selector evidence.
- Queue essentially empty throughout (p50 6 ms) ⇒ the gap must have come from a
  collapsed `kv_avail`, not from waiting-queue footprint.

---

## 4. Core finding: on hybrid-GDN models the "KV usage" gauge is a concurrency meter

Live probe on a fresh identical server (`tools/kv_gauge_probe.py`), fixed token
volume, varying concurrency:

| Load | In-flight tokens (% of 216,512) | `kv_cache_usage_perc` |
|---|---|---|
| idle | 0 | 0.00% |
| 8 concurrent small reqs | 0.8% | **19.12%** |
| 32 concurrent small reqs | 3.2% | **76.48%** (linear, ≈2.4%/seq) |
| 64 attempted | 6.4% | **97.99%, engine ran only ≤41 concurrently** |
| 2 × 8k-token prompts | 7.9% | 3.40% |
| after each phase, idle | 0 | 0.00% |

Corroborating server-log lines (same launch): `num_gpu_blocks_override=512`;
`GPU KV cache size: 216,512 tokens`; `Maximum concurrency for 262,144 tokens per
request: 3.07x` (a pure full-attention model would show 216,512/262,144 = 0.83×);
`enable_prefix_caching=False` (auto-disabled for hybrid models — prefix caching
is NOT the explanation).

**Interpretation.** Qwen3.6-35B-A3B is hybrid GDN(linear-attention)+full-attention.
The GDN layers hold a fixed-size recurrent state **per sequence**, allocated from
a 512-block pool; each running sequence costs ≈2.4% of the pool regardless of
length, so the pool — and the gauge — saturates at ~41–42 concurrent sequences.
Per-token attention KV contributes comparatively little (~1% per 8k tokens;
2-point estimate, proper fit is queue item 1).

**Consequences.**

1. *Leg-1 mechanism*: `KVMonitor` converted the slot-meter into fake "headroom
   tokens" `(1−u)×216512`, which collapsed to ~30k once ~35 sequences ran. The
   gap therefore fired exactly at the engine's true saturation knee, and v3
   operated as an adaptive concurrency governor + big-request filter. Right
   trigger point, wrong units: on a slot-bound engine, kicking one 900-token
   request frees the same one slot as kicking a 50-token request, while the
   token-denominated `release_target` believes it freed 18× more. A different
   workload shape (many small requests) could make the cover computation
   systematically over/under-shed. The win is real; the safety margin is luck.
2. *For the team*: `max_num_seqs=128` is not achievable on this model — true
   ceiling ≈41. Every saturation result on it (including "compute-bound"
   interpretations of extreme_burst) is state-slot-bound and should be re-read
   accordingly.
3. *For the paper*: engine "KV usage" semantics are architecture-dependent
   (tokens vs. per-sequence state). Routers that assume token units are wrong on
   hybrid models — none of the related work handles this. This is a
   contribution, not just a bug.

---

## 5. Historical design fork (Option B chosen on 2026-07-13)

The bullets below preserve the July-12 fork for provenance; they are not the
current TTFT-trigger design. Option B was executed in Section 5a. The later
failure of a token-KV trigger on a slot/compute-bound cell motivated the
explicit TTFT model in Sections 5b–5d rather than another unit conversion.

- **Option A (historical proposal): resource-generic units.** The policy works in
  *fractions of the binding pool*: `footprint_frac(req) = a + b·(local_prompt +
  expected_decode)`, `displacement = footprint_frac × residence_s`,
  `gap_frac = Σ waiting_frac + inflight_remaining_frac − (1−u)`. Constants
  `(a, b)` come from a one-time per-model calibration probe. On a pure
  full-attention model `a≈0, b=1/capacity` and the formulas reduce **exactly**
  to the current token·s design — it is a strict generalization. Trigger,
  ordering, hysteresis, runner: unchanged.
- **Option B: keep token semantics**, run the KV-bound story on a pure
  full-attention model, declare hybrid architectures out of scope.

A and B were not exclusive: the dense full-attention cell supplied the sanity
anchor where token-KV gauge semantics are exact.

The historical leg-1 win was not a valid comparison against a naive
concurrency spill. Section 5d now supplies an audit-valid exploratory
`newest` comparison under a common TTFT trigger; repeated/full matrices are
still owed before a paper claim.

---

## 5a. ADDENDUM 2026-07-13 — dense-model campaign (Option B executed) and the binding-resource principle

Murphy's decision: validate on a pure full-attention model (Qwen3-32B dense)
using the SAME team workload (ShareGPT + BurstGPT timestamps); the rednote
slice is dropped from the near-term plan. Executed same day; all results under
`$MSCRATCH/router_v32b/`, scripts `v32b_accept.sh` / `v32b_eb.sh` /
`v32b_rand.sh` in `$MSCRATCH`.

**Server**: Qwen3-32B on physical GPU2, FLASH_ATTN, `--max-num-seqs 128`,
`gpu-memory-utilization 0.95` → `GPU KV cache size: 112,064 tokens`,
`Maximum concurrency for 40,960 tokens per request: 2.74x` (= 112,064/40,960 —
self-consistent, unlike the hybrid's 3.07×). Note: GPU0 wedged a THIRD time
(box renumbers live CUDA devices; vLLM v0.19 rejects UUIDs in
`CUDA_VISIBLE_DEVICES`) — the launch script now resolves physical GPU2's
ordinal by UUID via torch at startup, and an NVML `sitecustomize` shim
(`$MSCRATCH/nvml_shim/`) lets vLLM import past the dead NVML handle.

**Gauge probe on dense (perfect inversion of the hybrid)**: usage == token
fraction (8-way small → 1.50%, 32-way → 5.95%, 64-way → 11.89% with all 64
running; 2×8k prompts → 8.14% ≈ one resident 8k prompt's share). The gauge is
an honest token meter here; no per-sequence slot ceiling (all 64 ran vs the
hybrid's hard 41).

**Cell results (current-harness semantics: `max_tokens` = trace value)**:

- `burst_1200` (n=2,725): does NOT saturate — all_local TTFT p50 227 ms, 0%
  violations, peak_inflight 78/128. The teammate's historical 2,262.9 s p50 on
  this cell came from her FIRST-generation harness which did not cap output
  length (`eval_count` 221 vs trace 131; reasoning-mode free-run, TPOT
  degraded to 304 ms). Under capped semantics the cell is calm. v3 neutrality
  re-confirmed: 2,734 ticks, 0 kicks, numbers identical to all_local.
- `extreme_burst_1200` (n=11,605) four arms:

| arm | outsourced | local TTFT p50 / p95 | local SLO viol (5 s) | both-sides viol* | cost |
|---|---|---|---|---|---|
| all_local | 0% | **620.3 s** / 1,255.7 s | 98.2% | 98.2% | $0 |
| random @ matched (38%) | 38.0% | 163.7 s / 358.3 s | 96.1% | 97.6% | $1.82 |
| **v3 nimbus** | 38.6% (self) | **6.40 s** / 11.2 s | **64.1%** | 78.0% | $2.70 |

*both-sides = kicked counted as violations (Section 6 pessimistic bound).
Engine was slot-bound throughout: peak_inflight 128 pinned, KV peaked ≈69%
(never the binding resource). TPOT stayed ~101–105 ms in all arms.

**Historical reading (superseded by the July-14 audit).** v3 beat matched-
fraction random by 25× on local p50. The runs behaved differently, but the gap
cannot be attributed to displacement because payload tokens differed from the
controller metadata and same-second candidates were not adjudicated together.
Separately, local violations stalled at 64%: the trigger "fit the waiting set
into free KV"
stabilizes the queue at exactly free-KV depth (~35k tokens ≈ 59 requests ≈ 6 s
of wait at this service rate) and stops shedding, while the actual binding
resource (compute slots) needs a near-empty queue to meet a 5 s TTFT SLO.
Honest units, wrong resource → systematic under-shed. Cost note: at matched
fraction v3's victims cost more per head than random's ($2.70 vs $1.82) —
displacement ordering deliberately exports the biggest requests.

**Historical July-13 proposal (superseded as the active path by Section 5b).**
The two campaigns compose into one principle. Hybrid 35B: the gauge
accidentally measured the binding resource (GDN state slots) → trigger landed
on the true knee → 0.32 s / 0%. Dense 32B: the gauge honestly measures tokens,
but slots bind → trigger lands on the wrong bar → 6.4 s / 64%. **v3.1 must set
the admission bar on the binding resource at the operating point** — per-
resource headroom meters (KV tokens, sequence slots, compute/batch slots) with
gap taken on the tightest one. The cost/displacement layer is still a hypothesis
to re-test on aligned payloads; the old ×25 comparison is not proof.
Option A above is subsumed: resource-generic units are necessary but not
sufficient — resource *identification* is the missing half. Design note for
Murphy's sign-off before implementing.

### 5b. ADDENDUM 2026-07-14 — TTFT-trigger rerun and audit contract

The design decision for the next leg is explicit:

- **Trigger:** predict from-arrival TTFT for the waiting queue and shed only
  while a retained request is predicted to exceed the 5 s SLO (minus a
  held-out calibration/tick guard). This is the requested invariant.
- **Selector under that fixed trigger:** compare `newest` (naive spill),
  `cost_cachedisp_old` (cost density whose denominator is the exact original
  v2 weight), and
  `cost_disp_current` (current footprint/residence proxy). `waiting_random`
  seeds are a later variance baseline. The `*_cachedisp_old` ordering is a
  greedy online selector, not the historical full 0/1 DP; an offline DP oracle
  remains an ablation.
- **No-cache scope:** `cached_tokens=0`, vLLM launched with
  `--no-enable-prefix-caching`. This leg tests TTFT triggering and residence
  ranking, not the cached-token term. A cache-aware trace is a separate leg.
- **Workload contract:** prompts are deterministic and tokenizer-sized;
  prompt target is capped at 32,768, decode at 1,024, and context at 40,960.
  `temperature=0, ignore_eos=true` makes actual decode residence equal the
  scheduler cap. The manifest reports how many original rows changed; this is
  a transformed workload, not verbatim ShareGPT.
- **Fairness/audit contract:** complete same-timestamp cohorts before choosing;
  discard a policy decision if it crosses the next arrival; warm twice with
  independent prompts; bind the matrix to the active server PID/log, endpoint,
  trace hash, tokenizer fingerprint, KV capacity, max sequences, and held-out
  profile. Every completed arm requires exact local prompt/decode usage and an
  atomic completion marker.

The separation of responsibilities is therefore:

| Question | July-14 answer | Unit |
|---|---|---|
| When must Nimbus shed? | `ttft_pred`: a waiting survivor would cross `SLO − guard` | seconds |
| Whom should it shed first? | selector ordering, including `cloud_cost / old_v2_weight` | relative score |
| How many should it shed? | minimum prefix of that ordering that makes every waiting survivor safe | request count determined by prediction |

In-flight requests affect future slot/prefill availability, but cannot be
recalled after dispatch; the guarantee is consequently scoped to waiting
survivors, not every already-local request.

Implementation checkpoints are `b930ab2` (first TTFT path), `78846da`
(token/cohort/server audit hardening), and `bcda9d0` (shared-prefill predictor
and schema-v2 profile). Post-run commit `93bf9f1` replaces the volatile raw
`/v1/models` hash with a stable endpoint-identity hash for **future matrices**.
The completed `bcda9d0` arms remain valid because their markers/hashes were
independently audited, but that existing OUTDIR cannot be resumed in place:
the old volatile fingerprint changes on every endpoint query and the new commit
also intentionally changes the run fingerprint.

### 5c. CALIBRATION AUDIT 2026-07-14 — 128 slots are not 128 prefill lanes

The first same-server profile was structurally valid: 60/60 measured blocks and
1,420/1,420 requests succeeded with exact prompt/decode usage. It nevertheless
produced `prefill=52.504 tok/s`, `TPOT=118.411 ms`, and a `7,792 ms` recommended
guard. The 5 s matrix correctly refused this artifact because its guard already
exceeded the SLO.

This was not a bad server. In the held-out `P=512, B=128` block, service TTFT
ranged from 159 ms to 17.412 s in roughly 16-request waves. The active engine
reported `max_num_batched_tokens=8192`, exactly 16 × 512: sequence slots admit
the requests, but one bounded prefill-compute lane determines when each wave
gets a first token. The old predictor assigned all 128 free-slot requests the
same TTFT and the old profiler divided one request's 512 tokens by the batch
median; both abstractions were wrong.

An independent 512-request, 52-cohort token-aligned slice established the
queue-level all-local anchor: 512/512 success, exact usage, TTFT p50/p95 =
56.007/152.686 s, 88.09% local 5 s violations, queue-delay p50 = 51.091 s.
This is a real pressure slice, not a no-op smoke.

The revised predictor does not change the original v2 selector formula. It
models sequence-slot releases plus one aggregate shared FIFO prefill-work lane;
unfinished in-flight prompts conservatively retain their full work until exact
stream usage proves first-token completion. This is an effective online model,
not a claim that vLLM literally executes only one prompt at a time. Profile
schema v2 separately fits shared prefill throughput and fixed first-token
overhead, keeps TPOT for decode residence/the old weight, and compares TTFT
order statistics inside each homogeneous-prompt cell instead of pretending
request ids reveal engine service order.

Schema-v2 profile result (same server lifecycle):

| Check | Result |
|---|---|
| Structural/token audit | 60/60 blocks; 1,420/1,420 success and exact usage |
| Effective shared prefill | 3,264.2498 tokens/s |
| Decode TPOT / first-token overhead | 151.8412 ms / 455.1077 ms |
| Fit | weighted R² = 0.957835 |
| Guard | calibration p99 residual 1,433 ms + one 250 ms tick = **1,683 ms** |
| Final held-out classifier | TP 192, FN 0, FP 22, TN 70 |

One held-out 32k-prompt point underpredicted by more than the full guard, but it
was still classified as a violation, so the acceptance-critical false-negative
count remained zero. Profile SHA256 is
`549dd0bf6e54587bb1188fd420431de6062c39526e186d4593a7e7208536e23f`.

The queue-level `newest` smoke also passed: 512/512 success; 314 routed, 198
local; local TTFT p50/p95/p99 = 1.947/2.449/2.663 s; 0/198 local violations;
pessimistic combined = 314/512 = 61.33%; cost = $0.142655. Decision audit found
210/226 snapshots with more than one candidate, maximum snapshot 17, and 46
multi-request applied rounds (up to 13 victims). All 314 applied victim ids
were unique and exactly matched routed rows; post-kick prediction never
exceeded `5 − 1.683 = 3.317 s`.

Token accuracy means exact agreement with the **materialized** workload. The
manifest records 52/512 rows whose requested materialization target could not
be hit because of chat-template minimum overhead. Relative to the original
trace, 53/512 prompts changed: those 52 plus one prompt capped at 32,768. This
does not break internal alignment, but the slice must not be described as
verbatim ShareGPT.

### 5d. EXPLORATORY SELECTOR COMPARISON 2026-07-14 — old weight wins this pass

One fixed-order pass used the same 512 requests, server, profile, TTFT trigger,
5 s SLO, 1.683 s guard, and 250 ms tick for all arms. Order was generated from
seed 78846 before launch: old-v2 ordering → `newest` → current displacement,
with a 20 s cooldown after each arm. Every arm completed 512/512 with exact
materialized prompt/decode usage.

| selector | routed / local | local TTFT p50 / p95 / p99 | local viol | pessimistic combined* | cost |
|---|---:|---:|---:|---:|---:|
| **`cost_cachedisp_old`** | **257 / 255** | **1.865 / 2.561 / 2.573 s** | **0/255** | **50.20%** | **$0.137257** |
| `cost_disp_current` | 266 / 246 | 1.894 / 2.641 / 2.839 s | 0/246 | 51.95% | $0.147803 |
| `newest` | 326 / 186 | 1.957 / 2.606 / 2.714 s | 0/186 | 63.67% | $0.149767 |

*The NullCloud pessimistic bound treats every routed request as a violation; it
is not measured cloud TTFT.

This pass supports two narrow conclusions. First, TTFT is a workable trigger:
all retained requests met the SLO while all-local violated 88.09%. Second,
selection matters even under the same stop rule: the old-v2 signal retained 69
more requests than `newest` and paid less. It does **not** prove the cached-token
term: no-cache reduces the old weight to
`P × (P/prefill_tput + D×TPOT)`. It also does not yet distinguish old-v2 from
current displacement conclusively (only 9 routed requests / 1.76 points apart)
or provide error bars. Repeat randomized-order matrices and random baselines
before any paper claim. A separate same-config `newest` smoke routed 314 rather
than this pass's 326, directly showing 12-request run-to-run variability; the
old-v2 gap to `newest` is much larger, while old-v2 vs current is not yet above
that observed noise floor.

The mechanism is consistent with the calibrated bottleneck. Old-v2 ordering
routed 291,981 prompt tokens with 257 requests (routed prompt p50 958), current
displacement routed 288,272 with 266 (p50 940), and `newest` only 270,277 with
326 (p50 554). The old no-cache formula's outer `P` and prefill `P/tput` term
strongly favor removing long prompts, buying more shared-prefill relief per
victim. This is a plausible explanation to test, not causal proof from one
ordered pass.

Evidence audit: all three completion markers share fingerprint
`a0e9b1484441bfae478850230305ce2c501d653b1a26031d545032e6cbce23a1`;
their raw/summary/decision hashes and line counts match; decision snapshots
reached 20/18/15 candidates (old/current/newest). Multi-request decisions were
44/55, 47/62, and 51/58 applied rounds respectively. Applied victim ids were
unique and exactly equal to each arm's routed rows. The matrix is explicitly
labeled `single-pass exploratory`.

### 5e. PRE-REGISTERED 2026-07-15 — balanced replicate matrix (design + decision rule)

Registered before the runs; the runs must be judged against this contract.

**Question.** Is the 9-request old-vs-current gap real (the observed
same-selector noise floor is 12 requests, Section 5d), and does either
deterministic selector beat chance under the identical `ttft_pred` trigger?

**Design.** 3×3 Latin square on the same heldout512 slice, one fresh no-cache
server lifecycle, and a fresh same-recipe profile — the PID/log binding makes
the retired v2 profile unusable by construction, so the r2 fit doubles as a
calibration-stability check against v2 (3264.25 tok/s, 455.11 ms overhead,
151.84 ms TPOT, 1683 ms guard). Arms are passed as explicit argv per block
(`ARM_ORDER_MODE=explicit` in each manifest):

| Block | Pos 1 | Pos 2 | Pos 3 |
|---|---|---|---|
| 1 | old | current | waiting_random(101) |
| 2 | current | waiting_random(202) | old |
| 3 | waiting_random(303) | old | current |

Every arm occupies every position exactly once. `waiting_random` sits inside
the square because it is the chance control for the selection layer; `newest`
is a real policy (naive admission spill), not a chance control, and is not
replicated — its single-pass deficit (63.67% vs 50.20%) dwarfs the noise
floor; it returns in the full cell. The 2026-07-14 pass is run 0
(order-confounded; reported separately, not averaged in).

**Decision rule (registered).**
- Primary metric: pessimistic combined violation. With 0 local violations this
  equals the kicked fraction — intended: fewer kicks at equal local safety =
  more relief per kick. Secondary: NullCloud cost; the winner must not be >5%
  worse on mean cost.
- Adopt old-v2 as the primary selector iff it wins ≥2/3 blocks on the primary
  metric AND its mean advantage exceeds the max intra-arm spread across
  blocks. Mirrored condition for current displacement.
- Tie (mean gap ≤ intra-arm spread): adopt old-v2 anyway — it is the original
  design and the simpler story ("the weight was right; the trigger was
  wrong") — and demote current displacement to an ablation arm. Record the
  tie explicitly; the paper claim then becomes "old ≈ current, both ≫ chance
  and naive spill", not "old > current".
- Falsification: if any waiting_random block lands within 5 pp of the best
  deterministic selector, the selection-layer claim is void regardless of the
  old/current ordering.

**Explicit non-goals.** This matrix cannot show generalization — it reuses
the same 512 requests as run 0, so it measures run-to-run noise and order
effects only. Generalization is assigned to the full 11,605-request cell
(11,093 requests untouched by any tuning), gated on this matrix, with
same-server anchor arms rerun from scratch: `kv_gap + cost_disp_current` (the
v3 default) and `all_local` — the July-13 dense anchors came from a
mis-calibrated TPOT (78 ms vs measured ~103 ms), a cache-enabled server, and
a non-token-aligned trace, and must not be cited for comparison. Debts that
block any default flip are unchanged (Section 7 item 7): estimated-vs-oracle
decode (trigger and weight both consume oracle decode lengths today), a
real-cloud leg (pessimistic combined ranks arms fairly only because every arm
shares NullCloud), the `max_cachedisp_old` ablation (is the cost division
load-bearing?), and the cache-aware `(P−C)` term.

---

## 6. Cloud-side accounting (headline-metric decision)

The default cloud is a zero-latency fake sink (`--cloud null`): kicked rows are
`routed_only=true` and excluded from SLO stats (`slo_measured_n` is the explicit
denominator). For paper headlines, the recommended primary metric is the
**pessimistic bound: every kicked request counts as an SLO violation** (real
cloud p50 ≈ 10 s > 5 s SLO), with local-only violation as the secondary metric.
Concretely,
`pessimistic_combined = (local_slo_violations + routed_only) / N`. This is an
upper bound under NullCloud, not observed cloud latency. A real-cloud leg
(`--cloud real …`) is only needed once, pre-submission.

---

## 7. Experiment queue (REVISED 2026-07-14 after the payload/cohort audit)

1. **COMPLETED — audit contract.** Atomic token-aligned materializer,
   complete-cohort/stale-arrival protection, local usage telemetry, no-cache
   server/profile binding, and completion markers; 75/75 tests, `py_compile`,
   `bash -n`, and `git diff --check` passed.
2. **COMPLETED — one bound Qwen3-32B no-cache lifecycle.** PID, log prefix,
   endpoint, KV 112,656, and `max_num_seqs=128` were bound to every artifact.
   After all audits, only recorded parent PID `904943` was terminated; child
   `905088` and port 8010 were verified gone.
3. **COMPLETED — materialization and schema-v2 profile.** Trace/manifest hashes,
   line counts, tokenizer, exact usage, held-out guard, and zero-FN classifier
   all passed; see Section 5c.
4. **COMPLETED — queue-level all-local anchor and TTFT smoke.** Both had 512/512
   success; the policy smoke had real multi-item choices and 0/198 retained
   violations.
5. **COMPLETED (exploratory) — three-selector comparison.** All markers and
   configuration bindings passed. Old-v2 ordering beat `newest` clearly in one
   pass. Its 9-request lead over current is smaller than the observed
   12-request same-selector difference, so that comparison remains unresolved.
6. **RUNNING — establish repeatability before changing the default.** Design
   and decision rule are pre-registered in Section 5e (`waiting_random` moved
   inside the Latin square as the chance control; `newest` dropped from the
   rotation because its margin dwarfs the noise floor — it returns in the full
   cell). Uses the stable endpoint fingerprint from `93bf9f1`; report paired
   per-block distributions, not only one aggregate table. If old-v2 passes the
   5e rule, run the full 11,605-request cell with same-server `kv_gap` v3 and
   `all_local` anchor arms.
7. **THEN restore missing semantics:** cache-aware trace for the cached-token
   term; estimated-vs-oracle decode; historical exact 0/1-DP/offline oracle;
   load/guard sweeps; real-cloud latency leg. The hybrid binding-resource result
   remains a separate architecture study, not the TTFT trigger definition.

---

## 8. Environment gotchas (GPU box) — read before running anything

- **Read [`../AGENTS.md`](../AGENTS.md) first**: kill every server you start
  (by recorded PID, then verify children gone via `nvidia-smi`); default to
  `CUDA_VISIBLE_DEVICES=2` (GPU0 is an intermittently failing card — wedged
  twice in July 2026, `lspci` shows `rev ff` when wedged, recovers to `rev a1`
  only after reboot; a wedged device poisons CUDA init box-wide).
- The owner's home dir is **over disk quota**: every launch must redirect
  `TMPDIR`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
  `XDG_CACHE_HOME`, `HF_HOME` to the scratch volume or vLLM dies at startup with a
  misleading "Engine core initialization failed" (real error: `Errno 122`).
  The launch script in `$MSCRATCH` already does all of this — use it.
- Python for the runner needs `aiohttp`: use the teammate's vllm conda env
  (in `PATH` inside the launch script).
- MTP means **one SSE chunk ≠ one token**. Progress tracking uses vLLM
  `stream_options.continuous_usage_stats` (exact cumulative `completion_tokens`
  per chunk) and fails explicitly if the endpoint omits it. Never count chunks.
- Trace data: BurstGPT/ShareGPT trace and the teammate's result sets live under
  `$JSCRATCH`; rednote slice + v3 results under `$MSCRATCH` (see Section 9).

## 9. Artifact inventory

| Where | What |
|---|---|
| repo `router/`, `tools/`, `docs/` | framework, policy, tests, probes, this record |
| `$MSCRATCH/router_v3/` | `v3_nimbus_eb1200.{jsonl,summary.json}`, `v3_all_local_rednote.*`, `v3_nimbus_rednote.*` |
| `$MSCRATCH/v3_accept.sh`, `v3_accept.log` | the 3-leg acceptance run (leg commands + outputs) |
| `$MSCRATCH/start_vllm_qwen36.sh` | canonical server launch (all workarounds baked in) |
| `$MSCRATCH/kv_gauge_probe.py` | box copy of the probe (repo `tools/` is authoritative) |
| `$MSCRATCH/nimbus/data/rednote_slice.jsonl` | 80-request long-prompt slice |
| `$MSCRATCH/router_ttft_nocache_78846da/heldout512_token_aligned.jsonl{,.manifest.json}` | Independent 512-request materialized trace and provenance |
| `$MSCRATCH/router_ttft_nocache_78846da/ttft_profile_v2_bcda9d0.json` | Accepted schema-v2 profile (SHA256 `549dd0bf…e23f`) |
| `$MSCRATCH/router_ttft_nocache_78846da/heldout512_all_local/` | Queue-pressure all-local anchor |
| `$MSCRATCH/router_ttft_nocache_78846da/heldout512_ttft_smoke_bcda9d0/` | Audited `newest` smoke and completion marker |
| `$MSCRATCH/router_ttft_nocache_78846da/heldout512_selector_compare_bcda9d0/` | Three-arm exploratory matrix, decisions, summaries, markers, manifest |
| `$MSCRATCH/router_ttft_nocache_78846da/vllm_qwen32b_nocache.log` | Server configuration/lifecycle log for this matrix (KV 112,656; distinct from the July-13 server) |
| `$JSCRATCH/workloads/…` | ShareGPT+BurstGPT trace (leg-1 `$DATA`) |
| `$JSCRATCH/initial_result/` | 14,537-row real OpenRouter measurements (cloud-latency calibration) |
| `$JSCRATCH/vllmresult/` | teammate's open-loop sweep on the same model (parity reference) |
