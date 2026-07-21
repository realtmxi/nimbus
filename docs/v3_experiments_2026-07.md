# Nimbus v3 — experiment record & handoff (July 2026)

Self-contained status document: what was built, what was measured, what broke our
assumptions, and the exact queue of next experiments. Written so a person or agent
(e.g. Codex) can continue without access to prior conversations. Historical
shipped-v3 (`kv_gap`) design baseline:
[`notion_algorithm_design_v3.md`](notion_algorithm_design_v3.md); the current
July-14–21 TTFT experiment contract and results are authoritative in Sections
5b–5k below. Framework usage: [`../router/README.md`](../router/README.md).
Chinese chronological ledger:
[`nimbus_experiment_ledger_2026-07.zh-CN.md`](nimbus_experiment_ledger_2026-07.zh-CN.md).

**Path convention** (per [`../AGENTS.md`](../AGENTS.md), machine names / usernames /
absolute scratch paths are never committed): `$MSCRATCH` = the project owner's
scratch dir on the GPU box; `$JSCRATCH` = the scratch dir of the teammate who
owns the workloads/serving setup. Actual values are configured out-of-band.

---

## 1. TL;DR

1. The `router/` framework (external FIFO + work-conserving dispatcher + policies)
   remains validated against the trusted open-loop harness (parity 1.003; queue
   neutrality 110 vs 111 ms). The E12 live execution tree was frozen at
   `98cab54`; validator-only timestamp repairs are `15979bb` and `132e012`, and
   the cumulative-retry/final-audit guards are `ee1a7f6` and `f3583a3`. At the
   latter checkpoint the clean tree passes **287/287** offline tests; commands
   are in Section 2.
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
5. The July-15 repeatability campaign completed **6 blocks / 24 arms** on one
   fresh dense-32B lifecycle. Every arm passed 512/512 requests, exact token
   accounting, and **0 measured local violations**. Old-v2 ordering beat
   `newest` in all 6 paired blocks (mean **50.17 fewer routes**, 9.80 points),
   while old-v2 and current displacement were route-equivalent at this scale
   (paired delta mean **-1.5 requests**, range -14 to +2). Old-v2 was cheaper
   than current in all 6 blocks (mean **$0.010716 / 7.2%** lower). This promotes
   `ttft_pred + cost_cachedisp_old` to the primary **experimental candidate**,
   with current displacement retained as the unresolved full-cell comparator;
   it does not change the repository default. The no-cache leg still does not
   validate the cached term, provider cache pricing, or prefix-cache
   displacement.
6. The pre-registered 11,605-request full cell then separated trigger safety
   from victim ordering. `ttft_pred + cost_cachedisp_old` (A) and
   `ttft_pred + cost_disp_current` (C) each had **0 local violations**; A and C
   were route-equivalent (6,748 vs 6,729 routed, delta +19 inside the frozen
   ±363-row band), while A cost less ($3.34405 vs $3.51914). Naive `newest`
   (B) had 39/3,014 local violations, so the frozen A/B/C-zero safety gate
   failed; the shipped `kv_gap` anchor (K) had 5,249/5,545 local violations and
   is not a TTFT-safe trigger on this cell. This keeps TTFT violation as the
   design objective, keeps old V2 as a supported ordering signal, and adds an
   explicit support-envelope/resource-model hardening step before any default
   change. The all-local pressure-anchor result and five-marker audit are in
   Section 5g.
7. A July-16 real-OpenRouter probe added a fixed-provider, no-fallback,
   first-token cancellation path (commit `e0b6686`). On 16 burst arrivals to
   DeepInfra, all 16 client streams stopped after the first generated token,
   with TTFT p50/p95/p99 **467/929/1,167 ms** and 0/16 above the 5 s SLO. The
   probe also corrected an important semantic mismatch: OpenRouter separates
   Qwen reasoning into `delta.reasoning`, while the bound local vLLM deployment
   (no reasoning parser) emits the same tokens through `content`; Nimbus now
   treats either as the first generated token. This validates the measurement
   path, not a full cloud-latency distribution or completed-response SLO. Under
   the explicitly temporary assumption that cancellation does not change cloud
   load, the next headline metric can be observed local+cloud TTFT rather than
   the NullCloud pessimistic bound.
8. The separately pre-registered ShareGPT current-turn local gate has completed
   (Section 5j). The all-local anchor had **11,400/11,604** local 5 s TTFT
   violations, so retokenization did not remove real pressure. With the same
   `ttft_pred` trigger, old-v2 A routed 5,109 requests and current C routed
   4,099; both retained **zero** local violations. A routed 1,010 more requests
   but had lower modeled full-response cost ($0.500634 vs $0.516213) and lower
   observed local p99 (2.176 vs 2.490 s). Because this is one fixed A→C order
   with NullCloud, it validates retained-local safety, not a selector winner or
   observed hybrid/cloud TTFT. External requests and actual cloud spend were
   both zero; original ShareGPT text did not leave the team host.
9. The registered real OpenRouter/DeepInfra experiment now has **two distinct
   lifecycles** (Section 5k). The first is permanently **TERMINAL PARTIAL**: its
   canary+A completed, C never launched, and its settled **$0.02670220** remains
   part of the global `$3` spend. A fresh retry then completed the ordered A→C
   pair and passed the final text-free audit. A routed 5,097 and had 11/11,604
   overall 5 s violations; C routed 4,057 and had 3/11,604. Both retained-local
   sets had zero violations. C used 1,040 fewer cloud routes but had worse
   overall TTFT p50/p95/p99 by 180.46/139.06/44.62 ms. Retry spend including
   canary was **$0.04693796**; cumulative spend including the terminal partial
   lifecycle was **$0.07364016**, leaving **$2.92635984** under the authorized
   cap. Because this is one A→C pair and the violation-rate difference is only
   0.06894 percentage points, the preregistered `<1 pp` rule declares the
   selector comparison **unresolved**; reverse-order/repeated lifecycles are
   required before naming a winner.

---

## 2. What is built and verified

| Artifact | What it is |
|---|---|
| `router/run.py` | Single entry point: external FIFO, work-conserving dispatcher, KV monitor, inflight-KV tracker, CLI |
| `router/common.py` | Trace loading (BurstGPT windows byte-identical to the trusted `vllm/run.py`; `--scenario full` for arbitrary traces), payloads, SSE client, NullCloud sink, billing, summaries |
| `router/nimbus.py` | Shipped KV-gap baseline plus orthogonal `kv_gap` / `ttft_pred` triggers and selector ablations |
| `router/test_*.py` | 105 unit tests, no network/GPU (`python3 -m unittest discover -s router -p 'test_*.py'`) |
| `tools/kv_gauge_probe.py` | Live probe that established the Section-4 finding (stdlib only) |
| `tools/analyze_eb1200.py` | Timeline reconstruction that flagged the anomaly from a result JSONL |
| `tools/materialize_token_aligned_trace.py` | Atomic, tokenizer-fingerprinted no-cache trace materializer |
| `tools/materialize_sharegpt_current_turn_trace.py` | Verbatim current-turn retokenizer with source/decode provenance, overflow policy, and text-free manifest |
| `tools/profile_ttft_batch.py` | Same-payload warmup/profile with a held-out shared-prefill-lane calibration artifact |
| `tools/analyze_ttft_repeatability.py` | Parameterized block-level validator/aggregator; no request-level pseudo-replication |
| `tools/analyze_ttft_full_cell.py` | Exact five-marker/full-cell integrity audit and frozen-gate scorer |
| `tools/analyze_ttft_violation_context.py` | Bound request/decision/profile/server-context diagnostic; explicitly cannot replay selector causality |
| `tools/analyze_ttft_current_turn_gate.py` | Text-free E12 L/A/C auditor: recomputes fingerprints; binds raw/summary/decision/marker/events, payload mode, and token exactness; enforces order/cooldown |
| `tools/openrouter_deepinfra_price_snapshot.py` | Public, non-inference endpoint metadata snapshot with exact DeepInfra price/context validation |
| `tools/check_e12_live_budget.py` | E12 trace/manifest/price validator and exact full-decode two-arm static budget attestation |
| `tools/openrouter_usage_snapshot.py` | Whitelisted non-secret OpenRouter key limit/usage snapshot; never persists the key or provider error body |
| `tools/check_openrouter_stage_budget.py` | Offline `$3` canary/A/C staged-budget gate over usage snapshots |
| `tools/check_e12_retry_budget.py` | Cumulative retry guard: carries prior spend into launch/final accounting while requiring a fresh settled pair |
| `tools/run_openrouter_ttft_canary.py` | One fixed public synthetic first-token-cancel request; never reads the restricted trace |
| `tools/check_e12_stage_launch.py` | Reproducible A/C launch authorization plus last-moment live-usage verification receipt; binds completed A before C |
| `tools/audit_e12_live.py` | Final text-free A/C integrity, provider/cancel, usage, and budget auditor |
| `tools/ttft_matrix_evidence.py` | Tested arm parser and semantic/hash completion-marker validator, including `all_local` anchors |
| `experiments/run_ttft_selector_matrix.sh` | Server/profile/trace-bound runner with E12 staged launch receipts, exact trace-byte SHA enforcement, fail-fast cloud gates, and completion markers |

At final-audit tooling checkpoint `f3583a3`, the clean tracked tree passes
**287/287** offline tests: 105 router tests plus 182 tools/evidence tests:

```bash
python3 -m unittest discover -s router -p 'test_*.py'
python3 -m unittest discover -s tools -p 'test_*.py'
```

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
| random matched | target 0.256, realized 25.3%* | 108.7 s | 97.0% |

\*Provenance RESOLVED 2026-07-15 from the on-box summaries (`$MSCRATCH/
router_eb/`, all three arms' raw+summary intact, hashed, and mirrored — see the
Chinese ledger §6.4/§6.13): 25.6% is nimbus's realized self-selected fraction
(and the random arm's `target_fraction=0.256`); 25.3% is the random arm's
realized i.i.d. fraction (`actual_fraction=0.25274`). Both historical records
were correct about different quantities. Random summary re-check:
`ttft_p50_ms=108,651.19`, `slo_violation_pct=97.013` — matches this table.
Still exploratory (pre-token-alignment replay), but now re-computable.

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

A July-16 source/runner audit clarifies what those 14,537 rows mean. The sender
read `prompt_text` and issued exactly one user message per trace row; it did not
reconstruct cumulative conversation history. Therefore that dataset is evidence
about original ShareGPT **current-turn payloads**, not the long cumulative token
metadata stored beside them. Section 5j adopts that same payload semantic while
recomputing scheduler-visible tokens from the bytes actually sent.

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

Team decision: validate on a pure full-attention model (Qwen3-32B dense)
using the SAME team workload (ShareGPT + BurstGPT timestamps); the rednote
slice is dropped from the near-term plan. Executed same day; all results under
`$MSCRATCH/router_v32b/`, scripts `v32b_accept.sh` / `v32b_eb.sh` /
`v32b_rand.sh` in `$MSCRATCH`.

“Same workload” here means the same source rows/timestamps, not necessarily the
same HTTP bytes: the historical sender used only each row's current-turn
`prompt_text`, while later token-aligned cells synthesize payloads matching the
cumulative token metadata. Section 5j makes that distinction explicit.

**Server**: Qwen3-32B on physical GPU2, FLASH_ATTN, `--max-num-seqs 128`,
`gpu-memory-utilization 0.95` → `GPU KV cache size: 112,064 tokens`,
`Maximum concurrency for 40,960 tokens per request: 2.74x` (= 112,064/40,960 —
self-consistent, unlike the hybrid's 3.07×). Note: GPU0 wedged again
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
the project owner's sign-off before implementing.

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

**Execution note (2026-07-15, added while the campaign's profile stage was
still running — no matrix data existed yet).** A concurrent session had
already deployed `cac4a5d` (one commit before this registration) to the GPU
box and started the item-6 campaign: fresh server lifecycle (same recipe, KV
112,656 confirmed) plus a fresh lifecycle-1 profile. That session will likely
execute item 6 as written at `cac4a5d`: Latin square over
old/`newest`/current plus ≥3 separate `waiting_random` seeds — i.e. `newest`
inside the rotation and random arms not position-balanced. To avoid duplicate
GPU load and cross-contamination of its calibration, this session did NOT
launch a second campaign. The **decision rule above is frozen as-is** and
applies unchanged to that design (it needs paired old-vs-current blocks,
intra-arm spread, and ≥3 random seeds — all present); the random arms'
lack of position balancing is recorded as a limitation, and the executed arm
table will be documented from the run manifests, not assumed.

**Explicit non-goals.** This matrix cannot show generalization — it reuses
the same 512 requests as run 0, so it measures run-to-run noise and order
effects only. Generalization is assigned to the full 11,605-request cell
(11,093 requests untouched by any tuning), gated on this matrix, with
same-server anchor arms rerun from scratch: `kv_gap + cost_disp_current` (the
v3 default) and `all_local` — the July-13 dense anchors came from a
mis-calibrated TPOT (78 ms vs measured ~103 ms), a cache-enabled server, and
a non-token-aligned trace, and must not be cited for comparison. Debts that
block any default flip are unchanged (Section 7 item 8): estimated-vs-oracle
decode (trigger and weight both consume oracle decode lengths today), a
real-cloud leg (pessimistic combined ranks arms fairly only because every arm
shares NullCloud), the `max_cachedisp_old` ablation (is the cost division
load-bearing?), and the cache-aware `(P−C)` term.

### 5f. EXECUTED 2026-07-15 — six-block repeatability and order-sensitivity campaign

This section records what was actually executed; Section 5e is preserved as
the earlier registration rather than rewritten after seeing data. The exact
experiment checkout was clean, detached commit `cac4a5d`. One fresh
Qwen3-32B no-cache server, one profile, one trace, and one parent PID were used
throughout all 24 arms. A completed block-1 command and the final block-6
command were each repeated after completion: all four arms were hash-validated
and skipped, exercising the stable endpoint-identity/resume fix from
`93bf9f1`.

The algorithm states must not be conflated:

| Status | Trigger (when/how much) | Selector (whom) |
|---|---|---|
| Shipped repository default | `kv_gap` | `cost_disp_current` |
| Primary experimental candidate after this screen | `ttft_pred` | `cost_cachedisp_old` |
| Required comparators | `ttft_pred` | `cost_disp_current`, `newest`, `waiting_random` |

The candidate preserves the exact old V2 quantity

`P × (U / prefill_tput + D × TPOT)`,

where `P` is full prompt tokens, `U=P−cached_tokens`, and `D` is expected
decode tokens. It is used only as the denominator in
`cloud_cost / old_v2_weight` victim ordering. It is **not** the TTFT trigger,
not a capacity budget, and not the historical 0/1 DP. In pseudocode:

```text
if max predicted waiting TTFT > SLO - guard:
    rank waiting candidates by cloud_cost / old_v2_weight
    remove the minimum ranking prefix
    until every waiting survivor is predicted safe
```

The fresh lifecycle profile was independently valid and close to the July-14
profile, which is evidence for calibration stability rather than a reason to
pool the two server lifecycles:

| Profile check | July-15 lifecycle-1 result |
|---|---:|
| Profile SHA256 | `ab703ddc11bc63a2e1a3ca5e7b2367dc8232eb37f623a8e625a306021278c819` |
| Structural/token audit | 60/60 measured blocks; 1,420/1,420 measured requests exact |
| Effective shared prefill | 3,255.0044 tokens/s |
| Decode TPOT / first-token overhead | 152.3152 ms / 440.4170 ms |
| Fit | weighted R² = 0.958213 |
| Guard | calibration p99 residual 1,462 ms + one 250 ms tick = **1,712 ms** |
| Final held-out classifier | TP 193, FN 0, FP 21, TN 70 |

Define A=`cost_cachedisp_old`, B=`newest`, C=`cost_disp_current`, and Rn=
`waiting_random(seed=n)`, all under `ttft_pred`. Each block first used R as a
wash-in/run-state reference, then rotated A/B/C. The initial cyclic screen was
`ABC / BCA / CAB`; A−C routed direction changed from -14 to +2, triggering the
pre-specified reverse-order sensitivity complement `ACB / CBA / BAC`. Random
therefore always occupies the first position and is not an unbiased fourth
treatment.

| Block | Executed order | A routed / cost | B routed / cost | C routed / cost | R routed / cost |
|---|---|---:|---:|---:|---:|
| 1 | R0 → A → B → C | 265 / $0.137411 | 314 / $0.145494 | 279 / $0.147099 | 316 / $0.144312 |
| 2 | R1 → B → C → A | 265 / $0.137340 | 316 / $0.146687 | 263 / $0.148247 | 316 / $0.140795 |
| 3 | R2 → C → A → B | 265 / $0.137411 | 314 / $0.145494 | 263 / $0.148447 | 312 / $0.143266 |
| 4 | R3 → A → C → B | 264 / $0.137962 | 316 / $0.146687 | 264 / $0.148332 | 318 / $0.144277 |
| 5 | R4 → C → B → A | 264 / $0.137619 | 314 / $0.145494 | 264 / $0.149080 | 320 / $0.145183 |
| 6 | R5 → B → A → C | 265 / $0.137500 | 315 / $0.146480 | 264 / $0.148332 | 319 / $0.146248 |

Every row above is 512/512 successful with exact materialized prompt/decode
usage and zero measured local TTFT violations. Consequently the pessimistic
combined percentage is exactly `routed/512`; it is a reporting transform, not
a second independent safety result.

Whole-run distributions (the statistical unit is one 512-request run, never an
individual request):

| Arm | Routed mean / median / range | Cost mean / range |
|---|---:|---:|
| A old V2 | 264.667 / 265 / 264–265 | $0.137541 / $0.137340–0.137962 |
| B newest | 314.833 / 314.5 / 314–316 | $0.146056 / $0.145494–0.146687 |
| C current | 266.167 / 264 / 263–279 | $0.148256 / $0.147099–0.149080 |
| R wash-in | 316.833 / 317 / 312–320 | $0.144013 / $0.140795–0.146248 |

Paired A-minus-comparator results:

| Pair / metric | Mean | Median | Range | A / tie / other wins |
|---|---:|---:|---:|---:|
| A−B routed | -50.167 | -50 | -52…-49 | 6 / 0 / 0 |
| A−B pessimistic combined | -9.798 pp | -9.766 pp | -10.156…-9.570 pp | 6 / 0 / 0 |
| A−B cost | -$0.008516 | -$0.008404 | -$0.009347…-$0.007875 | 6 / 0 / 0 |
| A−C routed | -1.500 | +0.5 | -14…+2 | 1 / 2 / 3 |
| A−C pessimistic combined | -0.293 pp | +0.098 pp | -2.734…+0.391 pp | 1 / 2 / 3 |
| A−C cost | -$0.010716 | -$0.010870 | -$0.011461…-$0.009688 | 6 / 0 / 0 |

Interpretation: A robustly beats naive `newest`. A and C are route-equivalent
within the observed run/order band; the negative routed mean is driven by
block 1 and does not justify `old > current`. A is nevertheless consistently
cheaper than C (about 7.2% on mean cost), and its local TTFT p99 is lower in all
six blocks. Under Section 5e's frozen tie branch, A becomes the primary
candidate for the full-cell gate while C remains mandatory. No default changes.

Independent audit covered all 24 completion markers, all 72 raw/summary/
decision hashes and nonempty-line counts, fingerprints, common server/profile/
trace/config binding, exact local token usage, unique applied victims equal to
cloud rows, and the recorded `pre > 3.288 s`, `post <= 3.288 s` invariant. The
evidence boundary is also explicit: decisions do not save complete snapshot
IDs/ages/in-flight release state, so an external auditor can verify recorded
pre/post consistency but cannot reconstruct selector ranking or the minimum
prefix without trusting code at the exact commit. A future decision schema
should persist that replay state.

Reproduce the block-level table from copied result directories with:

```bash
python3 tools/analyze_ttft_repeatability.py "$CAMPAIGN"/block*
```

The tool exits 2 on failed requests, inexact local token accounting, or any
measured local violation; its JSON mode is intended for downstream tables.

### 5g. PRE-REGISTERED AND EXECUTED 2026-07-15 — 11,605-request generalization gate

Registered and committed before any full-cell arm was launched. The existing
full token-aligned trace contains 11,605 requests across the complete 1,199 s
`extreme_burst_1200` horizon; 11,093 were not in the repeatability slice. It is
compatible with the same live server/profile and therefore requires neither a
new materialization nor recalibration.

Five treatments use one explicit, server/profile/trace-bound matrix. Starting
from canonical `[A, C, B, K, L]`, independent pre-existing seed 78846 passed to
Python's `random.Random(...).shuffle` fixes the order before outcomes:

1. B = `ttft_pred:newest:0`
2. K = `kv_gap:cost_disp_current:0` (shipped-v3 anchor)
3. C = `ttft_pred:cost_disp_current:0`
4. A = `ttft_pred:cost_cachedisp_old:0` (candidate)
5. L = `anchor:all_local:0` (pressure anchor)

All arms fully drain and cool down for 20 s. One pass is a bounded held-out
generalization gate, not a full-scale replicate or a chance-control result.
There is no outcome-based early stopping: finish all five unless a preflight/
PID/endpoint/log/hash/config check fails, the engine/GPU fails, a request fails,
token/decode alignment is inexact, K records a KV-read failure, or a completion
marker cannot be produced.

Acceptance rules, frozen before launch:

- All 5 markers; every arm 11,605/11,605 successful; exact materialized prompt
  and controlled decode usage for every retained-local row.
- A, B, and C each have zero measured local 5 s violations, the expected
  waiting-only/shared-prefill prediction telemetry, and recorded maximum
  post-kick waiting TTFT no greater than `5−1.712 = 3.288 s` (float epsilon).
- Let the observed 512-screen maximum intra-arm band be
  `16/512 = 3.125 pp` (363 full-cell rows). If `|A−C|` is within that band,
  retain the route-equivalence claim. If C beats A by more than the band,
  reject the candidate/default flip. If A beats C by more than the band, the
  gate passes but it is only a one-pass full-cell advantage, not replicated
  proof that old > current. In every case A must cost no more than `1.05 × C`.
- A must beat B by at least 5 percentage points pessimistic combined, or the
  selection signal does not generalize.
- K must have zero KV-read failures. A must Pareto-improve K on local safety
  and pessimistic combined (no worse on either, strict on at least one). If K
  is safe with fewer routes, there is no trigger/default-flip case.
- L must confirm genuine pressure: at least 90% local violations and local
  TTFT p50 above 5 s. A failed outcome criterion falsifies the corresponding
  claim but is not an infrastructure reason to terminate later arms.

#### Executed result

The matrix ran at detached clean commit `c6de62a` in the frozen B/K/C/A/L
order, with no outcome-based stop. All five arms used the same token-aligned
trace, dense Qwen3-32B no-cache server lifecycle, schema-v2 profile, model,
prices, 128-request client cap, and 5 s SLO. The exact marker/artifact and gate
audit is reproduced by `tools/analyze_ttft_full_cell.py`; the B diagnostic is
reproduced by `tools/analyze_ttft_violation_context.py`.

| Arm | Routed / local | Local TTFT p50 / p95 / p99 | Local violations | Pessimistic combined | Cost |
|---|---:|---:|---:|---:|---:|
| A: TTFT + old V2 | 6,748 / 4,857 | 1.795 / 2.612 / 2.867 s | **0 / 4,857** | 58.147% | **$3.344049** |
| B: TTFT + newest | 8,591 / 3,014 | 2.063 / 3.316 / 5.354 s | **39 / 3,014 (1.294%)** | 74.364% | $3.643557 |
| C: TTFT + current displacement | 6,729 / 4,876 | 1.959 / 2.752 / 2.974 s | **0 / 4,876** | **57.984%** | $3.519140 |
| K: shipped `kv_gap` | 6,060 / 5,545 | 18.277 / 21.946 / 23.537 s | **5,249 / 5,545 (94.662%)** | 97.449% | $3.477171 |
| L: all-local pressure anchor | 0 / 11,605 | 1,951.571 / 3,818.696 / 3,982.782 s | **11,544 / 11,605 (99.474%)** | 99.474% | $0 |

A and C are equivalent in **route count**, not in selected identities. Their
routed sets intersect on 5,452 requests but differ on 2,573 (Jaccard 0.679;
A-only 1,296, C-only 1,277). The A-only set has prompt/decode p50 527/160;
the C-only set has 186/315. This is consistent with old V2's outer `P` versus
current displacement's outer `(P+D)`, and the two exclusive sets account for
A's $0.175091 lower cost. It is not an independent per-snapshot ranking proof:
C ran before A in one lifecycle, dynamic feedback differs, and the decision
schema lacks the state needed to replay either ordering.

The pre-registered comparisons (whole-arm results; the 11,605 requests are not
treated as independent replicates) are:

| Frozen gate | Result | Observation |
|---|---:|---|
| Evidence integrity | **PASS** | Five exact markers/hashes; each arm 11,605/11,605 successful with exact local tokens |
| L pressure anchor | **PASS** | 99.474% local violations and p50 1,951.571 s >5 s |
| A/B/C local safety | **FAIL** | A=0, B=39, C=0 local violations |
| A/B/C post-kick bound | **PASS** | maxima 3.287906 / 3.287984 / 3.287990 s ≤3.288 s |
| A vs C equivalence | **PASS** | A-C=+19 rows (+0.1637 pp), inside inclusive ±363-row band |
| A cost vs C | **PASS** | A/C cost ratio 0.950246 ≤1.05 |
| A vs B signal | **PASS** | A improves pessimistic combined by 16.217 pp |
| A Pareto vs K | **PASS** | A is strictly better on both local and pessimistic-combined violations |

The full gate therefore fails the frozen selector-independent safety claim
even though both resource-sensitive orderings A and C are locally safe. B's 39
violations are real, but the decision schema cannot replay selector ranking or
assign request-level selector causality. The bound diagnostic establishes the
following narrower chain:

- 32/39 had service TTFT alone above 5 s; the other 7 crossed 5 s only after
  adding client queue delay. Queue delay alone explains none of the 39.
- All 39 were dispatched with 126–128 client-visible active local HTTP
  requests and planned `prompt + requested decode` commitment above both the
  profile's 98,304-token cell maximum and the 112,656-token declared-capacity
  proxy. That planned commitment is not measured KV occupancy.
- B-arm local-request TPOT p50 was 1.259× its calibrated value; among violating
  requests it was 1.371×. Yet the closest recorded pre-dispatch decisions still
  reported post-kick maxima of 2.543–3.285 s.
- The engine-reported cache-usage gauge was 98.1–100% near all 39 first-token
  timestamps, but that gauge is architecture-dependent and the explicit clock
  alignment reuses periodic samples. It is correlation, not a token-KV causal
  claim.

**Algorithm conclusion.** The exact old V2 token-seconds formula survives as a
victim-ordering signal: A is safe, route-equivalent to C under the frozen band,
and about 5% cheaper in this full cell after also costing less in all six
repeatability blocks. What this cell falsifies is `kv_gap` as a sufficient
TTFT-safety trigger: K routed 52.2% but left 94.7% of retained-local requests in
violation. TTFT violation remains the correct objective/trigger direction;
the predictor must gain an explicit calibrated support envelope and
deployment-resource fallback so a future selector-independent safety claim
does not rely on the resulting local mix. No repository default is changed by
this experiment, and the cache/decode-estimator/real-cloud/sweep debts still
block one.

The next implementation target is therefore the following control loop; it is
a design target, not code shipped by this commit:

```text
state = calibrated deployment features
        (waiting age/work, active sequence occupancy, shared-prefill work,
         planned commitment/workload mix, and any deployment-validated gauge)

if state is outside the profile support envelope:
    apply a conservative admission/spill fallback
elif max predicted waiting TTFT > SLO - guard:
    rank waiting requests by cloud_cost / exact_old_v2_weight
    spill the shortest prefix whose survivors are predicted safe
else:
    admit normally
```

This keeps the original v2 `P × (U/prefill_tput + D × TPOT)` formula in
the decision, but in the place its unit supports: victim ordering. It does not
invent a token-second capacity budget. TTFT remains the stop condition, while
KV/sequence/compute measurements describe deployment state and predictor
support.

Reproduce the frozen gate and the bound B diagnosis. The gate command exits 1
for this valid-but-failed outcome; exit 2 means invalid evidence:

```bash
FULL=$MSCRATCH/router_ttft_full_c6de62a_20260715/full11605_bkcal
PROFILE=$MSCRATCH/router_ttft_repeat_cac4a5d_20260715/ttft_profile_v2_cac4a5d_lifecycle1.json
SERVER_LOG=$MSCRATCH/router_ttft_repeat_cac4a5d_20260715/vllm_qwen32b_nocache.log

python3 tools/analyze_ttft_full_cell.py "$FULL" \
  --json-out "$FULL/full_cell_gate.audit.json" \
  --markdown-out "$FULL/full_cell_gate.audit.md"

python3 tools/analyze_ttft_violation_context.py \
  --raw "$FULL/extreme_burst_1200_ttft_pred_newest_seed0.jsonl" \
  --decisions "$FULL/extreme_burst_1200_ttft_pred_newest_seed0.decisions.jsonl" \
  --summary "$FULL/extreme_burst_1200_ttft_pred_newest_seed0.summary.json" \
  --marker "$FULL/extreme_burst_1200_ttft_pred_newest_seed0.complete.json" \
  --manifest "$FULL/matrix_manifest.txt" --profile "$PROFILE" \
  --server-log "$SERVER_LOG" --server-log-arm-start '07-15 06:14:14' \
  --json-out "$FULL/B_violation_context.audit.json" \
  --markdown-out "$FULL/B_violation_context.audit.md"
```

### 5h. EXECUTED 2026-07-16 — OpenRouter first-token cancellation probe

**Decision question.** Can a real-cloud leg measure Nimbus's TTFT SLO and stop
after the first generated token without pretending that an incomplete response
has E2E/TPOT/complete-usage evidence? The implementation captured by commit
`e0b6686` adds three opt-in cloud controls: a fixed OpenRouter provider order,
fallback disablement, and `--cloud-stop-after-first-token`. A probe row is a
successful TTFT measurement but explicitly has `response_completed=false`,
`e2e_ms=null`, `tpot_ms=null`, and pending cost until provider metadata exists.
Default/full-drain behavior is unchanged. The resulting tree passed 81 router
tests and 36 evidence tests (**117/117**), plus `py_compile` and
`git diff --check`.

**TTFT semantic correction.** The first attempt stopped only on non-empty
`delta.content`. That is not comparable to the bound local deployment: local
vLLM ran without a reasoning parser and therefore streams Qwen `<think>` tokens
through `content`, whereas OpenRouter exposes them separately through
`delta.reasoning` (or `reasoning_content`). The content-only request exhausted
its 261-token cap entirely in reasoning, produced no visible content, completed
instead of cancelling, and recorded client E2E 7.373 s / client-estimated cost
$0.0000958 (the generation record reported $0.000094842). The corrected Nimbus
definition is therefore **time to the first non-empty generated-token delta,
whether reasoning or content**. `first_token_kind` records which boundary fired;
`first_content_ttft_ms` remains separate when available.

**Protocol.** Real model `qwen/qwen3-32b`; provider pinned to `deepinfra`;
`provider.allow_fallbacks=false`; streaming; temperature 0; 5 s SLO; no
OpenRouter response-cache header; cancel immediately after the first reasoning
or content token. The burst smoke used the first 16 rows of the existing
token-aligned held-out trace: 7,658 materialized prompt tokens, 4,501 total
decode-token cap, and a 1 s trace-arrival span. This is a transport/measurement
smoke, not a random sample or a policy comparison.

| Check | Result |
|---|---:|
| Requests / HTTP successes / errors | 16 / 16 / 0 |
| Client abort requested / full response completed | 16 / 0 |
| First-token kind | 16 reasoning / 0 content |
| TTFT min / p50 / p95 / p99 / max | 349 / 467 / 929 / 1,167 / 1,167 ms |
| Observed TTFT violations at 5 s | **0 / 16** |
| Raw `cost_pending_n` | 16 (the final usage SSE is intentionally not read) |

The client-side transport result is 16/16, but it would be incorrect to claim
16/16 provider-side cancellations. A delayed generation audit found only one
of the 16 records. That record named DeepInfra, had `cancelled=true`, 847 native
prompt tokens, 6 completion tokens, and cost $0.000026; the API-key cumulative
usage increased by the same $0.000026. The other 15 generation IDs returned
not-found at audit time. They remain **unknown/pending**, not zero-cost rows.

Evidence lives under `$MSCRATCH/openrouter_ttft_cancel_20260716/`, with a
repo-sibling local mirror outside git. The 16-row evidence hashes are:

- raw JSONL: `e9bc6c8c72e98707ca08fd65b87a58523e6bff042ee1538a341e2453f1ccb9b5`;
- summary: `0a45bebed96e00fa19fcdd4ea81f379f33946cd02f627bf36b1c4b81980bc4d0`;
- billing audit: `e147071f5622a8b74a5c351ad3a525b5bddfa99c6c84c2deaeb60b2984033cab`.

**Claim boundary.** The fixed-provider abort/TTFT path and local/cloud
first-token semantics are validated. The observed latency distribution is
exploratory (n=16, one provider, one short burst). It does not measure E2E,
TPOT, visible-answer completion, mid-stream reliability, or complete cost. The
working assumption that first-token cancellation does not change upstream cloud
load is an explicit experiment assumption authorized on July 16, **not a result
of this probe**. Under that assumption, TTFT-cancel rows may be used in the next
observed-combined TTFT cells; any paper claim that needs completed-service
behavior still requires a full-drain sensitivity leg.

### 5i. PRE-REGISTERED 2026-07-16 — direct full real-cloud A/C

**Decision and deviation from the queue.** The project owner chose to skip both the
512-request frozen-route shadow and the 512-request live pilot and proceed
directly to the complete 11,605-request live-hybrid A/C pair. This increases
spend and makes a bad configuration more expensive, but it does not change the
measurement contract. The run must not start until the split local/cloud
`ignore_eos` controls, split cloud-wait telemetry, real-cloud marker validator,
and this pre-registration are committed. The three code prerequisites are
captured by `4c18ae6` (83 router tests plus 41 tools tests passed). The
operational OpenRouter budget target is **$1.20–$1.50** for both arms under the
prompt-dominated cancellation behavior seen in E10; this is not a guaranteed
provider-side cap, and pending generation records must not be reported as zero
cost.

**Frozen inputs and order.** Use the existing full token-aligned trace, exactly
11,605 rows over 1,199 s, SHA256
`465ef070d2a4a399ad41142b9e40bd9c505599d05af2f9d4dd56f9eb02024c52`;
its manifest SHA256 is
`c5621d3e45f7b1e2ee49485f7948a267dde65d29b34a377247061f7cccc72f7a`.
Start a fresh no-prefix-cache Qwen3-32B lifecycle on the default healthy shared
GPU, generate a fresh schema-v2 profile bound to that exact PID/log/code tree,
and run one pair in the frozen order **A then C** (order-seed label
`20260716`):

- A: `ttft_pred + cost_cachedisp_old`, seed 0;
- C: `ttft_pred + cost_disp_current`, seed 0.

Both arms use the same server lifecycle, profile, trace, 5 s TTFT SLO,
temperature 0, and 20 s inter-arm cooldown. Local requests use the controlled
residence workload (`local_ignore_eos=true`) and must exactly match materialized
prompt and requested decode tokens. Cloud requests use
`qwen/qwen3-32b`, provider order `[deepinfra]`,
`allow_fallbacks=false`, no response-cache opt-in, `cloud_ignore_eos=false`, a
16-request pre-first-token concurrency gate, and client abort after the first
non-empty reasoning or content token. Freeze the July-16 DeepInfra list prices
shown by [OpenRouter](https://openrouter.ai/qwen/qwen3-32b/pricing)—input
`$0.08/M`, output `$0.28/M`—as `in_price=0.08` and `out_price=0.28` for both
billing estimates and selector scores. A later price change does not alter
these registered arms.

**Frozen metric.** For every request, TTFT begins at trace arrival. A cloud
row's value is:

```text
arrival_to_first_token_ms
  = pre_route_queue_ms + cloud_gate_wait_ms + service_ttft_ms
```

The primary outcome is `summary.overall.slo_violation_pct`. An error, timeout,
or stream ending without a generated token is a measured violation. The
historical `pessimistic_combined` field is not a real-cloud headline. Report
local and cloud violation counts and TTFT distributions separately, plus route
fraction, pre-route wait, cloud-gate wait, HTTP/error classes, requested and
observed provider metadata, and known/pending cost coverage.

**Integrity contract.** Each arm must have exactly 11,605 unique raw rows and a
complete marker binding the raw/summary/decision hashes, trace, manifest,
profile, server PID/log and non-secret cloud configuration. It must satisfy
`overall.slo_measured_n == overall.n == 11605` and
`cloud.routed_only == 0`; applied victim IDs must equal cloud request IDs;
successful local rows must be token-exact; and every successful cloud row must
have finite arrival-to-first-token TTFT, `stream_abort_requested=true`, and
`response_completed=false`. Endpoint failures remain in the denominator and
are never silently retried away. Stop only for an invalid experiment—binding or
hash mismatch, systematic authentication/configuration rejection, corrupt
output, or local-server death—not for an unfavorable SLO outcome.

**Outcome interpretation frozen before launch.** Zero retained-local
violations is a safety target, not an integrity prerequisite. This is one
ordered A/C pair, so it can establish feasibility and give an observed-combined
effect estimate, not a replicated selector-superiority claim. If the absolute
A/C difference is below 1 percentage point, call it unresolved and run a
reverse-order pair before naming a winner; a larger one-pass difference is
still preliminary and must be reported with the order limitation. The run does
not measure completed-response E2E, TPOT, answer quality, mid-stream
reliability, or full-response deployment cost. It adopts the user-authorized
temporary assumption that first-token cancellation does not change upstream
cloud load; that assumption remains unvalidated.

**Execution checkpoint — profile complete; formal arms not started.** A clean
detached `2b53ff3` checkout started a fresh Qwen3-32B lifecycle on the selected
healthy GPU. The bound server used bfloat16, FlashAttention, no prefix cache,
`max_model_len=40960`, `max_num_seqs=128`, and a startup-log KV capacity of
112,064 tokens. Parent PID `1833193` and children `1833493,1833494` were
recorded. The fresh schema-v2 profile passed all local gates:

| Profile check | 2026-07-16 lifecycle result |
|---|---:|
| Profile SHA256 | `8a4c0057697112a21778365b8e00f00960953c94911db349fca3cd21b9d21c3e` |
| Structural/token audit | 60/60 blocks; 1,420/1,420 success, prompt exact, and decode exact |
| Effective shared prefill | 3,268.6112 tokens/s |
| Decode TPOT / first-token overhead | 151.7528 ms / 444.5446 ms |
| Fit | weighted R² = 0.957901 |
| Guard | calibration p99 residual 1,448 ms + one 250 ms tick = **1,698 ms** |
| Final held-out classifier | TP 195, FN 0, FP 19, TN 70 |

The live OpenRouter endpoint inventory still showed DeepInfra at `$0.08/M`
input and `$0.28/M` output, FP8, context 40,960. The selected API key had
`$187.559401086` of its own limit remaining. These were non-inference metadata
GETs only.

The formal matrix launch was then stopped by the external-data-export safety
gate **before its remote command executed**. The aligned trace does not contain
the source ShareGPT dialogue: `tools/materialize_token_aligned_trace.py`
replaced every prompt with a deterministic 12-hex nonce plus repeated
`calibration` filler, and the HTTP payload does not include `session_id`.
Nevertheless, sending 11,605 synthetic prompts, token-size distribution and
arrival schedule to OpenRouter/DeepInfra is still a bulk third-party transfer
and now requires explicit informed user approval. Post-block usage audit proved
key usage delta `$0`, account usage delta `$0`, no matrix PID/output directory,
and **zero formal inference requests**.

Because the matrix did not start, the registered A/C gates and status remain
unchanged and there is no real-cloud outcome to interpret. The server was
terminated immediately by its recorded PID; both recorded children and port
8010 were verified gone and GPU memory returned to the pre-launch level. The
profile is PID/log-bound, so a later approved run must start a new lifecycle
and re-profile rather than reuse this artifact. Evidence is under
`$MSCRATCH/router_realcloud_full_2b53ff3_20260716T1435Z/` with a verified local
repo-sibling mirror outside git at
`artifacts/realcloud_full_prerun_2026-07-16/`.

### 5j. POST-RUN STATUS LABEL — PRE-REGISTERED 2026-07-16; LOCAL GATE EXECUTED — ShareGPT current-turn external-validity leg

*Post-run annotation: the block beginning with the status below and ending at
the export authorization boundary is the unchanged pre-registration frozen at
commit `78da644`; the executed outcome is appended only after that block.
Artifact/event dates use UTC: the formal stages ran 2026-07-16 17:10–18:38
UTC, which is 2026-07-17 01:10–02:38 in Asia/Shanghai (UTC+08:00).*

**Status: TRACE MATERIALIZED / LOCAL ARMS NOT RUN / NO ORIGINAL TEXT
EXPORTED.** This cell was defined and hashed before starting any inference arm.
It does not retroactively relabel E6–E11 or merge their outcomes.

**Why there are two legs.** The source trace contains two properties that the
available payload cannot preserve simultaneously:

- `prompt_text` is the original ShareGPT current user turn;
- `num_prefill_tokens` is historical cumulative-conversation metadata.

The source file does not carry the exact cumulative chat payload that produced
that metadata. E6/E9/E11 therefore replace text with deterministic unique
filler so HTTP payload tokens match the cumulative sizes; those cells answer
whether the scheduling mechanism works under the target pressure distribution.
E12 instead preserves each original current-turn string byte-for-byte and
recomputes its tokens; it answers whether the policy generalizes to real
current-turn text and the different pressure mix that text induces. These are
two different estimands, not two standards for accepting the same result, and
their outcomes must never be averaged.

An audit of the teammate's historical workload and sender fixed the payload
contract before materialization: the sender selected `prompt_text` and wrapped
it as `messages=[{"role":"user","content": prompt_text}]`. It did not replay a
full conversation. “Current-turn-payload compatible” in this section refers
only to that semantic; the visible sender postdates the historical results and
its decode/runtime behavior differs, so E12 is not an exact reproduction.

**Frozen transformation and provenance.** Implementation commit `6054b32`
adds `tools/materialize_sharegpt_current_turn_trace.py` (tool SHA256
`39315185886e993bdf8b6fd6b0456017a3b6c7d50c926cb35bec953996227f4a`)
and extends matrix preflight to bind both current-turn and historical synthetic
materializers. The source ShareGPT+BurstGPT file has SHA256
`bf790b87eb61ba486a21155d0b6a417ad7ba6fb6abe0ff33a60ca155ace1ad0f`.
Selection reuses the inclusive `extreme_burst_1200` window and stable
`(arrived_at, source_index)` order. For each selected row:

```text
messages = [{"role": "user", "content": source.prompt_text}]
P = len(Qwen3-32B_tokenizer.apply_chat_template(
          messages, tokenize=True, add_generation_prompt=True))

prompt_text             = source.prompt_text       # verbatim
num_prefill_tokens      = P
uncached_prompt_tokens  = P
num_cached_tokens       = 0
num_decode_tokens       = min(source.num_decode_tokens, 1024)
```

No prompt is truncated. A row is dropped only when `P + D > 40,960`, and the
drop is recorded by source index and token counts in the text-free manifest.
The pre-arm audit found exactly one such row: zero-based source index 15,944 has
`P=221,051`, `D=17`, total 221,068. It was discovered before any policy outcome
and is dropped for every arm. Five emitted rows have source decode length above
1,024 and use the frozen cap while retaining `trace_num_decode_tokens`.

The final restricted trace is under
`$MSCRATCH/sharegpt_current_turn_6054b32_20260716/`; it is not committed or
copied into a general artifact mirror because it contains original text.

| Frozen trace check | Value |
|---|---:|
| Selected / emitted rows | 11,605 / **11,604** |
| Arrival range / span | 1,260,532–1,261,731 / 1,199 s |
| Actual `P` min / p50 / p95 / p99 / max | 9 / 26 / 463 / 1,699 / 10,795 |
| Actual emitted prompt-token sum | **1,289,405** |
| Historical cumulative prompt-token sum on emitted rows | 7,887,915 |
| Emitted decode-token p50 / p95 / max / sum | 238 / 630 / 1,024 / 3,038,796 |
| Decode-cap / context-drop affected rows | 5 / 1 |
| Trace SHA256 | `e838016a8e55660c565dadb1ad019770f6b88f878d8ca29f165c30887d2cb410` |
| Manifest SHA256 | `698bb94a82d133b0c54aa87d8badd4f181140c352cb29c1e726cfe2b291bf9a8` |

The manifest binds the materializer, `router/common.py`, shared tokenizer
helpers, source/output hashes, tokenizer vocab SHA, and Qwen chat-template SHA.
All 11,604 emitted rows passed preservation/alignment checks; the full tree
passed 83 router plus 53 tools tests (**136/136**), `py_compile`, `bash -n`, and
`git diff --check` before this registration.

**Frozen local-only gate.** Before any third-party POST, start one fresh
Qwen3-32B lifecycle with bfloat16, no prefix cache, `max_model_len=40960`, and
`max_num_seqs=128`. Generate a fresh schema-v2 profile bound to that PID, log,
endpoint, tokenizer, and checkout. The target profile cells are:

```text
1x16,1x32,1x512,1x4096,1x32768,
8x16,8x32,8x512,8x4096,
16x512,16x4096,32x2048,64x1024,128x32,128x512
```

Use five repeats, decode 256, seed 0, the profile's held-out zero-FN gate, and
the same 250 ms Nimbus tick. Then run the complete trace as three staged runner
invocations in frozen order:

1. L — exact arm `anchor:all_local:0`;
2. A — exact arm `ttft_pred:cost_cachedisp_old:0`;
3. C — exact arm `ttft_pred:cost_disp_current:0`.

Each invocation has its own `OUT_DIR`, matrix manifest, and fingerprint because
the runner binds the exact arm list. All three nevertheless share the same
lifecycle/profile/trace and identical `time_scale=1.0`, 5 s TTFT SLO,
temperature 0, `local_ignore_eos=true`, `CLOUD=null`, `timeout_s=7200`, and
prices `$0.08/M` input plus `$0.28/M` output. Validate the completed stage and
wait at least 20 s before starting the next. NullCloud is only a non-network
sink for selected victims; no HTTP request leaves the team machine.

Each arm must have 11,604 unique rows and a valid marker/hash binding. L must
have 11,604 successful local rows with exact prompt and decode usage. A/C
successful local rows must be token-exact, and their applied victim IDs must
exactly equal their NullCloud row IDs. L is the observed pressure anchor: the
pre-arm queueing calculation suggests pressure remains sequence/decode-bound,
but that calculation is not accepted as an outcome. If L has zero 5 s TTFT
violations, report that this deployment needs no offload for this workload and
do not start A or C. Otherwise start A after its cooldown; if A retains any
local violation, report that the safety gate failed and do not start C.
Otherwise run C; any retained-local C violation fails the final safety gate.
Do not tune the guard after seeing outcomes under this registration.

**Claim and export boundary.** A passing local gate shows only that the
TTFT-triggered selectors safely shed this real current-turn workload on one
deployment. It does not validate full-conversation replay, the cached-token
term, prefix-cache displacement, answer quality, cloud TTFT, or a selector
winner from one ordered A/C pair. The single 221k drop must remain explicit in
all denominators and comparisons.

A later live A/C requires a new lifecycle/profile, a separately frozen live
contract, and this explicit authorization before the first POST:

> I confirm that I have the right to send the selected 11,604 original
> ShareGPT current-turn texts and their request timing to OpenRouter/DeepInfra,
> understand that they may contain personal or sensitive content, and authorize
> the registered first-token-cancel experiment.

Agreement with the two-leg methodology is not that export authorization. Until
it is received, original prompts remain on restricted scratch and only the
local-only L/A/C gate is permitted.

#### Executed local-gate result (appended after completion)

The text above is the frozen pre-registration. The outcome below was appended
only after the three formal stages completed. The formal profile and L/A/C
stages ran on clean execution commit
`78da6448af18159cb0a755626f3dbee42a90361e`; the analyzer was added only
afterward. Authoritative evidence-audit commit `cdf7f16` contains
`tools/analyze_ttft_current_turn_gate.py`; the analyzer
recomputed each runner fingerprint, verified every marker against the current
raw/summary/decision bytes, checked common trace/profile/lifecycle inputs,
enforced current-turn/no-cache payload identity and exact token usage, confirmed
that applied victims exactly equal NullCloud rows, and bound the stage-event
order and cooldowns. Its text-free JSON and Markdown reports have SHA256
`7bdeb256372f62f5cd8fdca10a1b91d2a7877c9546ce0f0e30f26b90f5fc7548`
and `dfa916eceea3c4c59a6194e127299bfb49bfeb6b231233842ea6b3aae38b5dee`.

**Lifecycle/profile.** The first launch accidentally retained vLLM's default
prefix-cache setting. It was detected from the log and terminated before any
profile cell or inference arm; this configuration drift is retained as an
aborted lifecycle, not hidden. The second launch explicitly used
`--no-enable-prefix-caching`: Qwen3-32B, bfloat16, `max_model_len=40960`,
`max_num_seqs=128`, FLASH_ATTN, measured KV capacity 112,656 tokens. All 15
registered cells completed 75/75 blocks and 2,105/2,105 measured requests with
exact prompt/decode usage. The frozen profile fit was:

| Profile field | Result |
|---|---:|
| Prefill throughput | 3,242.2097 tokens/s |
| TPOT | 151.6079 ms |
| First-token overhead | 353.0143 ms |
| Weighted R² / guard | 0.963029 / 1,735 ms |
| Held-out n / TP-FN-FP-TN | 421 / 188-0-26-207 |

One held-out request had underprediction larger than the recommended guard;
therefore the valid claim is zero held-out **5 s classification** false
negatives, not pointwise guard coverage of every request.

**Formal L→A→C outcome.** Every stage had 11,604/11,604 successful unique rows
and exact retained-local prompt/decode accounting. Prices were frozen at
$0.08/M input and $0.28/M output; the dollar column is a modeled
full-response outsource cost for selected rows, not actual spend. A separate
event-log audit within the analyzer confirmed L→A→C order and cross-matrix
cooldowns of 94 s and 62 s, both above the registered 20 s minimum.

| Arm | NullCloud routes | Retained local | Local 5 s violations | Local TTFT p50 / p95 / p99 | Peak waiting | Modeled cost |
|---|---:|---:|---:|---:|---:|---:|
| L `all_local` | 0 | 11,604 | **11,400 / 11,604** | 644,578 / 1,304,490 / 1,363,365 ms | 6,149 | $0 |
| A `old-v2` | 5,109 (44.028%) | 6,495 | **0 / 6,495** | 1,393 / 2,003 / 2,176 ms | 26 | $0.500634 |
| C `current` | 4,099 (35.324%) | 7,505 | **0 / 7,505** | 1,490 / 2,165 / 2,490 ms | 28 | $0.51621328 |

The frozen pressure and retained-local safety gates therefore pass. A routed
1,010 more rows than C (+8.704 percentage points; 24.64% relative to C), yet
its modeled cost was $0.01557928 (3.018%) lower because it selected more but
cheaper victims. Conversely, C's NullCloud selected-route count was 19.77%
lower than A's; if mapped one-for-one to a future live run, that would imply
fewer potential API calls and text exposures. Their victim sets materially
differ: intersection 2,574,
A-only 2,535, C-only 1,525, union 6,634, Jaccard 0.3880. This directly confirms
that the two selectors made materially different choices in E12, but it still
does not identify a winner: the cell is one fixed A→C order. A's 314.177 ms
lower observed local p99 may include order or
within-lifecycle temporal/server-state drift and is not a causal selector
estimate.

**Strict boundary.** The result says that `ttft_pred` plus either registered
selector safely sheds enough load for the requests retained locally on this
one current-turn, no-cache deployment. It does **not** say that all 11,604
requests met an end-to-end SLO: the 5,109/4,099 routed rows were NullCloud rows
and have no cloud TTFT. It also does not validate the cached-token term
(`U=P` here), full-conversation replay, the dropped 221k outlier, answer
quality, real provider behavior, or the repository's shipped `kv_gap` default.
Known decode lengths were capped from the trace, so an oracle-estimate boundary
also remains. External POSTs were zero, actual cloud spend was $0, and original
text never left the team host.

**Evidence and cleanup.** The experiment root is
`$MSCRATCH/sharegpt_current_turn_local_78da644_20260716/`. Important SHA256
bindings are:

| Artifact | SHA256 |
|---|---|
| Lifecycle environment / aborted lifecycle-1 log / lifecycle-2 server log | `de4d3a75e88a542aae3b00b4131ba8ff8c846aad8e08485bd58452225abe0b54` / `aa578bd3d43383f76a579f66c17c7d327b5ac60cb9b78b98e4a38a1af1e577f0` / `46e9c68eabb61a39d4791005b5c810b882e9507db3a6d66849c792032e974bc9` |
| Profile / profile stdout | `bb35dcc01fd661e8b9a1b428cdcd898dbed2f84f041a30bd30c444098d81e4cb` / `ee295077c12e724b935025935248b8d78339a84df2619327282dfc1ffab40a99` |
| L raw / summary / marker | `d182eeb2468273a0de0a7a93b33af9ac0ac26d920ff42f6844da4b7f76c32ce2` / `71e3ff1364dfb4a2d9a2ce2588281c48dab633d7d1fe5abb6bca4dac7c5fcf1a` / `63cec2778f1a17728fbd15b4c26a3f53e82b55b4d03eb078c77cdefa53166301` |
| A raw / summary / decisions / marker | `ee26d46dbe3023d7cd64ef8c6f83cc20ea7af0877469f9d34d256056661f2c1b` / `7721c91cfce7af8e078e3e71b47659aca4624167a621e0c63a3cd172a538bf67` / `5d5ecac2784f2456828162d74302601fab2b6b132325aabb21bb8c32449505c8` / `42846a87186366605f036eb8b0d4d3b48aea2fa8b42ed8a11c05274778571d32` |
| C raw / summary / decisions / marker | `c965c02470151b6960bd52fa4cd9457a4e306905cc219501d0761c973f11cbf` / `d5b1b0e9cf8b855a7cc62026edeb13b73e26ed258f9b05b4029debace0cfd2d4` / `7ab9b533d41e3fcd6e73c0665cbc0f11ae412b417b2b90ed83198957e5562dde` / `7f0a789d485fc0407fbb9d8a321a647c8c73ae2fba79e89d3c1556f253d4c713` |
| L / A / C matrix events | `371dec09ba42053cf6516ec82e9faf51b3509eee47c6ce772b075057e10216bb` / `439b46169af6f126568f23f20726705bc2482193500ad93cf76a7c5e43f16ff7` / `135c4fc75788869881bf67707bdcfeecad944cd9290d0fe224b56098a47fd11f` |

To reproduce the aggregate, prompt-free gate audit on the experiment host:

```bash
RUN="$MSCRATCH/sharegpt_current_turn_local_78da644_20260716"
python3 tools/analyze_ttft_current_turn_gate.py \
  --l-dir "$RUN/stage_L_all_local" \
  --a-dir "$RUN/stage_A_old_v2" \
  --c-dir "$RUN/stage_C_current" \
  --expected-n 11604 \
  --expected-trace-sha256 e838016a8e55660c565dadb1ad019770f6b88f878d8ca29f165c30887d2cb410 \
  --expected-trace-manifest-sha256 698bb94a82d133b0c54aa87d8badd4f181140c352cb29c1e726cfe2b291bf9a8 \
  --expected-profile-sha256 bb35dcc01fd661e8b9a1b428cdcd898dbed2f84f041a30bd30c444098d81e4cb \
  --min-cooldown-s 20 \
  --json-out "$RUN/e12_current_turn_local_gate.audit.json" \
  --markdown-out "$RUN/e12_current_turn_local_gate.audit.md"
```

All recorded server/profile/stage processes were stopped, port 8010 was free,
and GPU memory returned to the free baseline. At that local-gate checkpoint,
the tree passed 83 router plus 64 tools tests (**147/147**), including 11/11
focused analyzer tests, plus `py_compile` and `git diff --check`; the current
E12-live total and commands are recorded in Section 2.

### 5k. E12 live current-turn A/C first-token-cancel — old lifecycle TERMINAL PARTIAL; fresh retry COMPLETE

**Status (updated 2026-07-21): lifecycle 1 is TERMINAL PARTIAL; lifecycle 2
completed A→C and its final audit passed.** These are independent lifecycles
with independent profiles and run fingerprints. They share the frozen trace,
deployment, provider, selector, and `$3` authorization contract below, but an
arm from one lifecycle must never be paired with an arm from the other. The user
supplied the following informed authorization on 2026-07-18:

> 我确认有权将这 11,604 条原始 ShareGPT current-turn 文本及其到达时序发送给
> OpenRouter/DeepInfra；我了解其中可能包含个人或敏感内容，并授权运行 A/C 两臂的
> 首-token-cancel 实验，费用上限为 3 美元。

This authorization covered the two registered live trace arms below and was
reconfirmed specifically for C on 2026-07-20 after the validator repair. It
does not erase the sensitivity of the source data: the trace and request-level
artifacts remain restricted, are never committed, and must not be copied to a
general-purpose mirror. In lifecycle 1 the one-request synthetic canary and A
were sent, while C was blocked before any POST. In lifecycle 2 a new canary and
both A and C were sent to the fixed OpenRouter/DeepInfra marketplace route.
Read-only public endpoint/price metadata checks are not inference requests.

The experiment node and all three GPUs are healthy. During the final launch
preflight, the historical dense Qwen3-32B weight directory was found to have
been removed; no quantized or substitute model was accepted. The exact public
revision `9216db5781bf21249d130ec9da846c4624c16137` was restored into the
private scratch area and independently verified byte-for-byte: 27 files,
17 safetensors shards, 65,540,298,478 bytes, 707 BF16 tensors, exact config and
index, and official-manifest SHA256
`6597f6b6ebb926354d721f44fa3cc5ea97f61c4d645858440a08f46cbfd9020e`.
The source trace and original tokenizer snapshot remain unchanged. No server
was started during download or verification.

A funded inference key passed the static metadata for the explicit
marketplace-v2 contract: exact server limit `$5`, no reset,
`include_byok_in_limit=false`, more than the full-pair bound remaining, and
zero recorded BYOK usage at preflight. The synthetic generation subsequently
proved `is_byok=false`, and the settled A snapshots showed positive marketplace
usage with unchanged BYOK usage. The default remains the strict server-cap
contract. The experiment-only marketplace-v2 exception is not a general
relaxation and cannot be used with another provider, BYOK billing, an unfunded
account, or an unregistered deployment.

#### Frozen trace identity and completed manifest binding

The live trace is the 11,604-row, original-current-turn, no-cache trace from
Section 5j. Its frozen content-level identity is:

| Field | Frozen value |
|---|---:|
| Trace SHA256 | `e838016a8e55660c565dadb1ad019770f6b88f878d8ca29f165c30887d2cb410` |
| Rows | 11,604 |
| Prompt-token sum | 1,289,405 |
| Per-row decode-cap sum | 3,038,796 |
| Payload/cache mode | original current-turn text; `cached_tokens=0` |

Commit `446bf56` was transferred as a verified git bundle into a private clean
detached checkout. From that checkout, the trace was re-materialized from the
same restricted source and Qwen3-32B tokenizer with scenario
`extreme_burst_1200`, decode cap 1,024, context cap 40,960, and overflow policy
`drop`. It reproduced the exact trace SHA and all counts above. This initial
pre-execution manifest was later superseded for live A because its exact
dependency binding no longer matched; it is preserved under SHA:

```text
0fbc544e2e9e37befe1a7e9a3bbaf54fa26eaed52d11d4d00e718924592a31c2
```

Before the live arm, the execution tree advanced to `98cab54` for marketplace
guards and evidence-limit repairs. A canary-only callback had changed
`router/common.py`'s whole-file SHA without changing trace materialization
semantics, so the original manifest correctly failed the exact dependency
check. The trace was therefore re-materialized into a new sibling path from
the same restricted source, tokenizer, and arguments. Its bytes were identical
to the old trace (`cmp` pass; the same `e838016a…cb410` SHA), while the new
manifest honestly binds the current dependency SHA. The active manifest is
common to both execution lifecycles. Their separate budget-attestation
identities are:

```text
manifest  dc0430c11faca9077f75f88cea8ab66ed5e2424351057819fb27d20dc3db9b9a
lifecycle 1 budget  c3631bd4073e87a8945e5bee91a782e1ab423405b117e7634419ddf961856310
lifecycle 2 budget  21812bb145deb29e692d0e178ce8bd65d5890954a5a7d49279bf1a2ae6ea5ac9
```

The old trace/manifest were preserved rather than overwritten. The active
trace and manifest are mode `0600` inside a mode-`0700` restricted
directory. Strict schema validation passed; an independent pass recomputed
11,604 rows, prompt/decode sums `1,289,405/3,038,796`, no-cache alignment for
all rows, and the unchanged output SHA. A differing trace SHA, row count, token
sum, payload mode, or overflow policy invalidates the registered run; it is not
fixed by silently changing this contract. The historical Section-5j manifest
remains valid for the completed local gate but cannot substitute for this
launch-checkout manifest.

#### Frozen deployment, profile, arms, and cloud behavior

The frozen pre-run contract required one fresh Qwen3-32B lifecycle with
bfloat16, prefix caching disabled, `max_model_len=40960`, and
`max_num_seqs=128`, plus a new schema-v2 profile bound to that exact checkout,
PID, server log, endpoint, tokenizer, and trace. It forbade reuse of the
completed local-gate profile. The profile cells were frozen to:

```text
1x16,1x32,1x512,1x4096,1x32768,
8x16,8x32,8x512,8x4096,
16x512,16x4096,32x2048,64x1024,128x32,128x512
```

Use five repeats, decode 256, seed 0, the held-out 5 s classifier gate with
zero false negatives, and the same 250 ms Nimbus tick. Both live arms use that
single fresh lifecycle/profile and exact trace arrival schedule with
`time_scale=1.0`, 5 s TTFT SLO, temperature 0, `local_ignore_eos=true`, and a
600 s per-request timeout. The frozen order and exact arm identifiers are:

1. A — `ttft_pred:cost_cachedisp_old:0`;
2. wait for provider usage to settle and pass the next-stage budget gate;
3. C — `ttft_pred:cost_disp_current:0`.

Run A and C as separate staged invocations so a completed A audit and usage
gate are prerequisites for C. Do not stop merely because A's latency result is
unfavorable; stop only for an integrity, configuration, provider, server, or
budget failure defined here.

For every routed row, use OpenRouter model `qwen/qwen3-32b`, provider
`DeepInfra` only, `allow_fallbacks=false`, no response-cache opt-in,
temperature 0, and a pre-first-token client concurrency cap of 16. Set cloud
`max_tokens` to that row's frozen capped decode length, but cancel the client
stream immediately after the first non-empty `reasoning` or `content` delta.
Use `cloud_ignore_eos=false`. There is no automatic or manual request retry; a
failure, timeout, or clean stream with no generated token remains that row's
measured outcome. The returned provider and model must match the requested
fixed route.

#### Price, budget, canary, and staged usage gates

Before launch, a same-day, public, non-inference OpenRouter endpoint snapshot
must verify that the only selected DeepInfra endpoint for this model has at
least 40,960 context and prices of **$0.08/M input** and **$0.28/M output**.
The live endpoint currently does not advertise a per-request fee; this is
recorded as `request_price_source=absent_not_advertised` and conservatively
bound to zero. An explicit nonzero/malformed request or optional text-price
field fails closed. Any mismatch stops the experiment and requires a new
pre-registration; do not silently recalculate after seeing a different price.
At those verified prices, the conservative static bound that routes every row
in both arms and bills the entire capped decode is:

```text
2 × (1,289,405 × $0.08/M + 3,038,796 × $0.28/M)
  = $1.90803056
```

This bound is deliberately independent of first-token cancellation and is a
spend guard, not a latency metric. The user-authorized total experiment cap is
**$3**, including the synthetic canary. The default
`strict_server_cap_v1` contract requires `0 < limit <= 3`,
`limit_reset=null`, and `include_byok_in_limit=true`. For this E12 run only,
`e12_marketplace_deepinfra_no_byok_v1` accepts the exact observed key metadata
`limit=5`, `limit_reset=null`, and `include_byok_in_limit=false`, but only under
the separately frozen OpenRouter-marketplace/DeepInfra/no-fallback route. This
exception does **not** raise the user budget: every delta-plus-future-bound gate
and the final audit remain capped at $3, while the static full-pair bound remains
$1.90803056. The exception additionally requires `byok_usage` to be present and
unchanged in every usage snapshot, positive metered usage for the canary and
each arm, and the canary's official generation record to report
`is_byok=false`. The raw generation id exists only in memory for that metadata
GET; evidence stores only its SHA256 and `is_byok=false`.

Both modes require a non-management, non-provisioning, non-free-tier inference
key whose expiry is null or at least six hours beyond launch verification.
The marketplace exception must be selected explicitly in every snapshot,
stage-gate, launch, runner, and final-audit command; the runner binding is:

```text
E12_LIVE_KEY_CONTRACT_MODE=e12_marketplace_deepinfra_no_byok_v1
E12_LIVE_KEY_LIMIT_MAX_USD=5
```

Omitting the mode retains the strict `<=3`/BYOK-in-limit behavior. Reject an
absent, mismatched, or resettable limit, missing/changing `byok_usage`, a BYOK
canary generation, insufficient remaining budget, a key-type mismatch,
non-monotone usage, or any snapshot that cannot be reconciled.

The exact pre-trace sequence is: capture a baseline key-usage snapshot; run one
fixed public synthetic prompt through the same model/provider/no-fallback
first-token-cancel path with `max_tokens=1`; wait for its usage to settle; then
prove that observed experiment delta plus the full two-arm bound remains at
most $3 before starting A. No trace text is used by the canary. After A, wait
until two consecutive key-usage snapshots at least 60 seconds apart are equal,
then require observed delta plus the frozen full-C bound of **$0.95401528** to
remain at most $3 before starting C. If usage does not settle or the gate fails,
do not start C. After C, take a settled final snapshot and require total
baseline-to-final experiment usage to be at most $3. Pending or delayed usage
is never treated as zero. Immediately before each arm, persist a fresh
`e12_live_current_usage.json` and an
`e12_stage_launch_verify_receipt.json`; both are mode `0600`, refuse overwrite,
and are SHA-bound into the run fingerprint, manifest, and every event. The C
authorization additionally revalidates the complete A bundle and A launch
receipt. The final auditor requires one baseline plus three settled usage
pairs (seven baseline/settlement snapshots total), as well as both arm-specific
launch-time usage snapshots and receipts.

Redirects are forbidden at every authenticated request boundary. Any 3xx,
401, 402, 403, 404, 405, or 422 is an immediate stop and forbids
starting/resuming a stage. HTTP 400 is systemic when it accounts for at least
80% of three or more completed cloud rows; non-429 HTTP failures are systemic
when they account for at least 80% of ten or more rows. An arm that routed
cloud work but produced zero successful cloud TTFT samples is invalid. Online,
the sender also stops after three completed non-429 HTTP failures while cloud
successes remain zero, limiting exposure before the completed-arm audit. Such
failures are not retried.
Provider error bodies, stream error messages, prompts, API secrets,
authorization headers, raw generation IDs, and provider-controlled metadata
must not be written to raw rows, logs, summaries, or audit output. Generation
IDs are hashed at ingestion; provider/model strings are persisted only as
request-derived allowlisted constants. Any mismatch becomes a fixed
`ProtocolMismatch` and stops the stage. Secrets must not appear in command-line
arguments or shell tracing.

#### Frozen integrity checks, headline, and claim boundary

Each arm must produce exactly 11,604 unique request rows plus a complete marker
binding trace, new manifest, profile, lifecycle, arm fingerprint, raw bytes,
summary, decisions, events, price snapshot, budget attestation, usage snapshots,
and launch receipts. The sender hashes and parses one identical in-memory read
of the trace, then verifies the frozen SHA before any request can start; this
closes path-replacement TOCTOU. Materializer manifests and summaries use strict
schemas, and E12 events contain only the fixed command marker
`router.run_config_bound_by_fingerprint`, never raw argv. Applied victim IDs
must exactly equal cloud request IDs; every successful retained-local row must
be prompt/decode-token exact; every successful cloud row must have finite
from-arrival first-token TTFT, `stream_abort_requested=true`,
`response_completed=false`, exact provider/model, and a canonical SHA256
generation-ID hash. Missing or failed rows remain in the denominator. A server
death, hash/binding mismatch, duplicate/missing ID, token mismatch,
provider/model drift, unredacted error, or failed budget gate invalidates/stops
the run.

The primary outcome for each arm is observed overall TTFT violation rate:

```text
overall TTFT violation
  = count(TTFT > 5 s, request failure, or missing TTFT) / 11,604
```

TTFT begins at trace arrival and therefore includes pre-route queueing,
cloud-concurrency-gate waiting, and provider service TTFT. A valid headline
requires `overall.slo_measured_n == overall.n == 11604` and
`cloud.routed_only == 0`. Report overall, local, and cloud counts/distributions,
route fraction, waits, status/error classes, provider/model conformance, abort
coverage, and known/pending spend separately. **Do not use
`pessimistic_combined` as the live headline** and do not reinterpret every
routed request as a violation.

This is one exploratory ordered A→C pair. It can estimate observed hybrid TTFT
and compare the registered selectors on this lifecycle, but it cannot by itself
establish order-robust superiority. An absolute A/C headline difference below
1 percentage point is explicitly unresolved and calls for a reverse-order
replicate; a larger one-pass difference remains preliminary. First-token
cancel does not measure completed-response E2E, TPOT, answer quality,
mid-stream reliability, or full-response cost, and the authorized assumption
that cancellation does not change upstream cloud load remains an assumption.

#### Lifecycle 1 — TERMINAL PARTIAL: A complete, C never launched

The bound Qwen3-32B lifecycle used execution commit `98cab54`, PID `112997`,
FlashAttention v2, BF16, prefix caching disabled, `max_model_len=40960`,
`max_num_seqs=128`, and 112,032 KV tokens. The accepted 75/75-block profile
had 2,105/2,105 successful requests, prefill throughput 3,239.221 tokens/s,
TPOT 151.516 ms, fixed first-token overhead 358.018 ms, and a 1,722 ms TTFT
guard. The synthetic DeepInfra canary returned HTTP 200 with 385.425 ms TTFT,
`is_byok=false`, and a settled cost of `$0.00000236`.

Stage A (`ttft_pred:cost_cachedisp_old:0`) completed at `time_scale=1.0`:

| Metric | A result |
|---|---:|
| Total / successful | 11,604 / 11,604 |
| Local / cloud | 6,541 / 5,063 |
| Routed fraction | 43.6315% |
| Overall TTFT p50 / p95 / p99 | 1,125.168 / 1,951.594 / 2,187.860 ms |
| Overall 5 s violations | 4 / 11,604 = **0.03447%** |
| Local violations | **0 / 6,541** |
| Cloud violations | 4 / 5,063 = 0.07900% |
| HTTP failures / 429 | 0 / 0 |
| A marketplace spend | **$0.02669984** |
| Baseline-to-post-A total (canary + A) | **$0.02670220** |

Every cloud row was a successful fixed-DeepInfra first-token-cancel
measurement; every successful local row was token-exact. Queue pressure reached
128 in flight and 24 waiting. The raw, decisions, summary, marker, event, and
manifest SHAs are respectively `eb593136…b97bf`, `db3274b0…2d49`,
`3d0b6d79…ccdd`, `5a571990…7db`, `c1c82d0e…4a79b`, and
`d774339f…2014`; launch attestation SHA is `e09f4554…336c`.

C did not execute. Its first authorization attempt exposed a validator precision
bug: the receipt timestamp `09:04:33.728166Z` was compared literally with
whole-second shell manifest/event time `09:04:33Z`, despite filesystem order
showing that the receipt was written first. Commit `15979bb` fixes only this
boundary by comparing the receipt at shell precision; a next-second inversion
still fails. Focused launch tests passed 12/12 and the tools suite passed
175/175 including private transport tests. A clean, SHA-locked validator
checkout at `15979bb` was then separated from the unchanged execution checkout
at `98cab54`; only `tools.check_e12_stage_launch` was dispatched there. Commit
`132e012` subsequently applied the same narrow precision rule to the final
auditor without changing the execution tree.

The next C preflight stopped because its usage snapshot became older than ten
minutes while explicit approval was pending. A new settled pair was captured,
but the local execution policy then rejected the external process before SSH
or any trace POST, even after the user explicitly reconfirmed authorization.
Consequently `stage_c` and `authorize_c_v3` do not exist; `authorize_c` and the
empty `authorize_c_v2` are retained as failed-preflight evidence. **There is no
paired A/C result and no final E12 verdict from this lifecycle.** A is a valid
descriptive arm, but it must not be used to claim that old-v2 beats current
displacement.

After the external-policy rejection, recorded server PID `112997` and children
`113279/113280` were terminated, port 8010 was verified closed, and GPU2 memory
fell from about 94.8 GiB to 125 MiB. The bound server and profile were destroyed,
so this lifecycle is permanently **TERMINAL PARTIAL**: it cannot be resumed, and
its A must never be paired with a C from a later lifecycle. Its settled canary+A
spend of **$0.02670220** remains charged against the global `$3` authorization.

#### Lifecycle 2 — fresh retry A→C: COMPLETE / FINAL AUDIT PASS

The retry used the same frozen execution commit
`98cab54a29e3b8ff066e474d26ddbdbf0394b2b8`, trace, manifest, deployment, and
arm order, but started a new server/profile lifecycle. Retry-budget tooling was
frozen at `ee1a7f648aed8bb19eba52fa223944a0b647ccae`; the final delayed-settlement
audit used `f3583a329fbbd2099a24b3ef9425ee6cdf8979cc`. The contract ID is
`e12-live-current-turn-retry-pair-20260720-marketplace-8512bc423b3d095b1c426c345d9095d2a75188634e1eb2d93cd6b4f59d2bab40`.

The new lifecycle ran under PID `137586`, with 112,032 KV tokens and a valid
15-cell profile (SHA256
`b5e29af776ecf82edf94ef3aa9388ee00c20a70546d416dac199b0bc1a134d47`).
All 2,105 profile samples succeeded; the held-out set had 421 samples and zero
false negatives. Its calibrated parameters were prefill throughput
`3247.990163` tokens/s, TPOT `151.213123 ms`, fixed first-token overhead
`372.804523 ms`, and TTFT guard `1712 ms`. The fixed DeepInfra canary passed at
`319.492 ms`, requested stream cancellation, did not complete the response, and
reported marketplace rather than BYOK billing; its evidence SHA256 is
`70bbb70bbbd551b2bffdc6f3aee57c462817a24750c6c4a75f7633c42d50aa10`.

Both registered arms then completed at `time_scale=1.0`:

| Metric | A — `cost_cachedisp_old` | C — `cost_disp_current` |
|---|---:|---:|
| Total / successful | 11,604 / 11,601 | 11,604 / 11,604 |
| Local / cloud | 6,507 / 5,097 | 7,547 / 4,057 |
| Routed fraction | 43.9245% | 34.9621% |
| Overall TTFT p50 / p95 / p99 | 1,126.482 / 1,983.963 / 2,451.606 ms | 1,306.939 / 2,123.023 / 2,496.230 ms |
| Local TTFT p50 / p95 / p99 | 1,377.188 / 2,007.805 / 2,204.826 ms | 1,470.152 / 2,147.514 / 2,372.112 ms |
| Cloud TTFT p50 / p95 / p99 | 474.991 / 1,830.916 / 2,941.866 ms | 567.699 / 1,938.093 / 2,972.899 ms |
| Overall 5 s violations | 11 / 11,604 = **0.0947949%** | 3 / 11,604 = **0.0258532%** |
| Retained-local violations | **0 / 6,507** | **0 / 7,547** |
| Cloud timeout / HTTP 429 | 3 / 0 | 0 / 0 |
| Settled marketplace spend | **$0.02720312** | **$0.01973248** |

The TTFT quantiles use successful rows with a finite observed TTFT, so A's
three timeout rows are absent from its quantiles. They are nevertheless
retained as failures and violations in the fixed 11,604-request SLO
denominator. The spend values come from settled key/account usage deltas, not
the per-run summary fields: first-token cancellation left cloud costs pending
there, with `known_cost_usd=0`.

All successful retained-local rows were prompt/decode-token exact; applied
victims exactly matched cloud request IDs. The complete text-free evidence
identities are:

| Evidence | A SHA256 | C SHA256 |
|---|---|---|
| raw | `2bc944209cbfbe26e8a8916b7922d0766e8b13156ff513291ba576be6ba54877` | `19cfa07772809c8872ac0e4a345d62a74c82f05508e268315bc314f9eb942d65` |
| summary | `c2455fec91d833c753562975fddb8437097d520f58806fbfbe8e718b7d9fbd49` | `de028203af6244a606db797b49dd7ee8af8ef8b459be6537be29ed8fc0db6d95` |
| decisions | `652cb27643360c3abbd33d152c63c66b935b54fd0547abc85e744904aa0bde63` | `a5fa2810b4ac46cae9efa7c96951ef093ea07f846389612ef37a270e45707a4f` |
| completion marker | `8b9429d49c54f8431e9bbe28c6fd5cbdff161a4aa16bd6b08e30deb7d860625e` | `fd34d7a0ed9dca89886ef6ce09a8295f4f21de4ff11365dcf744455596ee48e5` |
| events | `e7d64c4011b8f0696a28b4994e83482bf81a52a9e1f7e5ec2d2171c53d9aaf48` | `a3d0353615dab460487c2d76bd0ca63de9ebbc0379ff925d4b99e8dc60ac4bd4` |
| run fingerprint | `967b3ac418bc55b8db9049bfd4353882ea4e7902c047b465b16acbceb747dba6` | `3c4d23ebcb98aa8bc70d03b66e820ff020bc53885738d888de29c39628c5c10d` |

The retry canary cost `$0.00000236`; A+C cost `$0.04693560`, so retry spend was
**$0.04693796**. Including lifecycle 1's terminal partial spend, the global
experiment total was **$0.07364016** and headroom was **$2.92635984**; BYOK
usage remained `$0`. The public price snapshot, retry budget attestation,
pre-A/pre-C/final cumulative guards, final stage gate, and final audit have
SHA256 values respectively:

```text
price              690cb986740769b711a67d3d1cb848aace618c54263a273aca94f6a89f158a02
budget             21812bb145deb29e692d0e178ce8bd65d5890954a5a7d49279bf1a2ae6ea5ac9
pre-A cumulative   8512bc423b3d095b1c426c345d9095d2a75188634e1eb2d93cd6b4f59d2bab40
pre-C cumulative   c741b6091a789263ec04183d1a21a84f78df7f4a9db2efefbf23fc612cd33751
final cumulative   105e211721ad8e1074dcac6b1551e361d5fca5a2d33b615bca3be7e85c458e25
final stage gate   ae409ab1e3c060c8cfd7d3c7b8a70e286df2ede60f2418727b69fca91b255b19
final audit        ed669378d416c220e5afe87d03744b1e62105fc6a9efa0b5ab7b69e36cdaeaf4
```

The final audit verdict is **PASS**. At final-gate creation, both snapshots in
the final settlement pair had to be no more than 10 minutes old, exactly equal,
and at least 60 seconds apart. A later offline audit replays that freshness
check against the recorded final-current timestamp; it does not compare old,
immutable evidence with the later audit wall clock. Final mode may retain an
old global/retry accounting baseline after delayed provider settlement. This
relaxation is final-audit-only: every paid launch uses `final=False`, still
requires a fresh retry baseline and fresh settled pair, and cannot reuse the
retrospective exception. This preserves the global anchor and therefore never
drops lifecycle 1's spend from accounting.

For schema compatibility, `final_budget_gate.json` retains the frozen
`next_stage_full_upper_bound_usd=$0.95401528` field. Because its mode is
`final`, `required_from_baseline_usd` is only the observed retry delta
`$0.04693796`; it does not authorize or reserve another stage. The cumulative
final guard is authoritative for global accounting: its future bound is `$0`,
its global spend is `$0.07364016`, and its headroom is `$2.92635984`.

C used 1,040 fewer cloud routes than A (**-8.9624 percentage points**) and had
8 fewer violations (**-0.06894 percentage points**), but its overall TTFT
p50/p95/p99 were worse by 180.46/139.06/44.62 ms. This is only one ordered A→C
pair. Because the absolute violation-rate difference is below the preregistered
1-point threshold, the result is explicitly **unresolved**, not a selector win;
a fresh reverse-order run and repeated/balanced blocks are required.

Finally, PID `137586` and all children were terminated, port 8010 was closed,
and GPU2 returned to 122 MiB. The cleanup check passed. The final auditor binds
the pre-run server-log prefix SHA256
`12c996bd9ad380e6218a1130d9c8dad5d3242a910f60785fa31ceea0f770e219`;
after shutdown messages had flushed, the complete `server.log` SHA256 was
rechecked as
`9d667b22d50a558de46687bdb6053c2405c821742758bf4d628328d990823c41`.
The restricted request-level artifacts remain
under `$MSCRATCH/e12_retry_pair_20260720/live_run/` and must not enter git.

---

## 6. Cloud-side accounting (headline-metric decision)

The default cloud is a zero-latency fake sink (`--cloud null`): kicked rows are
`routed_only=true` and excluded from SLO stats (`slo_measured_n` is the explicit
denominator). In a NullCloud cell, report local observed SLO, route fraction,
and estimated token cost. `pessimistic_combined =
(local_slo_violations + routed_only) / N` remains a conservative **assumed
upper bound** and a historical route-efficiency transform. It is not the
headline for a real-cloud experiment. The frozen Sections 5e–5g gates retain
their preregistered pessimistic rules; this accounting change does not rewrite
their historical decision criteria.

For `--cloud real`, the primary metric is now the observed, from-arrival TTFT
violation rate already emitted as:

```text
observed_combined_ttft = summary.overall.slo_violation_pct
                       = count(local/cloud error, missing TTFT, or TTFT > SLO) / N
```

A valid real-cloud headline must verify
`overall.slo_measured_n == overall.n == N` and `cloud.routed_only == 0`, then
report overall, local, and cloud violations/TTFT separately. The existing
`pessimistic_combined` field still marks every cloud row as a violation even in
a real-cloud summary; ignore it for that headline. Renaming it to an explicit
`nullcloud_upper_bound` is reporting-layer debt.

For the July-16 TTFT-cancel mode, “first token” means the first non-empty
reasoning or content delta, matching the no-reasoning-parser local vLLM stream.
Under the explicit temporary assumption that cancellation does not change cloud
load, such rows are valid observed TTFT measurements and can enter
`overall.slo_violation_pct`. They are not completed-service measurements:
`response_completed=false`, E2E/TPOT are absent, and cost may remain pending.
Errors or a stream that ends without any generated token still count as TTFT
violations. A full-drain sensitivity leg is required before making E2E,
mid-stream reliability, or complete billing claims.

When every retained-local request is safe, the NullCloud pessimistic bound
equals the routed fraction exactly; those are two views of one number, not
independent evidence. Dollar cost can disagree with route count because
selectors choose different prompt/decode mixes, so both remain paired
whole-run diagnostics even after observed cloud TTFT becomes the headline.

---

## 7. Experiment queue (REVISED 2026-07-21 after completed E12 retry)

1. **COMPLETED — audit contract.** Atomic token-aligned materializer,
   complete-cohort/stale-arrival protection, local usage telemetry, no-cache
   server/profile binding, and completion markers; 81/81 router tests,
   `py_compile`, `bash -n`, and `git diff --check` passed.
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
6. **COMPLETED — six-block repeatability/order screen.** The actual design was
   R wash-in plus `ABC / BCA / CAB`, conditionally extended after an A/C sign
   flip with `ACB / CBA / BAC`; see Section 5f rather than assuming the earlier
   5e table. All 24 arms passed. Old-v2 clearly beat `newest`; old/current are
   route-equivalent within run/order spread, while old-v2 cost less in 6/6.
7. **COMPLETED — full 11,605-request generalization gate.** The exact
   pre-registered B/K/C/A/L order fully ran; evidence integrity and the pressure
   anchor passed, but the aggregate held-out gate failed because B had 39 local
   violations. A and C had zero; A remained route-equivalent to C and cheaper,
   while shipped K was grossly TTFT-unsafe. See Section 5g for the exact claim
   boundary.
8. **COMPLETED (probe only) — fixed-provider real-cloud TTFT cancellation.**
   Commit `e0b6686` pins provider order, disables fallback, aborts after the
   first reasoning/content token, preserves pending cost, and records generation
   metadata. The 16-row DeepInfra smoke was 16/16 client-aborted, 0 errors, and
   0/16 above 5 s; see Section 5h for the strict claim boundary.
9. **SKIPPED BY DECISION — 512 frozen-route cloud shadow (A then C,
   separately).** The planned stitch/provider check remains a valid cheaper
   diagnostic, but the project owner chose on July 16 to proceed directly to the full live
   pair. The skipped design would replay the existing A and C routed IDs at
   their recorded kick times, not at original
   trace arrival. For each cloud request compute `pre-route wait + cloud-gate
   wait + OpenRouter service TTFT`; combine those rows with the corresponding
   measured local rows. Pin DeepInfra, disable fallback, cap pre-first-token
   client concurrency at 16, and keep pending costs pending. Budget about $0.06
   for 523 calls. Acceptance: exact frozen IDs/times, no duplicates/missing
   rows, no `routed_only`, every success has TTFT, error/429 rate no greater than
   1%, and an explicit observed-combined/local/cloud report. This is a cheap
   provider/stitch check, not the final hybrid result.
10. **SKIPPED BY DECISION — 512 live hybrid A/C TTFT-cancel pilot.** The project owner
    accepted the extra spend/risk and selected the direct full run. Its two
    required code prerequisites—split local/cloud `ignore_eos` and split
    pre-route/cloud-gate waits—are incorporated into the full-cell contract.
    The original pilot would run A and C on the same bound local
    lifecycle/profile with fixed DeepInfra and no fallback. Its acceptance was:
    512/512 unique measured rows; exact local token alignment; A/C local
    violations remain zero; applied victims equal cloud rows; every cloud row
    has measured first-generated-token TTFT; and `overall.slo_measured_n=512`.
    The headline is `overall.slo_violation_pct`; never the real-run
    `pessimistic_combined` field.
11. **SUPERSEDED BY E12 — direct full 11,605 synthetic live hybrid A/C.**
    Section 5i preserves the A-then-C design and complete integrity/outcome
    contract. Its first lifecycle completed a fresh profile but sent zero formal
    cloud requests; it was shut down at the bulk-transfer gate. The later E12
    methodology decision selected the 11,604-row original-current-turn workload
    instead, so this synthetic leg was not resumed. It remains a separately
    registered fallback, not part of the E12 result. If ever reactivated, use a
    new bound lifecycle/profile and the consent applicable to that exact payload;
    retain complete raw/decision/marker evidence and require reverse-order
    replication before a close A/C winner claim.
12. **COMPLETED — E12 local-only current-turn L/A/C.** The 11,604-row
    verbatim current-turn trace, fresh short-prompt-aware profile, and full
    `all_local → old-v2 → current` order completed with zero third-party POSTs.
    L established heavy pressure; A/C both passed retained-local safety but
    exposed a selected-route-count (and therefore potential live exposure)
    versus modeled-dollar tradeoff. Section 5j records the audited result and
    its NullCloud claim boundary.
13. **NEXT — pre-register reverse-order/repeated fresh lifecycles.** Repeat the
    live current-turn comparison as C→A and preferably balanced blocks; a new
    export authorization, same-day price snapshot, and independent cumulative
    budget gate are required before any additional POST. Include `newest` or
    waiting-random only if the preregistered question needs a naive comparator.
    Acceptance must estimate order/route/cost/overall-TTFT stability and retain
    the zero-local-violation gate. The single A→C pair cannot select a winner.
14. **COMPLETE RETRY / OLD LIFECYCLE TERMINAL PARTIAL — E12 live current-turn
    A/C.** Lifecycle 1 ended after canary+A and charged `$0.02670220`; its C never
    launched and its server/profile were destroyed. Lifecycle 2 freshly profiled
    the same execution tree, completed A→C, and passed the final audit. A/C routed
    5,097/4,057 and had 11/3 overall violations, with zero retained-local
    violations in both arms. Retry spend was `$0.04693796`; cumulative spend was
    `$0.07364016`. The `<1 pp` preregistration rule makes the comparison
    unresolved. Section 5k records the complete identities and cleanup.
15. **THEN harden the TTFT trigger outside the calibration support envelope.**
    Profile deployment binding resources, log full decision snapshots/scores,
    and add a conservative fallback when live state leaves calibrated support.
    Engine cache gauge may enter only as a deployment-profiled resource feature,
    never as a universally token-denominated TTFT trigger.
16. **Restore remaining semantics:** cache-aware trace for the cached-token
    term; estimated-vs-oracle decode; historical exact 0/1-DP/offline oracle;
    load/guard sweeps; and at least one full-drain cloud sensitivity leg for
    E2E/mid-stream reliability. The hybrid binding-resource result remains a
    separate architecture study, not the TTFT trigger definition.

---

## 8. Environment gotchas (GPU box) — read before running anything

- **Read [`../AGENTS.md`](../AGENTS.md) first**: kill every server you start
  (by recorded PID, then verify children gone via `nvidia-smi`); default to
  `CUDA_VISIBLE_DEVICES=2` (GPU0 is an intermittently failing card;
  `lspci` shows `rev ff` when wedged and recovers to `rev a1`
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
| `$MSCRATCH/router_ttft_nocache_78846da/extreme_token_aligned.jsonl{,.manifest.json}` | Full 11,605-request aligned trace; SHA256 `465ef070…24c52`, manifest `c5621d3e…72f7a` |
| `$MSCRATCH/router_ttft_nocache_78846da/ttft_profile_v2_bcda9d0.json` | Accepted schema-v2 profile (SHA256 `549dd0bf…e23f`) |
| `$MSCRATCH/router_ttft_nocache_78846da/heldout512_all_local/` | Queue-pressure all-local anchor |
| `$MSCRATCH/router_ttft_nocache_78846da/heldout512_ttft_smoke_bcda9d0/` | Audited `newest` smoke and completion marker |
| `$MSCRATCH/router_ttft_nocache_78846da/heldout512_selector_compare_bcda9d0/` | Three-arm exploratory matrix, decisions, summaries, markers, manifest |
| `$MSCRATCH/router_ttft_nocache_78846da/vllm_qwen32b_nocache.log` | Server configuration/lifecycle log for this matrix (KV 112,656; distinct from the July-13 server) |
| `$MSCRATCH/router_ttft_repeat_cac4a5d_20260715/` | July-15 lifecycle profile/log/server log plus `block01_r0_abc/` … `block06_r5_bac/`; 24 audited arms |
| `$MSCRATCH/router_ttft_repeat_cac4a5d_20260715/ttft_profile_v2_cac4a5d_lifecycle1.json` | Repeatability profile, SHA256 `ab703ddc…c819` |
| `$MSCRATCH/router_ttft_full_c6de62a_20260715/full11605_bkcal/` | Completed pre-registered B/K/C/A/L matrix: raw/decision/summary/markers, frozen manifest, full-cell gate audit, and bound B violation-context audit |
| `$MSCRATCH/openrouter_ttft_cancel_20260716/` | July-16 fixed-DeepInfra TTFT-cancel smokes and 16-row burst; raw/summary plus delayed generation/billing audit (Section 5h) |
| `$MSCRATCH/router_realcloud_full_2b53ff3_20260716T1435Z/` | E11 pre-arm lifecycle: valid fresh profile, server log, non-secret OpenRouter baseline, and blocked-launch zero-usage audit; no formal A/C rows |
| `$MSCRATCH/sharegpt_current_turn_6054b32_20260716/` | Restricted E12 original-current-turn trace and text-free manifest; 11,604 emitted rows, output SHA `e838016a…cb410`, manifest SHA `698bb94a…bf9a8` |
| `$MSCRATCH/e12_launch_446bf56_20260720/` | Preserved pre-execution clean checkout plus first re-materialized restricted E12 trace; 11,604 rows, output SHA `e838016a…cb410`, old bound manifest SHA `0fbc544e…a31c2`, files 0600/directories 0700; no POST from this checkpoint |
| `$MSCRATCH/e12_private_20260720/live_run/` | Restricted lifecycle-1 evidence, permanently **TERMINAL PARTIAL**: execution commit `98cab54`, active trace SHA `e838016a…cb410`, manifest `dc0430c…b9b9a`, profile, canary, complete A evidence, no `stage_c`; never pair this A with another lifecycle |
| `$MSCRATCH/e12_retry_pair_20260720/live_run/` | Restricted lifecycle-2 evidence: fresh profile, canary, complete A/C raw/decision/summary/marker/events, usage/cumulative guards, and final audit `ed669378…daeaf4` (**PASS**); never commit request-level artifacts |
| `$MSCRATCH/sharegpt_current_turn_local_78da644_20260716/` | Completed E12 no-export lifecycle/profile and L/A/C raw/decision/summary/marker evidence plus text-free gate audit; 0 external POSTs, actual cloud spend $0 |
| repo-sibling `artifacts/realcloud_full_prerun_2026-07-16/` | Verified local mirror of the six E11 pre-arm evidence files; outside git |
| `$JSCRATCH/workloads/…` | ShareGPT+BurstGPT trace (leg-1 `$DATA`) |
| `$JSCRATCH/initial_result/` | 14,537 successful historical provider observations using single-current-turn `prompt_text`; not token-aligned long-context calibration or an exact E12 reproduction |
| `$JSCRATCH/vllmresult/` | teammate's open-loop sweep on the same model (parity reference) |
