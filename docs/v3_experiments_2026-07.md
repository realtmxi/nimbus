# Nimbus v3 — experiment record & handoff (July 2026)

Self-contained status document: what was built, what was measured, what broke our
assumptions, and the exact queue of next experiments. Written so a person or agent
(e.g. Codex) can continue without access to prior conversations. Authoritative
design spec: [`notion_algorithm_design_v3.md`](notion_algorithm_design_v3.md);
framework usage: [`../router/README.md`](../router/README.md).

**Path convention** (per [`../AGENTS.md`](../AGENTS.md), machine names / usernames /
absolute scratch paths are never committed): `$MSCRATCH` = the project owner's
scratch dir on the GPU box; `$JSCRATCH` = the scratch dir of the teammate who
owns the workloads/serving setup. Actual values are configured out-of-band.

---

## 1. TL;DR

1. The `router/` framework (external FIFO + work-conserving dispatcher + policies)
   remains validated against the trusted open-loop harness (parity 1.003; queue
   neutrality 110 vs 111 ms). **70/70** unit tests pass without network/GPU.
2. The current experiment keeps **TTFT violation as the trigger**. KV is not the
   objective and did not replace the original algorithm. The original v2 weight
   `prompt × (uncached_prompt/prefill_tput + decode×TPOT)` is preserved exactly
   as selector `cost_cachedisp_old`; trigger and selector are independently
   switchable.
3. The July 12–13 headline runs are now **exploratory only, not selector proof**.
   A July 14 audit found that the scheduler used cumulative trace-token metadata
   while HTTP often sent only a tiny current-turn prompt; simultaneous arrivals
   were also decided row-by-row, often leaving the selector one candidate. The
   recorded latencies are real for those payloads, but claims that displacement
   caused the win are not identified.
4. The independent gauge finding remains valid: hybrid Qwen3.6-35B-A3B exposes
   a per-sequence state pool (~41 concurrent), while dense Qwen3-32B exposes
   token KV. That explains why a KV-gauge trigger is architecture-dependent; it
   does **not** imply Nimbus should replace its SLO trigger with KV.
5. The active evidence path is a token-aligned, synthetic, capped, no-prefix-
   cache dense-32B rerun: same payload semantics for profiling and matrix;
   held-out TTFT calibration; then `newest` vs original-v2 displacement vs the
   current displacement selector under one TTFT stop rule. See Section 5b/7.

---

## 2. What is built and verified

| Artifact | What it is |
|---|---|
| `router/run.py` | Single entry point: external FIFO, work-conserving dispatcher, KV monitor, inflight-KV tracker, CLI |
| `router/common.py` | Trace loading (BurstGPT windows byte-identical to the trusted `vllm/run.py`; `--scenario full` for arbitrary traces), payloads, SSE client, NullCloud sink, billing, summaries |
| `router/nimbus.py` | v3 policy: gap trigger + ascending cost/displacement shedding |
| `router/test_*.py` | 70 unit tests, no network/GPU (`python3 -m unittest router.test_common router.test_run router.test_nimbus`) |
| `tools/kv_gauge_probe.py` | Live probe that established the Section-4 finding (stdlib only) |
| `tools/analyze_eb1200.py` | Timeline reconstruction that flagged the anomaly from a result JSONL |
| `tools/materialize_token_aligned_trace.py` | Atomic, tokenizer-fingerprinted no-cache trace materializer |
| `tools/profile_ttft_batch.py` | Same-payload warmup/profile with a held-out scalar calibration artifact |

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

## 5. Design fork (decision pending — Murphy)

- **Option A (recommended): resource-generic units.** The policy works in
  *fractions of the binding pool*: `footprint_frac(req) = a + b·(local_prompt +
  expected_decode)`, `displacement = footprint_frac × residence_s`,
  `gap_frac = Σ waiting_frac + inflight_remaining_frac − (1−u)`. Constants
  `(a, b)` come from a one-time per-model calibration probe. On a pure
  full-attention model `a≈0, b=1/capacity` and the formulas reduce **exactly**
  to the current token·s design — it is a strict generalization. Trigger,
  ordering, hysteresis, runner: unchanged.
- **Option B: keep token semantics**, run the KV-bound story on a pure
  full-attention model, declare hybrid architectures out of scope.

A and B are not exclusive: A is the algorithm fix; a full-attention cell is
still wanted as a sanity anchor where the token math is exact.

Also still owed (either option): the leg-1 win has **not** yet been shown to beat
a *naive* concurrency spill ("pool full → shed the newest arrival, no
selection"). v3's selection should win on cost and on which requests keep local
latency — that baseline arm is queue item 3.

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

**The two campaigns compose into one principle.** Hybrid 35B: the gauge
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
  `cost_cachedisp_old` (the original formula exactly), and
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

The first framework checkpoint is local commit `b930ab2`. The audit-hardening
checkpoint and GPU results are recorded after they complete; do not substitute
the historical dense numbers above for them.

---

## 6. Cloud-side accounting (headline-metric decision)

The default cloud is a zero-latency fake sink (`--cloud null`): kicked rows are
`routed_only=true` and excluded from SLO stats (`slo_measured_n` is the explicit
denominator). For paper headlines, the recommended primary metric is the
**pessimistic bound: every kicked request counts as an SLO violation** (real
cloud p50 ≈ 10 s > 5 s SLO), with local-only violation as the secondary metric.
A real-cloud leg (`--cloud real …`) is only needed once, pre-submission.

---

## 7. Experiment queue (REVISED 2026-07-14 after the payload/cohort audit)

1. **Finish and commit the audit contract.** Atomic token-aligned materializer,
   complete-cohort/stale-arrival protection, local usage telemetry, no-cache
   server/profile binding, and completion markers. *Accept:* 70/70 tests,
   `py_compile`, `bash -n`, and `git diff --check` green.
2. **Start one dense Qwen3-32B no-cache server and keep one lifecycle.** Record
   the exact parent PID; verify `enable_prefix_caching=False`, KV capacity, and
   `max_num_seqs=128`. Warm once/discard and warm independently again. *Accept:*
   second warm batch below the configured cold-path ceiling; kill only this PID
   after all arms and verify its children are gone.
3. **Materialize/profile the transformed extreme trace.** Use the caps and
   sampling contract in Section 5b, with scratch-backed `TMPDIR`, `HF_HOME`, and
   `XDG_CACHE_HOME`. *Accept:* trace/manifest hash and line count agree; profile
   produces positive parameters and a held-out error/tick guard.
4. **Queue-level held-out smoke before the full matrix.** Run all-local on an
   independent multi-cohort slice, then one TTFT arm. *Accept:* 100% success,
   exact prompt/decode usage, no cold-path recurrence, and decision snapshots
   contain real multi-item choices.
5. **High-information exploratory comparison:** under the identical TTFT
   trigger run `newest`, `cost_cachedisp_old`, and `cost_disp_current`, with
   pessimistic-combined violation and cost as the primary Pareto readout.
   *Accept:* every arm completion marker valid; no config/trace/profile drift.
6. **Only if item 5 has separation:** run `waiting_random` with ≥3 seeds and
   repeat full matrices with randomized arm order. Single-pass deterministic
   arms are exploratory, not paper error bars.
7. **Then restore missing semantics:** cache-aware trace for the cached-token
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
| `$JSCRATCH/workloads/…` | ShareGPT+BurstGPT trace (leg-1 `$DATA`) |
| `$JSCRATCH/initial_result/` | 14,537-row real OpenRouter measurements (cloud-latency calibration) |
| `$JSCRATCH/vllmresult/` | teammate's open-loop sweep on the same model (parity reference) |
