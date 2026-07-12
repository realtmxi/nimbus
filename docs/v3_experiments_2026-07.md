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
   is validated layer-by-layer against the team's trusted open-loop harness
   (parity ratio 1.003; queue neutrality 110 vs 111 ms). 56/56 unit tests, no GPU needed.
2. v3 nimbus on the team's hardest cell (`extreme_burst_1200`, n=11,605) produced
   a spectacular headline: **34% self-selected outsourcing, local TTFT p50
   325 s → 0.32 s, SLO violations 97.9% → 0.0%**, cost $1.79.
3. **But a post-hoc audit found the trigger fired for a different reason than
   designed** (Section 4): on Qwen3.6-35B-A3B — a hybrid GDN/linear-attention
   model — vLLM's `kv_cache_usage_perc` gauge tracks a **per-sequence state-slot
   pool (~2.4%/seq, saturating at ~41 concurrent)**, not tokens. v3 accidentally
   became an adaptive concurrency governor. The numbers are real; the mechanism
   is not the token-KV story in the design doc.
4. Direct corollary the team should know: on this model the engine's true
   concurrency ceiling is **~41, not `max_num_seqs=128`** — all prior saturation
   experiments on it are state-slot-bound, not compute-bound as we assumed.
5. Pending decision (Murphy): generalize the algorithm's units to "fraction of
   the binding resource" (recommended, small change) vs. moving the KV-bound
   story to a pure full-attention model. See Section 5.

---

## 2. What is built and verified

| Artifact | What it is |
|---|---|
| `router/run.py` | Single entry point: external FIFO, work-conserving dispatcher, KV monitor, inflight-KV tracker, CLI |
| `router/common.py` | Trace loading (BurstGPT windows byte-identical to the trusted `vllm/run.py`; `--scenario full` for arbitrary traces), payloads, SSE client, NullCloud sink, billing, summaries |
| `router/nimbus.py` | v3 policy: gap trigger + ascending cost/displacement shedding |
| `router/test_*.py` | 56 unit tests, no network/GPU (`python3 -m unittest router.test_common router.test_run router.test_nimbus`) |
| `tools/kv_gauge_probe.py` | Live probe that established the Section-4 finding (stdlib only) |
| `tools/analyze_eb1200.py` | Timeline reconstruction that flagged the anomaly from a result JSONL |

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
- Selection worked as designed: kicked prompt p50 = 644 vs kept p50 = 28
  (footprint p50 920 vs 289) — big requests went to the cloud.
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

## 6. Cloud-side accounting (headline-metric decision)

The default cloud is a zero-latency fake sink (`--cloud null`): kicked rows are
`routed_only=true` and excluded from SLO stats (`slo_measured_n` is the explicit
denominator). For paper headlines, the recommended primary metric is the
**pessimistic bound: every kicked request counts as an SLO violation** (real
cloud p50 ≈ 10 s > 5 s SLO), with local-only violation as the secondary metric.
A real-cloud leg (`--cloud real …`) is only needed once, pre-submission.

---

## 7. Experiment queue (in order, each with acceptance criteria)

1. **Calibrate `(a, b)`** for Qwen3.6-35B-A3B: sweep concurrency at fixed token
   volume and token volume at fixed concurrency; fit `usage ≈ a·seqs + b·tokens`.
   Extend `tools/kv_gauge_probe.py`. *Accept:* R² > 0.99; `a ≈ 0.024` reproduced;
   ceiling `⌊1/a⌋ ≈ 41` matches observed max running.
2. **Resource-generic patch (Option A)**: `router/nimbus.py` quantities → pool
   fractions; `router/run.py` KVMonitor returns `(1−u)` directly; CLI grows
   `--resource-a/--resource-b` (token mode = `a 0 --resource-b 1/capacity`
   preserving today's flags). *Accept:* all unit tests green; token-mode
   configuration reproduces current behavior on the existing regression tests.
3. **Re-run leg 1 + baseline arms**: v3(A-units) vs **naive-spill** (usage ≥ τ →
   kick newest, no selection) vs random@matched-fraction. *Accept:* v3 ≥
   naive-spill on cost at equal local SLO, or the honest negative is recorded.
4. **Paper-grade KV/slot-bound cell**: larger rednote slice (500–1,000 requests,
   natural timestamps, 2–3 load levels via `--time-scale`), 4 arms
   (all_local / v3 / random@matched / all_cloud), ≥3 seeds, frontier data
   (violation-vs-cost as shed aggressiveness varies).
5. **Adopt the Section-6 headline metric** and recompute all cells under it
   (e.g. rednote probe becomes 66.3% vs 67.5% — the wash is the motivation for
   frontier tuning, not an embarrassment).
6. **Ablations**: oracle vs estimated decode length; cost-aware ordering vs pure
   displacement; offline exact-cover DP bound vs greedy; hysteresis `h` sweep.
7. Deferred review minors: #6 zero-footprint kick guard, #8 `on_tick` recompute
   cleanup, #9 gauge-parse early-exit.

---

## 8. Environment gotchas (GPU box) — read before running anything

- **Read [`../AGENTS.md`](../AGENTS.md) first**: kill every server you start
  (by recorded PID, then verify children gone via `nvidia-smi`); default to
  `CUDA_VISIBLE_DEVICES=2` (GPU0 is an intermittently failing card — wedged
  twice in July 2026, `lspci` shows `rev ff` when wedged, recovers to `rev a1`
  only after reboot; a wedged device poisons CUDA init box-wide).
- The owner's home dir is **over disk quota**: every launch must redirect
  `TMPDIR`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
  `XDG_CACHE_HOME` to the scratch volume or vLLM dies at startup with a
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
