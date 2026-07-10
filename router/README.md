**English** | [简体中文](README.zh-CN.md)

# router/ — hybrid routing framework

Dispatches trace requests between a local vLLM and a cloud sink according to a
policy, recording latency, cost, and the split. Single entry point:
`python -m router.run`. Policies: three baselines (`all_local` / `all_cloud` /
`random`) plus the **nimbus** shedding policy (cache-displacement, see below).

## Files

| File | Role |
|---|---|
| `run.py` | **The entry point**: external FIFO + work-conserving dispatcher + KV monitor + CLI |
| `common.py` | Shared library: `one_request` / `load_trace` / `SCENARIOS` (line-for-line from `vllm/run.py` @ `dff1a81`), `Endpoint`, `Policy`, `NullCloud`, billing, `summarize` |
| `nimbus.py` | Nimbus v3: physical KV-gap trigger + cost/displacement density ordering (`--policy nimbus --kv-capacity-tokens N`) |
| `test_run.py` / `test_common.py` / `test_nimbus.py` | 56 unit tests; no network / aiohttp / GPU needed |

## Architecture

```
arrival ──Policy (decided at arrival)──cloud──> fake sink (default) / real cloud
            │local
            v
      [external FIFO] ──work-conserving dispatcher──> vLLM (internal queue ≈ empty)
```

- **Policy**: `all_local` / `all_cloud` / `random --fraction f --seed s` — one
  rule with p = 0 / 1 / f, an i.i.d. coin flip reading no system state; the
  same seed reproduces every per-request decision
- **Dispatcher**: admit whenever a slot is free (`inflight < --max-inflight`,
  align with the server's `--max-num-seqs`) ⇒ vLLM's internal queue stays ~empty
- No pressure ⇒ the queue is always empty ⇒ behavior degrades to open-loop
  (≡ how `vllm/run.py` runs)
- Under pressure ⇒ overflow waits in **our** queue, with full identity — the
  set the nimbus shedding policy operates on. Why own the queue: engines
  expose only queue *counts* (three `/metrics` gauges, verified against vLLM
  v0.19 source), never the identity of waiting requests; selective offloading
  needs names
- **KV awareness lives in the nimbus policy**, not the dispatcher: the kick
  check runs BEFORE dispatch, and admission is frozen while a shed decision
  is computing (so a completing request can never admit a victim mid-decision)

## Cloud sinks

`--cloud null` (**default**): a fake request — records "this one was routed to
cloud" plus the trace's token counts and cost (`--max-tokens` semantics are
identical to the real payload: it replaces the trace value). No latency is
modeled; such rows carry `routed_only=true` and are excluded from SLO stats.

`--cloud real --cloud-url … --cloud-model … --cloud-api-key-env KEY`: real
streaming calls; `--cloud-max-concurrency` (default 32) guards against
self-inflicted 429s under burst. Use only when the experiment needs real cloud
latency. (Jialu has 14k real OpenRouter measurements on the GPU host under
`/scratch/jialu/initial_result/` for estimating distributions; measured
qwen3-32b TTFT p50 ≈ 10 s — reasoning + provider queueing; the cloud is not "fast".)

## Usage (GPU host; use a python env with aiohttp; run from the repo root)

```bash
python -m router.run --data <trace.jsonl> --scenario burst_300 \
  --policy random --fraction 0.3 --seed 0 \
  --local-url http://127.0.0.1:8010/v1/chat/completions --local-model Qwen3.6-35B-A3B \
  --max-inflight 128        # align with the server's --max-num-seqs
```

Output: per-request JSONL (`ttft_ms = queue_delay_ms + service_ttft_ms`,
measured from the trace arrival time, comparable with open-loop) plus
`.summary.json` (overall/local/cloud sections + queue telemetry). Each section
reports `slo_measured_n` — the explicit SLO denominator after `routed_only`
rows are excluded. Billing: failed requests cost $0; the local side always $0.

Local tests (no network/GPU): `python3 -m unittest router.test_common router.test_run router.test_nimbus`

## Baseline & validation record

The baseline is **Jialu's `vllm/run.py`** (the team-verified open-loop load
generator). An in-schema open-loop runner existed inside this package long
enough to establish the parity anchor, then was removed (first principles).

All measured on the GPU host, Qwen3.6-35B-A3B, `burst_300`, n = 756:

| # | Check | Result |
|---|---|---|
| 1 | **Parity anchor**: open-loop `all_local` ≡ `vllm/run.py` (same server, back-to-back) | ✅ paired TTFT p50 ratio = 1.003; the fat tail was first-leg JIT warmup only (measured with the since-removed in-schema runner) |
| 2 | **Queue neutrality**: this framework's `all_local` ≈ open-loop | ✅ p50 110 vs 111 ms, p99 233 vs 223 ms, queue_delay max 2 ms |
| 3 | Pacing under pressure: queueing stays client-side, engine never drowns, nothing leaks | ✅ queue_delay p50 = 83.8 s while engine service TTFT p50 = 90 ms (756/756 success); the mechanism is now a pure concurrency gate, same property covered by the `max_inflight=1` serialization unit test |
| 4 | `random` end-to-end: fraction 29.4 % vs target 30 %, billing recomputes exactly, local bills $0 | ✅ |
| 5 | Null cloud: `routed_only` excluded from SLO stats; `--max-tokens` mirrors the payload | ✅ unit tests |
| 6 | **nimbus neutrality**: with free capacity, 0 kicks, ≡ all_local | ✅ p50 113 vs 112 ms (measured pre-revert; at no pressure old and corrected policies are identical — 0 kicks either way; ticks now fire per arrival since kick-before-dispatch) |
| 7 | Historical pre-v3 max-displacement run on compute-bound `extreme_burst_1200`: nimbus 25.6 % outsourced, local p50 12.4 s vs random-at-25.3 % 108.7 s (all-local 325 s), but all policies still had very high local SLO violations | ⚠️ selection signal only, **not v3 validation**: the workload was compute/slot-bound, outside v3's explicitly KV-bound scope |

## Non-goals (the boundary is the design)

| Not here | Why | Returns when |
|---|---|---|
| cloud latency modeling | routing reads no cloud-side metric; a fake sink proves the architecture | when plotting cost-vs-SLO frontiers |
| sweep / plotting drivers | first target cell only just measured | with the frontier experiments |

(nimbus + KV-aware admission were in this table until 2026-07-09 — both now
implemented: the nimbus kick check runs BEFORE dispatch, so under
`--policy nimbus` a saturated engine sheds new arrivals even with free slots.)

## The Nimbus v3 policy (`--policy nimbus --kv-capacity-tokens N`)

Queue-level shedding, adjudicated BEFORE local dispatch on every
arrival/completion. The authoritative design is
[`docs/notion_algorithm_design_v3.md`](../docs/notion_algorithm_design_v3.md).
The online rule is:

```
footprint(req)    = local_prompt_tokens + expected_decode                 [tokens]
residence(req)    = local_prompt_tokens / prefill_tput + expected_decode × TPOT
displacement(req) = footprint(req) × residence(req)                      [token·s]

G = max(0, Σ footprint(waiting) + Σ remaining_decode(inflight) - K_headroom)
release_target = G + 0.05 × K_headroom

if G > 0:
    kick by ascending cloud_cost(req) / displacement(req)
    until Σ footprint(kicked) >= release_target
```

Footprint and real `/metrics` headroom share the same unit and decide whether
the queue fits. Displacement enters only the victim ordering; cloud price makes
that ordering cost-aware. A 5 % release margin prevents immediate retriggering.
For Nimbus local calls, the sender enables vLLM continuous usage stats and
tracks exact cumulative generated tokens; it never treats MTP content chunks
as individual tokens.

This policy is deliberately **KV-bound**. Compute/slot-bound overload requires a
separate trigger and is not silently treated as KV pressure. No online
knapsack solver runs in v3; an exact cover-form DP is planned as an offline
evaluation reference and is not yet implemented.
