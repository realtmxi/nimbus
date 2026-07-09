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
| `nimbus.py` | The nimbus shedding policy: kick the largest cache-displacers when the waiting set exceeds real KV headroom (`--policy nimbus --kv-capacity-tokens N`); knapsack solver retained for the cost-aware ablations |
| `test_run.py` / `test_common.py` / `test_nimbus.py` | 47 unit tests; no network / aiohttp / GPU needed |

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
  set a shedding policy (nimbus) will operate on. Why own the queue: engines
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
| 7 | **nimbus under pressure** (slots=4, KV budget 3k): self-selected 29.0 % outsourcing, local SLO violations **0 %** vs **91.1 %** for all_local under the identical constraint (p50 435 ms vs 32.8 s) | ⚠️ measured with the PRE-REVERT ($-knapsack) policy — mechanism demonstration only; re-measure with the corrected CacheDisp policy on the extreme_burst cell (queued, pending GPU reset) |

## Non-goals (the boundary is the design)

| Not here | Why | Returns when |
|---|---|---|
| cloud latency modeling | routing reads no cloud-side metric; a fake sink proves the architecture | when plotting cost-vs-SLO frontiers |
| sweep / plotting drivers | first target cell only just measured | with the frontier experiments |

(nimbus + KV-aware admission were in this table until 2026-07-09 — both now
implemented: the nimbus kick check runs BEFORE dispatch, so under
`--policy nimbus` a saturated engine sheds new arrivals even with free slots.)

## The nimbus policy (`--policy nimbus --kv-capacity-tokens N`)

Queue-level shedding, adjudicated BEFORE local dispatch on every
arrival/completion. Semantics — the CacheDisp core (reaffirmed 2026-07-09
after review caught an implementation drift):

```
displacement(req) ≈ prompt_tokens × (prefill_time + decode_tokens × TPOT)   [token·s]
while Σ token_footprint(waiting) > K_avail(/metrics):
    kick the LARGEST displacer
```

Shed the requests that pin the most cache for the longest — API cost does not
enter the shed rule (it belongs to the frontier comparison). Cost-aware
knapsack variants (minimize cloud $ s.t. enough displacement released, or
maximize released displacement s.t. a cloud budget) are planned ablations;
`solve_knapsack`/`value_usd` are retained for them. Remaining open knob: the
trigger (fits-in-KV vs SLO-bound head wait = Notion decision 2).
