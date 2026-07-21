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
| `nimbus.py` | Nimbus v3 baseline plus orthogonal KV-gap / predicted-TTFT triggers and victim-selector ablations |
| `test_run.py` / `test_common.py` / `test_nimbus.py` | 105 unit tests; no network / aiohttp / GPU needed |

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
- Requests with the same trace timestamp form one arrival cohort. The policy
  sees the complete cohort before any member is dispatched, so a selector is
  never benchmarked on a sequence of artificial one-item queues.

## Cloud sinks

`--cloud null` (**default**): a fake request — records "this one was routed to
cloud" plus the trace's token counts and cost (`--max-tokens` semantics are
identical to the real payload: it replaces the trace value). No latency is
modeled; such rows carry `routed_only=true` and are excluded from SLO stats.

`--cloud real --cloud-url … --cloud-model … --cloud-api-key-env KEY`: real
streaming calls; `--cloud-max-concurrency` (default 32) guards against
self-inflicted 429s under burst. Use only when the experiment needs real cloud
latency. (A teammate has 14k real OpenRouter measurements under
`$JSCRATCH/initial_result/` for estimating distributions; measured
qwen3-32b TTFT p50 ≈ 10 s — reasoning + provider queueing; the cloud is not "fast".)

For the TTFT-only OpenRouter probe, additionally use
`--cloud-provider deepinfra --cloud-no-fallbacks
--cloud-stop-after-first-token`. The last flag aborts after the first non-empty
reasoning or content delta (the same generated-token boundary as the bound local
vLLM deployment). It records TTFT but deliberately leaves E2E/TPOT absent and
cost pending when the final usage event is not received. This is an opt-in probe;
ordinary real-cloud calls still drain to `[DONE]`.

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
rows are excluded. `pessimistic_combined` also counts every cloud route as an
SLO violation, so NullCloud cannot reward over-shedding; it is a NullCloud upper
bound, not the headline for real cloud. For real cloud use `overall.slo_*` after
verifying `overall.slo_measured_n == overall.n` and `cloud.routed_only == 0`.
Billing: failed
requests cost $0; the local side always $0. `--decision-log FILE` optionally
records each applied/stale Nimbus decision and its victim ordering. Raw rows
also retain `scheduler_prompt_tokens` next to endpoint-reported
`prompt_tokens`; the summary's `token_alignment` section makes a trace/payload
unit mismatch visible instead of silently accepting it.

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
arrival/completion. This historical shipped baseline is specified in
[`docs/notion_algorithm_design_v3.md`](../docs/notion_algorithm_design_v3.md);
the current algorithm/evidence entry point is
[`docs/nimbus_algorithm_and_results_2026-07.zh-CN.md`](../docs/nimbus_algorithm_and_results_2026-07.zh-CN.md).
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

On a deployment whose gauge probe verifies token-KV semantics, footprint and
`/metrics` headroom share the same unit and decide whether the queue fits. The
metric name alone is not proof: hybrid-model gauges may instead represent
per-sequence state. Displacement enters only the victim ordering; cloud price
makes that ordering cost-aware. A 5 % release margin prevents immediate retriggering.
For Nimbus local calls, the sender enables vLLM continuous usage stats and
tracks exact cumulative generated tokens; it never treats MTP content chunks
as individual tokens.

This policy is deliberately **KV-bound**. Compute/slot-bound overload requires a
separate trigger and is not silently treated as KV pressure. No online
knapsack solver runs in v3. The implementation uses heuristic
`cost/displacement` ordering to cover a footprint target; it is not a standard
minimum-cost-cover density algorithm. An exact cover-form DP is planned as an
offline evaluation reference and is not yet implemented.

## Experimental predicted-TTFT trigger

`--nimbus-trigger ttft_pred` predicts FCFS admission over two resources: local
sequence slots and one shared prefill-compute lane. This matters because 128
free sequence slots do not make 128 concurrent prompts receive their first
token simultaneously. The model includes waiting age, exact decode progress,
conservative unfinished in-flight prefill work, cumulative waiting prefill,
and a fitted fixed first-token overhead. It does not read the KV gauge. Its
absolute timing parameters must be explicitly calibrated:

```bash
python -m router.run ... --policy nimbus \
  --nimbus-trigger ttft_pred --nimbus-selector cost_cachedisp_old \
  --prefill-tput 3270 --tpot-ms 152 --first-token-overhead-ms 461 --slo-s 5 \
  --ttft-guard-ms 1685 --nimbus-tick-ms 250
```

The numbers above only illustrate the separate parameters; use the values from
the same-server profile artifact. `first-token-overhead-ms` affects only TTFT.
`tpot-ms` estimates decode slot residence and remains part of the original v2
weight, so the two quantities must not be conflated.

Selectors are `newest`, `waiting_random`, `max_cachedisp_old`,
`cost_cachedisp_old`, and `cost_disp_current`. The two `*_cachedisp_old`
variants reuse the original v2 weight formula but are heuristic orderings, not
the historical exact 0/1-knapsack implementation. The current TTFT stop rule
is diagnostic: it makes the retained **waiting-queue survivors** predicted-safe;
already in-flight requests are outside that post-kick claim, which is why the
decision log and summary label the scope `waiting_only`. In NullCloud runs the
pessimistic-combined field is only an assumed upper bound; a real-cloud run uses
observed `overall.slo_violation_pct`. Decode length is still the trace cap (oracle); estimator and
combined-objective ablations remain follow-up work. The reproducible driver is
`experiments/run_ttft_selector_matrix.sh`; an explicit
`anchor:all_local:0` arm uses the same bound manifest/marker contract without
pretending the anchor has a Nimbus trigger or decision log.

**Current evidence boundary (2026-07-15).** In the completed pre-registered
11,605-request dense-32B no-cache cell, `ttft_pred + cost_cachedisp_old` and
`ttft_pred + cost_disp_current` each retained zero local violations; old V2 was
within the frozen route-equivalence band and cost 4.98% less. `ttft_pred +
newest` left 39 violations; all had planned commitment above the maximum
profiled cell, consistent with leaving calibration coverage (no runtime
support-envelope classifier exists yet), while `kv_gap + cost_disp_current`
left 5,249.
Consequently `ttft_pred` is still experimental and needs an explicit
support-envelope/resource fallback; `kv_gap` remains the trigger default under
`--policy nimbus` for compatibility, not a demonstrated TTFT-safety guarantee.
See Section 5g of
[`../docs/v3_experiments_2026-07.md`](../docs/v3_experiments_2026-07.md).

### Token-aligned no-cache experiment prerequisite

The primary ShareGPT/BurstGPT file stores cumulative conversation lengths in
`num_prefill_tokens`, while `prompt_text` is only the current user turn. Direct
replay can therefore predict 500 tokens and send fewer than 20. Do not use that
payload for a displacement-selector claim. Materialize a self-consistent
no-cache trace with the deployment tokenizer, run vLLM with prefix caching
disabled, and profile the same deployment before selecting parameters:

```bash
python tools/materialize_token_aligned_trace.py \
  --input <sharegpt-burstgpt.jsonl> --output <aligned.jsonl> \
  --scenario extreme_burst_1200 --tokenizer <model-path> \
  --salt dense32b-nocache-v1 \
  --max-prompt-tokens 32768 --max-decode-tokens 1024 \
  --max-context-tokens 40960 --overflow-policy error

python tools/profile_ttft_batch.py \
  --base-url http://127.0.0.1:8010 --model qwen3-32b \
  --tokenizer <model-path> --server-log <active-vllm.log> \
  --server-pid <recorded-pid> \
  --kv-capacity-tokens <startup-log-token-capacity> \
  --slo-s 5 \
  --output <profile.json>

DATA=<aligned.jsonl> BASE_URL=http://127.0.0.1:8010 MODEL=qwen3-32b \
  PROFILE=<profile.json> SERVER_LOG=<active-vllm.log> SERVER_PID=<recorded-pid> \
  OUT_DIR=<new-results-dir> experiments/run_ttft_selector_matrix.sh
```

This leg has `cached_tokens = 0` by construction. A cache-aware workload is a
separate experiment; these results must not be presented as validating the
cached-token term of the original formula. The prompt/decode/context caps make
this a transformed synthetic workload; report the manifest's affected-row
counts with every result. On a quota-constrained GPU host, point `TMPDIR`,
`HF_HOME`, and `XDG_CACHE_HOME` at scratch before either command.
