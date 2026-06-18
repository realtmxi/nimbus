# Nimbus: Cache-Displacement-Aware Outsourcing for Hybrid LLM Inference

Nimbus is a hybrid LLM inference system that combines local GPU deployment with
serverless cloud APIs. When burst traffic exceeds local capacity, Nimbus
intelligently selects which requests to outsource to cloud APIs based on their
**cache displacement** — the product of KV memory footprint and residence time
— rather than compute cost (FLOPs).

**Key result**: At 15% outsource fraction, Nimbus achieves **11.7x lower TTFT
p50** than FLOP-based outsourcing on the ShareGPT+BurstGPT trace
(Qwen2.5-7B on RTX PRO 6000 Blackwell).

## Repository Structure

```
nimbus/                          Core algorithm (Python module)
  decision.py                    OutsourcingEngine: KV-time budget + iterative knapsack
  knapsack.py                    KnapsackSolver: dp_scaled, fractional, dp, random
  violation_detection.py         Legacy FLOP TTFT detector (baseline only)
  candidate_selection.py         Candidate request filtering
  cost_calculator.py             API cost model
  flop_calculator.py             Compute cost (Nimbus v0)
  profiled_flop_calculator.py    Profiled compute cost
  request.py                     OutsourcingRequestInfo dataclass
  request_tracker.py             Outsourced request tracking
  queue.py                       WaitingQueueInterface
  adapters.py                    Waiting-queue adapter (legacy SGLang name)

experiments/                     End-to-end trace replay against live serving engines
  run_engine.py                  Online Nimbus vs baselines with shared accounting
  run_engine_sweep.py            Online iso-SLO cost/violation sweep driver
  run_offload_strategies.py      Fixed-fraction baseline/oracle replay infra
  metrics_collector.py           Time-series KV/queue/throughput metrics
  run_cachedisp_knee.sh          Knee sweep across outsource fractions
  run_cachedisp_repeats.sh       Multi-seed repeats for error bars

scripts/analysis/                Offline analysis (no GPU required)
  exp_knapsack_vs_sorting.py     Knapsack vs greedy comparison
  exp_motivation_figure.py       Memory-bottleneck motivation figure
  plot_engine_sweep.py           Online iso-SLO cost/violation plot

docs/
  nimbus_v2_pitch.md             Paper pitch (Cache Displacement story)
  tcpo_oracle_design.md          Trace-Clairvoyant Pressure Oracle design
  exp_knapsack_vs_sorting_design.md

scripts/download_data.sh         Compose/copy trace data
data/                            Local trace files (gitignored)
```

## Outsourcing Strategies (Baselines + Ours)

Nimbus's online decision path is `experiments/run_engine.py`, which calls the
core `nimbus.decision.OutsourcingEngine` with the default v2 token-seconds
cache-displacement weight. `experiments/run_offload_strategies.py` implements
fixed-fraction baselines/oracles behind a unified `OffloadStrategy` interface:

### Intuitive baselines (oblivious / extremes)

| Strategy | CLI name | Description |
|----------|----------|-------------|
| AllLocalStrategy | `all_local` | No outsourcing (cost lower bound, latency upper bound) |
| AllCloudStrategy | `all_cloud` | Outsource everything (cost upper bound, no local pressure) |
| FIFOStrategy | `fifo` | Outsource the first N% requests by arrival order |
| RandomRequestStrategy | `random_request` | Outsource each request i.i.d. with probability `fraction` |

### System-state baselines (use system signals, not request features)

| Strategy | CLI name | Description |
|----------|----------|-------------|
| PressureGatedStrategy | `pressure_gated` | Outsource only when KV pressure exceeds threshold |
| SessionAwareStrategy | `session_aware` | Outsource entire sessions to preserve prefix continuity |
| GatedSessionAwareStrategy | `gated_session_aware` | Pressure gate + session-sticky decisions |

### Feature-aware baselines (use per-request features)

| Strategy | CLI name | Description |
|----------|----------|-------------|
| SizeOutsourceLongStrategy | `size_long` | Outsource largest-prefill requests |
| SizeOutsourceShortStrategy | `size_short` | Outsource smallest-prefill requests (worst case) |

### Fixed-Fraction Heuristic

| Strategy | CLI name | Description |
|----------|----------|-------------|
| **CacheDispStrategy** | `cache_disp` | Weight = `prefill_tokens * decode_tokens` (memory-time product) |

## Quick Start

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Validate Core Logic
```bash
python -m unittest discover -s tests -v
python -m compileall nimbus experiments scripts tests
python experiments/run_engine.py --synthetic-burst --local mock --weight v2 \
    --output-dir logs/engine_smoke
```

### Download trace data
```bash
bash scripts/download_data.sh           # Primary trace (~491 MB)
bash scripts/download_data.sh --all     # All traces (~20 GB)
```
See `data/README.md` for details on available datasets.

### Run Offline Analysis (no GPU needed)
```bash
# Knapsack vs greedy sorting comparison
python scripts/analysis/exp_knapsack_vs_sorting.py

# Motivation: memory is the bottleneck
python scripts/analysis/exp_motivation_figure.py
```

### Run Online Nimbus Smoke Test (no GPU)
```bash
python experiments/run_engine.py \
    --synthetic-burst \
    --local mock \
    --weight v2 \
    --output-dir logs/engine_smoke
```

### Run Online Nimbus on Real vLLM
vLLM is the default serving backend for the current experimental path. On a GPU
host with the model available locally, use the smoke driver to start a temporary
vLLM server, run both normal-capacity and forced-pressure Nimbus checks, analyze
the pressure run, and stop the server:
```bash
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \
    bash experiments/run_vllm_smoke.sh logs/vllm_smoke
```

The low-level harness also defaults to vLLM:
```bash
python experiments/run_engine.py \
    --policy nimbus \
    --weight v2 \
    --local real \
    --serving-url http://127.0.0.1:18200 \
    --model qwen2.5-0.5b \
    --local-kv-tokens 1786208 \
    --trace-file data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl \
    --time-scale 5.0 \
    --output-dir logs/engine_vllm
```
For a tighter pressure check, override the SLO:
```bash
SLO_S=1.0 bash experiments/run_vllm_smoke.sh logs/vllm_smoke_slo1
```
For an outsource-favorable synthetic smoke, make the simulated cloud TTFT
explicit:
```bash
SLO_S=0.5 CLOUD_TTFT_MS=300 \
    bash experiments/run_vllm_smoke.sh logs/vllm_smoke_slo05_cloud300
```
The vLLM smoke uses `SYNTHETIC_PROMPT_MODE=sized` by default, so synthetic
token metadata and prompt text are capped together before hitting the real
server. Override `SYNTHETIC_PROMPT_TOKEN_CAP` if you change `--max-model-len`.
Treat this as an integration smoke, not a paper-grade policy comparison: the
temporary 0.5B vLLM server may be fast enough that `all_local` also meets loose
synthetic SLOs. Use the iso-SLO sweep on a real trace or mock-mode stress trace
for algorithm comparisons.

To profile real decode TPOT first and feed that profile into the V2 decision:
```bash
RUN_TPOT_PROFILE=1 TPOT_BATCH_SIZES="1 2 4" \
    bash experiments/run_vllm_smoke.sh logs/vllm_smoke_profiled
```
This writes `logs/vllm_smoke_profiled/tpot_profile.json`. You can also run the
profiler directly against any OpenAI-compatible endpoint:
```bash
python experiments/profile_serving_tpot.py \
    --serving-url http://127.0.0.1:18200 \
    --model qwen2.5-0.5b \
    --engine vllm \
    --gpu 1 \
    --batch-sizes 1 2 4 8 \
    --prompt-tokens 512 \
    --decode-tokens 128 \
    --output logs/tpot_profile.json
```
If the profile contains `prefill_throughput_tokens_per_s`, Nimbus uses it for
V2 prefill timing instead of `--prefill-tput`.

The driver writes:
- `logs/vllm_smoke/normal/engine_summary.csv`
- `logs/vllm_smoke/pressure/engine_summary.csv`
- `logs/vllm_smoke/pressure/engine_sweep_normalized.csv`
- `logs/vllm_smoke/sweep/engine_summary.csv`
- `logs/vllm_smoke/sweep/engine_sweep_normalized.csv`
- `logs/vllm_smoke/tpot_profile.json` when `RUN_TPOT_PROFILE=1`
- `logs/vllm_smoke/manifest.json`

The underlying Blackwell-compatible vLLM launch command is:
```bash
CUDA_VISIBLE_DEVICES=1 \
vllm serve /path/to/Qwen2.5-0.5B-Instruct \
    --served-model-name qwen2.5-0.5b \
    --host 127.0.0.1 \
    --port 18200 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.25 \
    --trust-remote-code \
    --enforce-eager \
    --attention-backend TRITON_ATTN
```

Then run Nimbus against vLLM's OpenAI-compatible API and `/metrics` endpoint:
```bash
python experiments/run_engine.py \
    --synthetic-burst \
    --synthetic-n 8 \
    --policy nimbus \
    --weight v2 \
    --local real \
    --serving-engine vllm \
    --serving-url http://127.0.0.1:18200 \
    --model qwen2.5-0.5b \
    --local-kv-tokens 1786208 \
    --output-dir logs/engine_vllm_smoke
```

`--serving-engine vllm` directly reads `vllm:kv_cache_usage_perc` from the
serving engine's `/metrics` text endpoint and maps it onto the configured
`--local-kv-tokens` capacity. This does not require running a Prometheus server.
`--serving-engine sglang` reads `sglang:num_used_tokens` and
`sglang:max_total_num_tokens` directly.

SGLang remains available as a compatibility path by passing
`--serving-engine sglang --serving-url http://localhost:8200`.

### Real cloud sink (`--cloud real`)

By default outsourced requests use `--cloud sim` (modeled TTFT + real `$` from
token counts). To measure **real** cloud TTFT and bill real usage, stream from
an OpenAI-compatible endpoint. The API key is read from an environment variable
— never hardcode it.
```bash
export OPENROUTER_API_KEY1=...        # or: source .env
python experiments/run_engine.py \
    --policy nimbus --weight v2 \
    --local real --serving-url http://127.0.0.1:18200 --model qwen3-32b \
    --cloud real \
    --cloud-url https://openrouter.ai/api/v1/chat/completions \
    --cloud-model qwen3-32b \
    --cloud-api-key-env OPENROUTER_API_KEY1 \
    --trace-file data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl \
    --output-dir logs/engine_realcloud
```
For an apples-to-apples hybrid comparison, serve the **same model** in the cloud
as locally (`--cloud-model` == `--model`); a different cloud model confounds the
cost/latency comparison. The streaming client (TTFT capture, usage-based cost,
error handling) is ported from the `vllm/` open-loop baseline.

For providers with explicit prompt-cache TTL and cached-input pricing, add:
```bash
    --remote-cache-ttl-s 300 \
    --cached-in-price 0.03
```

In `--local real` mode, Nimbus reads live KV pressure from
`<serving-url>/metrics`. Transient metrics failures use the last good KV sample;
if the first read fails, the harness fails closed by treating local KV as full.
`engine_summary.csv` records these events in `metrics_read_failures`.

Request-level CSVs report end-to-end `ttft_ms` as arrival-to-first-token time,
including time spent waiting in Nimbus's local queue before a request is admitted
locally or kicked to cloud. The service-only component is kept separately as
`service_ttft_ms`, with `queue_delay_ms` showing the waiting component.

### Run Online Iso-SLO Sweep
```bash
python experiments/run_engine_sweep.py \
    --policies nimbus cachedisp_oracle random all_local all_cloud \
    --fractions 0.10 0.15 0.20 0.25 0.30 \
    --nimbus-weights v2 \
    --local real \
    --serving-engine vllm \
    --serving-url http://127.0.0.1:18200 \
    --local-kv-tokens 1786208 \
    --trace-file data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl \
    --time-scale 5.0 \
    --output-dir logs/engine_sweep
```

Analyze the sweep:
```bash
python scripts/analysis/plot_engine_sweep.py logs/engine_sweep \
    --ttft-slo-ms 5000 \
    --max-violation-pct 0.0
```

For quick no-GPU algorithm iteration, use the mock stress sweep:
```bash
bash experiments/run_mock_stress_sweep.sh logs/mock_stress_sweep
```
This writes `engine_summary.csv`, `engine_sweep_normalized.csv`, and a manifest
under the output directory. Mock mode is the right place to compare policy
behavior under synthetic KV pressure because token metadata directly drives the
local timing model. By default it uses the V2 timing model: local TTFT is
`prefill_tokens / --prefill-tput`, while local slot/KV occupancy lasts through
`prefill + decode_tokens * TPOT`. The script defaults to `TIME_SCALE=50`; much
larger values can make Python/OS scheduling overhead appear as synthetic queue
delay. `CLOUD_TTFT_GUARD_MULTIPLIER` defaults to `1.5` so Nimbus makes cloud
handoff decisions before modeled cloud jitter consumes the TTFT deadline.

### Run Bistability Validation
The old `--mode hysteresis` ramp is only a quick exploratory smoke. For paper
numbers, use the steady-state hold plus trigger-removal protocol:
```bash
python experiments/run_offload_strategies.py \
    --mode bistability \
    --sglang-url http://localhost:8200 \
    --trace-file data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl \
    --fractions 0.30 0.35 0.40 0.45 \
    --bistability-trigger-fraction 0.0 \
    --bistability-hold-requests 2000 \
    --bistability-max-holds 4 \
    --bistability-steady-windows 3 \
    --bistability-steady-cv 0.10 \
    --output-dir logs/bistability
```
Only cite rows where `clean_steady` and `post_trigger_steady` are both true.

### Run Fixed-Fraction Baseline Knee Sweep (legacy SGLang path)
```bash
# Start SGLang on a GPU box, then:
python experiments/run_offload_strategies.py \
    --sglang-url http://localhost:8200 \
    --mode knee \
    --fractions 0.0 0.15 0.20 0.25 0.30 0.35 0.50 \
    --strategies cache_disp session_aware oracle_size \
    --output-dir logs/cachedisp_knee
```

The Nimbus main path is the online `run_engine.py --policy nimbus --weight v2`
experiment above.

## Data

Trace files are not committed (multi-GB). `scripts/download_data.sh` composes
the primary ShareGPT+BurstGPT workload locally from public raw traces, following
RouteWise's data-prep path. Remote mirror copying is still available with
`scripts/download_data.sh --remote` after setting `NIMBUS_TRACE_SSH` and
`NIMBUS_REMOTE_DATA_ROOT`. See `data/README.md` for schema and dataset details.

- ShareGPT+BurstGPT: BurstGPT arrivals/token counts with ShareGPT prompt text
- RouteWise traces: long-context rednote agent + production freeinference logs

## Citation

```
@inproceedings{nimbus2026,
  title  = {Nimbus: Cache-Displacement-Aware Outsourcing for Hybrid LLM Inference},
  author = {Murphy and Yiyan Zhai and Yiyu Liu and Juncheng Yang},
  booktitle = {EuroSys},
  year = {2026}
}
```
