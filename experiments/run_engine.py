#!/usr/bin/env python3
"""Unified online harness: run our engine OR any baseline through the SAME loop /
cost meter / SLO accounting, for an apples-to-apples iso-SLO comparison.

REAL-SIGNAL DESIGN (no latency prediction): this is a real-machine experiment, not
a simulation, so the engine is driven by REAL serving-engine signals, NOT a FLOP/Vidur-style
TTFT predictor:
  - The knapsack BUDGET = real available local KV headroom (`K_avail` tokens, read
    live from the serving engine's /metrics) x SLO horizon -> token-seconds.
  - WHICH to outsource = knapsack over the waiting queue, weight = cache displacement
    (--weight v2 = token-seconds, unit-matched to the budget), value = API $ saved.
  - The loop is Notion's `while TTFT_violation_exists(): solve knapsack -> kick 1 -> recheck`,
    where TTFT_violation_exists() == (sum of waiting weight > budget) -- a REAL-signal check
    that responds to kicking, so NO latency prediction is needed.
  - The FLOP violation detector is NOT used on the decision path.
This also closes the old "budget units" question: the budget is the real measured KV
capacity x time, in the same token-seconds unit as the V2 weight.

Baselines (`--policy <other>`) decide per-arrival via the strategy classes from
run_offload_strategies, through the SAME loop + SimCloud cost meter + iso-SLO accounting.

Fidelities:
  --local mock : modeled local TTFT + modeled KV (used = in-flight prompt tokens); no GPU.
  --local real : admit to a live OpenAI-compatible server, measure real TTFT,
                 read real KV pressure from /metrics.
  --cloud sim  : outsourced -> real $ (token counts) + modeled TTFT (real cloud = later).
"""

import argparse
import asyncio
import contextlib
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `import nimbus`
sys.path.insert(0, str(Path(__file__).resolve().parent))      # experiments/ -> sibling module

# aiohttp + run_offload_strategies imported lazily inside functions (unit-testable w/o aiohttp).
from nimbus.adapters import SGLangWaitingQueueAdapter  # noqa: E402
from nimbus.cost_calculator import APICostCalculator  # noqa: E402
from nimbus.decision import OutsourcingEngine  # noqa: E402
from nimbus.flop_calculator import SimpleFLOPCalculator  # noqa: E402
from nimbus.request import OutsourcingRequestInfo  # noqa: E402
from nimbus.tpot_profile import TPOTProfile  # noqa: E402

BASELINE_POLICIES = {
    "all_local", "all_cloud", "random", "fifo", "pressure_gated",
    "session_aware", "size_long", "flop_oracle", "cachedisp_oracle",
}


WEIGHT_MODES = {
    "v0": "v0_flops",
    "v1": "v1_cache_displacement",
    "v2": "v2_token_seconds",
}


@contextlib.asynccontextmanager
async def null_session():
    yield None


class SimCloud:
    """Simulated cloud sink: real $ from token counts, modeled TTFT.

    Optional remote-prefix cache accounting follows the Notion API cost model:
    cached input tokens are billed at a separate price when the same session was
    previously outsourced within an explicit provider TTL.
    """

    def __init__(
        self,
        in_price: float,
        cached_in_price: float | None,
        out_price: float,
        ttft_mean_ms: float,
        remote_cache_ttl_s: float = 0.0,
        jitter: float = 0.3,
        seed: int = 0,
    ):
        self._cost = APICostCalculator(in_price, out_price, cached_in_price)
        self._ttft_mean = ttft_mean_ms
        self._remote_cache_ttl_s = max(0.0, remote_cache_ttl_s)
        self._remote_cache: dict[str, tuple[float, int]] = {}
        self._jitter = jitter
        self._rng = random.Random(seed)

    def serve(self, req: OutsourcingRequestInfo, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        ttft = self._ttft_mean * (1.0 + self._rng.uniform(-self._jitter, self._jitter))
        remote_cached_tokens = self.estimate_remote_cached_tokens(req, now)
        breakdown = self._cost.calculate_cost_breakdown(
            input_tokens=req.num_prompt_tokens,
            cached_input_tokens=remote_cached_tokens,
            output_tokens=req.num_output_tokens,
        )
        self._remember_remote_prompt(req, now)
        return {
            "ttft_ms": ttft,
            "cost_usd": breakdown["total_cost_usd"],
            "success": True,
            "remote_cached_tokens": remote_cached_tokens,
            **breakdown,
        }

    def estimate_remote_cached_tokens(
        self,
        req: OutsourcingRequestInfo,
        now: float,
    ) -> int:
        if "remote_cached_tokens" in req.metadata:
            return min(req.num_prompt_tokens, max(0, int(req.metadata["remote_cached_tokens"])))

        ttl_cached = 0
        session_id = req.metadata.get("session_id")
        if self._remote_cache_ttl_s > 0 and session_id is not None:
            cached = self._remote_cache.get(str(session_id))
            if cached is not None:
                cached_at, cached_prompt_tokens = cached
                if now - cached_at <= self._remote_cache_ttl_s:
                    ttl_cached = min(cached_prompt_tokens, req.num_prompt_tokens)
        return min(req.num_prompt_tokens, max(0, ttl_cached))

    def _remember_remote_prompt(self, req: OutsourcingRequestInfo, now: float) -> None:
        if self._remote_cache_ttl_s <= 0:
            return
        session_id = req.metadata.get("session_id")
        if session_id is None:
            return
        self._remote_cache[str(session_id)] = (now, req.num_prompt_tokens)


class RealCloud:
    """Real cloud sink: streams from an OpenAI-compatible endpoint, measures
    real TTFT, and bills real $ from returned usage (falling back to planned
    token counts).

    Same ``serve``-shaped result dict as :class:`SimCloud` so outsourced-request
    accounting is identical across ``--cloud sim`` and ``--cloud real``. The
    remote-prefix cache (TTL) model is also mirrored so cost stays comparable.
    Ported from the open-loop baseline client (``vllm/run.py``).
    """

    def __init__(
        self,
        url: str,
        model: str,
        api_key_env: str | None,
        in_price: float,
        cached_in_price: float | None,
        out_price: float,
        remote_cache_ttl_s: float = 0.0,
        timeout_s: float = 600.0,
        temperature: float = 0.0,
    ):
        self._url = url
        self._model = model
        self._api_key_env = api_key_env
        self._cost = APICostCalculator(in_price, out_price, cached_in_price)
        self._remote_cache_ttl_s = max(0.0, remote_cache_ttl_s)
        self._remote_cache: dict[str, tuple[float, int]] = {}
        self._timeout_s = timeout_s
        self._temperature = temperature

    def estimate_remote_cached_tokens(self, req: OutsourcingRequestInfo, now: float) -> int:
        if "remote_cached_tokens" in req.metadata:
            return min(req.num_prompt_tokens, max(0, int(req.metadata["remote_cached_tokens"])))
        ttl_cached = 0
        session_id = req.metadata.get("session_id")
        if self._remote_cache_ttl_s > 0 and session_id is not None:
            cached = self._remote_cache.get(str(session_id))
            if cached is not None:
                cached_at, cached_prompt_tokens = cached
                if now - cached_at <= self._remote_cache_ttl_s:
                    ttl_cached = min(cached_prompt_tokens, req.num_prompt_tokens)
        return min(req.num_prompt_tokens, max(0, ttl_cached))

    def _remember_remote_prompt(self, req: OutsourcingRequestInfo, now: float) -> None:
        if self._remote_cache_ttl_s <= 0:
            return
        session_id = req.metadata.get("session_id")
        if session_id is None:
            return
        self._remote_cache[str(session_id)] = (now, req.num_prompt_tokens)

    def cost_dict(
        self,
        prompt_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        ttft_ms: float | None,
        success: bool,
    ) -> dict:
        """Build the SimCloud-shaped result dict from measured token counts."""
        breakdown = self._cost.calculate_cost_breakdown(
            input_tokens=prompt_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )
        return {
            "ttft_ms": ttft_ms,
            "cost_usd": breakdown["total_cost_usd"],
            "success": bool(success and ttft_ms is not None),
            "remote_cached_tokens": min(prompt_tokens, max(0, cached_input_tokens)),
            **breakdown,
        }

    async def serve_async(self, session, req: OutsourcingRequestInfo, now: float) -> dict:
        import os

        headers = {"Content-Type": "application/json"}
        if self._api_key_env:
            key = os.environ.get(self._api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"

        payload = {
            "model": self._model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self._temperature,
            "max_tokens": max(1, int(req.num_output_tokens)),
            "messages": [{"role": "user", "content": req.metadata.get("prompt_text", "")}],
        }
        remote_cached = self.estimate_remote_cached_tokens(req, now)

        start = time.perf_counter()
        first_token_time = None
        usage: dict = {}
        completion_chunks = 0
        done = False
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self._timeout_s)
            async with session.post(
                self._url, headers=headers, json=payload, timeout=timeout
            ) as resp:
                if resp.status < 400:
                    buffer = ""
                    async for raw in resp.content.iter_chunked(8192):
                        buffer += raw.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line or line.startswith(":") or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                done = True
                                break
                            obj = json.loads(data)
                            if obj.get("usage"):
                                usage = obj["usage"]
                            choices = obj.get("choices") or []
                            delta = choices[0].get("delta") if choices else {}
                            token = delta.get("content") if isinstance(delta, dict) else ""
                            if token:
                                first_token_time = first_token_time or time.perf_counter()
                                completion_chunks += 1
                        if done:
                            break
        except Exception:
            first_token_time = first_token_time  # fall through; success gated on first token

        ttft_ms = None if first_token_time is None else (first_token_time - start) * 1000.0
        prompt_tokens = int(usage.get("prompt_tokens") or req.num_prompt_tokens)
        completion_tokens = int(
            usage.get("completion_tokens") or (completion_chunks if first_token_time else 0)
        )
        self._remember_remote_prompt(req, now)
        return self.cost_dict(
            prompt_tokens=prompt_tokens,
            cached_input_tokens=min(prompt_tokens, remote_cached),
            output_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            success=first_token_time is not None,
        )


def _sized_synthetic_prompt(prompt_tokens: int) -> str:
    """Return a simple prompt whose word count tracks the synthetic token count."""
    return " ".join(["hello"] * max(1, int(prompt_tokens)))


def synthetic_burst_trace(
    n: int,
    seed: int,
    prompt_mode: str = "stub",
    prompt_token_cap: int = 2048,
) -> list[dict]:
    """Bursty synthetic trace (dense cluster in the middle) for no-GPU/no-trace smoke tests."""
    rng = random.Random(seed)
    rows, t = [], 0.0
    for i in range(n):
        in_burst = (0.3 * n) < i < (0.7 * n)
        t += rng.expovariate(1 / (0.04 if in_burst else 0.4))
        prompt_tokens = rng.choice([512, 2048, 8000, 32000, 64000])
        if prompt_mode == "sized":
            prompt_tokens = min(prompt_tokens, max(1, int(prompt_token_cap)))
            prompt_text = _sized_synthetic_prompt(prompt_tokens)
        else:
            prompt_text = "ping"
        rows.append({
            "request_id": i, "ts_sec": t, "arrived_at": t,
            "num_prefill_tokens": prompt_tokens,
            "num_decode_tokens": rng.choice([64, 256, 512, 1024]),
            "num_cached_tokens": 0, "session_id": i % max(1, n // 5),
            "prompt_text": prompt_text,
        })
    return rows


def make_decider(args: argparse.Namespace, trace: list[dict]):
    """Per-arrival baseline strategy, or None for nimbus (real-KV-budget knapsack)."""
    p, f, s = args.policy, args.fraction, args.seed
    if p == "nimbus":
        return None
    from run_offload_strategies import (
        AllLocalStrategy, AllCloudStrategy, RandomRequestStrategy, FIFOStrategy,
        PressureGatedStrategy, SessionAwareStrategy, SizeOutsourceLongStrategy,
        FlopBasedStrategy, CacheDispStrategy,
    )
    return {
        "all_local": lambda: AllLocalStrategy(f, s),
        "all_cloud": lambda: AllCloudStrategy(f, s),
        "random": lambda: RandomRequestStrategy(f, s),
        "fifo": lambda: FIFOStrategy(f, s, trace),
        "pressure_gated": lambda: PressureGatedStrategy(f, s, kv_threshold=args.admit_kv),
        "session_aware": lambda: SessionAwareStrategy(f, s, trace),
        "size_long": lambda: SizeOutsourceLongStrategy(f, s, trace),
        "flop_oracle": lambda: FlopBasedStrategy(f, s, trace),
        "cachedisp_oracle": lambda: CacheDispStrategy(f, s, trace),
    }[p]()


def _parse_kv_from_metrics(
    text: str,
    serving_engine: str = "sglang",
    fallback_max_tokens: float = 0.0,
) -> tuple[float, float]:
    """Return (used_tokens, max_tokens) from serving-engine /metrics text."""
    if serving_engine == "vllm":
        usage: float | None = None
        for line in text.splitlines():
            if (
                line.startswith("vllm:kv_cache_usage_perc")
                or line.startswith("vllm:gpu_cache_usage_perc")
            ):
                usage = float(line.split()[-1])
        if usage is None:
            raise ValueError("metrics missing vLLM KV cache usage")
        max_tokens = max(0.0, float(fallback_max_tokens))
        if max_tokens <= 0:
            raise ValueError("vLLM KV parsing requires positive fallback_max_tokens")
        used = min(max(0.0, usage), 1.0) * max_tokens
        return used, max_tokens

    if serving_engine != "sglang":
        raise ValueError(f"Unknown serving_engine: {serving_engine}")

    used: float | None = None
    max_tok = 0.0
    for line in text.splitlines():
        if line.startswith("sglang:num_used_tokens"):
            used = float(line.split()[-1])
        elif line.startswith("sglang:max_total_num_tokens"):
            max_tok = float(line.split()[-1])
    if used is None:
        raise ValueError("metrics missing sglang:num_used_tokens")
    return used, max_tok


class KVMetricsState:
    """Conservative KV metrics fallback for real-engine replay.

    A transient /metrics failure should not make Nimbus believe local KV is empty.
    Use the last good sample when possible; before the first good sample, fail closed
    by reporting the configured capacity as fully used.
    """

    def __init__(self, fallback_max_tokens: float, serving_engine: str = "sglang"):
        self._fallback_max_tokens = max(0.0, float(fallback_max_tokens))
        self._serving_engine = serving_engine
        self._last: tuple[float, float] | None = None
        self.failures = 0

    def record_text(self, text: str) -> tuple[float, float]:
        used, max_tokens = _parse_kv_from_metrics(
            text,
            serving_engine=self._serving_engine,
            fallback_max_tokens=self._fallback_max_tokens,
        )
        return self.record(used, max_tokens)

    def record(self, used_tokens: float, max_tokens: float) -> tuple[float, float]:
        max_tokens = float(max_tokens or self._fallback_max_tokens)
        if max_tokens <= 0:
            snapshot = (0.0, 0.0)
        else:
            used_tokens = min(max(0.0, float(used_tokens)), max_tokens)
            snapshot = (used_tokens, max_tokens)
        self._last = snapshot
        return snapshot

    def fallback_on_error(self) -> tuple[float, float]:
        self.failures += 1
        if self._last is not None:
            return self._last
        if self._fallback_max_tokens <= 0:
            return 0.0, 0.0
        return self._fallback_max_tokens, self._fallback_max_tokens


class LocalAdmissionState:
    """Synchronous local-slot/KV reservation for the replay admission loop."""

    def __init__(self) -> None:
        self.inflight = 0
        self.mock_kv_used = 0.0

    def reserve(self, req: OutsourcingRequestInfo, local_mode: str) -> None:
        self.inflight += 1
        if local_mode == "mock":
            self.mock_kv_used += req.num_prompt_tokens

    def release(self, req: OutsourcingRequestInfo, local_mode: str) -> None:
        self.inflight = max(0, self.inflight - 1)
        if local_mode == "mock":
            self.mock_kv_used = max(0.0, self.mock_kv_used - req.num_prompt_tokens)


def _estimate_decode_batch_size(
    local_state: LocalAdmissionState,
    waiting: list[OutsourcingRequestInfo],
    used_tokens: float,
    max_tokens: float,
    admit_kv: float,
    max_inflight: int,
) -> int:
    """Estimate near-term decode batch for TPOT lookup.

    Only requests that can be admitted immediately should affect this estimate.
    Counting the whole waiting queue overstates TPOT during backlog, because most
    queued requests are not decoding yet and may be outsourced before admission.
    """
    slots = max(0, int(max_inflight) - local_state.inflight)
    if slots <= 0:
        return max(1, local_state.inflight)

    kv_headroom = max(0.0, float(admit_kv) * float(max_tokens) - float(used_tokens))
    if max_tokens <= 0:
        kv_headroom = float("inf")

    admitted = 0
    reserved = 0.0
    for req in waiting:
        if admitted >= slots:
            break
        prompt_tokens = max(0.0, float(req.num_prompt_tokens))
        if reserved + prompt_tokens > kv_headroom:
            break
        reserved += prompt_tokens
        admitted += 1

    return max(1, local_state.inflight + admitted)


def _queue_delay_ms(
    req: OutsourcingRequestInfo,
    now: float,
    time_scale: float,
    scale_wait: bool,
) -> float:
    elapsed_s = max(0.0, now - req.arrival_time)
    if scale_wait:
        elapsed_s *= max(0.0, time_scale)
    return elapsed_s * 1000.0


def _loop_sleep_seconds(tick_s: float, time_scale: float, scale_time: bool) -> float:
    if not scale_time:
        return max(0.0, tick_s)
    return max(0.0, tick_s) / max(time_scale, 1e-9)


def _effective_tpot_seconds(
    tpot_profile: TPOTProfile | None,
    fallback_tpot_s: float,
    predicted_decode_batch_size: int,
) -> float:
    if tpot_profile is None:
        return fallback_tpot_s
    return tpot_profile.tpot_seconds_for_batch(predicted_decode_batch_size)


def _mock_service_times(
    req: OutsourcingRequestInfo,
    args: argparse.Namespace,
    flop: SimpleFLOPCalculator,
    eff_flops: float,
    tpot_profile: TPOTProfile | None,
    decode_batch_size: int,
) -> tuple[float, float]:
    """Return (ttft_seconds, occupancy_seconds) for mock local execution."""
    if getattr(args, "mock_service_model", "v2") == "flops":
        prefill_s = (
            flop.compute_prefill_flops(req, req.remaining_prompt_tokens) / eff_flops
        ) if eff_flops > 0 else 0.0
        return prefill_s, prefill_s

    prefill_tput = float(
        getattr(
            args,
            "effective_prefill_tput",
            getattr(args, "prefill_tput", 0.0),
        )
    )
    prefill_s = (
        req.remaining_prompt_tokens
        / max(prefill_tput, 1e-9)
    )
    decode_s = req.remaining_output_tokens * _effective_tpot_seconds(
        tpot_profile,
        float(getattr(args, "tpot_s", 0.0)),
        decode_batch_size,
    )
    return prefill_s, prefill_s + decode_s


async def replay(args: argparse.Namespace, trace: list[dict]) -> list[dict]:
    flop = SimpleFLOPCalculator(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_attention_heads=args.num_heads, device_tflops=args.device_tflops,
    )
    tpot_profile_path = getattr(args, "tpot_profile", None)
    tpot_profile = TPOTProfile.from_json(tpot_profile_path) if tpot_profile_path else None
    effective_prefill_tput = (
        tpot_profile.prefill_throughput_tokens_per_s
        if tpot_profile is not None and tpot_profile.prefill_throughput_tokens_per_s
        else args.prefill_tput
    )
    args.effective_prefill_tput = effective_prefill_tput
    tpot_samples_s: list[float] = []
    serving_engine = getattr(args, "serving_engine", "vllm")
    adapter = SGLangWaitingQueueAdapter(metrics_url=f"{args.sglang_url.rstrip('/')}/metrics")
    if args.cloud == "real":
        cloud = RealCloud(
            url=args.cloud_url,
            model=args.cloud_model,
            api_key_env=args.cloud_api_key_env,
            in_price=args.in_price,
            cached_in_price=args.cached_in_price,
            out_price=args.out_price,
            remote_cache_ttl_s=args.remote_cache_ttl_s,
            timeout_s=args.cloud_timeout_s,
        )
    else:
        cloud = SimCloud(
            in_price=args.in_price,
            cached_in_price=args.cached_in_price,
            out_price=args.out_price,
            ttft_mean_ms=args.cloud_ttft_ms,
            remote_cache_ttl_s=args.remote_cache_ttl_s,
            seed=args.seed,
        )
    decider = make_decider(args, trace)
    engine = None
    if args.policy == "nimbus":
        engine = OutsourcingEngine(
            waiting_queue=adapter, flop_calculator=flop,
            input_price_per_million=args.in_price, output_price_per_million=args.out_price,
            cached_input_price_per_million=args.cached_in_price,
            knapsack_strategy="dp_scaled",
            weight_mode=WEIGHT_MODES[args.weight],
            prefill_throughput_tokens_per_s=effective_prefill_tput,
            tpot_seconds=_effective_tpot_seconds(tpot_profile, args.tpot_s, 1),
        )

    results: dict[str, dict] = {}
    req_objs: dict[str, OutsourcingRequestInfo] = {}
    local_state = LocalAdmissionState()
    kv_metrics = KVMetricsState(args.local_kv_tokens, serving_engine=serving_engine)
    eff_flops = flop.get_effective_flops_per_second(args.util)
    url = f"{args.sglang_url.rstrip('/')}/v1/chat/completions"
    metrics_url = f"{args.sglang_url.rstrip('/')}/metrics"
    first_ts, clock0, i, kv = trace[0]["ts_sec"], time.time(), 0, 0.0
    pending: list[asyncio.Task] = []

    def replay_time() -> float:
        if args.local == "mock":
            return (time.time() - clock0) * args.time_scale
        return time.time()

    def replay_arrival_time(row: dict) -> float:
        if args.local == "mock":
            return max(0.0, float(row["ts_sec"]) - first_ts)
        return time.time()

    def record_outsourced(req: OutsourcingRequestInfo) -> None:
        if req.request_id in results:
            return
        decision_now = replay_time()
        queue_delay_ms = _queue_delay_ms(
            req,
            decision_now,
            args.time_scale,
            scale_wait=False,
        )

        def _write(out: dict) -> None:
            service_ttft_ms = out["ttft_ms"] if out["success"] else None
            ttft_ms = None if service_ttft_ms is None else queue_delay_ms + service_ttft_ms
            results[req.request_id] = {
                "id": req.request_id, "outsourced": True, "success": out["success"],
                "ttft_ms": ttft_ms,
                "queue_delay_ms": queue_delay_ms,
                "service_ttft_ms": service_ttft_ms,
                "cost_usd": out["cost_usd"],
                "remote_cached_tokens": out["remote_cached_tokens"],
                "uncached_input_tokens": out["uncached_input_tokens"],
                "cached_input_tokens": out["cached_input_tokens"],
                "output_tokens": out["output_tokens"],
                "prefill": req.num_prompt_tokens, "decode": req.num_output_tokens,
            }

        if args.cloud == "real":
            # A real cloud call is async and takes wall-clock time. Claim the
            # request synchronously (so dedup + the per-tick nimbus guard treat
            # it as handled now), then patch in measured TTFT/cost when the
            # streaming call returns -- mirrors the admit_local task pattern.
            _write({
                "ttft_ms": None, "success": False, "cost_usd": 0.0,
                "remote_cached_tokens": 0, "uncached_input_tokens": 0,
                "cached_input_tokens": 0, "output_tokens": 0,
            })

            async def _outsource_task() -> None:
                try:
                    _write(await cloud.serve_async(session, req, decision_now))
                except Exception:
                    pass  # leave the failed-claim row (success=False -> SLO violation)

            pending.append(asyncio.create_task(_outsource_task()))
        else:
            _write(cloud.serve(req, now=decision_now))

    # aiohttp is needed for real local serving and/or a real cloud sink;
    # --local mock --cloud sim runs without it.
    send_request = None
    session_cm = null_session()
    if args.local == "real" or args.cloud == "real":
        import aiohttp
        session_cm = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None),
        )
    if args.local == "real":
        from run_offload_strategies import send_request
    async with session_cm as session:

        async def read_kv() -> tuple[float, float]:
            """Return (used_tokens, max_tokens). Real: from /metrics; mock: modeled."""
            if args.local == "real":
                try:
                    async with session.get(metrics_url) as resp:
                        return kv_metrics.record_text(await resp.text())
                except Exception:
                    return kv_metrics.fallback_on_error()
            return local_state.mock_kv_used, args.local_kv_tokens

        async def admit_local(req: OutsourcingRequestInfo) -> None:
            res = {"success": False, "ttft_ms": None}
            try:
                if args.local == "mock":
                    ttft_s, occupancy_s = _mock_service_times(
                        req,
                        args,
                        flop,
                        eff_flops,
                        tpot_profile,
                        max(1, local_state.inflight),
                    )
                    await asyncio.sleep(
                        min(occupancy_s / max(args.time_scale, 1e-9), args.mock_cap_s)
                    )
                    res = {
                        "success": True,
                        "ttft_ms": ttft_s * 1000.0,
                        "latency_ms": occupancy_s * 1000.0,
                    }
                else:
                    res = await send_request(session, url, args.model,
                                             req.metadata.get("prompt_text", ""),
                                             req.num_output_tokens, req.request_id)
            except Exception as exc:
                res = {"success": False, "ttft_ms": None, "error": str(exc)}
            finally:
                local_state.release(req, args.local)
            finished_now = replay_time()
            admitted_at = float(req.metadata.get("local_admitted_at", finished_now))
            queue_delay_ms = _queue_delay_ms(
                req,
                admitted_at,
                args.time_scale,
                scale_wait=False,
            )
            service_ttft_ms = res["ttft_ms"] if res["success"] else None
            ttft_ms = None if service_ttft_ms is None else queue_delay_ms + service_ttft_ms
            results[req.request_id] = {
                "id": req.request_id, "outsourced": False, "success": res["success"],
                "ttft_ms": ttft_ms,
                "queue_delay_ms": queue_delay_ms,
                "service_ttft_ms": service_ttft_ms,
                "cost_usd": 0.0,
                "remote_cached_tokens": 0,
                "uncached_input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "prefill": req.num_prompt_tokens, "decode": req.num_output_tokens,
            }

        while i < len(trace) or adapter.get_length() > 0 or local_state.inflight > 0 or pending:
            now_el = time.time() - clock0

            # 1) arrivals
            while i < len(trace) and (trace[i]["ts_sec"] - first_ts) / args.time_scale <= now_el:
                r = trace[i]
                i += 1
                req = OutsourcingRequestInfo(
                    request_id=str(r.get("request_id", i)),
                    arrival_time=replay_arrival_time(r),
                    num_prompt_tokens=int(r.get("num_prefill_tokens") or 0),
                    num_output_tokens=int(r.get("num_decode_tokens") or 0),
                    num_cached_tokens=int(r.get("num_cached_tokens") or 0),
                    prefill_slo_seconds=args.slo_s,
                    input_price_per_token=args.in_price / 1e6,
                    output_price_per_token=args.out_price / 1e6,
                )
                req.metadata["prompt_text"] = r.get("prompt_text", "")
                req.metadata["session_id"] = r.get("session_id")
                if "remote_cached_tokens" in r:
                    req.metadata["remote_cached_tokens"] = int(r.get("remote_cached_tokens") or 0)
                    req.metadata["remote_cached_tokens_explicit"] = True
                req_objs[req.request_id] = req
                if engine is not None:
                    adapter.add_request(req)            # nimbus decides per tick (below)
                elif decider.should_outsource(r, kv):  # baseline decides at arrival
                    record_outsourced(req)
                else:
                    adapter.add_request(req)

            used, kv_max = await read_kv()
            kv = (used / kv_max) if kv_max > 0 else 0.0
            kv_avail = max(0.0, args.admit_kv * kv_max - used)

            # 2) nimbus: Notion-faithful iterative loop (no prediction). While the waiting set
            #    exceeds admission KV headroom (K_avail x SLO horizon), solve knapsack & kick 1.
            if engine is not None:
                while True:
                    now = replay_time()
                    waiting = adapter.get_all_waiting()
                    predicted_decode_batch = _estimate_decode_batch_size(
                        local_state,
                        waiting,
                        used,
                        kv_max,
                        args.admit_kv,
                        args.max_inflight,
                    )
                    engine.tpot_seconds = _effective_tpot_seconds(
                        tpot_profile,
                        args.tpot_s,
                        predicted_decode_batch,
                    )
                    tpot_samples_s.append(engine.tpot_seconds)
                    for req in waiting:
                        if not req.metadata.get("remote_cached_tokens_explicit"):
                            req.metadata.pop("remote_cached_tokens", None)
                        req.metadata["remote_cached_tokens"] = cloud.estimate_remote_cached_tokens(
                            req,
                            now,
                        )
                    decision = engine.decide_by_kv_time_budget(
                        kv_avail,
                        args.slo_s,
                        max_iterations=1,
                        current_time=now,
                        wait_time_scale=1.0,
                        deadline_guard_seconds=(
                            args.cloud_ttft_ms
                            / 1000.0
                            * args.cloud_ttft_guard_multiplier
                        ),
                    )
                    if not decision.requests_to_outsource:
                        break
                    for rid in decision.requests_to_outsource:
                        req = req_objs.get(rid)
                        if req is not None and rid not in results:
                            record_outsourced(req)

            # 3) admission backpressure (shared): admit kept reqs locally while KV has headroom
            while (
                adapter.get_length() > 0
                and local_state.inflight < args.max_inflight
                and kv < args.admit_kv
            ):
                head = adapter.peek()
                if head is None:
                    break
                adapter.remove_requests({head.request_id})
                head.metadata["local_admitted_at"] = replay_time()
                local_state.reserve(head, args.local)
                pending.append(asyncio.create_task(admit_local(head)))
                used, kv_max = await read_kv()
                kv = (used / kv_max) if kv_max > 0 else 0.0

            pending = [t for t in pending if not t.done()]
            await asyncio.sleep(
                _loop_sleep_seconds(
                    args.tick_s,
                    args.time_scale,
                    scale_time=(args.local == "mock"),
                )
            )

        await asyncio.gather(*pending, return_exceptions=True)

    args.metrics_read_failures = kv_metrics.failures
    if tpot_profile is not None and args.policy == "nimbus":
        args.tpot_profile_label = tpot_profile.label()
        args.tpot_profile_points = len(tpot_profile.points)
    else:
        args.tpot_profile_label = "-"
        args.tpot_profile_points = 0
    if tpot_samples_s:
        args.tpot_ms_min = min(tpot_samples_s) * 1000.0
        args.tpot_ms_mean = (sum(tpot_samples_s) / len(tpot_samples_s)) * 1000.0
        args.tpot_ms_max = max(tpot_samples_s) * 1000.0
    else:
        fallback_ms = args.tpot_s * 1000.0
        args.tpot_ms_min = fallback_ms
        args.tpot_ms_mean = fallback_ms
        args.tpot_ms_max = fallback_ms
    return list(results.values())


def summarize(results: list[dict], args: argparse.Namespace) -> dict:
    n = len(results)
    slo_ms = args.slo_s * 1000.0
    outs = [r for r in results if r["outsourced"]]
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None and r["success"]]
    violations = sum(1 for r in results
                     if (not r["success"]) or r["ttft_ms"] is None or r["ttft_ms"] > slo_ms)
    remote_cached = sum(int(r.get("remote_cached_tokens") or 0) for r in outs)
    remote_input = sum(int(r.get("prefill") or 0) for r in outs)

    def pct(v, p):
        return sorted(v)[min(int(len(v) * p), len(v) - 1)] if v else 0.0

    fraction = effective_fraction(args)

    return {
        "policy": args.policy, "weight": (args.weight if args.policy == "nimbus" else "-"),
        "fraction": fraction,
        "slo_s": args.slo_s, "total": n, "outsourced": len(outs),
        "outsource_pct": round(len(outs) / max(n, 1), 4),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "remote_cached_input_tokens": remote_cached,
        "remote_cache_hit_pct": round(remote_cached / max(remote_input, 1), 4),
        "metrics_read_failures": int(getattr(args, "metrics_read_failures", 0)),
        "serving_engine": getattr(args, "serving_engine", "vllm"),
        "tpot_profile": getattr(args, "tpot_profile_label", "-"),
        "tpot_profile_points": int(getattr(args, "tpot_profile_points", 0)),
        "tpot_ms_min": round(float(getattr(args, "tpot_ms_min", args.tpot_s * 1000.0)), 3),
        "tpot_ms_mean": round(float(getattr(args, "tpot_ms_mean", args.tpot_s * 1000.0)), 3),
        "tpot_ms_max": round(float(getattr(args, "tpot_ms_max", args.tpot_s * 1000.0)), 3),
        "slo_violation_pct": round(violations / max(n, 1), 4),
        "ttft_p50_ms": round(pct(ttfts, 0.50), 1), "ttft_p99_ms": round(pct(ttfts, 0.99), 1),
    }


def effective_fraction(args: argparse.Namespace) -> str | float:
    if args.policy == "nimbus":
        return "self"
    if args.policy == "all_cloud":
        return 1.0
    if args.policy == "all_local":
        return 0.0
    return args.fraction


async def run_engine(args: argparse.Namespace) -> None:
    if args.synthetic_burst or not args.trace_file:
        trace = synthetic_burst_trace(
            args.synthetic_n,
            args.seed,
            prompt_mode=args.synthetic_prompt_mode,
            prompt_token_cap=args.synthetic_prompt_token_cap,
        )
        print(f"[synthetic-burst] {len(trace)} requests")
    else:
        from run_offload_strategies import load_trace
        trace = load_trace(args.trace_file, args.max_requests, args.duration_hours, args.start_hours)
        print(f"[trace] {len(trace)} requests from {args.trace_file}")
    if not trace:
        print("No requests!")
        return

    print(f"=== policy={args.policy} local={args.local} cloud={args.cloud} "
          f"engine={getattr(args, 'serving_engine', 'vllm')} "
          f"weight={args.weight} frac={effective_fraction(args)} SLO={args.slo_s}s "
          f"max_inflight={args.max_inflight} kv_tokens={args.local_kv_tokens} "
          f"remote_cache_ttl={args.remote_cache_ttl_s}s ===")
    results = await replay(args, trace)
    summary = summarize(results, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{args.policy}_{args.weight}"
        if args.policy == "nimbus"
        else f"{args.policy}_f{int(float(summary['fraction']) * 100):03d}"
    )
    with open(out_dir / f"requests_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "outsourced",
                "success",
                "ttft_ms",
                "queue_delay_ms",
                "service_ttft_ms",
                "cost_usd",
                "prefill",
                "decode",
                "remote_cached_tokens",
                "uncached_input_tokens",
                "cached_input_tokens",
                "output_tokens",
            ],
        )
        w.writeheader()
        w.writerows(results)
    summ_path = out_dir / "engine_summary.csv"
    write_header = not summ_path.exists()
    with open(summ_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)

    print("\n=== RESULT ===")
    for k, v in summary.items():
        print(f"  {k:>18}: {v}")
    print(f"\nAppended to {summ_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Nimbus online engine + baseline comparison harness")
    p.add_argument("--policy", default="nimbus", choices=["nimbus", *sorted(BASELINE_POLICIES)])
    p.add_argument("--fraction", type=float, default=0.2, help="outsource fraction (baselines only)")
    p.add_argument(
        "--sglang-url",
        "--serving-url",
        dest="sglang_url",
        default="http://localhost:8000",
        help="OpenAI-compatible serving base URL; --sglang-url is a legacy alias",
    )
    p.add_argument(
        "--serving-engine",
        choices=["sglang", "vllm"],
        default="vllm",
        help="metrics dialect for --local real; both use OpenAI-compatible chat API",
    )
    p.add_argument("--model", default="Qwen2.5-7B-Instruct")
    p.add_argument("--trace-file", default=None)
    p.add_argument("--synthetic-burst", action="store_true", help="in-memory bursty trace (no GPU/trace)")
    p.add_argument("--synthetic-n", type=int, default=400)
    p.add_argument(
        "--synthetic-prompt-mode",
        choices=["stub", "sized"],
        default="stub",
        help="stub uses a tiny prompt; sized creates word-count prompts matching capped synthetic token metadata",
    )
    p.add_argument(
        "--synthetic-prompt-token-cap",
        type=int,
        default=2048,
        help="cap synthetic prompt metadata/text length when --synthetic-prompt-mode sized",
    )
    p.add_argument("--output-dir", default="logs/engine")
    p.add_argument("--local", choices=["real", "mock"], default="mock")
    p.add_argument("--cloud", choices=["sim", "real"], default="sim",
                   help="cloud sink: sim=modeled TTFT + real $ from token counts; "
                        "real=stream from an OpenAI-compatible endpoint, measure real TTFT")
    p.add_argument("--weight", choices=["v0", "v1", "v2"], default="v2",
                   help="cache-displacement weight; v2 (token-seconds) is unit-matched to the KV budget")
    p.add_argument("--start-hours", type=float, default=0.0)
    p.add_argument("--duration-hours", type=float, default=1.0)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--max-requests", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slo-s", type=float, default=5.0, help="TTFT SLO (s); also the KV-budget horizon")
    p.add_argument("--max-inflight", type=int, default=32, help="local concurrency cap")
    p.add_argument("--admit-kv", type=float, default=0.90, help="admit while KV pressure below this")
    p.add_argument("--local-kv-tokens", type=float, default=200000.0,
                   help="local KV capacity in tokens; vLLM maps usage percent onto this, SGLang can report capacity directly")
    p.add_argument("--tick-s", type=float, default=0.25)
    p.add_argument("--mock-cap-s", type=float, default=2.0)
    p.add_argument(
        "--mock-service-model",
        choices=["v2", "flops"],
        default="v2",
        help="mock local timing model; v2 uses --prefill-tput and --tpot-s/profile",
    )
    p.add_argument("--hidden-dim", type=int, default=3584)   # Qwen2.5-7B
    p.add_argument("--num-layers", type=int, default=28)
    p.add_argument("--num-heads", type=int, default=28)
    p.add_argument("--device-tflops", type=float, default=200.0, help="only used by --weight v0 / mock local time")
    p.add_argument("--util", type=float, default=0.8)
    p.add_argument("--prefill-tput", type=float, default=50000.0, help="tokens/s (v2 weight)")
    p.add_argument("--tpot-s", type=float, default=0.03, help="seconds/token (v2 weight)")
    p.add_argument(
        "--tpot-profile",
        default=None,
        help="JSON profile mapping decode batch size to TPOT for v2 weights",
    )
    p.add_argument("--in-price", type=float, default=0.15, help="cloud input $/M tokens")
    p.add_argument(
        "--cached-in-price",
        type=float,
        default=None,
        help="cloud cached-input $/M tokens; defaults to --in-price when omitted",
    )
    p.add_argument("--out-price", type=float, default=1.20, help="cloud output $/M tokens")
    p.add_argument(
        "--remote-cache-ttl-s",
        type=float,
        default=0.0,
        help="explicit remote prompt-cache TTL in real replay seconds; 0 disables TTL cache accounting",
    )
    p.add_argument("--cloud-ttft-ms", type=float, default=900.0)
    p.add_argument(
        "--cloud-ttft-guard-multiplier",
        type=float,
        default=1.5,
        help="Nimbus deadline guard multiplier for simulated cloud TTFT",
    )
    # --cloud real sink (OpenAI-compatible streaming endpoint)
    p.add_argument(
        "--cloud-url",
        default="https://openrouter.ai/api/v1/chat/completions",
        help="real cloud OpenAI-compatible endpoint (used when --cloud real)",
    )
    p.add_argument(
        "--cloud-model",
        default="qwen3-32b",
        help="cloud model id; use the SAME model as local for a clean comparison",
    )
    p.add_argument(
        "--cloud-api-key-env",
        default="OPENROUTER_API_KEY1",
        help="env var holding the cloud API key (never hardcode keys)",
    )
    p.add_argument("--cloud-timeout-s", type=float, default=600.0)
    args = p.parse_args()
    asyncio.run(run_engine(args))


if __name__ == "__main__":
    main()
