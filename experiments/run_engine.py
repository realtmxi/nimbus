#!/usr/bin/env python3
"""Unified online harness: run our ACTUAL engine OR any baseline through the
SAME loop / cost meter / SLO accounting, so the comparison is apples-to-apples.

Unlike run_offload_strategies.py (whose cache_disp/flop are OFFLINE oracles and
which records no cost), this drives the REAL online engine
(nimbus.decision.OutsourcingEngine) and measures cost + SLO for every policy
identically, enabling an iso-SLO cost-vs-violation comparison.

The decision differs by policy; everything else (admission backpressure, local
serving, cloud cost, SLO accounting) is shared:
  --policy nimbus   : per-TICK engine triage of an external queue (knapsack).
  --policy <other>  : per-ARRIVAL baseline decision (reuses the strategy classes
                      from run_offload_strategies: all_local/all_cloud/random/
                      fifo/pressure_gated/session_aware/size_long/flop_oracle/
                      cachedisp_oracle). Baselines take --fraction; sweep it to
                      get a curve. nimbus self-selects its fraction.

Admission backpressure (the bit that makes it real): kept requests are admitted
to the local engine ONLY while there is capacity (--max-inflight / --admit-kv).
Under burst the external queue backs up, which is the only way the engine ever
sees a backlog to triage. Without it you get a false "no advantage".

Fidelities:
  --local mock : modeled local TTFT, no SGLang (M0 wiring/logic smoke; NOT a result)
  --local real : admit to a live SGLang and measure real TTFT (M1)
  --cloud sim  : outsourced -> real $ (token counts) + modeled TTFT (real cloud = later)

CAVEATS: the engine's TTFT predictor is a FLOP model; CALIBRATE --device-tflops/
--util against real SGLang (M2) before trusting the advantage number. Cloud
latency/cost is simulated (first-order until a real backend, M5).
"""

import argparse
import asyncio
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `import nimbus`
sys.path.insert(0, str(Path(__file__).resolve().parent))      # experiments/ -> sibling module

# NOTE: aiohttp and run_offload_strategies (which imports aiohttp) are imported
# LAZILY inside the functions that need them, so this module imports cleanly for
# unit tests even without aiohttp installed.

from nimbus.adapters import SGLangWaitingQueueAdapter  # noqa: E402
from nimbus.cost_calculator import APICostCalculator  # noqa: E402
from nimbus.decision import OutsourcingEngine  # noqa: E402
from nimbus.flop_calculator import SimpleFLOPCalculator  # noqa: E402
from nimbus.request import OutsourcingRequestInfo  # noqa: E402

BASELINE_POLICIES = {
    "all_local", "all_cloud", "random", "fifo", "pressure_gated",
    "session_aware", "size_long", "flop_oracle", "cachedisp_oracle",
}


class NimbusEngine(OutsourcingEngine):
    """OutsourcingEngine with a pluggable knapsack weight (v0 FLOP / v1 / v2)."""

    def __init__(self, *a, weight: str = "v1", prefill_throughput: float = 50000.0,
                 tpot_s: float = 0.03, **k):
        super().__init__(*a, **k)
        self._weight_kind = weight
        self._prefill_tput = prefill_throughput
        self._tpot_s = tpot_s

    def _build_knapsack_item(self, req: OutsourcingRequestInfo) -> dict:
        if self._weight_kind == "v0":
            return super()._build_knapsack_item(req)  # prefill_flops + 0.6*decode_flops
        p, d = req.num_prompt_tokens, req.num_output_tokens
        if self._weight_kind == "v1":
            weight = float(p * d)  # cache displacement: footprint x residence
        else:  # v2: token-seconds = footprint(tokens) x residence(prefill_time + decode_time, s)
            prefill_t = max(0, p - req.num_cached_tokens) / self._prefill_tput
            decode_t = d * self._tpot_s
            weight = float(p * (prefill_t + decode_t))
        api_cost = self._cost_calculator.calculate_cost(
            req.remaining_prompt_tokens, req.remaining_output_tokens
        )
        return {"id": req.request_id, "weight": max(1, int(weight)), "value": max(1, int(api_cost * 1e6))}


class SimCloud:
    """Simulated cloud sink: real $ from token counts, modeled TTFT."""

    def __init__(self, in_price: float, out_price: float, ttft_mean_ms: float,
                 jitter: float = 0.3, seed: int = 0):
        self._cost = APICostCalculator(in_price, out_price)
        self._ttft_mean = ttft_mean_ms
        self._jitter = jitter
        self._rng = random.Random(seed)

    def serve(self, req: OutsourcingRequestInfo) -> dict:
        ttft = self._ttft_mean * (1.0 + self._rng.uniform(-self._jitter, self._jitter))
        cost = self._cost.calculate_cost(req.num_prompt_tokens, req.num_output_tokens)
        return {"ttft_ms": ttft, "cost_usd": cost, "success": True}


def synthetic_burst_trace(n: int, seed: int) -> list[dict]:
    """Bursty synthetic trace (dense cluster in the middle) for no-GPU/no-trace smoke tests."""
    rng = random.Random(seed)
    rows, t = [], 0.0
    for i in range(n):
        in_burst = (0.3 * n) < i < (0.7 * n)
        t += rng.expovariate(1 / (0.04 if in_burst else 0.4))  # dense arrivals mid-trace
        rows.append({
            "request_id": i, "ts_sec": t, "arrived_at": t,
            "num_prefill_tokens": rng.choice([512, 2048, 8000, 32000, 64000]),
            "num_decode_tokens": rng.choice([64, 256, 512, 1024]),
            "num_cached_tokens": 0, "session_id": i % max(1, n // 5),
            "prompt_text": "ping",
        })
    return rows


def make_decider(args: argparse.Namespace, trace: list[dict]):
    """Return a per-arrival baseline strategy, or None for the nimbus (per-tick) engine."""
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


async def replay(args: argparse.Namespace, trace: list[dict]) -> list[dict]:
    import aiohttp
    from run_offload_strategies import send_request, probe_kv_pressure
    flop = SimpleFLOPCalculator(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_attention_heads=args.num_heads, device_tflops=args.device_tflops,
    )
    adapter = SGLangWaitingQueueAdapter(metrics_url=f"{args.sglang_url.rstrip('/')}/metrics")
    cloud = SimCloud(args.in_price, args.out_price, args.cloud_ttft_ms, seed=args.seed)
    decider = make_decider(args, trace)
    engine = None
    if args.policy == "nimbus":
        engine = NimbusEngine(
            waiting_queue=adapter, flop_calculator=flop,
            prefill_slo_base_seconds=args.slo_s, utilization_target=args.util,
            input_price_per_million=args.in_price, output_price_per_million=args.out_price,
            knapsack_strategy="dp_scaled", enable_iterative_outsourcing=True,
            weight=args.weight, prefill_throughput=args.prefill_tput, tpot_s=args.tpot_s,
        )

    results: dict[str, dict] = {}
    req_objs: dict[str, OutsourcingRequestInfo] = {}
    inflight = {"n": 0}
    pending: list[asyncio.Task] = []
    eff_flops = flop.get_effective_flops_per_second(args.util)
    url = f"{args.sglang_url.rstrip('/')}/v1/chat/completions"
    first_ts, clock0, i, kv = trace[0]["ts_sec"], time.time(), 0, 0.0

    def record_outsourced(req: OutsourcingRequestInfo) -> None:
        out = cloud.serve(req)
        results[req.request_id] = {
            "id": req.request_id, "outsourced": True, "success": out["success"],
            "ttft_ms": out["ttft_ms"], "cost_usd": out["cost_usd"],
            "prefill": req.num_prompt_tokens, "decode": req.num_output_tokens,
        }

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=None)) as session:

        async def admit_local(req: OutsourcingRequestInfo) -> None:
            inflight["n"] += 1
            try:
                if args.local == "mock":
                    svc_s = (flop.compute_prefill_flops(req, req.remaining_prompt_tokens) / eff_flops) if eff_flops > 0 else 0.0
                    await asyncio.sleep(min(svc_s / max(args.time_scale, 1e-9), args.mock_cap_s))
                    res = {"success": True, "ttft_ms": svc_s * 1000.0}
                else:
                    res = await send_request(session, url, args.model,
                                             req.metadata.get("prompt_text", ""),
                                             req.num_output_tokens, req.request_id)
            finally:
                inflight["n"] -= 1
            results[req.request_id] = {
                "id": req.request_id, "outsourced": False, "success": res["success"],
                "ttft_ms": res["ttft_ms"], "cost_usd": 0.0,
                "prefill": req.num_prompt_tokens, "decode": req.num_output_tokens,
            }

        while i < len(trace) or adapter.get_length() > 0 or inflight["n"] > 0 or pending:
            now_el = time.time() - clock0

            # 1) arrivals
            while i < len(trace) and (trace[i]["ts_sec"] - first_ts) / args.time_scale <= now_el:
                r = trace[i]
                i += 1
                req = OutsourcingRequestInfo(
                    request_id=str(r.get("request_id", i)), arrival_time=time.time(),
                    num_prompt_tokens=int(r.get("num_prefill_tokens") or 0),
                    num_output_tokens=int(r.get("num_decode_tokens") or 0),
                    num_cached_tokens=int(r.get("num_cached_tokens") or 0),
                    prefill_slo_seconds=args.slo_s,
                    input_price_per_token=args.in_price / 1e6,
                    output_price_per_token=args.out_price / 1e6,
                )
                req.metadata["prompt_text"] = r.get("prompt_text", "")
                req_objs[req.request_id] = req
                if engine is not None:
                    adapter.add_request(req)            # engine decides later (per tick)
                elif decider.should_outsource(r, kv):  # baseline decides now (per arrival)
                    record_outsourced(req)
                else:
                    adapter.add_request(req)            # kept -> admission buffer

            # 2) nimbus: per-tick engine triage of the external queue
            if engine is not None:
                decision = engine.should_outsource(time.time())
                for rid in decision.requests_to_outsource:
                    req = req_objs.get(rid)
                    if req is not None and rid not in results:
                        record_outsourced(req)

            # 3) admission backpressure (shared): admit kept reqs locally while capacity remains
            kv = (await probe_kv_pressure(args.sglang_url)) if args.local == "real" else 0.0
            while adapter.get_length() > 0 and inflight["n"] < args.max_inflight and kv < args.admit_kv:
                head = adapter.peek()
                if head is None:
                    break
                adapter.remove_requests({head.request_id})
                pending.append(asyncio.create_task(admit_local(head)))
                if args.local == "real":
                    kv = await probe_kv_pressure(args.sglang_url)

            pending = [t for t in pending if not t.done()]
            await asyncio.sleep(args.tick_s)

        await asyncio.gather(*pending, return_exceptions=True)

    return list(results.values())


def summarize(results: list[dict], args: argparse.Namespace) -> dict:
    n = len(results)
    slo_ms = args.slo_s * 1000.0
    outs = [r for r in results if r["outsourced"]]
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None and r["success"]]
    violations = sum(1 for r in results
                     if (not r["success"]) or r["ttft_ms"] is None or r["ttft_ms"] > slo_ms)

    def pct(v, p):
        return sorted(v)[min(int(len(v) * p), len(v) - 1)] if v else 0.0

    return {
        "policy": args.policy, "weight": (args.weight if args.policy == "nimbus" else "-"),
        "fraction": ("self" if args.policy == "nimbus" else args.fraction),
        "slo_s": args.slo_s, "total": n, "outsourced": len(outs),
        "outsource_pct": round(len(outs) / max(n, 1), 4),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "slo_violation_pct": round(violations / max(n, 1), 4),
        "ttft_p50_ms": round(pct(ttfts, 0.50), 1), "ttft_p99_ms": round(pct(ttfts, 0.99), 1),
    }


async def run_engine(args: argparse.Namespace) -> None:
    from run_offload_strategies import load_trace
    if args.synthetic_burst or not args.trace_file:
        trace = synthetic_burst_trace(args.synthetic_n, args.seed)
        print(f"[synthetic-burst] {len(trace)} requests")
    else:
        trace = load_trace(args.trace_file, args.max_requests, args.duration_hours, args.start_hours)
        print(f"[trace] {len(trace)} requests from {args.trace_file}")
    if not trace:
        print("No requests!")
        return

    print(f"=== policy={args.policy} local={args.local} cloud={args.cloud} "
          f"weight={args.weight} frac={args.fraction} SLO={args.slo_s}s "
          f"max_inflight={args.max_inflight} ===")
    results = await replay(args, trace)
    summary = summarize(results, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.policy}" + ("" if args.policy == "nimbus" else f"_f{int(args.fraction * 100):03d}")
    with open(out_dir / f"requests_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "outsourced", "success", "ttft_ms", "cost_usd", "prefill", "decode"])
        w.writeheader()
        w.writerows(results)
    summ_path = out_dir / "engine_summary.csv"
    write_header = not summ_path.exists()
    with open(summ_path, "a", newline="") as f:  # append so a sweep accumulates rows
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
    p.add_argument("--sglang-url", default="http://localhost:30000")
    p.add_argument("--model", default="Qwen2.5-7B-Instruct")
    p.add_argument("--trace-file", default=None)
    p.add_argument("--synthetic-burst", action="store_true", help="use in-memory bursty trace (no GPU/trace needed)")
    p.add_argument("--synthetic-n", type=int, default=400)
    p.add_argument("--output-dir", default="logs/engine")
    p.add_argument("--local", choices=["real", "mock"], default="mock")
    p.add_argument("--cloud", choices=["sim"], default="sim")
    p.add_argument("--weight", choices=["v0", "v1", "v2"], default="v1")
    p.add_argument("--start-hours", type=float, default=0.0)
    p.add_argument("--duration-hours", type=float, default=1.0)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--max-requests", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slo-s", type=float, default=5.0, help="TTFT SLO (seconds)")
    p.add_argument("--max-inflight", type=int, default=32, help="local concurrency cap (backpressure)")
    p.add_argument("--admit-kv", type=float, default=0.90, help="admit while KV pressure below this")
    p.add_argument("--tick-s", type=float, default=0.25)
    p.add_argument("--mock-cap-s", type=float, default=2.0)
    p.add_argument("--hidden-dim", type=int, default=3584)   # Qwen2.5-7B
    p.add_argument("--num-layers", type=int, default=28)
    p.add_argument("--num-heads", type=int, default=28)
    p.add_argument("--device-tflops", type=float, default=200.0, help="CALIBRATE to real GPU")
    p.add_argument("--util", type=float, default=0.8)
    p.add_argument("--prefill-tput", type=float, default=50000.0, help="tokens/s (v2 weight)")
    p.add_argument("--tpot-s", type=float, default=0.03, help="seconds/token (v2 weight)")
    p.add_argument("--in-price", type=float, default=0.15, help="cloud input $/M tokens")
    p.add_argument("--out-price", type=float, default=1.20, help="cloud output $/M tokens")
    p.add_argument("--cloud-ttft-ms", type=float, default=900.0)
    args = p.parse_args()
    asyncio.run(run_engine(args))


if __name__ == "__main__":
    main()
