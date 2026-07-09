"""The hybrid router: external FIFO queue + work-conserving dispatcher.

Architecture:

    arrival --> [policy: cloud?] --yes--> cloud sink (NullCloud fake or real)
                     |no
                     v
              [external FIFO queue] --dispatcher--> local vLLM (internal queue ~ empty)

The dispatcher is WORK-CONSERVING: whenever the local engine has a free slot,
the queue head is dispatched immediately; requests are never held back
gratuitously. Headroom = concurrency slots: inflight < --max-inflight (align it
with the server's --max-num-seqs so vLLM's internal queue stays ~empty).
KV-aware admission is deliberately NOT here — it returns with the nimbus
policy in Step 3, where KV is actually part of the algorithm.
Consequences:
  - no pressure  -> queue is always empty -> behavior degrades to open-loop,
    i.e. equivalent to the verified baseline load generator vllm/run.py
    (queue-neutrality: validated on gpu1, 110 vs 111 ms TTFT p50);
  - burst        -> overflow waits in OUR queue (not inside vLLM), which is
    exactly the set a shedding policy (nimbus, later) will operate on.

TTFT accounting: ttft_ms = queue_delay_ms + service_ttft_ms, i.e. measured from
the request's trace arrival time, comparable with an open-loop run (vllm/run.py).

Policies here are the same arrival-time baselines (all_local / all_cloud /
random). The queue only paces local dispatch for them; a queue-level shedding
policy (knapsack) plugs into this file later as an on-tick hook.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from router.common import (
    DEFAULT_TIMEOUT_S,
    Endpoint,
    Policy,
    SCENARIOS,
    _null_session,
    build_cloud_sink,
    load_trace,
    one_request,
    resolve_output_path,
    summarize,
)

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None


def metrics_url_of(chat_url: str) -> str:
    """http://host:port/v1/chat/completions -> http://host:port/metrics"""
    base = chat_url.split("/v1/", 1)[0].rstrip("/")
    return base + "/metrics"


class KVMonitor:
    """Real KV availability from the engine's /metrics (no client-side model).

    Reads `vllm:kv_cache_usage_perc` (verified gauge name, vLLM v0.19) and
    converts to available tokens against the engine's reported capacity
    (--kv-capacity-tokens, from the startup log "GPU KV cache size: N tokens").
    Responses are cached for ttl_s; on read failure the last value is reused
    (and counted) so a transient scrape error cannot stall the tick loop.
    """

    GAUGE = "vllm:kv_cache_usage_perc"

    def __init__(self, session, metrics_url: str, capacity_tokens: float,
                 ttl_s: float = 0.25):
        self.session = session
        self.metrics_url = metrics_url
        self.capacity = float(capacity_tokens)
        self.ttl_s = ttl_s
        self._avail = capacity_tokens   # optimistic until first read
        self._at = float("-inf")
        self.read_failures = 0

    async def available_tokens(self) -> float:
        now = time.monotonic()
        if now - self._at < self.ttl_s:
            return self._avail
        try:
            async with self.session.get(self.metrics_url) as resp:
                text = await resp.text()
            usage = None
            for line in text.splitlines():
                if line.startswith(self.GAUGE):
                    usage = float(line.split()[-1])
            if usage is None:
                raise ValueError(f"{self.GAUGE} not found in /metrics")
            self._avail = self.capacity * (1.0 - usage)
            self._at = now
        except Exception as exc:
            self.read_failures += 1
            if self.read_failures == 1:
                print(f"[kv-monitor] WARNING: /metrics read failed ({exc}); "
                      f"reusing last value")
        return self._avail


class LocalAdmission:
    """Concurrency gate for the local engine: admit while inflight < max_inflight."""

    def __init__(self, max_inflight: int):
        self.max_inflight = int(max_inflight)
        self.inflight = 0
        self.peak_inflight = 0   # telemetry

    def fits(self, req: dict[str, Any]) -> bool:
        return self.inflight < self.max_inflight

    def reserve(self, req: dict[str, Any]) -> None:
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)

    def release(self, req: dict[str, Any]) -> None:
        self.inflight -= 1


async def replay_queued(
    args: argparse.Namespace,
    trace: list[dict[str, Any]],
    policy: Policy,
    local: Endpoint | None,
    cloud: Endpoint | None,
    sink: 'NullCloud | None' = None,
    send_local: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    send_cloud: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    kv_monitor: "KVMonitor | None" = None,
) -> tuple[list[dict[str, Any]], LocalAdmission, "KVMonitor | None"]:
    """Queued replay. `send_local`/`send_cloud` are injectable for tests
    (default to the verified streaming primitive one_request)."""
    out = resolve_output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)

    admission = LocalAdmission(args.max_inflight)
    queue: list[tuple[dict[str, Any], float]] = []   # (req, arrival_due) FIFO
    results: list[dict[str, Any]] = []
    pending: set[asyncio.Task] = set()

    needs_http = ((local is not None and send_local is None)
                  or (cloud is not None and sink is None and send_cloud is None))
    if needs_http and aiohttp is None:
        raise RuntimeError("aiohttp is required to send requests (run on a host that has it)")
    session_cm = (aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
                  if needs_http else _null_session())

    cloud_gate = (asyncio.Semaphore(args.cloud_max_concurrency)
                  if getattr(args, "cloud_max_concurrency", 0) > 0 else None)

    async with session_cm as session:
        if (kv_monitor is None and hasattr(policy, "on_tick")
                and session is not None and args.local_url):
            kv_monitor = KVMonitor(session, metrics_url_of(args.local_url),
                                   args.kv_capacity_tokens)
            # fail fast: a dead KV signal would silently turn nimbus into
            # all_local (never-succeeded reads return optimistic full capacity)
            await kv_monitor.available_tokens()
            if kv_monitor.read_failures:
                raise RuntimeError(
                    f"KV metrics unreadable at startup ({kv_monitor.metrics_url}, "
                    f"gauge {KVMonitor.GAUGE}) — refusing to run nimbus blind")
        if send_local is None:
            async def send_local(endpoint, req, due):  # noqa: F811 - default sender
                return await one_request(session, endpoint, req, due,
                                         max_tokens_override=args.max_tokens,
                                         timeout_s=args.timeout_s)
        if send_cloud is None:
            async def send_cloud(req, due):  # noqa: F811 - default sender
                return await one_request(session, cloud, req, due,
                                         max_tokens_override=args.max_tokens,
                                         timeout_s=args.timeout_s)

        run_start = time.perf_counter()
        f = out.open("w", encoding="utf-8")

        def record(res: dict[str, Any]) -> None:
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

        async def serve_local(req: dict[str, Any], arrival_due: float) -> None:
            dispatch_t = time.perf_counter()
            try:
                res = await send_local(local, req, dispatch_t)
            except Exception as exc:  # defensive: a sender bug must not leak reservation
                res = {"request_id": req["request_id"], "arrived_at": req["arrived_at"],
                       "relative_arrival_s": req["relative_arrival_s"],
                       "endpoint": "local", "model": getattr(local, "model", None),
                       "success": False, "error": str(exc),
                       "error_type": type(exc).__name__, "http_status": None,
                       "ttft_ms": None, "e2e_ms": None, "tpot_ms": None,
                       "chunks": 0, "prompt_tokens": None, "completion_tokens": None,
                       "output_chars": 0, "cost_usd": 0.0, "scheduled_lag_ms": 0.0}
            finally:
                admission.release(req)
            queue_delay_ms = max(0.0, (dispatch_t - arrival_due) * 1000)
            service_ttft = res.get("ttft_ms")
            res["queue_delay_ms"] = queue_delay_ms
            res["service_ttft_ms"] = service_ttft
            res["ttft_ms"] = None if service_ttft is None else queue_delay_ms + service_ttft
            if res.get("e2e_ms") is not None:
                res["e2e_ms"] = queue_delay_ms + res["e2e_ms"]
            record(res)
            maybe_dispatch()
            await maybe_kick()

        def spawn_local(req: dict[str, Any], arrival_due: float) -> None:
            admission.reserve(req)
            task = asyncio.create_task(serve_local(req, arrival_due))
            pending.add(task)
            task.add_done_callback(pending.discard)

        def maybe_dispatch() -> None:
            # work-conserving: admit the head while a slot is free
            while queue and admission.fits(queue[0][0]):
                req, due = queue.pop(0)
                spawn_local(req, due)

        def _finalize_cloud(res: dict[str, Any], queue_delay_ms: float) -> None:
            # same from-arrival accounting as the local path: a kicked request
            # carries the time it waited in our queue before being shed
            service = res.get("ttft_ms")
            res["queue_delay_ms"] = queue_delay_ms
            res["service_ttft_ms"] = service
            if service is not None:
                res["ttft_ms"] = queue_delay_ms + service
            if res.get("e2e_ms") is not None:
                res["e2e_ms"] = queue_delay_ms + res["e2e_ms"]

        def route_cloud(req: dict[str, Any], due: float,
                        queue_delay_ms: float = 0.0) -> None:
            if sink is not None:
                res = sink.serve(req, due, max_tokens_override=args.max_tokens)
                _finalize_cloud(res, queue_delay_ms)
                record(res)
            else:
                async def cloud_task():
                    if cloud_gate is not None:
                        async with cloud_gate:
                            res = await send_cloud(req, due)
                    else:
                        res = await send_cloud(req, due)
                    _finalize_cloud(res, queue_delay_ms)
                    record(res)
                task = asyncio.create_task(cloud_task())
                pending.add(task)
                task.add_done_callback(pending.discard)

        # ---- queue-level policy hook (nimbus): kick overflow to the cloud ----
        has_tick = hasattr(policy, "on_tick")
        tick_busy = False

        async def maybe_kick() -> None:
            nonlocal tick_busy
            if not has_tick or tick_busy or not queue:
                return
            tick_busy = True
            try:
                kv_avail = (await kv_monitor.available_tokens()
                            if kv_monitor is not None else float("inf"))
                snapshot = [r for r, _ in queue]
                # heavy DP runs off-thread so a deep-queue solve cannot stall
                # arrival pacing / completions / the KV scrape
                victims = await asyncio.get_running_loop().run_in_executor(
                    None, policy.on_tick, snapshot, kv_avail)
                for v in victims:
                    # tolerant lookup: the queue kept moving while we decided —
                    # a victim may have been dispatched already; skip it
                    idx = next((i for i, (r, _) in enumerate(queue)
                                if r["request_id"] == v["request_id"]), None)
                    if idx is None:
                        continue
                    req, due = queue.pop(idx)
                    waited_ms = max(0.0, (time.perf_counter() - due) * 1000)
                    policy.n_outsourced += 1
                    route_cloud(req, due, queue_delay_ms=waited_ms)
            finally:
                tick_busy = False

        try:
            for req in trace:
                due = run_start + req["relative_arrival_s"]
                wait = due - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)

                if policy.outsource(req):
                    route_cloud(req, due)
                else:
                    queue.append((req, due))
                    maybe_dispatch()
                    await maybe_kick()

            # drain: everything left in queue/in flight completes through the
            # same dispatcher (completions re-trigger maybe_dispatch)
            while queue or pending:
                if pending:
                    done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                else:  # queue non-empty, nothing in flight -> dispatch now
                    maybe_dispatch()
        finally:
            f.close()

    print(f"raw: {out}")
    return results, admission, kv_monitor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--policy", choices=["all_local", "all_cloud", "random", "nimbus"], required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--local-url", default=None)
    parser.add_argument("--local-model", default=None)
    parser.add_argument("--max-inflight", type=int, default=128,
                        help="local concurrency cap; match the server --max-num-seqs")

    # nimbus policy (see router/nimbus.py)
    parser.add_argument("--kv-capacity-tokens", type=float, default=None,
                        help="engine KV capacity in tokens, from the vLLM startup "
                             "log 'GPU KV cache size: N tokens' (required for --policy nimbus)")
    parser.add_argument("--prefill-tput", type=float, default=20000.0,
                        help="prefill throughput tokens/s for the displacement estimate "
                             "(kick priority only — relative ordering is what matters)")
    parser.add_argument("--tpot-ms", type=float, default=9.5,
                        help="per-output-token time (ms) for the displacement estimate; "
                             "9.5ms measured on the 35B-A3B+MTP setup")

    parser.add_argument("--cloud", choices=["null", "real"], default="null")
    parser.add_argument("--cloud-url", default=None)
    parser.add_argument("--cloud-model", default=None)
    parser.add_argument("--cloud-api-key-env", default=None)
    parser.add_argument("--cloud-max-concurrency", type=int, default=32,
                        help="cap concurrent REAL cloud requests (0 = unlimited); "
                             "guards against self-inflicted 429s under burst")
    parser.add_argument("--in-price", type=float, default=0.15)
    parser.add_argument("--out-price", type=float, default=1.20)

    parser.add_argument("--slo-s", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.max_inflight <= 0:
        parser.error("--max-inflight must be >= 1 (0 would deadlock the dispatcher)")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be >= 1 (it replaces the trace decode count "
                     "in the payload; 0 breaks fake/real parity)")
    needs_local = args.policy in ("all_local", "random", "nimbus")
    needs_cloud = args.policy in ("all_cloud", "random", "nimbus")
    if args.policy == "nimbus" and (args.kv_capacity_tokens is None
                                    or args.kv_capacity_tokens <= 0):
        parser.error("--policy nimbus requires --kv-capacity-tokens > 0 "
                     "(read it from the vLLM startup log)")
    if needs_local and not (args.local_url and args.local_model):
        parser.error(f"--policy {args.policy} requires --local-url and --local-model")
    if needs_cloud and args.cloud == "real" and not args.cloud_url:
        parser.error(f"--policy {args.policy} --cloud real requires --cloud-url")
    if needs_cloud and args.cloud == "real" and not (args.cloud_model or args.local_model):
        parser.error("--cloud real requires --cloud-model (or --local-model to inherit); "
                     "refusing to send a placeholder model name to a real endpoint")

    return args


def queue_stats(results: list[dict[str, Any]], admission: LocalAdmission,
                policy=None, kv_monitor=None) -> dict[str, Any]:
    delays = sorted(r["queue_delay_ms"] for r in results
                    if r.get("endpoint") == "local" and "queue_delay_ms" in r)

    def pct(q: float) -> float | None:
        if not delays:
            return None
        return delays[min(len(delays) - 1, max(0, int(round(q / 100 * (len(delays) - 1)))))]

    stats = {
        "queue_delay_p50_ms": pct(50),
        "queue_delay_p99_ms": pct(99),
        "queue_delay_max_ms": delays[-1] if delays else None,
        "peak_inflight": admission.peak_inflight,
    }
    if policy is not None and hasattr(policy, "on_tick"):
        stats["nimbus_ticks"] = policy.ticks
        stats["nimbus_kick_rounds"] = policy.kick_rounds
        stats["nimbus_kicked"] = policy.n_outsourced
    if kv_monitor is not None:
        stats["kv_read_failures"] = kv_monitor.read_failures
    return stats


async def main() -> None:
    args = parse_args()
    trace = load_trace(args.data, args.scenario)
    print(f"loaded {len(trace)} requests  scenario={args.scenario} policy={args.policy} "
          f"max_inflight={args.max_inflight}")

    if args.policy == "nimbus":
        from router.nimbus import NimbusPolicy
        policy = NimbusPolicy(args.in_price, args.out_price,
                              args.prefill_tput, args.tpot_ms / 1000.0, seed=args.seed)
    else:
        policy = Policy(args.policy, args.fraction, args.seed)
    needs_cloud = args.policy in ("all_cloud", "random", "nimbus")
    local, cloud, sink = build_cloud_sink(args, needs_cloud)

    results, admission, kv_mon = await replay_queued(args, trace, policy, local, cloud, sink)
    summary = summarize(results, policy, args.slo_s)
    summary["queue"] = queue_stats(results, admission, policy, kv_mon)

    out = resolve_output_path(args)
    summary_path = out.with_name(out.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
