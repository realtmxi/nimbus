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
The dispatcher itself stays KV-blind; KV awareness lives in the nimbus policy,
whose kick check runs BEFORE dispatch on every arrival/completion — so with a
real KV signal, admission is effectively KV-aware under --policy nimbus.
Consequences:
  - no pressure  -> queue is always empty -> behavior degrades to open-loop,
    i.e. equivalent to the verified baseline load generator vllm/run.py
    (queue-neutrality: validated on gpu1, 110 vs 111 ms TTFT p50);
  - burst        -> overflow waits in OUR queue (not inside vLLM), which is
    exactly the set a shedding policy (nimbus, later) will operate on.

TTFT accounting: ttft_ms = queue_delay_ms + service_ttft_ms, i.e. measured from
the request's trace arrival time, comparable with an open-loop run (vllm/run.py).

Policies: the arrival-time baselines (all_local / all_cloud / random, for whom
the queue only paces local dispatch) and the queue-level nimbus shedding policy
(router/nimbus.py), hooked in via on_tick.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable

from router.common import (
    DEFAULT_KV_HYSTERESIS_FRACTION,
    DEFAULT_TIMEOUT_S,
    Endpoint,
    Policy,
    SCENARIOS,
    _null_session,
    build_cloud_sink,
    effective_decode,
    load_trace,
    local_prompt_tokens,
    one_request,
    resolve_output_path,
    summarize,
)
from router.nimbus import (
    DecisionContext,
    InflightPrefill,
    NIMBUS_SELECTORS,
    NIMBUS_TRIGGERS,
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

    def invalidate(self) -> None:
        """Drop the TTL cache so the next read scrapes fresh state."""
        self._at = float("-inf")

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
        self.peak_waiting = 0

    def fits(self, req: dict[str, Any]) -> bool:
        return self.inflight < self.max_inflight

    def reserve(self, req: dict[str, Any]) -> None:
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)

    def release(self, req: dict[str, Any]) -> None:
        self.inflight -= 1


@dataclass
class _InflightKVState:
    prompt_tokens: int
    expected_decode: int
    generated_tokens: int = 0

    @property
    def remaining_decode(self) -> int:
        return self.expected_decode - self.generated_tokens

    @property
    def current_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens

class _InflightKVTracker:
    """Client-known KV state, updated from exact cumulative engine usage."""

    def __init__(self, max_tokens_override: int | None):
        self.max_tokens_override = max_tokens_override
        self._states: dict[Any, _InflightKVState] = {}
        self._version = 0

    def add(self, req: dict[str, Any]) -> None:
        self._states[req["request_id"]] = _InflightKVState(
            prompt_tokens=local_prompt_tokens(req),
            expected_decode=effective_decode(req, self.max_tokens_override),
        )
        self._version += 1

    def update_generated(self, request_id: Any, generated_tokens: int) -> None:
        state = self._states.get(request_id)
        if state is None:
            return
        generated = min(state.expected_decode, max(0, int(generated_tokens)))
        # Cumulative stream usage should be monotone. Keep that invariant even
        # if an endpoint repeats or reorders a progress update.
        updated = max(state.generated_tokens, generated)
        if updated != state.generated_tokens:
            state.generated_tokens = updated
            self._version += 1

    def remove(self, request_id: Any) -> None:
        if self._states.pop(request_id, None) is not None:
            self._version += 1

    @property
    def version(self) -> int:
        return self._version

    @property
    def remaining_decode_tokens(self) -> int:
        return sum(state.remaining_decode for state in self._states.values())

    @property
    def predicted_current_tokens(self) -> int:
        return sum(state.current_tokens for state in self._states.values())

    @property
    def unfinished_prefill_tokens(self) -> int:
        """Conservative shared-lane work not known to have reached first token."""
        return sum(
            state.prompt_tokens
            for state in self._states.values()
            if state.generated_tokens == 0
        )

    @property
    def inflight_prefills(self) -> tuple[InflightPrefill, ...]:
        return tuple(
            InflightPrefill(state.prompt_tokens, state.expected_decode)
            for state in self._states.values()
            if state.generated_tokens == 0
        )

    def decode_remaining_service_s(self, tpot_s: float) -> tuple[float, ...]:
        return tuple(
            state.remaining_decode * tpot_s
            for state in self._states.values()
            if state.generated_tokens > 0
        )


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
    has_tick = hasattr(policy, "on_tick")
    needs_kv = has_tick and getattr(policy, "trigger", "kv_gap") == "kv_gap"
    # Baselines keep their original payload and pay no Nimbus bookkeeping cost.
    # Injected test senders retain each full commitment until completion.
    inflight_kv = _InflightKVTracker(args.max_tokens) if has_tick else None

    needs_http = ((local is not None and send_local is None)
                  or (cloud is not None and sink is None and send_cloud is None))
    if needs_http and aiohttp is None:
        raise RuntimeError("aiohttp is required to send requests (run on a host that has it)")
    session_cm = (aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
                  if needs_http else _null_session())

    cloud_gate = (asyncio.Semaphore(args.cloud_max_concurrency)
                  if getattr(args, "cloud_max_concurrency", 0) > 0 else None)

    async with session_cm as session:
        if (kv_monitor is None and needs_kv
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
                progress = (
                    partial(inflight_kv.update_generated, req["request_id"])
                    if inflight_kv is not None
                    else None
                )
                return await one_request(session, endpoint, req, due,
                                         max_tokens_override=args.max_tokens,
                                         timeout_s=args.timeout_s,
                                         on_output_progress=progress,
                                         temperature=args.temperature,
                                         ignore_eos=args.local_ignore_eos)
        if send_cloud is None:
            async def send_cloud(req, due):  # noqa: F811 - default sender
                return await one_request(session, cloud, req, due,
                                         max_tokens_override=args.max_tokens,
                                         timeout_s=args.timeout_s,
                                         temperature=args.temperature,
                                         ignore_eos=args.cloud_ignore_eos,
                                         provider_order=args.cloud_provider,
                                         allow_fallbacks=(
                                             False
                                             if args.cloud_no_fallbacks
                                             else None
                                         ),
                                         stop_after_first_token=(
                                             args.cloud_stop_after_first_token
                                         ))

        run_start = time.perf_counter()
        f = out.open("w", encoding="utf-8")
        decision_f = None
        if args.decision_log is not None:
            decision_path = args.decision_log
            if not decision_path.is_absolute():
                decision_path = args.out_dir / decision_path
            decision_path.parent.mkdir(parents=True, exist_ok=True)
            decision_f = decision_path.open("w", encoding="utf-8", buffering=1)

        def record(res: dict[str, Any]) -> None:
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

        def record_decision(row: dict[str, Any]) -> None:
            if decision_f is not None:
                decision_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        def prediction_telemetry() -> dict[str, Any]:
            if getattr(policy, "trigger", None) != "ttft_pred":
                return {}
            return {
                "prediction_scope": "waiting_only",
                "prediction_model": "seq_slots_shared_prefill_lane_v1",
                "waiting_predicted_max_ttft_s": getattr(
                    policy, "last_predicted_max_ttft_s", None
                ),
                "waiting_post_kick_max_ttft_s": getattr(
                    policy, "last_post_kick_max_ttft_s", None
                ),
            }

        def prediction_context_telemetry(
            context: DecisionContext | None,
        ) -> dict[str, Any]:
            if context is None:
                return {}
            return {
                "inflight_decode_n": len(context.inflight_remaining_s),
                "inflight_prefill_n": len(context.inflight_prefills),
                "inflight_prefill_tokens": sum(
                    state.prompt_tokens for state in context.inflight_prefills
                ),
            }

        def annotate_scheduler_estimates(
            res: dict[str, Any], req: dict[str, Any]
        ) -> None:
            """Keep the controller's token view beside endpoint-reported usage.

            A trace/payload mismatch otherwise silently invalidates both the
            TTFT predictor and displacement ranking while the request itself
            still succeeds.
            """
            res["scheduler_prompt_tokens"] = int(req.get("prompt_tokens") or 0)
            res["scheduler_uncached_prompt_tokens"] = local_prompt_tokens(req)
            res["scheduler_decode_tokens"] = effective_decode(req, args.max_tokens)
            for key in (
                "trace_prompt_tokens", "trace_decode_tokens",
                "payload_mode", "cache_mode", "source_request_index",
            ):
                if req.get(key) is not None:
                    res[key] = req[key]

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
                if inflight_kv is not None:
                    inflight_kv.remove(req["request_id"])
                # A completion changes both current KV usage and committed
                # future growth; force the v3 decision to read fresh headroom.
                invalidate = getattr(kv_monitor, "invalidate", None)
                if invalidate is not None:
                    invalidate()
            annotate_scheduler_estimates(res, req)
            queue_delay_ms = max(0.0, (dispatch_t - arrival_due) * 1000)
            service_ttft = res.get("ttft_ms")
            res["queue_delay_ms"] = queue_delay_ms
            res["service_ttft_ms"] = service_ttft
            res["ttft_ms"] = None if service_ttft is None else queue_delay_ms + service_ttft
            if res.get("e2e_ms") is not None:
                res["e2e_ms"] = queue_delay_ms + res["e2e_ms"]
            record(res)
            await maybe_kick()
            maybe_dispatch()

        def spawn_local(req: dict[str, Any], arrival_due: float) -> None:
            admission.reserve(req)
            if inflight_kv is not None:
                inflight_kv.add(req)
            task = asyncio.create_task(serve_local(req, arrival_due))
            pending.add(task)
            task.add_done_callback(pending.discard)

        def maybe_dispatch() -> None:
            # policy first: while a shed decision is in flight, admission is
            # frozen — otherwise a completing request could dispatch a victim
            # the policy is about to kick (race caught in review)
            if has_tick and tick_busy:
                return
            # work-conserving: admit the head while a slot is free
            while queue and admission.fits(queue[0][0]):
                req, due = queue.pop(0)
                spawn_local(req, due)

        def _finalize_cloud(
            res: dict[str, Any],
            pre_route_queue_ms: float,
            cloud_gate_wait_ms: float,
        ) -> None:
            # Keep the two pre-service waits separate.  ``queue_delay_ms`` is
            # retained as their aggregate for old result consumers, while the
            # explicit fields make it possible to distinguish time spent in
            # Nimbus's local queue from throttling at the cloud concurrency
            # gate.  Both are part of arrival-to-first-token latency.
            service = res.get("ttft_ms")
            pre_route_queue_ms = max(0.0, pre_route_queue_ms)
            cloud_gate_wait_ms = max(0.0, cloud_gate_wait_ms)
            total_wait_ms = pre_route_queue_ms + cloud_gate_wait_ms
            res["pre_route_queue_ms"] = pre_route_queue_ms
            res["cloud_gate_wait_ms"] = cloud_gate_wait_ms
            res["queue_delay_ms"] = total_wait_ms
            res["service_ttft_ms"] = service
            if service is not None:
                res["ttft_ms"] = total_wait_ms + service
            if res.get("e2e_ms") is not None:
                res["e2e_ms"] = total_wait_ms + res["e2e_ms"]

        def route_cloud(req: dict[str, Any], due: float,
                        pre_route_queue_ms: float = 0.0) -> None:
            if sink is not None:
                res = sink.serve(req, due, max_tokens_override=args.max_tokens)
                annotate_scheduler_estimates(res, req)
                _finalize_cloud(res, pre_route_queue_ms, 0.0)
                record(res)
            else:
                cloud_queued_at = time.perf_counter()
                async def cloud_task():
                    send_started_at = cloud_queued_at
                    try:
                        if cloud_gate is not None:
                            async with cloud_gate:
                                send_started_at = time.perf_counter()
                                res = await send_cloud(req, due)
                        else:
                            res = await send_cloud(req, due)
                    except Exception as exc:  # a sender bug must not lose the row
                        res = {"request_id": req["request_id"],
                               "arrived_at": req["arrived_at"],
                               "relative_arrival_s": req["relative_arrival_s"],
                               "endpoint": "cloud",
                               "model": getattr(cloud, "model", None),
                               "success": False, "error": str(exc),
                               "error_type": type(exc).__name__, "http_status": None,
                               "ttft_ms": None, "e2e_ms": None, "tpot_ms": None,
                               "chunks": 0, "prompt_tokens": None,
                               "completion_tokens": None, "output_chars": 0,
                               "cost_usd": 0.0, "scheduled_lag_ms": 0.0}
                    cloud_gate_wait_ms = max(
                        0.0, (send_started_at - cloud_queued_at) * 1000
                    )
                    annotate_scheduler_estimates(res, req)
                    _finalize_cloud(
                        res, pre_route_queue_ms, cloud_gate_wait_ms
                    )
                    record(res)
                task = asyncio.create_task(cloud_task())
                pending.add(task)
                task.add_done_callback(pending.discard)

        # ---- queue-level policy hook (nimbus): kick overflow to the cloud ----
        # Invariant: the policy adjudicates BEFORE any local admission. While a
        # shed decision is computing off-thread, admission is FROZEN (see
        # maybe_dispatch); events that land meanwhile set tick_rerun so the
        # decision re-runs on the fresh queue before admission resumes.
        tick_busy = False
        tick_rerun = False
        decision_seq = 0

        async def maybe_kick(
            apply_before_s: float | None = None,
        ) -> bool:
            """Adjudicate the current queue without crossing a future arrival.

            The arrival producer and policy call share one coroutine.  A
            policy call that starts before the next cohort but finishes after
            it must not apply an irreversible cloud route to the old snapshot;
            return ``False`` so the caller can enqueue every now-due cohort and
            re-adjudicate the combined candidate set first.
            """
            nonlocal tick_busy, tick_rerun, decision_seq
            if not has_tick:
                return True
            assert inflight_kv is not None
            if tick_busy:
                tick_rerun = True       # missed wakeup: re-adjudicate after
                return True
            if not queue:
                return True
            if (apply_before_s is not None
                    and time.perf_counter() >= apply_before_s):
                return False
            tick_busy = True
            try:
                retries = 0
                while True:
                    tick_rerun = False
                    kv_avail = (
                        await kv_monitor.available_tokens()
                        if needs_kv and kv_monitor is not None
                        else float("inf")
                    )
                    # The 250ms metrics cache can briefly predate newly
                    # dispatched prompts. Clamp headroom with commitments we
                    # already know about so a burst cannot slip through blind.
                    if needs_kv and args.kv_capacity_tokens is not None:
                        predicted_headroom = max(
                            0.0,
                            float(args.kv_capacity_tokens)
                            - inflight_kv.predicted_current_tokens,
                        )
                        kv_avail = min(kv_avail, predicted_headroom)
                    snapshot = [r for r, _ in queue]
                    decision_seq += 1
                    this_decision = decision_seq
                    snapshot_ids = (
                        [r["request_id"] for r in snapshot]
                        if decision_f is not None else None
                    )
                    snapshot_hash = (
                        hashlib.sha256(repr(snapshot_ids).encode("utf-8")).hexdigest()[:16]
                        if decision_f is not None else None
                    )
                    remaining_decode = inflight_kv.remaining_decode_tokens
                    context = None
                    progress_version = None
                    if getattr(policy, "trigger", None) == "ttft_pred":
                        decision_now = time.perf_counter()
                        progress_version = inflight_kv.version
                        context = DecisionContext(
                            waiting_age_s={
                                r["request_id"]: max(0.0, decision_now - due)
                                for r, due in queue
                            },
                            inflight_remaining_s=(
                                inflight_kv.decode_remaining_service_s(
                                    policy.tpot_s
                                )
                            ),
                            max_inflight=args.max_inflight,
                            inflight_prefills=inflight_kv.inflight_prefills,
                        )
                    # heavy work runs off-thread so a deep-queue decision cannot
                    # stall arrival pacing / completions / the KV scrape
                    on_tick_args = (
                        (snapshot, kv_avail, remaining_decode, context)
                        if context is not None
                        else (snapshot, kv_avail, remaining_decode)
                    )
                    decision_started = time.perf_counter()
                    victims = await asyncio.get_running_loop().run_in_executor(
                        None, policy.on_tick, *on_tick_args
                    )
                    decision_ms = (time.perf_counter() - decision_started) * 1000
                    if hasattr(policy, "decision_calls"):
                        policy.decision_calls += 1
                        policy.decision_total_ms += decision_ms
                        policy.decision_max_ms = max(policy.decision_max_ms, decision_ms)
                    # The producer could not enqueue a future cohort while it
                    # awaited this executor call.  If that cohort is due now,
                    # discard even a non-empty victim set before it becomes an
                    # irreversible cloud route.  The producer immediately
                    # absorbs due cohorts and calls us again on the fresh set.
                    if (apply_before_s is not None
                            and time.perf_counter() >= apply_before_s):
                        if hasattr(policy, "stale_decisions"):
                            policy.stale_decisions += 1
                        record_decision({
                            "decision_id": this_decision,
                            "at_s": time.perf_counter() - run_start,
                            "status": "stale_future_arrival",
                            "trigger": getattr(policy, "trigger", None),
                            "selector": getattr(policy, "selector", None),
                            "snapshot_n": len(snapshot),
                            "snapshot_hash": snapshot_hash,
                            "inflight_n": (
                                context.inflight_n
                                if context else admission.inflight
                            ),
                            **prediction_context_telemetry(context),
                            **prediction_telemetry(),
                            "proposed_victim_ids": [
                                v["request_id"] for v in victims
                            ],
                            "decision_ms": decision_ms,
                        })
                        return False
                    if (progress_version is not None
                            and inflight_kv.version != progress_version):
                        tick_rerun = True
                    # a decision computed on a stale window must not be applied:
                    # a completion may have freed KV (stale kicks over-outsource)
                    # or arrivals changed the set. Discard and re-decide on
                    # fresh state — bounded, then apply the latest anyway so a
                    # busy system cannot livelock the shed path.
                    if tick_rerun and queue and retries < 3:
                        if hasattr(policy, "stale_decisions"):
                            policy.stale_decisions += 1
                        record_decision({
                            "decision_id": this_decision,
                            "at_s": time.perf_counter() - run_start,
                            "status": "stale_retry",
                            "trigger": getattr(policy, "trigger", None),
                            "selector": getattr(policy, "selector", None),
                            "snapshot_n": len(snapshot),
                            "snapshot_hash": snapshot_hash,
                            "inflight_n": (
                                context.inflight_n
                                if context else admission.inflight
                            ),
                            **prediction_context_telemetry(context),
                            **prediction_telemetry(),
                            "proposed_victim_ids": [v["request_id"] for v in victims],
                            "decision_ms": decision_ms,
                        })
                        retries += 1
                        if kv_monitor is not None:
                            kv_monitor.invalidate()
                        continue
                    applied_ids = []
                    for v in victims:
                        idx = next((i for i, (r, _) in enumerate(queue)
                                    if r["request_id"] == v["request_id"]), None)
                        if idx is None:      # defensive; admission is frozen,
                            continue         # so victims should still be here
                        req, due = queue.pop(idx)
                        waited_ms = max(0.0, (time.perf_counter() - due) * 1000)
                        policy.n_outsourced += 1
                        applied_ids.append(req["request_id"])
                        route_cloud(req, due, pre_route_queue_ms=waited_ms)
                    if applied_ids and hasattr(policy, "applied_kick_rounds"):
                        policy.applied_kick_rounds += 1
                    record_decision({
                        "decision_id": this_decision,
                        "at_s": time.perf_counter() - run_start,
                        "status": (
                            "applied_stale_bounded"
                            if tick_rerun else ("applied" if applied_ids else "no_op")
                        ),
                        "trigger": getattr(policy, "trigger", None),
                        "selector": getattr(policy, "selector", None),
                        "snapshot_n": len(snapshot),
                        "snapshot_hash": snapshot_hash,
                        "inflight_n": (
                            context.inflight_n
                            if context else admission.inflight
                        ),
                        **prediction_context_telemetry(context),
                        **prediction_telemetry(),
                        "proposed_victim_ids": [v["request_id"] for v in victims],
                        "applied_victim_ids": applied_ids,
                        "decision_ms": decision_ms,
                    })
                    if not tick_rerun or not queue:
                        break               # queue unchanged since snapshot
                    retries = 0             # applied; handle the new events
            finally:
                tick_busy = False
            return True

        periodic_tick_s = (
            args.nimbus_tick_ms / 1000.0
            if getattr(policy, "needs_periodic_tick", False)
            else None
        )

        async def wait_for_arrival(due: float) -> None:
            """Sleep to an arrival while periodically rechecking TTFT risk."""
            while True:
                wait = due - time.perf_counter()
                if wait <= 0:
                    return
                if periodic_tick_s is None:
                    await asyncio.sleep(wait)
                    return
                await asyncio.sleep(min(wait, periodic_tick_s))
                if queue:
                    if not await maybe_kick(due):
                        return
                    maybe_dispatch()

        try:
            # BurstGPT timestamps are integer seconds, so several requests can
            # be genuinely simultaneous.  Form the whole arrival cohort before
            # adjudication: deciding and dispatching one row at a time makes the
            # first rows irrevocably in-flight and leaves selectors with a
            # one-item "choice" (an invalid selector experiment).
            cohort_start = 0
            while cohort_start < len(trace):
                due = run_start + (
                    trace[cohort_start]["relative_arrival_s"] * args.time_scale
                )
                await wait_for_arrival(due)

                # Absorb every cohort due as of one clock sample. If policy
                # computation crosses the following arrival, maybe_kick()
                # returns False without applying victims and this loop expands
                # the candidate set before trying again.
                while True:
                    now = time.perf_counter()
                    while cohort_start < len(trace):
                        relative_arrival_s = trace[cohort_start][
                            "relative_arrival_s"
                        ]
                        cohort_due = run_start + (
                            relative_arrival_s * args.time_scale
                        )
                        if cohort_due > now:
                            break
                        cohort_end = cohort_start + 1
                        while (cohort_end < len(trace)
                               and trace[cohort_end]["relative_arrival_s"]
                               == relative_arrival_s):
                            cohort_end += 1
                        for req in trace[cohort_start:cohort_end]:
                            if policy.outsource(req):
                                route_cloud(req, cohort_due)
                            else:
                                queue.append((req, cohort_due))
                        admission.peak_waiting = max(
                            admission.peak_waiting, len(queue)
                        )
                        cohort_start = cohort_end

                    next_due = (
                        run_start
                        + trace[cohort_start]["relative_arrival_s"] * args.time_scale
                        if cohort_start < len(trace) else None
                    )
                    if await maybe_kick(next_due):
                        break

                # Only a decision that completed before the next arrival may
                # make the retained cohort irrevocably in-flight.
                maybe_dispatch()

            # drain: everything left in queue/in flight completes through the
            # same dispatcher (completions re-trigger maybe_dispatch)
            next_periodic_tick_at = (
                time.perf_counter() + periodic_tick_s
                if periodic_tick_s is not None
                else None
            )
            while queue or pending:
                if pending:
                    timeout = (
                        max(0.0, next_periodic_tick_at - time.perf_counter())
                        if next_periodic_tick_at is not None
                        else None
                    )
                    done, _ = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=timeout,
                    )
                    now = time.perf_counter()
                    if (next_periodic_tick_at is not None
                            and now >= next_periodic_tick_at):
                        while next_periodic_tick_at <= now:
                            next_periodic_tick_at += periodic_tick_s
                    else:
                        now = None
                    if now is not None and queue:
                        await maybe_kick()
                        maybe_dispatch()
                else:  # queue non-empty, nothing in flight -> dispatch now
                    maybe_dispatch()
        finally:
            f.close()
            if decision_f is not None:
                decision_f.close()

    print(f"raw: {out}")
    return results, admission, kv_monitor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "full"], required=True,
                        help="BurstGPT window, or 'full' = replay the whole file")
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
                             "log 'GPU KV cache size: N tokens' (required for kv_gap)")
    parser.add_argument("--prefill-tput", type=float, default=None,
                        help="shared prefill-lane tokens/s for the deployment "
                             "(required calibration for ttft_pred; kv_gap "
                             "otherwise defaults to 20000)")
    parser.add_argument("--tpot-ms", type=float, default=None,
                        help="effective per-request TPOT at the operating batch "
                             "(required calibration for ttft_pred; kv_gap otherwise "
                             "defaults to the 35B/MTP profile, 9.5ms)")
    parser.add_argument("--first-token-overhead-ms", type=float, default=None,
                        help="fixed TTFT intercept for the shared prefill model "
                             "(required calibration for ttft_pred; distinct "
                             "from decode TPOT)")
    parser.add_argument("--kv-hysteresis-fraction", type=float,
                        default=DEFAULT_KV_HYSTERESIS_FRACTION,
                        help="extra KV headroom fraction released per v3 shedding "
                             "round (default: %(default)s)")
    parser.add_argument("--nimbus-trigger", choices=NIMBUS_TRIGGERS, default="kv_gap",
                        help="queue shedding trigger (default preserves Nimbus v3)")
    parser.add_argument("--nimbus-selector", choices=NIMBUS_SELECTORS,
                        default="cost_disp_current",
                        help="victim ordering, independent of the trigger")
    parser.add_argument("--ttft-guard-ms", type=float, default=None,
                        help="trigger this many ms before the TTFT SLO (required "
                             "for ttft_pred; cover tick + decision latency)")
    parser.add_argument("--nimbus-tick-ms", type=float, default=250.0,
                        help="wall-clock recheck interval for ttft_pred")

    parser.add_argument("--cloud", choices=["null", "real"], default="null")
    parser.add_argument("--cloud-url", default=None)
    parser.add_argument("--cloud-model", default=None)
    parser.add_argument("--cloud-api-key-env", default=None)
    parser.add_argument("--cloud-max-concurrency", type=int, default=32,
                        help="cap concurrent REAL cloud requests (0 = unlimited); "
                             "guards against self-inflicted 429s under burst")
    parser.add_argument("--cloud-provider", action="append", default=None,
                        help="OpenRouter provider slug in priority order; repeat "
                             "the flag to supply more than one")
    parser.add_argument("--cloud-no-fallbacks", action="store_true",
                        help="send OpenRouter provider.allow_fallbacks=false")
    parser.add_argument("--cloud-stop-after-first-token", action="store_true",
                        help="TTFT-only probe: abort the real cloud stream after "
                             "the first non-empty content or reasoning delta")
    parser.add_argument("--in-price", type=float, default=0.15)
    parser.add_argument("--out-price", type=float, default=1.20)

    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="multiply arrival offsets (<1 compresses the trace; "
                             "1.0 = replay at recorded speed)")
    parser.add_argument("--slo-s", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None,
                        help="optional sampling override sent to real endpoints")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="legacy shorthand: force both local and cloud "
                             "generation to the requested cap")
    local_ignore_eos = parser.add_mutually_exclusive_group()
    local_ignore_eos.add_argument(
        "--local-ignore-eos", dest="local_ignore_eos", action="store_true",
        default=None, help="force only local generation to the requested cap",
    )
    local_ignore_eos.add_argument(
        "--no-local-ignore-eos", dest="local_ignore_eos", action="store_false",
        help="allow local EOS (overrides legacy --ignore-eos)",
    )
    cloud_ignore_eos = parser.add_mutually_exclusive_group()
    cloud_ignore_eos.add_argument(
        "--cloud-ignore-eos", dest="cloud_ignore_eos", action="store_true",
        default=None, help="force only cloud generation to the requested cap",
    )
    cloud_ignore_eos.add_argument(
        "--no-cloud-ignore-eos", dest="cloud_ignore_eos", action="store_false",
        help="allow cloud EOS (overrides legacy --ignore-eos)",
    )
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--decision-log", type=Path, default=None,
                        help="optional JSONL audit log for Nimbus decisions")

    args = parser.parse_args(argv)

    # Preserve --ignore-eos exactly for existing commands, while allowing a
    # full hybrid run to hold local requests to the cap and let cloud requests
    # stop after their first token.  Explicit endpoint flags win over legacy.
    if args.local_ignore_eos is None:
        args.local_ignore_eos = args.ignore_eos
    if args.cloud_ignore_eos is None:
        args.cloud_ignore_eos = args.ignore_eos

    if args.slo_s <= 0 or args.timeout_s <= 0:
        parser.error("--slo-s and --timeout-s must be > 0")
    if args.policy == "nimbus" and args.nimbus_trigger == "ttft_pred":
        missing = [
            flag for flag, value in (
                ("--prefill-tput", args.prefill_tput),
                ("--tpot-ms", args.tpot_ms),
                ("--first-token-overhead-ms", args.first_token_overhead_ms),
                ("--ttft-guard-ms", args.ttft_guard_ms),
            ) if value is None
        ]
        if missing:
            parser.error("ttft_pred requires explicit calibrated " + ", ".join(missing))
    if args.prefill_tput is None:
        args.prefill_tput = 20_000.0
    if args.tpot_ms is None:
        args.tpot_ms = 9.5
    if args.first_token_overhead_ms is None:
        args.first_token_overhead_ms = args.tpot_ms
    if args.ttft_guard_ms is None:
        args.ttft_guard_ms = 0.0

    if args.max_inflight <= 0:
        parser.error("--max-inflight must be >= 1 (0 would deadlock the dispatcher)")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be >= 1 (it replaces the trace decode count "
                     "in the payload; 0 breaks fake/real parity)")
    if args.prefill_tput <= 0:
        parser.error("--prefill-tput must be > 0")
    if args.tpot_ms < 0:
        parser.error("--tpot-ms must be >= 0")
    if args.first_token_overhead_ms < 0:
        parser.error("--first-token-overhead-ms must be >= 0")
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if not 0.0 <= args.kv_hysteresis_fraction < 1.0:
        parser.error("--kv-hysteresis-fraction must be in [0,1)")
    if args.ttft_guard_ms < 0 or args.ttft_guard_ms >= args.slo_s * 1000.0:
        parser.error("--ttft-guard-ms must be in [0, --slo-s)")
    if args.nimbus_tick_ms <= 0:
        parser.error("--nimbus-tick-ms must be > 0")
    if args.in_price < 0 or args.out_price < 0:
        parser.error("--in-price/--out-price must be >= 0")
    if args.cloud_max_concurrency < 0:
        parser.error("--cloud-max-concurrency must be >= 0 (0 = unlimited)")
    if args.time_scale <= 0:
        parser.error("--time-scale must be > 0")
    needs_local = args.policy in ("all_local", "random", "nimbus")
    needs_cloud = args.policy in ("all_cloud", "random", "nimbus")
    if args.cloud_provider:
        args.cloud_provider = [provider.strip() for provider in args.cloud_provider]
        if any(not provider for provider in args.cloud_provider):
            parser.error("--cloud-provider must not be empty")
        if len(set(args.cloud_provider)) != len(args.cloud_provider):
            parser.error("--cloud-provider entries must be unique")
    cloud_openrouter_options = (
        args.cloud_provider is not None
        or args.cloud_no_fallbacks
        or args.cloud_stop_after_first_token
    )
    if cloud_openrouter_options and (not needs_cloud or args.cloud != "real"):
        parser.error("OpenRouter cloud provider/cancellation options require a "
                     "cloud-using policy with --cloud real")
    if args.cloud_stop_after_first_token:
        if not args.cloud_provider or len(args.cloud_provider) != 1:
            parser.error("--cloud-stop-after-first-token requires exactly one "
                         "--cloud-provider")
        if not args.cloud_no_fallbacks:
            parser.error("--cloud-stop-after-first-token requires "
                         "--cloud-no-fallbacks")
        if not args.cloud_api_key_env:
            parser.error("--cloud-stop-after-first-token requires "
                         "--cloud-api-key-env")
        if args.cloud_ignore_eos:
            parser.error("--cloud-stop-after-first-token cannot be combined "
                         "with cloud ignore-EOS generation")
    if (args.policy == "nimbus" and args.nimbus_trigger == "kv_gap"
            and (args.kv_capacity_tokens is None
                 or args.kv_capacity_tokens <= 0)):
        parser.error("--policy nimbus --nimbus-trigger kv_gap requires "
                     "--kv-capacity-tokens > 0 (read it from the vLLM startup log)")
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
        "peak_waiting": admission.peak_waiting,
    }
    if policy is not None and hasattr(policy, "on_tick"):
        stats["nimbus_ticks"] = policy.ticks
        stats["nimbus_kick_rounds"] = policy.kick_rounds
        stats["nimbus_kicked"] = policy.n_outsourced
        stats["nimbus_trigger"] = getattr(policy, "trigger", None)
        stats["nimbus_selector"] = getattr(policy, "selector", None)
        stats["nimbus_applied_kick_rounds"] = getattr(
            policy, "applied_kick_rounds", None
        )
        stats["nimbus_stale_decisions"] = getattr(policy, "stale_decisions", None)
        decision_calls = getattr(policy, "decision_calls", 0)
        stats["nimbus_decision_calls"] = decision_calls
        stats["nimbus_decision_mean_ms"] = (
            getattr(policy, "decision_total_ms", 0.0) / decision_calls
            if decision_calls else 0.0
        )
        stats["nimbus_decision_max_ms"] = getattr(policy, "decision_max_ms", 0.0)
        if getattr(policy, "trigger", None) == "ttft_pred":
            stats["nimbus_prediction_scope"] = "waiting_only"
            stats["nimbus_prediction_model"] = (
                "seq_slots_shared_prefill_lane_v1"
            )
            stats["nimbus_max_waiting_predicted_ttft_s"] = getattr(
                policy, "max_predicted_ttft_s_seen", None
            )
            stats["nimbus_max_waiting_post_kick_ttft_s"] = getattr(
                policy, "max_post_kick_ttft_s_seen", None
            )
    if kv_monitor is not None:
        stats["kv_read_failures"] = kv_monitor.read_failures
    return stats


def token_alignment_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare controller token metadata with endpoint-reported usage."""
    local_success = [
        r for r in results
        if r.get("success") and r.get("endpoint") == "local"
        and not r.get("routed_only")
    ]
    prompt_pairs = [
        (int(r["scheduler_prompt_tokens"]), int(r["prompt_tokens"]))
        for r in local_success
        if r.get("scheduler_prompt_tokens") is not None
        and r.get("prompt_tokens") is not None
    ]
    decode_pairs = [
        (int(r["scheduler_decode_tokens"]), int(r["completion_tokens"]))
        for r in local_success
        if r.get("scheduler_decode_tokens") is not None
        and r.get("completion_tokens") is not None
    ]
    if not prompt_pairs and not decode_pairs:
        return {"measured_n": 0}

    absolute_errors = sorted(
        abs(actual - expected) for expected, actual in prompt_pairs
    )
    relative_errors = sorted(
        abs(actual - expected) / expected
        for expected, actual in prompt_pairs
        if expected > 0
    )
    ratios = sorted(
        actual / expected for expected, actual in prompt_pairs if expected > 0
    )
    decode_ratios = sorted(
        actual / expected for expected, actual in decode_pairs if expected > 0
    )

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]

    return {
        "local_success_n": len(local_success),
        "measured_n": len(prompt_pairs),
        "missing_prompt_usage_n": len(local_success) - len(prompt_pairs),
        "prompt_exact_n": sum(
            expected == actual for expected, actual in prompt_pairs
        ),
        "prompt_exact_fraction": (
            sum(expected == actual for expected, actual in prompt_pairs)
            / len(prompt_pairs) if prompt_pairs else None
        ),
        "absolute_error_p50_tokens": pct(absolute_errors, 0.50),
        "absolute_error_p95_tokens": pct(absolute_errors, 0.95),
        "absolute_error_max_tokens": (
            absolute_errors[-1] if absolute_errors else None
        ),
        "relative_error_p50": pct(relative_errors, 0.50),
        "relative_error_p95": pct(relative_errors, 0.95),
        "actual_over_scheduler_p50": pct(ratios, 0.50),
        "actual_over_scheduler_p95": pct(ratios, 0.95),
        "decode_measured_n": len(decode_pairs),
        "missing_completion_usage_n": len(local_success) - len(decode_pairs),
        "decode_cap_hit_n": sum(
            expected == actual for expected, actual in decode_pairs
        ),
        "decode_cap_hit_fraction": (
            sum(expected == actual for expected, actual in decode_pairs)
            / len(decode_pairs) if decode_pairs else None
        ),
        "completion_over_scheduler_p50": pct(decode_ratios, 0.50),
        "completion_over_scheduler_p05": pct(decode_ratios, 0.05),
    }


async def main() -> None:
    args = parse_args()
    trace = load_trace(args.data, args.scenario)
    print(f"loaded {len(trace)} requests  scenario={args.scenario} policy={args.policy} "
          f"max_inflight={args.max_inflight}")

    if args.policy == "nimbus":
        from router.nimbus import NimbusPolicy
        policy = NimbusPolicy(args.prefill_tput, args.tpot_ms / 1000.0, seed=args.seed,
                              max_tokens_override=args.max_tokens,
                              in_price_mtok=args.in_price,
                              out_price_mtok=args.out_price,
                              hysteresis_fraction=args.kv_hysteresis_fraction,
                              trigger=args.nimbus_trigger,
                              selector=args.nimbus_selector,
                              slo_s=args.slo_s,
                              ttft_guard_s=args.ttft_guard_ms / 1000.0,
                              first_token_overhead_s=(
                                  args.first_token_overhead_ms / 1000.0
                              ))
    else:
        policy = Policy(args.policy, args.fraction, args.seed)
    needs_cloud = args.policy in ("all_cloud", "random", "nimbus")
    local, cloud, sink = build_cloud_sink(args, needs_cloud)

    results, admission, kv_mon = await replay_queued(args, trace, policy, local, cloud, sink)
    summary = summarize(results, policy, args.slo_s)
    summary["queue"] = queue_stats(results, admission, policy, kv_mon)
    summary["token_alignment"] = token_alignment_stats(results)
    summary["config"] = {
        "scenario": args.scenario,
        "seed": args.seed,
        "time_scale": args.time_scale,
        "max_inflight": args.max_inflight,
        "max_tokens_override": args.max_tokens,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
        "local_ignore_eos": args.local_ignore_eos,
        "cloud_ignore_eos": args.cloud_ignore_eos,
        "nimbus_trigger": args.nimbus_trigger,
        "nimbus_selector": args.nimbus_selector,
        "prefill_tput": args.prefill_tput,
        "tpot_ms": args.tpot_ms,
        "first_token_overhead_ms": args.first_token_overhead_ms,
        "slo_s": args.slo_s,
        "ttft_guard_ms": args.ttft_guard_ms,
        "nimbus_tick_ms": args.nimbus_tick_ms,
        "kv_capacity_tokens": args.kv_capacity_tokens,
        "kv_hysteresis_fraction": args.kv_hysteresis_fraction,
        "cloud": args.cloud,
        "cloud_url": args.cloud_url,
        "cloud_model": args.cloud_model or args.local_model,
        "cloud_api_key_env": args.cloud_api_key_env,
        "cloud_max_concurrency": args.cloud_max_concurrency,
        "cloud_provider_order": args.cloud_provider,
        "cloud_no_fallbacks": args.cloud_no_fallbacks,
        "cloud_stop_after_first_token": args.cloud_stop_after_first_token,
        "local_url": args.local_url,
        "local_model": args.local_model,
        "in_price": args.in_price,
        "out_price": args.out_price,
    }

    out = resolve_output_path(args)
    summary_path = out.with_name(out.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
