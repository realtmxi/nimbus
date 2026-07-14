"""Nimbus queue shedding with independently selectable triggers and selectors.

The default remains the shipped v3 policy:

    trigger  = kv_gap
    selector = cost_disp_current

The experimental ``ttft_pred`` trigger reuses the same queue hook but predicts
FCFS admission over two local resources: sequence slots and one shared prefill
compute lane.  It includes time already waited, estimated releases of in-flight
slots, unfinished in-flight prefill work, each waiting request's prefill, and
the calibrated fixed first-token overhead.  Trigger and selector are
deliberately orthogonal so a selector comparison uses the same candidate set
and stop rule.

See docs/notion_algorithm_design_v3.md for the v3 baseline and
docs/v3_experiments_2026-07.md for the experiment record.
"""
from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass
from typing import Any, Mapping

from router.common import (
    DEFAULT_KV_HYSTERESIS_FRACTION,
    effective_decode,
    local_prompt_tokens,
    token_cost_usd,
)


NIMBUS_TRIGGERS = ("kv_gap", "ttft_pred")
NIMBUS_SELECTORS = (
    "cost_disp_current",
    "cost_cachedisp_old",
    "max_cachedisp_old",
    "newest",
    "waiting_random",
)


@dataclass(frozen=True)
class InflightPrefill:
    """Admitted request still occupying the shared prefill lane."""

    prompt_tokens: int
    decode_tokens: int


@dataclass(frozen=True)
class DecisionContext:
    """Runtime state needed by the predicted-TTFT trigger.

    ``inflight_remaining_s`` contains one estimated slot-release time per
    request already known to be decoding.  ``inflight_prefills`` contains
    admitted requests that have not produced a first token, in dispatch order;
    the simulator derives both their shared-lane completion and slot release
    instead of counting their prefill twice.  Missing slots up to
    ``max_inflight`` are free now.  Waiting ages are keyed by request id so
    selector reordering cannot accidentally change deadline accounting.
    """

    waiting_age_s: Mapping[Any, float]
    inflight_remaining_s: tuple[float, ...]
    max_inflight: int
    inflight_prefills: tuple[InflightPrefill, ...] = ()

    @property
    def inflight_n(self) -> int:
        return len(self.inflight_remaining_s) + len(self.inflight_prefills)


def token_footprint(req: dict[str, Any], max_tokens_override: int | None = None) -> int:
    """Peak marginal KV commitment: local prompt + expected full decode."""
    return local_prompt_tokens(req) + effective_decode(req, max_tokens_override)


def displacement_token_s(req: dict[str, Any], prefill_tput: float, tpot_s: float,
                         max_tokens_override: int | None = None) -> float:
    """Conservative online proxy for ``integral KV_tokens(t) dt``."""
    prompt = local_prompt_tokens(req)
    decode = effective_decode(req, max_tokens_override)
    residence_s = prompt / max(prefill_tput, 1e-9) + decode * tpot_s
    return token_footprint(req, max_tokens_override) * residence_s


def classic_cachedisp_token_s(
    req: dict[str, Any],
    prefill_tput: float,
    tpot_s: float,
    max_tokens_override: int | None = None,
) -> float:
    """The original Nimbus v2 formula from commit ``587547a``.

    ``prompt * (uncached_prompt / prefill_tput + decode * TPOT)``.  The outer
    prompt is the full KV footprint pinned by the request; cached/processed
    prompt tokens reduce only the remaining prefill time.
    """
    prompt = int(req.get("prompt_tokens") or 0)
    if req.get("uncached_prompt_tokens") is not None:
        remaining_prefill = max(0, int(req["uncached_prompt_tokens"]))
    else:
        processed = int(
            req.get("num_processed_tokens")
            or req.get("processed_prompt_tokens")
            or 0
        )
        cached = int(
            req.get("num_cached_tokens")
            or req.get("cached_prompt_tokens")
            or req.get("cached_tokens")
            or 0
        )
        remaining_prefill = max(0, prompt - processed - cached)
    decode = effective_decode(req, max_tokens_override)
    residence_s = remaining_prefill / max(prefill_tput, 1e-9) + decode * tpot_s
    return prompt * residence_s


def cloud_cost_usd(req: dict[str, Any], in_price_mtok: float, out_price_mtok: float,
                   max_tokens_override: int | None = None) -> float:
    """Expected API charge if this request is outsourced."""
    prompt = int(req.get("prompt_tokens") or 0)
    return token_cost_usd(
        prompt,
        effective_decode(req, max_tokens_override),
        in_price_mtok,
        out_price_mtok,
    )


def shedding_score(
    req: dict[str, Any],
    prefill_tput: float,
    tpot_s: float,
    in_price_mtok: float,
    out_price_mtok: float,
    max_tokens_override: int | None = None,
) -> float:
    """Dollars paid per token-second released; lower is kicked first."""
    displacement = displacement_token_s(
        req, prefill_tput, tpot_s, max_tokens_override
    )
    cost = cloud_cost_usd(
        req, in_price_mtok, out_price_mtok, max_tokens_override
    )
    if displacement <= 0:
        return 0.0 if cost <= 0 else math.inf
    return cost / displacement


def classic_cachedisp_score(
    req: dict[str, Any],
    prefill_tput: float,
    tpot_s: float,
    in_price_mtok: float,
    out_price_mtok: float,
    max_tokens_override: int | None = None,
) -> float:
    """Cloud dollars paid per original-v2 token-second released."""
    displacement = classic_cachedisp_token_s(
        req, prefill_tput, tpot_s, max_tokens_override
    )
    cost = cloud_cost_usd(
        req, in_price_mtok, out_price_mtok, max_tokens_override
    )
    if displacement <= 0:
        return 0.0 if cost <= 0 else math.inf
    return cost / displacement


def _stable_random_priority(seed: int, request_id: Any) -> bytes:
    """A retry-stable random key; stale-decision reruns keep the same order."""
    payload = f"{seed}\0{request_id!r}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def victim_order(
    waiting: list[dict[str, Any]],
    selector: str,
    *,
    prefill_tput: float,
    tpot_s: float,
    in_price_mtok: float,
    out_price_mtok: float,
    seed: int,
    max_tokens_override: int | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic victim priority for a fixed queue snapshot."""
    if selector == "cost_disp_current":
        return sorted(
            waiting,
            key=lambda r: shedding_score(
                r,
                prefill_tput,
                tpot_s,
                in_price_mtok,
                out_price_mtok,
                max_tokens_override,
            ),
        )
    if selector == "cost_cachedisp_old":
        return sorted(
            waiting,
            key=lambda r: classic_cachedisp_score(
                r,
                prefill_tput,
                tpot_s,
                in_price_mtok,
                out_price_mtok,
                max_tokens_override,
            ),
        )
    if selector == "max_cachedisp_old":
        return sorted(
            waiting,
            key=lambda r: classic_cachedisp_token_s(
                r, prefill_tput, tpot_s, max_tokens_override
            ),
            reverse=True,
        )
    if selector == "newest":
        return list(reversed(waiting))
    if selector == "waiting_random":
        return sorted(
            waiting,
            key=lambda r: _stable_random_priority(seed, r.get("request_id")),
        )
    raise ValueError(f"unknown Nimbus selector {selector!r}")


def predicted_waiting_ttfts_s(
    waiting: list[dict[str, Any]],
    context: DecisionContext,
    *,
    prefill_tput: float,
    tpot_s: float,
    first_token_overhead_s: float | None = None,
    max_tokens_override: int | None = None,
) -> list[float]:
    """Predict from-arrival TTFT over sequence slots plus shared prefill.

    This is a calibrated online approximation, not an engine-exact simulator.
    Decode length is the trace/request cap in this first experiment (an oracle
    input whose estimator ablation is intentionally separate).  Prefill work
    is serialized because continuous-batching engines share a bounded prefill
    token budget: giving 128 requests 128 free sequence slots does not make all
    128 first tokens simultaneous.
    """
    if context.max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    first_token_s = (
        tpot_s
        if first_token_overhead_s is None
        else float(first_token_overhead_s)
    )
    if first_token_s < 0:
        raise ValueError("first_token_overhead_s must be non-negative")
    if any(
        state.prompt_tokens < 0 or state.decode_tokens < 0
        for state in context.inflight_prefills
    ):
        raise ValueError("inflight prefill token counts must be non-negative")

    slots = [max(0.0, float(x)) for x in context.inflight_remaining_s]
    prefill_ready_s = 0.0
    for state in context.inflight_prefills:
        prefill_ready_s += (
            float(state.prompt_tokens) / max(prefill_tput, 1e-9)
        )
        heapq.heappush(
            slots,
            prefill_ready_s
            + first_token_s
            + max(0, state.decode_tokens - 1) * tpot_s,
        )
    # Runtime invariants keep len(inflight) <= max_inflight.  If an injected
    # context violates that, retain every known busy slot rather than silently
    # discarding work and becoming optimistic.
    slot_count = max(context.max_inflight, len(slots))
    slots.extend([0.0] * (slot_count - len(slots)))
    heapq.heapify(slots)

    # The tracker cannot observe partial prefill progress, so each admitted
    # prefill above retains its full work until exact output usage proves it
    # reached decode.  This is conservative without inventing engine progress.
    predictions: list[float] = []
    for req in waiting:
        slot_ready_s = heapq.heappop(slots)
        prompt = local_prompt_tokens(req)
        decode = effective_decode(req, max_tokens_override)
        prefill_s = prompt / max(prefill_tput, 1e-9)
        prefill_started_s = max(slot_ready_s, prefill_ready_s)
        prefill_done_s = prefill_started_s + prefill_s
        age_s = max(0.0, float(context.waiting_age_s.get(req.get("request_id"), 0.0)))
        # TTFT includes the fitted fixed first-token overhead. Slot residence
        # includes all requested decode steps after prefill before the next
        # queued request can be admitted; waiting prefill cannot overtake it.
        predictions.append(age_s + prefill_done_s + first_token_s)
        heapq.heappush(
            slots,
            prefill_done_s
            + first_token_s
            + max(0, decode - 1) * tpot_s,
        )
        prefill_ready_s = prefill_done_s
    return predictions


class NimbusPolicy:
    """Queue-level shedding policy. Interface-compatible with common.Policy
    (outsource/actual_fraction) plus the on_tick hook the runner calls."""

    name = "nimbus"
    p = None   # no target fraction: the policy self-selects its split

    def __init__(
        self,
        prefill_tput: float,
        tpot_s: float,
        seed: int = 0,
        max_tokens_override: int | None = None,
        in_price_mtok: float = 0.15,
        out_price_mtok: float = 1.20,
        hysteresis_fraction: float = DEFAULT_KV_HYSTERESIS_FRACTION,
        trigger: str = "kv_gap",
        selector: str = "cost_disp_current",
        slo_s: float = 5.0,
        ttft_guard_s: float = 0.0,
        first_token_overhead_s: float | None = None,
    ):
        if trigger not in NIMBUS_TRIGGERS:
            raise ValueError(f"unknown Nimbus trigger {trigger!r}")
        if selector not in NIMBUS_SELECTORS:
            raise ValueError(f"unknown Nimbus selector {selector!r}")
        if slo_s <= 0:
            raise ValueError("slo_s must be positive")
        if not 0.0 <= ttft_guard_s < slo_s:
            raise ValueError("ttft_guard_s must be in [0, slo_s)")
        self.prefill_tput = prefill_tput
        self.tpot_s = tpot_s
        self.mto = max_tokens_override
        self.in_price_mtok = in_price_mtok
        self.out_price_mtok = out_price_mtok
        self.hysteresis_fraction = hysteresis_fraction
        self.trigger = trigger
        self.selector = selector
        self.slo_s = slo_s
        self.ttft_guard_s = ttft_guard_s
        self.first_token_overhead_s = (
            tpot_s
            if first_token_overhead_s is None
            else first_token_overhead_s
        )
        if self.first_token_overhead_s < 0:
            raise ValueError("first_token_overhead_s must be non-negative")
        self.needs_periodic_tick = trigger == "ttft_pred"
        self.seed = seed                 # retained for CLI/interface parity
        self.n_total = 0
        self.n_outsourced = 0
        self.ticks = 0
        self.kick_rounds = 0
        self.last_gap_tokens = 0.0
        self.last_release_target_tokens = 0.0
        self.last_predicted_max_ttft_s = 0.0
        self.last_post_kick_max_ttft_s = 0.0
        self.max_predicted_ttft_s_seen = 0.0
        self.max_post_kick_ttft_s_seen = 0.0
        # Runner-owned audit counters (kept here so queue_stats can report one
        # self-contained policy record).
        self.decision_calls = 0
        self.decision_total_ms = 0.0
        self.decision_max_ms = 0.0
        self.stale_decisions = 0
        self.applied_kick_rounds = 0

    def outsource(self, req: dict[str, Any]) -> bool:
        """Arrival-time hook: nimbus never outsources at arrival — every
        request enters the queue; shedding happens from the queue on ticks."""
        self.n_total += 1
        return False

    @property
    def actual_fraction(self) -> float:
        return self.n_outsourced / max(self.n_total, 1)

    def on_tick(
        self,
        waiting: list[dict[str, Any]],
        kv_headroom_tokens: float,
        inflight_remaining_tokens: float = 0.0,
        context: DecisionContext | None = None,
    ) -> list[dict[str, Any]]:
        """Choose victims using the configured trigger and selector.

        Pure w.r.t. shared state (telemetry counters only) — safe to run in a
        worker thread; the caller accounts for actually-kicked requests.
        """
        self.ticks += 1

        if self.trigger == "ttft_pred":
            if context is None:
                raise ValueError("ttft_pred requires a DecisionContext")
            self.last_gap_tokens = 0.0
            self.last_release_target_tokens = 0.0
            initial = predicted_waiting_ttfts_s(
                waiting,
                context,
                prefill_tput=self.prefill_tput,
                tpot_s=self.tpot_s,
                first_token_overhead_s=self.first_token_overhead_s,
                max_tokens_override=self.mto,
            )
            self.last_predicted_max_ttft_s = max(initial, default=0.0)
            self.max_predicted_ttft_s_seen = max(
                self.max_predicted_ttft_s_seen,
                self.last_predicted_max_ttft_s,
            )
            deadline_s = self.slo_s - self.ttft_guard_s
            if self.last_predicted_max_ttft_s <= deadline_s:
                self.last_post_kick_max_ttft_s = self.last_predicted_max_ttft_s
                self.max_post_kick_ttft_s_seen = max(
                    self.max_post_kick_ttft_s_seen,
                    self.last_post_kick_max_ttft_s,
                )
                return []

            ranked = victim_order(
                waiting,
                self.selector,
                prefill_tput=self.prefill_tput,
                tpot_s=self.tpot_s,
                in_price_mtok=self.in_price_mtok,
                out_price_mtok=self.out_price_mtok,
                seed=self.seed,
                max_tokens_override=self.mto,
            )

            # For a fixed victim ranking, removing a longer prefix cannot make
            # any remaining FCFS request later.  Binary-search the minimum
            # prefix that resolves all predicted violations.
            def safe_after(k: int) -> bool:
                victim_ids = {r.get("request_id") for r in ranked[:k]}
                survivors = [
                    r for r in waiting if r.get("request_id") not in victim_ids
                ]
                predictions = predicted_waiting_ttfts_s(
                    survivors,
                    context,
                    prefill_tput=self.prefill_tput,
                    tpot_s=self.tpot_s,
                    first_token_overhead_s=self.first_token_overhead_s,
                    max_tokens_override=self.mto,
                )
                return max(predictions, default=0.0) <= deadline_s

            lo, hi = 1, len(ranked)
            while lo < hi:
                mid = (lo + hi) // 2
                if safe_after(mid):
                    hi = mid
                else:
                    lo = mid + 1
            kicked = ranked[:lo]
            victim_ids = {r.get("request_id") for r in kicked}
            survivors = [r for r in waiting if r.get("request_id") not in victim_ids]
            post = predicted_waiting_ttfts_s(
                survivors,
                context,
                prefill_tput=self.prefill_tput,
                tpot_s=self.tpot_s,
                first_token_overhead_s=self.first_token_overhead_s,
                max_tokens_override=self.mto,
            )
            self.last_post_kick_max_ttft_s = max(post, default=0.0)
            self.max_post_kick_ttft_s_seen = max(
                self.max_post_kick_ttft_s_seen,
                self.last_post_kick_max_ttft_s,
            )
            self.kick_rounds += 1
            return kicked

        self.last_predicted_max_ttft_s = 0.0
        self.last_post_kick_max_ttft_s = 0.0
        waiting_footprint = sum(token_footprint(r, self.mto) for r in waiting)
        gap = max(
            0.0,
            waiting_footprint
            + max(0.0, inflight_remaining_tokens)
            - kv_headroom_tokens,
        )
        self.last_gap_tokens = gap
        if gap <= 0:
            self.last_release_target_tokens = 0.0
            return []

        ranked = victim_order(
            waiting,
            self.selector,
            prefill_tput=self.prefill_tput,
            tpot_s=self.tpot_s,
            in_price_mtok=self.in_price_mtok,
            out_price_mtok=self.out_price_mtok,
            seed=self.seed,
            max_tokens_override=self.mto,
        )
        release_target = gap + (
            self.hysteresis_fraction * max(0.0, kv_headroom_tokens)
        )
        self.last_release_target_tokens = release_target
        self.kick_rounds += 1
        kicked: list[dict[str, Any]] = []
        released = 0
        for victim in ranked:
            if released >= release_target:
                break
            released += token_footprint(victim, self.mto)
            kicked.append(victim)
        return kicked
