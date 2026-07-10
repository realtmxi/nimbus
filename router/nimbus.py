"""Nimbus v3: cost-aware cache-displacement shedding over the waiting queue.

The policy keeps the three physical quantities separate:

    footprint(req)    = prompt + expected decode                    [tokens]
    residence(req)    = prefill time + expected decode * TPOT       [s]
    displacement(req) = footprint * residence                      [token*s]

At each tick:

    gap = max(0, sum(waiting footprints)
                 + sum(in-flight remaining decode)
                 - KV headroom)
    release_target = gap + hysteresis * KV headroom

Requests are kicked by ascending cloud_cost / displacement until their
footprints cover the release target. Footprint decides whether the queue fits;
displacement decides whom to kick; cloud cost makes that ordering
cost-sensitive. No online knapsack is involved.

See docs/notion_algorithm_design_v3.md. The design is intentionally KV-bound:
compute/slot-bound overload requires a separate trigger.
"""
from __future__ import annotations

import math
from typing import Any

from router.common import (
    DEFAULT_KV_HYSTERESIS_FRACTION,
    effective_decode,
    local_prompt_tokens,
    token_cost_usd,
)


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
    ):
        self.prefill_tput = prefill_tput
        self.tpot_s = tpot_s
        self.mto = max_tokens_override
        self.in_price_mtok = in_price_mtok
        self.out_price_mtok = out_price_mtok
        self.hysteresis_fraction = hysteresis_fraction
        self.seed = seed                 # retained for CLI/interface parity
        self.n_total = 0
        self.n_outsourced = 0
        self.ticks = 0
        self.kick_rounds = 0
        self.last_gap_tokens = 0.0
        self.last_release_target_tokens = 0.0

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
    ) -> list[dict[str, Any]]:
        """Cover the physical KV gap with cost-aware CacheDisp shedding.

        Pure w.r.t. shared state (telemetry counters only) — safe to run in a
        worker thread; the caller accounts for actually-kicked requests.
        """
        self.ticks += 1
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

        release_target = gap + (
            self.hysteresis_fraction * max(0.0, kv_headroom_tokens)
        )
        self.last_release_target_tokens = release_target
        self.kick_rounds += 1

        by_score = sorted(
            waiting,
            key=lambda r: shedding_score(
                r,
                self.prefill_tput,
                self.tpot_s,
                self.in_price_mtok,
                self.out_price_mtok,
                self.mto,
            ),
        )
        kicked: list[dict[str, Any]] = []
        released = 0
        for victim in by_score:
            if released >= release_target:
                break
            released += token_footprint(victim, self.mto)
            kicked.append(victim)
        return kicked
