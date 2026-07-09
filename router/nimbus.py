"""Nimbus policy: cache-displacement shedding over the waiting queue.

The core rule (Notion "Algorithm Design", reaffirmed by Murphy 2026-07-09 after
an implementation drift was caught in review):

    displacement(req) = KV footprint x residence time            [token*s]
                      ~ prompt_tokens x (prefill_time + decode_tokens x TPOT)

    while Σ token_footprint(waiting) > kv_available_tokens:      # fits-in-KV trigger
        kick the request with the LARGEST displacement

It answers: how much KV does this request pin, and for how long? Shedding the
largest displacers frees the most cache-time per kick. API cost does NOT enter
the shed rule — it belongs to the frontier comparison and to the planned
cost-aware knapsack ablations (Murphy's objectives: minimize cloud cost s.t.
enough displacement released / maximize released displacement s.t. cloud
budget), for which solve_knapsack/value_usd below are retained.

History note: between 2026-07-08 and 07-09 this file briefly implemented
"maximize kept API value under a token budget" — that was drift introduced
during the budget-unit fix, not the design; reverted.

Trigger/capacity stays physical (tokens, decision 2026-07: budget = K_avail
from real /metrics). No pressure -> zero kicks -> degrades to all_local.
"""
from __future__ import annotations

import math
import random
from typing import Any


def effective_decode(req: dict[str, Any], max_tokens_override: int | None = None) -> int:
    """Decode length as the payload will actually request it: --max-tokens
    REPLACES the trace value (same semantics as make_payload/NullCloud)."""
    return int(max_tokens_override) if max_tokens_override is not None else int(req["max_tokens"])


def token_footprint(req: dict[str, Any], max_tokens_override: int | None = None) -> int:
    """KV occupancy upper bound in tokens: full prompt + full decode."""
    return int(req.get("prompt_tokens") or 0) + effective_decode(req, max_tokens_override)


def displacement_token_s(req: dict[str, Any], prefill_tput: float, tpot_s: float,
                         max_tokens_override: int | None = None) -> float:
    """Cache displacement: KV footprint (tokens) x residence time (s).

    Uses the prompt as the footprint (the decode tail grows gradually and is
    small for most requests) and a two-term residence estimate. Constants are
    CLI-tunable; only the RELATIVE ordering matters for kick priority.
    """
    prompt = int(req.get("prompt_tokens") or 0)
    residence_s = (prompt / max(prefill_tput, 1e-9)
                   + effective_decode(req, max_tokens_override) * tpot_s)
    return max(prompt, 1) * residence_s


def value_usd(req: dict[str, Any], in_price_mtok: float, out_price_mtok: float,
              max_tokens_override: int | None = None) -> float:
    """API $ saved by serving this request locally (= cost if outsourced)."""
    prompt = int(req.get("prompt_tokens") or 0)
    return (prompt * in_price_mtok
            + effective_decode(req, max_tokens_override) * out_price_mtok) / 1e6


def solve_knapsack(
    items: list[tuple[str, int, float]],   # (id, weight_tokens, value_usd)
    budget: int,
    scale_to: int = 2048,
) -> set[str]:
    """0/1 knapsack: maximize kept value s.t. Σweight <= budget. Returns KEEP ids.

    Weights are scaled (ceil) down to <= scale_to buckets before the DP, so the
    kept set NEVER exceeds the true budget (ceil only over-counts weights).
    With budget <= scale_to the solution is exact; tests cross-check against
    brute force on small instances.
    """
    if budget <= 0 or not items:
        return set()
    feasible = [(i, w, v) for (i, w, v) in items if w <= budget]
    if not feasible:
        return set()

    scale = max(1, math.ceil(budget / scale_to))
    cap = budget // scale
    if cap <= 0:
        return set()
    # ceil-scaled weight, CLAMPED to cap: an item that truly fits (w <= budget)
    # must stay a DP candidate even when ceil pushes it past the floor-rounded
    # cap. Feasibility is preserved: a clamped item occupies the whole cap, so
    # the DP can only ever select it alone, and alone it fits by construction.
    scaled = [(i, min(max(1, math.ceil(w / scale)), cap), v) for (i, w, v) in feasible]

    n = len(scaled)
    # dp[w] = best value at exactly-or-under weight w; choice bits for reconstruction
    dp = [0.0] * (cap + 1)
    take = [[False] * (cap + 1) for _ in range(n)]
    for k, (_, w, v) in enumerate(scaled):
        for c in range(cap, w - 1, -1):
            cand = dp[c - w] + v
            if cand > dp[c]:
                dp[c] = cand
                take[k][c] = True
    keep: set[str] = set()
    c = cap
    for k in range(n - 1, -1, -1):
        if take[k][c]:
            keep.add(scaled[k][0])
            c -= scaled[k][1]
    return keep


class NimbusPolicy:
    """Queue-level shedding policy. Interface-compatible with common.Policy
    (outsource/actual_fraction) plus the on_tick hook the runner calls."""

    name = "nimbus"
    p = None   # no target fraction: the policy self-selects its split

    def __init__(self, prefill_tput: float, tpot_s: float, seed: int = 0,
                 max_tokens_override: int | None = None):
        self.prefill_tput = prefill_tput
        self.tpot_s = tpot_s
        self.mto = max_tokens_override
        self.rng = random.Random(seed)   # unused; parity with Policy
        self.n_total = 0
        self.n_outsourced = 0
        self.ticks = 0
        self.kick_rounds = 0

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
        kv_available_tokens: float,
    ) -> list[dict[str, Any]]:
        """Kick the largest cache-displacers until the waiting set fits KV.

        Pure w.r.t. shared state (telemetry counters only) — safe to run in a
        worker thread; the caller accounts for actually-kicked requests (a
        victim may have been dispatched while this computed).
        """
        self.ticks += 1
        total = sum(token_footprint(r, self.mto) for r in waiting)
        if total <= kv_available_tokens:
            return []
        self.kick_rounds += 1
        by_displacement = sorted(
            waiting,
            key=lambda r: displacement_token_s(r, self.prefill_tput, self.tpot_s,
                                               self.mto),
            reverse=True)                       # largest displacer first
        kicked: list[dict[str, Any]] = []
        for victim in by_displacement:
            if total <= kv_available_tokens:
                break
            total -= token_footprint(victim, self.mto)
            kicked.append(victim)
        return kicked
