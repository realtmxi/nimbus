"""Nimbus policy: knapsack-based shedding over the waiting queue.

The written algorithm (Notion "Algorithm Design"), with the budget-unit fix
decided 2026-07 (Murphy): the knapsack CAPACITY is physical — KV tokens — while
cache displacement (token·seconds) enters only as the KICK PRIORITY. This
answers the doc's open question ("if weight unit is token·seconds, what is the
new budget?"): weight and budget answer different questions and need not share
a unit.

    while Σ token_footprint(waiting) > kv_available_tokens:      # fits-in-KV trigger
        keep = knapsack(items, weight=token_footprint, value=$saved,
                        budget=kv_available_tokens)
        kick the request NOT kept with the worst $value per token·second of
        displacement (cheapest to buy back per unit of local cache-time freed)

Kicked requests go to the cloud sink; everything else stays queued for the
work-conserving dispatcher. No pressure (everything fits) -> zero kicks -> the
policy degrades to all_local.

Definitions per request (from trace fields):
  token_footprint = prompt_tokens + max_tokens        [tokens]   (KV upper bound)
  displacement    = prompt_tokens * residence_s       [token·s]
      residence_s = prompt_tokens/prefill_tput + max_tokens * tpot_s
  value           = prompt*in_price + max_tokens*out_price   [$ saved if local]
"""
from __future__ import annotations

import math
import random
from typing import Any


def token_footprint(req: dict[str, Any]) -> int:
    """KV occupancy upper bound in tokens: full prompt + full decode."""
    return int(req.get("prompt_tokens") or 0) + int(req["max_tokens"])


def displacement_token_s(req: dict[str, Any], prefill_tput: float, tpot_s: float) -> float:
    """Cache displacement: KV footprint (tokens) x residence time (s).

    Uses the prompt as the footprint (the decode tail grows gradually and is
    small for most requests) and a two-term residence estimate. Constants are
    CLI-tunable; only the RELATIVE ordering matters for kick priority.
    """
    prompt = int(req.get("prompt_tokens") or 0)
    residence_s = prompt / max(prefill_tput, 1e-9) + int(req["max_tokens"]) * tpot_s
    return max(prompt, 1) * residence_s


def value_usd(req: dict[str, Any], in_price_mtok: float, out_price_mtok: float) -> float:
    """API $ saved by serving this request locally (= cost if outsourced)."""
    prompt = int(req.get("prompt_tokens") or 0)
    return (prompt * in_price_mtok + int(req["max_tokens"]) * out_price_mtok) / 1e6


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

    def __init__(self, in_price_mtok: float, out_price_mtok: float,
                 prefill_tput: float, tpot_s: float, seed: int = 0):
        self.in_price = in_price_mtok
        self.out_price = out_price_mtok
        self.prefill_tput = prefill_tput
        self.tpot_s = tpot_s
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
        """Decide which waiting requests to kick to the cloud right now.

        Semantics = the Notion iterative loop (solve knapsack, kick worst
        $/displacement from the out-set, recheck), computed efficiently: on a
        static snapshot with a fixed budget, removing an out-set member leaves
        the kept set optimal, so one solve per round suffices — victims are the
        out-set in ascending density order, taken until the remainder fits.
        The outer while re-solves only if scaling slack leaves the remainder
        still over budget after the whole out-set is gone (rare).

        NOTE: this method is pure w.r.t. shared state (no counters beyond
        telemetry) and is safe to run in a worker thread; the caller accounts
        for actually-kicked requests (a victim may have been dispatched while
        this computed).
        """
        self.ticks += 1
        remaining = list(waiting)
        kicked: list[dict[str, Any]] = []

        def fits() -> bool:
            return sum(token_footprint(r) for r in remaining) <= kv_available_tokens

        def density(r: dict[str, Any]) -> float:
            return (value_usd(r, self.in_price, self.out_price)
                    / max(displacement_token_s(r, self.prefill_tput, self.tpot_s), 1e-9))

        while remaining and not fits():
            self.kick_rounds += 1
            items = [(str(r["request_id"]), token_footprint(r),
                      value_usd(r, self.in_price, self.out_price)) for r in remaining]
            keep = solve_knapsack(items, int(kv_available_tokens))
            out = [r for r in remaining if str(r["request_id"]) not in keep]
            if not out:                 # degenerate (e.g. budget <= 0 kept nothing)
                out = remaining
            for victim in sorted(out, key=density):   # worst density first
                if fits():
                    break
                remaining.remove(victim)
                kicked.append(victim)
        return kicked
