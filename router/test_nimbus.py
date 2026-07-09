"""Unit tests for router/nimbus.py + its integration hook. No network/GPU.

Run from the repo root:  python3 -m unittest router.test_nimbus -v
"""
from __future__ import annotations

import asyncio
import itertools
import random
import unittest

from router.common import Endpoint, NullCloud
from router.nimbus import (
    NimbusPolicy,
    displacement_token_s,
    solve_knapsack,
    token_footprint,
    value_usd,
)
from router.run import KVMonitor, replay_queued
from router.test_run import LOCAL, CLOUD, RecordingSender, mk_trace


def brute_force_keep(items, budget):
    """Exact reference: max Σvalue s.t. Σweight <= budget (n <= ~12)."""
    best_v, best_set = 0.0, set()
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            w = sum(c[1] for c in combo)
            v = sum(c[2] for c in combo)
            if w <= budget and v > best_v:
                best_v, best_set = v, {c[0] for c in combo}
    return best_v, best_set


class TestKnapsack(unittest.TestCase):
    def test_matches_brute_force_on_random_instances(self):
        rng = random.Random(7)
        for trial in range(30):
            n = rng.randint(1, 10)
            items = [(f"r{i}", rng.randint(1, 400), rng.uniform(0.001, 1.0))
                     for i in range(n)]
            budget = rng.randint(1, 1200)   # <= scale_to=2048 -> DP is exact
            keep = solve_knapsack(items, budget)
            got_v = sum(v for (i, w, v) in items if i in keep)
            got_w = sum(w for (i, w, v) in items if i in keep)
            best_v, _ = brute_force_keep(items, budget)
            self.assertLessEqual(got_w, budget)                    # feasible
            self.assertAlmostEqual(got_v, best_v, places=9)        # optimal

    def test_scaled_never_exceeds_budget(self):
        rng = random.Random(11)
        for trial in range(20):
            items = [(f"r{i}", rng.randint(100, 60000), rng.uniform(0.001, 1.0))
                     for i in range(rng.randint(5, 60))]
            budget = rng.randint(10_000, 300_000)                  # forces scaling
            keep = solve_knapsack(items, budget)
            self.assertLessEqual(sum(w for (i, w, v) in items if i in keep), budget)

    def test_edge_cases(self):
        self.assertEqual(solve_knapsack([], 100), set())
        self.assertEqual(solve_knapsack([("a", 10, 1.0)], 0), set())
        self.assertEqual(solve_knapsack([("a", 200, 1.0)], 100), set())  # oversized
        self.assertEqual(solve_knapsack([("a", 10, 1.0)], 100), {"a"})


class TestFormulas(unittest.TestCase):
    REQ = {"request_id": 1, "prompt_tokens": 1000, "max_tokens": 200}

    def test_hand_computed(self):
        self.assertEqual(token_footprint(self.REQ), 1200)
        # displacement = prompt * (prompt/tput + decode*tpot) = 1000*(1000/10000 + 200*0.01)
        self.assertAlmostEqual(
            displacement_token_s(self.REQ, prefill_tput=10000, tpot_s=0.01),
            1000 * (0.1 + 2.0))
        self.assertAlmostEqual(value_usd(self.REQ, 0.15, 1.20),
                               (1000 * 0.15 + 200 * 1.20) / 1e6)


def mk_policy(**kw):
    return NimbusPolicy(prefill_tput=20000, tpot_s=0.0095, **kw)


class TestOnTick(unittest.TestCase):
    def req(self, rid, prompt, decode):
        return {"request_id": rid, "prompt_tokens": prompt, "max_tokens": decode}

    def test_no_pressure_no_kicks(self):
        pol = mk_policy()
        waiting = [self.req(i, 500, 100) for i in range(5)]   # footprint 3000
        self.assertEqual(pol.on_tick(waiting, kv_available_tokens=10_000), [])
        self.assertEqual(pol.n_outsourced, 0)

    def test_kicks_until_fit_and_terminates(self):
        pol = mk_policy()
        waiting = [self.req(i, 1000, 200) for i in range(10)]  # 10 x 1200 = 12000
        kicked = pol.on_tick(waiting, kv_available_tokens=5000)
        remaining = [r for r in waiting if r not in kicked]
        self.assertLessEqual(sum(token_footprint(r) for r in remaining), 5000)
        self.assertGreater(len(kicked), 0)

    def test_kicks_largest_displacement_first(self):
        """The CacheDisp core (Murphy 2026-07-09): shed the request that pins
        the most cache-time, regardless of its API price."""
        pol = mk_policy()
        a = self.req("A", 2900, 100)    # displacement ~3,176 token·s
        b = self.req("B", 1000, 2000)   # displacement ~19,050 token·s -> kicked
        kicked = pol.on_tick([a, b], kv_available_tokens=3000)   # only one fits
        self.assertEqual([r["request_id"] for r in kicked], ["B"])
        kicked2 = pol.on_tick([a, b], kv_available_tokens=100)   # neither fits
        self.assertEqual([r["request_id"] for r in kicked2], ["B", "A"])

    def test_arrival_hook_never_outsources(self):
        pol = mk_policy()
        self.assertFalse(any(pol.outsource(self.req(i, 10, 10)) for i in range(50)))
        self.assertEqual(pol.actual_fraction, 0.0)


class TestReviewRegressions(unittest.TestCase):
    def test_scaling_boundary_item_stays_candidate(self):
        """Regression (review): an item that fits alone must never be dropped
        by ceil-scaling vs floor cap (budget=4097 -> scale=3, cap=1365,
        ceil(4097/3)=1366 used to be filtered out)."""
        a = ("A", 4097, 600e-6)
        b = ("B", 100, 75e-6)
        keep = solve_knapsack([a, b], budget=4097)
        self.assertEqual(keep, {"A"})            # optimal: keep the big one

    def test_matches_literal_kick_one_recheck(self):
        """Batched shedding == literal kick-max-displacement-then-recheck."""
        pol = mk_policy()
        rng = random.Random(3)
        waiting = [{"request_id": i, "prompt_tokens": rng.randint(50, 3000),
                    "max_tokens": rng.randint(10, 800)} for i in range(40)]
        budget = 20_000
        kicked = [r["request_id"] for r in pol.on_tick(list(waiting), budget)]
        remaining = list(waiting)
        ref = []
        while remaining and sum(token_footprint(r) for r in remaining) > budget:
            victim = max(remaining,
                         key=lambda r: displacement_token_s(r, 20000, 0.0095))
            remaining.remove(victim)
            ref.append(victim["request_id"])
        self.assertEqual(kicked, ref)

    def test_kicked_real_cloud_ttft_includes_queue_wait(self):
        """Regression (review): a kicked request served by REAL cloud must fold
        its queue wait into ttft_ms, like the local path."""
        import tempfile
        from pathlib import Path
        from router.run import parse_args, replay_queued
        from router.test_run import RecordingSender

        class SlowLocal(RecordingSender):
            def __init__(self):
                super().__init__(service_s=0.10)

        async def cloud_sender(req, due):
            return {"request_id": req["request_id"], "arrived_at": req["arrived_at"],
                    "relative_arrival_s": req["relative_arrival_s"],
                    "scheduled_lag_ms": 0.0, "endpoint": "cloud", "model": "m",
                    "success": True, "error": None, "error_type": None,
                    "http_status": 200, "ttft_ms": 300.0, "e2e_ms": 400.0,
                    "tpot_ms": 1.0, "chunks": 2, "prompt_tokens": 10,
                    "completion_tokens": 5, "output_chars": 4, "cost_usd": 0.0001}

        out_dir = Path(tempfile.mkdtemp())
        args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "nimbus",
                           "--local-url", "http://x", "--local-model", "m",
                           "--cloud", "real", "--cloud-url", "http://c",
                           "--cloud-model", "m",
                           "--kv-capacity-tokens", "1000000", "--max-inflight", "1",
                           "--out-dir", str(out_dir)])
        results, _, _ = asyncio.run(replay_queued(
            args, mk_trace(10), mk_policy(), LOCAL, CLOUD,
            sink=None, send_local=SlowLocal(), send_cloud=cloud_sender,
            kv_monitor=FakeKV(200)))    # tiny budget -> waiting reqs get kicked
        cloud = [r for r in results if r["endpoint"] == "cloud"]
        self.assertGreater(len(cloud), 0)
        for r in cloud:
            self.assertEqual(r["service_ttft_ms"], 300.0)
            self.assertAlmostEqual(
                r["ttft_ms"], r["queue_delay_ms"] + 300.0, places=6)
            self.assertGreater(r["queue_delay_ms"], 0.0)   # it actually waited


class TestCodexRegressions(unittest.TestCase):
    def test_kv_zero_kicks_even_with_free_slots(self):
        """Regression (codex P1): the kick check must run BEFORE dispatch —
        with 128 free slots but ZERO KV available, nothing may enter local."""
        results, pol, sender = run_nimbus2(mk_trace(20), kv_avail=0, max_inflight=128)
        cloud = [r for r in results if r["endpoint"] == "cloud"]
        self.assertEqual(len(cloud), 20)                       # all shed
        self.assertEqual(len(sender.dispatch_order), 0)        # engine untouched
        self.assertGreater(pol.ticks, 0)

    def test_max_tokens_override_changes_policy_arithmetic(self):
        """Regression (codex P2): footprint/value/displacement must use the
        effective decode (--max-tokens replaces trace), like payload & billing."""
        from router.nimbus import token_footprint, value_usd, displacement_token_s
        req = {"request_id": 1, "prompt_tokens": 100, "max_tokens": 1000}
        self.assertEqual(token_footprint(req), 1100)
        self.assertEqual(token_footprint(req, max_tokens_override=1), 101)
        self.assertAlmostEqual(value_usd(req, 0.15, 1.20, max_tokens_override=1),
                               (100 * 0.15 + 1 * 1.20) / 1e6)
        self.assertLess(displacement_token_s(req, 20000, 0.0095, max_tokens_override=1),
                        displacement_token_s(req, 20000, 0.0095))
        # policy-level: with override=1 the 20-req queue fits into 3000 tokens
        pol = mk_policy(max_tokens_override=1)                 # footprint 101 each
        waiting = [dict(req, request_id=i) for i in range(20)]
        self.assertEqual(pol.on_tick(waiting, kv_available_tokens=3000), [])
        pol2 = mk_policy()                                     # footprint 1100 each
        self.assertGreater(len(pol2.on_tick(waiting, kv_available_tokens=3000)), 0)

    def test_kicked_real_cloud_sender_exception_records_failure(self):
        """Regression (codex P2): a raising cloud sender must still produce a
        failure row for every kicked request (no lost results)."""
        import tempfile
        from pathlib import Path
        from router.run import parse_args, replay_queued
        from router.test_run import RecordingSender

        async def boom_cloud(req, due):
            raise RuntimeError("cloud down")

        out_dir = Path(tempfile.mkdtemp())
        args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "nimbus",
                           "--local-url", "http://x", "--local-model", "m",
                           "--cloud", "real", "--cloud-url", "http://c",
                           "--cloud-model", "m",
                           "--kv-capacity-tokens", "1000000", "--max-inflight", "2",
                           "--out-dir", str(out_dir)])
        results, _, _ = asyncio.run(replay_queued(
            args, mk_trace(10), mk_policy(), LOCAL, CLOUD,
            sink=None, send_local=RecordingSender(service_s=0.05),
            send_cloud=boom_cloud, kv_monitor=FakeKV(200)))
        self.assertEqual(len(results), 10)                     # nothing lost
        cloud = [r for r in results if r["endpoint"] == "cloud"]
        self.assertGreater(len(cloud), 0)
        for r in cloud:
            self.assertFalse(r["success"])
            self.assertEqual(r["error_type"], "RuntimeError")
            self.assertEqual(r["cost_usd"], 0.0)

    def test_bad_nimbus_args_rejected(self):
        """Regression (codex P3): reject nonsense constants at parse time."""
        from router.run import parse_args
        base = ["--data", "t", "--scenario", "normal", "--policy", "nimbus",
                "--local-url", "http://x", "--local-model", "m",
                "--kv-capacity-tokens", "1000"]
        for bad in (["--prefill-tput", "0"], ["--tpot-ms", "-1"],
                    ["--in-price", "-0.1"], ["--cloud-max-concurrency", "-1"],
                    ["--slo-s", "0"]):
            with self.assertRaises(SystemExit, msg=bad):
                parse_args(base + bad)


class FakeKV:
    """Injectable monitor: fixed available tokens."""
    def __init__(self, avail):
        self.avail = avail
        self.read_failures = 0
    async def available_tokens(self):
        return self.avail


def run_nimbus2(trace, kv_avail, max_inflight=2, sender=None):
    from router.run import parse_args
    import tempfile
    from pathlib import Path
    out_dir = Path(tempfile.mkdtemp())
    args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "nimbus",
                       "--local-url", "http://x", "--local-model", "m",
                       "--kv-capacity-tokens", "1000000",
                       "--max-inflight", str(max_inflight),
                       "--out-dir", str(out_dir)])
    pol = mk_policy()
    sender = sender or RecordingSender()
    results, admission, _ = asyncio.run(replay_queued(
        args, trace, pol, LOCAL, CLOUD, sink=NullCloud(CLOUD),
        send_local=sender, kv_monitor=FakeKV(kv_avail)))
    return results, pol, sender


def run_nimbus(trace, kv_avail, sender=None):
    from router.run import parse_args
    import tempfile
    from pathlib import Path
    out_dir = Path(tempfile.mkdtemp())
    args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "nimbus",
                       "--local-url", "http://x", "--local-model", "m",
                       "--kv-capacity-tokens", "1000000",
                       "--max-inflight", "2",   # force queue formation at t=0 burst
                       "--out-dir", str(out_dir)])
    pol = mk_policy()
    sender = sender or RecordingSender()
    results, admission, _ = asyncio.run(replay_queued(
        args, trace, pol, LOCAL, CLOUD, sink=NullCloud(CLOUD),
        send_local=sender, kv_monitor=FakeKV(kv_avail)))
    return results, pol, sender


class TestIntegration(unittest.TestCase):
    def test_no_pressure_equals_all_local(self):
        results, pol, sender = run_nimbus(mk_trace(20), kv_avail=1e9)
        self.assertEqual(len(results), 20)
        self.assertEqual(pol.n_outsourced, 0)
        self.assertEqual(len(sender.dispatch_order), 20)      # everything local
        self.assertTrue(all(r["endpoint"] == "local" for r in results))

    def test_pressure_kicks_to_cloud_with_wait_accounting(self):
        # max_inflight=2 -> 28 of 30 queue at t=0; their footprint 28x150=4200
        # exceeds kv_avail 600 -> nimbus sheds down to <= 4 queued (600/150)
        trace = mk_trace(30)                                   # prompt 100 + max 50
        results, pol, sender = run_nimbus(trace, kv_avail=600)
        cloud = [r for r in results if r["endpoint"] == "cloud"]
        local = [r for r in results if r["endpoint"] == "local"]
        self.assertEqual(len(cloud) + len(local), 30)
        self.assertGreater(len(cloud), 0)                      # kicks happened
        self.assertEqual(pol.n_outsourced, len(cloud))
        for r in cloud:
            self.assertTrue(r["routed_only"])                  # via null sink
            self.assertGreaterEqual(r["queue_delay_ms"], 0.0)  # waited-then-kicked
        # the survivors' footprint respects the budget at each decision point;
        # at least verify SOME requests stayed local under a 600-token budget
        self.assertGreater(len(local), 0)


class TestKVMonitor(unittest.TestCase):
    def test_parses_gauge_and_caches(self):
        class FakeResp:
            def __init__(self, text): self._t = text
            async def text(self): return self._t
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
        class FakeSession:
            def __init__(self): self.calls = 0
            def get(self, url):
                self.calls += 1
                return FakeResp('some_other 1.0\nvllm:kv_cache_usage_perc{engine="0"} 0.25\n')
        s = FakeSession()
        mon = KVMonitor(s, "http://x/metrics", capacity_tokens=1000, ttl_s=60)
        self.assertAlmostEqual(asyncio.run(mon.available_tokens()), 750.0)
        asyncio.run(mon.available_tokens())                    # within ttl
        self.assertEqual(s.calls, 1)                           # cached

    def test_failure_reuses_last_value(self):
        class BoomSession:
            def get(self, url): raise RuntimeError("down")
        mon = KVMonitor(BoomSession(), "http://x/metrics", capacity_tokens=1000, ttl_s=0)
        self.assertEqual(asyncio.run(mon.available_tokens()), 1000)   # optimistic init
        self.assertEqual(mon.read_failures, 1)


if __name__ == "__main__":
    unittest.main()
