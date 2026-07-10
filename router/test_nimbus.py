"""Unit tests for router/nimbus.py + its integration hook. No network/GPU.

Run from the repo root:  python3 -m unittest router.test_nimbus -v
"""
from __future__ import annotations

import asyncio
import random
import unittest

from router.common import DEFAULT_KV_HYSTERESIS_FRACTION, Endpoint, NullCloud
from router.nimbus import (
    NimbusPolicy,
    cloud_cost_usd,
    displacement_token_s,
    shedding_score,
    token_footprint,
)
from router.run import KVMonitor, _InflightKVTracker, replay_queued
from router.test_run import LOCAL, CLOUD, RecordingSender, mk_trace


class TestFormulas(unittest.TestCase):
    REQ = {"request_id": 1, "prompt_tokens": 1000, "max_tokens": 200}

    def test_hand_computed(self):
        self.assertEqual(token_footprint(self.REQ), 1200)
        # displacement = footprint * residence = 1200*(1000/10000 + 200*0.01)
        self.assertAlmostEqual(
            displacement_token_s(self.REQ, prefill_tput=10000, tpot_s=0.01),
            1200 * (0.1 + 2.0))
        self.assertAlmostEqual(cloud_cost_usd(self.REQ, 0.15, 1.20),
                               (1000 * 0.15 + 200 * 1.20) / 1e6)

    def test_prefix_aware_footprint_is_local_but_cloud_cost_uses_full_prompt(self):
        req = dict(self.REQ, uncached_prompt_tokens=100)
        self.assertEqual(token_footprint(req), 300)
        self.assertAlmostEqual(
            displacement_token_s(req, prefill_tput=10000, tpot_s=0.01),
            300 * (0.01 + 2.0))
        self.assertAlmostEqual(cloud_cost_usd(req, 0.15, 1.20),
                               (1000 * 0.15 + 200 * 1.20) / 1e6)


class TestInflightKVTracker(unittest.TestCase):
    def test_exact_mtp_progress_preserves_peak_commitment(self):
        tracker = _InflightKVTracker(max_tokens_override=None)
        req = {
            "request_id": "mtp",
            "prompt_tokens": 100,
            "uncached_prompt_tokens": 20,
            "max_tokens": 50,
        }
        tracker.add(req)
        self.assertEqual(tracker.predicted_current_tokens, 20)
        self.assertEqual(tracker.remaining_decode_tokens, 50)

        # One content delta may contain seven MTP-accepted tokens. The exact
        # cumulative usage, not the one chunk, advances current KV by seven.
        tracker.update_generated("mtp", 7)
        self.assertEqual(tracker.predicted_current_tokens, 27)
        self.assertEqual(tracker.remaining_decode_tokens, 43)
        self.assertEqual(
            tracker.predicted_current_tokens + tracker.remaining_decode_tokens,
            70,
        )

        tracker.update_generated("mtp", 5)  # out-of-order progress cannot regress
        self.assertEqual(tracker.remaining_decode_tokens, 43)
        tracker.remove("mtp")
        self.assertEqual(tracker.predicted_current_tokens, 0)
        self.assertEqual(tracker.remaining_decode_tokens, 0)


def mk_policy(**kw):
    return NimbusPolicy(prefill_tput=20000, tpot_s=0.0095, **kw)


class TestOnTick(unittest.TestCase):
    def req(self, rid, prompt, decode):
        return {"request_id": rid, "prompt_tokens": prompt, "max_tokens": decode}

    def test_no_pressure_no_kicks(self):
        pol = mk_policy()
        waiting = [self.req(i, 500, 100) for i in range(5)]   # footprint 3000
        self.assertEqual(pol.on_tick(waiting, kv_headroom_tokens=10_000), [])
        self.assertEqual(pol.n_outsourced, 0)

    def test_kicks_until_fit_and_terminates(self):
        pol = mk_policy()
        waiting = [self.req(i, 1000, 200) for i in range(10)]  # 10 x 1200 = 12000
        kicked = pol.on_tick(waiting, kv_headroom_tokens=5000)
        remaining = [r for r in waiting if r not in kicked]
        self.assertLessEqual(sum(token_footprint(r) for r in remaining), 5000)
        self.assertGreater(len(kicked), 0)

    def test_kicks_lowest_cost_per_displacement_first(self):
        """v3 is cost-aware, so max displacement alone does not choose."""
        pol = mk_policy(hysteresis_fraction=0)
        cheap_relief = self.req("A", 100, 500)       # lower displacement, lower score
        max_disp = self.req("B", 10_000, 10)         # larger displacement, higher score
        self.assertLess(
            shedding_score(cheap_relief, 20_000, 0.0095, 0.15, 1.20),
            shedding_score(max_disp, 20_000, 0.0095, 0.15, 1.20),
        )
        self.assertGreater(
            displacement_token_s(max_disp, 20_000, 0.0095),
            displacement_token_s(cheap_relief, 20_000, 0.0095),
        )
        kicked = pol.on_tick(
            [cheap_relief, max_disp], kv_headroom_tokens=10_010
        )
        self.assertEqual([r["request_id"] for r in kicked], ["A"])

    def test_inflight_growth_is_part_of_gap(self):
        pol = mk_policy(hysteresis_fraction=0)
        waiting = [self.req("waiting", 80, 20)]       # footprint 100
        self.assertEqual(pol.on_tick(waiting, 150, inflight_remaining_tokens=0), [])
        kicked = pol.on_tick(waiting, 150, inflight_remaining_tokens=60)
        self.assertEqual([r["request_id"] for r in kicked], ["waiting"])
        self.assertEqual(pol.last_gap_tokens, 10)

    def test_hysteresis_adds_headroom_margin(self):
        pol = mk_policy(hysteresis_fraction=0.05)
        waiting = [self.req(i, 80, 20) for i in range(3)]  # total footprint 300
        kicked = pol.on_tick(waiting, kv_headroom_tokens=250)
        self.assertEqual(pol.last_gap_tokens, 50)
        self.assertEqual(pol.last_release_target_tokens, 62.5)
        self.assertEqual(len(kicked), 1)                    # one 100-token item covers it

    def test_arrival_hook_never_outsources(self):
        pol = mk_policy()
        self.assertFalse(any(pol.outsource(self.req(i, 10, 10)) for i in range(50)))
        self.assertEqual(pol.actual_fraction, 0.0)


class TestReviewRegressions(unittest.TestCase):
    def test_matches_literal_v3_density_cover(self):
        """Policy output equals v3 score ordering until release_target is met."""
        pol = mk_policy()
        rng = random.Random(3)
        waiting = [{"request_id": i, "prompt_tokens": rng.randint(50, 3000),
                    "max_tokens": rng.randint(10, 800)} for i in range(40)]
        headroom = 20_000
        kicked = [r["request_id"] for r in pol.on_tick(list(waiting), headroom)]
        total = sum(token_footprint(r) for r in waiting)
        target = max(0, total - headroom) + 0.05 * headroom
        ordered = sorted(
            waiting,
            key=lambda r: shedding_score(r, 20_000, 0.0095, 0.15, 1.20),
        )
        ref = []
        released = 0
        for victim in ordered:
            if released >= target:
                break
            released += token_footprint(victim)
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
        """Regression (codex P2): footprint/cost/displacement must use the
        effective decode (--max-tokens replaces trace), like payload & billing."""
        from router.nimbus import cloud_cost_usd, token_footprint, displacement_token_s
        req = {"request_id": 1, "prompt_tokens": 100, "max_tokens": 1000}
        self.assertEqual(token_footprint(req), 1100)
        self.assertEqual(token_footprint(req, max_tokens_override=1), 101)
        self.assertAlmostEqual(cloud_cost_usd(req, 0.15, 1.20, max_tokens_override=1),
                               (100 * 0.15 + 1 * 1.20) / 1e6)
        self.assertLess(displacement_token_s(req, 20000, 0.0095, max_tokens_override=1),
                        displacement_token_s(req, 20000, 0.0095))
        # policy-level: with override=1 the 20-req queue fits into 3000 tokens
        pol = mk_policy(max_tokens_override=1)                 # footprint 101 each
        waiting = [dict(req, request_id=i) for i in range(20)]
        self.assertEqual(pol.on_tick(waiting, kv_headroom_tokens=3000), [])
        pol2 = mk_policy()                                     # footprint 1100 each
        self.assertGreater(len(pol2.on_tick(waiting, kv_headroom_tokens=3000)), 0)

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
        self.assertEqual(
            parse_args(base).kv_hysteresis_fraction,
            DEFAULT_KV_HYSTERESIS_FRACTION,
        )
        self.assertEqual(
            mk_policy().hysteresis_fraction,
            DEFAULT_KV_HYSTERESIS_FRACTION,
        )
        for bad in (["--prefill-tput", "0"], ["--tpot-ms", "-1"],
                    ["--in-price", "-0.1"], ["--cloud-max-concurrency", "-1"],
                    ["--slo-s", "0"], ["--kv-hysteresis-fraction", "1"]):
            with self.assertRaises(SystemExit, msg=bad):
                parse_args(base + bad)


class TestTickDispatchRace(unittest.TestCase):
    def test_completion_during_slow_tick_cannot_dispatch_the_victim(self):
        """Regression (codex round-3 P1): while a shed decision computes
        off-thread, a completing request must NOT admit the victim locally.
        Reproduction: r1 (huge displacer, queue head) is about to be kicked by
        a slow tick; r0 completes mid-tick; old code dispatched r1 local."""
        import time as _time
        from router.nimbus import NimbusPolicy

        class SleepyPolicy(NimbusPolicy):
            def on_tick(self, waiting, kv, inflight_remaining=0):
                if len(waiting) >= 2:
                    _time.sleep(0.12)          # slow decision window
                return super().on_tick(waiting, kv, inflight_remaining)

        # r0 small (dispatches first, completes during the slow tick),
        # r1 HUGE displacer at the queue head (the victim),
        # r2 small (should be the only other local request)
        trace = [
            {"request_id": 0, "arrived_at": 1477007, "relative_arrival_s": 0.0,
             "prompt": "r0", "max_tokens": 10, "prompt_tokens": 40, "session_id": 0},
            {"request_id": 1, "arrived_at": 1477007, "relative_arrival_s": 0.0,
             "prompt": "r1", "max_tokens": 2000, "prompt_tokens": 100, "session_id": 0},
            {"request_id": 2, "arrived_at": 1477007, "relative_arrival_s": 0.0,
             "prompt": "r2", "max_tokens": 10, "prompt_tokens": 40, "session_id": 0},
        ]
        import tempfile
        from pathlib import Path
        from router.run import parse_args, replay_queued
        out_dir = Path(tempfile.mkdtemp())
        args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "nimbus",
                           "--local-url", "http://x", "--local-model", "m",
                           "--kv-capacity-tokens", "1000000", "--max-inflight", "1",
                           "--out-dir", str(out_dir)])
        pol = SleepyPolicy(prefill_tput=20000, tpot_s=0.0095)
        sender = RecordingSender(service_s=0.05)   # r0 completes inside the 0.12s tick
        results, _, _ = asyncio.run(replay_queued(
            args, trace, pol, LOCAL, CLOUD, sink=NullCloud(CLOUD),
            send_local=sender, kv_monitor=FakeKV(150)))   # r1+r2 don't fit -> kick r1
        cloud_ids = {r["request_id"] for r in results if r["endpoint"] == "cloud"}
        self.assertIn(1, cloud_ids)                        # the victim went to CLOUD
        self.assertNotIn(1, sender.dispatch_order)         # never dispatched locally
        self.assertEqual(pol.n_outsourced, len(cloud_ids))


class TestStaleDecisionDiscard(unittest.TestCase):
    def test_kv_freed_during_tick_prevents_over_outsourcing(self):
        """Regression (codex round-4 P2): a completion that frees KV during a
        slow decision must invalidate that decision — r1 stays LOCAL because
        the fresh KV read (10000) fits it, even though the stale read (0)
        said kick."""
        import time as _time
        from router.nimbus import NimbusPolicy

        class SleepyPolicy(NimbusPolicy):
            def on_tick(self, waiting, kv, inflight_remaining=0):
                _time.sleep(0.12)                  # slow decision window
                return super().on_tick(waiting, kv, inflight_remaining)

        kv_cell = [100.0]     # r0 (footprint 50) fits; r1 (150) does not — yet

        class DynamicKV:
            read_failures = 0
            async def available_tokens(self):
                return kv_cell[0]
            def invalidate(self):
                pass

        class FreeingSender(RecordingSender):
            async def __call__(self, endpoint, req, due):
                res = await super().__call__(endpoint, req, due)
                kv_cell[0] = 10000.0               # completion frees KV
                return res

        trace = [
            {"request_id": 0, "arrived_at": 1477007, "relative_arrival_s": 0.0,
             "prompt": "r0", "max_tokens": 10, "prompt_tokens": 40, "session_id": 0},
            {"request_id": 1, "arrived_at": 1477007, "relative_arrival_s": 0.0,
             "prompt": "r1", "max_tokens": 50, "prompt_tokens": 100, "session_id": 0},
        ]
        import tempfile
        from pathlib import Path
        from router.run import parse_args, replay_queued
        out_dir = Path(tempfile.mkdtemp())
        args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "nimbus",
                           "--local-url", "http://x", "--local-model", "m",
                           "--kv-capacity-tokens", "1000000", "--max-inflight", "1",
                           "--out-dir", str(out_dir)])
        pol = SleepyPolicy(prefill_tput=20000, tpot_s=0.0095)
        sender = FreeingSender(service_s=0.05)     # r0 completes inside the tick
        results, _, _ = asyncio.run(replay_queued(
            args, trace, pol, LOCAL, CLOUD, sink=NullCloud(CLOUD),
            send_local=sender, kv_monitor=DynamicKV()))
        cloud = [r for r in results if r["endpoint"] == "cloud"]
        self.assertEqual(cloud, [])                       # nothing over-outsourced
        self.assertEqual(sender.dispatch_order, [0, 1])   # r1 served locally
        self.assertEqual(pol.n_outsourced, 0)


class FakeKV:
    """Injectable monitor: fixed available tokens."""
    def __init__(self, avail):
        self.avail = avail
        self.read_failures = 0
    async def available_tokens(self):
        return self.avail
    def invalidate(self):
        pass


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

    def test_runtime_passes_inflight_remaining_decode_to_policy(self):
        class CapturingPolicy(NimbusPolicy):
            def __init__(self):
                super().__init__(prefill_tput=20_000, tpot_s=0.0095)
                self.seen_remaining = []
            def on_tick(self, waiting, kv, inflight_remaining=0):
                self.seen_remaining.append(inflight_remaining)
                return super().on_tick(waiting, kv, inflight_remaining)

        policy = CapturingPolicy()
        trace = mk_trace(3)
        import tempfile
        from pathlib import Path
        from router.run import parse_args, replay_queued
        args = parse_args([
            "--data", "t", "--scenario", "normal", "--policy", "nimbus",
            "--local-url", "http://x", "--local-model", "m",
            "--kv-capacity-tokens", "1000000", "--max-inflight", "1",
            "--out-dir", str(Path(tempfile.mkdtemp())),
        ])
        asyncio.run(replay_queued(
            args, trace, policy, LOCAL, CLOUD, sink=NullCloud(CLOUD),
            send_local=RecordingSender(service_s=0.05), kv_monitor=FakeKV(1e9),
        ))
        self.assertIn(50, policy.seen_remaining)

    def test_stale_metrics_are_clamped_by_known_local_commitments(self):
        """A cached full-headroom scrape must not admit beyond capacity."""
        import tempfile
        from pathlib import Path
        from router.run import parse_args, replay_queued

        trace = mk_trace(2)  # each commits 100 prompt + 50 decode = 150 tokens
        args = parse_args([
            "--data", "t", "--scenario", "normal", "--policy", "nimbus",
            "--local-url", "http://x", "--local-model", "m",
            "--kv-capacity-tokens", "200", "--max-inflight", "1",
            "--out-dir", str(Path(tempfile.mkdtemp())),
        ])
        policy = mk_policy()
        sender = RecordingSender(service_s=0.05)
        results, _, _ = asyncio.run(replay_queued(
            args, trace, policy, LOCAL, CLOUD, sink=NullCloud(CLOUD),
            send_local=sender, kv_monitor=FakeKV(200),  # stale "all free"
        ))
        self.assertEqual(sender.dispatch_order, [0])
        self.assertEqual(
            [r["request_id"] for r in results if r["endpoint"] == "cloud"],
            [1],
        )


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
