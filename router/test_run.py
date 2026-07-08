"""Unit tests for router/run.py (the router entry point). No network: sender injected.

Run from the repo root:  python3 -m unittest router.test_run -v
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from router.common import Endpoint, Policy
from router.run import (
    LocalAdmission,
    parse_args,
    queue_stats,
    replay_queued,
)

LOCAL = Endpoint(name="local", url="http://x", model="m")
CLOUD = Endpoint(name="cloud", url="fake://", model="m",
                 input_price_per_mtok=0.15, output_price_per_mtok=1.20)

def mk_trace(n: int, prompt_tokens: int = 100, max_tokens: int = 50,
             gap_s: float = 0.0) -> list[dict]:
    return [{
        "request_id": i,
        "arrived_at": 1477007 + i,
        "relative_arrival_s": i * gap_s,
        "prompt": f"req {i}",
        "max_tokens": max_tokens,
        "prompt_tokens": prompt_tokens,
        "session_id": 0,
    } for i in range(n)]


def mk_args(policy: str = "all_local", max_inflight: int = 128,
            fraction: float = 0.5, extra: list | None = None) -> object:
    out_dir = Path(tempfile.mkdtemp())
    argv = ["--data", "t.jsonl", "--scenario", "normal", "--policy", policy,
            "--fraction", str(fraction),
            "--local-url", "http://x", "--local-model", "m",
            "--max-inflight", str(max_inflight),
            "--out-dir", str(out_dir)] + (extra or [])
    return parse_args(argv)


class RecordingSender:
    """Injectable local sender: records dispatch times/concurrency, sleeps a bit."""

    def __init__(self, service_s: float = 0.02):
        self.service_s = service_s
        self.dispatch_order: list[int] = []
        self.active = 0
        self.peak_active = 0

    async def __call__(self, endpoint, req, due):
        self.dispatch_order.append(req["request_id"])
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        await asyncio.sleep(self.service_s)
        self.active -= 1
        return {
            "request_id": req["request_id"], "arrived_at": req["arrived_at"],
            "relative_arrival_s": req["relative_arrival_s"], "scheduled_lag_ms": 0.0,
            "endpoint": "local", "model": "m", "success": True, "error": None,
            "error_type": None, "http_status": 200, "ttft_ms": 10.0,
            "e2e_ms": self.service_s * 1000, "tpot_ms": 5.0, "chunks": 3,
            "prompt_tokens": req.get("prompt_tokens"), "completion_tokens": req["max_tokens"],
            "output_chars": 12, "cost_usd": 0.0,
        }


def run(args, trace, policy, sink=None, sender=None, cloud_sender=None):
    sender = sender or RecordingSender()
    has_cloud = sink is not None or cloud_sender is not None
    results, admission = asyncio.run(replay_queued(
        args, trace, policy, LOCAL, CLOUD if has_cloud else None,
        sink=sink, send_local=sender, send_cloud=cloud_sender))
    return results, admission, sender


class TestAdmissionGate(unittest.TestCase):
    def test_concurrency_gate(self):
        adm = LocalAdmission(max_inflight=2)
        req = {"prompt_tokens": 100, "max_tokens": 50}
        self.assertTrue(adm.fits(req))
        adm.reserve(req)
        adm.reserve(req)
        self.assertFalse(adm.fits(req))          # slots full
        adm.release(req)
        self.assertTrue(adm.fits(req))
        adm.release(req)
        self.assertEqual(adm.inflight, 0)
        self.assertEqual(adm.peak_inflight, 2)


class TestWorkConserving(unittest.TestCase):
    def test_no_pressure_dispatches_immediately_fifo(self):
        """Queue neutrality at the unit level: with free slots every request
        dispatches the moment it arrives — no added queue delay."""
        args = mk_args(max_inflight=128)
        trace = mk_trace(20)
        results, admission, sender = run(args, trace, Policy("all_local", 0, 0))
        self.assertEqual(len(results), 20)
        self.assertEqual(sender.dispatch_order, list(range(20)))     # FIFO
        for r in results:
            self.assertLess(r["queue_delay_ms"], 20.0)               # ~0
            self.assertEqual(r["ttft_ms"], r["queue_delay_ms"] + r["service_ttft_ms"])

    def test_pressure_paces_and_accounts_queue_delay(self):
        """A single slot: dispatch serializes, later requests accrue queue
        delay, and ttft = queue_delay + service."""
        args = mk_args(max_inflight=1)
        trace = mk_trace(5)                               # all arrive at t=0
        results, admission, sender = run(args, trace, Policy("all_local", 0, 0),
                                         sender=RecordingSender(service_s=0.03))
        self.assertEqual(sender.peak_active, 1)                        # serialized
        self.assertEqual(sender.dispatch_order, list(range(5)))        # FIFO preserved
        by_id = {r["request_id"]: r for r in results}
        self.assertLess(by_id[0]["queue_delay_ms"], 15.0)
        self.assertGreater(by_id[4]["queue_delay_ms"], 100.0)          # waited ~4x30ms
        self.assertAlmostEqual(
            by_id[4]["ttft_ms"], by_id[4]["queue_delay_ms"] + 10.0, places=5)

    def test_slot_released_on_sender_exception(self):
        class Boom(RecordingSender):
            async def __call__(self, endpoint, req, due):
                raise RuntimeError("kaboom")
        args = mk_args()
        results, admission, _ = run(args, mk_trace(4), Policy("all_local", 0, 0),
                                    sender=Boom())
        self.assertEqual(len(results), 4)
        self.assertTrue(all(not r["success"] for r in results))
        self.assertEqual(admission.inflight, 0)           # nothing leaked


class TestCloudPath(unittest.TestCase):
    def test_random_split_with_null_cloud(self):
        from router.common import NullCloud
        args = mk_args(policy="random", fraction=0.4)
        sink = NullCloud(CLOUD)
        trace = mk_trace(200)
        results, _, sender = run(args, trace, Policy("random", 0.4, seed=1), sink=sink)
        cloud = [r for r in results if r["endpoint"] == "cloud"]
        local = [r for r in results if r["endpoint"] == "local"]
        self.assertEqual(len(cloud) + len(local), 200)
        self.assertAlmostEqual(len(cloud) / 200, 0.4, delta=0.12)
        self.assertEqual(len(sender.dispatch_order), len(local))  # cloud never local
        for r in cloud:
            self.assertEqual(r["queue_delay_ms"], 0.0)            # cloud skips queue
            self.assertTrue(r["routed_only"])                     # fake sink: routed & counted
            self.assertIsNone(r["ttft_ms"])                       # no latency claim
            self.assertGreater(r["cost_usd"], 0.0)
        for r in local:
            self.assertEqual(r["cost_usd"], 0.0)

    def test_results_are_json_serializable_and_persisted(self):
        args = mk_args()
        results, admission, _ = run(args, mk_trace(3), Policy("all_local", 0, 0))
        from router.common import resolve_output_path
        out = resolve_output_path(args)
        lines = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual(len(lines), 3)
        stats = queue_stats(results, admission)
        self.assertIn("queue_delay_p50_ms", stats)
        self.assertGreaterEqual(stats["peak_inflight"], 1)


class TestCloudConcurrencyGate(unittest.TestCase):
    def test_real_cloud_concurrency_capped(self):
        """Regression (codex review): real-cloud sends must respect
        --cloud-max-concurrency so burst outsourcing can't self-inflict 429s."""
        class CloudRecorder:
            def __init__(self):
                self.active = 0
                self.peak = 0
                self.n = 0
            async def __call__(self, req, due):
                self.active += 1
                self.peak = max(self.peak, self.active)
                await asyncio.sleep(0.02)
                self.active -= 1
                self.n += 1
                return {"request_id": req["request_id"], "arrived_at": req["arrived_at"],
                        "relative_arrival_s": req["relative_arrival_s"],
                        "scheduled_lag_ms": 0.0, "endpoint": "cloud", "model": "m",
                        "success": True, "error": None, "error_type": None,
                        "http_status": 200, "ttft_ms": 5.0, "e2e_ms": 20.0,
                        "tpot_ms": 1.0, "chunks": 2, "prompt_tokens": 10,
                        "completion_tokens": 5, "output_chars": 4, "cost_usd": 0.0001}
        rec = CloudRecorder()
        args = mk_args(policy="all_cloud",
                       extra=["--cloud", "real", "--cloud-url", "http://c",
                              "--cloud-model", "m", "--cloud-max-concurrency", "2"])
        results, _, _ = run(args, mk_trace(10), Policy("all_cloud", 1.0, 0),
                            cloud_sender=rec)
        self.assertEqual(rec.n, 10)
        self.assertEqual(len(results), 10)
        self.assertLessEqual(rec.peak, 2)                 # gate binds
        self.assertGreaterEqual(rec.peak, 2)              # and is actually exercised


class TestParseArgsQueued(unittest.TestCase):
    def test_random_default_cloud_is_null_sink(self):
        args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "random",
                           "--local-url", "http://l", "--local-model", "m"])
        self.assertEqual(args.cloud, "null")  # default: fake sink, no url/key needed

    def test_real_cloud_requires_url_and_model(self):
        base = ["--data", "t", "--scenario", "normal", "--policy", "all_cloud",
                "--cloud", "real"]
        with self.assertRaises(SystemExit):
            parse_args(base)                                   # no url
        with self.assertRaises(SystemExit):
            parse_args(base + ["--cloud-url", "http://c"])     # url but no model
        args = parse_args(base + ["--cloud-url", "http://c",
                                  "--cloud-model", "qwen3-32b"])
        self.assertEqual(args.cloud_model, "qwen3-32b")

    def test_cloud_model_defaults_to_local_model(self):
        from router.common import build_endpoints
        args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "random",
                           "--local-url", "http://l", "--local-model", "m",
                           "--cloud-url", "http://c"])
        _, cloud = build_endpoints(args)
        self.assertEqual(cloud.model, "m")

    def test_summary_lives_next_to_raw_output(self):
        from router.common import resolve_output_path
        args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "all_local",
                           "--local-url", "http://l", "--local-model", "m",
                           "--output", "day1/run.jsonl"])
        out = resolve_output_path(args)
        self.assertEqual(out, Path("results/day1/run.jsonl"))
        self.assertEqual(out.with_name(out.stem + ".summary.json"),
                         Path("results/day1/run.summary.json"))

    def test_nonpositive_max_tokens_rejected(self):
        """Regression (codex review): --max-tokens 0 sent payload max_tokens=0
        while NullCloud billed 1 — reject non-positive values at parse."""
        base = ["--data", "t", "--scenario", "normal", "--policy", "all_local",
                "--local-url", "http://x", "--local-model", "m"]
        for bad in ("0", "-5"):
            with self.assertRaises(SystemExit):
                parse_args(base + ["--max-tokens", bad])
        self.assertEqual(parse_args(base + ["--max-tokens", "1"]).max_tokens, 1)

    def test_zero_max_inflight_rejected(self):
        """Regression (PR #3 review): max_inflight=0 would tight-loop the drain."""
        with self.assertRaises(SystemExit):
            parse_args(["--data", "t", "--scenario", "normal", "--policy", "all_local",
                        "--local-url", "http://x", "--local-model", "m",
                        "--max-inflight", "0"])

    def test_minimal_args_suffice(self):
        args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "all_local",
                           "--local-url", "http://x", "--local-model", "m"])
        self.assertEqual(args.max_inflight, 128)


if __name__ == "__main__":
    unittest.main()
