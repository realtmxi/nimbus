"""Unit tests for router/run.py (the router entry point). No network: sender injected.

Run from the repo root:  python3 -m unittest router.test_run -v
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from router.common import Endpoint, Policy, summarize
from router.run import (
    CloudFatalError,
    LocalAdmission,
    TraceIntegrityError,
    load_verified_trace,
    parse_args,
    queue_stats,
    replay_queued,
    token_alignment_stats,
)

LOCAL = Endpoint(name="local", url="http://x", model="m")
CLOUD = Endpoint(name="cloud", url="fake://", model="m",
                 input_price_per_mtok=0.15, output_price_per_mtok=1.20)
EXPECTED_TRACE_SHA256 = "a" * 64

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
    results, admission, _ = asyncio.run(replay_queued(
        args, trace, policy, LOCAL, CLOUD if has_cloud else None,
        sink=sink, send_local=sender, send_cloud=cloud_sender))
    return results, admission, sender


class TestTraceIntegrity(unittest.TestCase):
    def test_hashes_and_parses_one_exact_in_memory_read(self):
        raw = (
            b'{"arrived_at":1477007,"prompt_text":"current turn",'
            b'"num_prefill_tokens":2,"num_decode_tokens":1}\n'
        )

        class OneReadSource:
            def __init__(self):
                self.calls = 0

            def read_bytes(self):
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError("trace path was reopened")
                return raw

        source = OneReadSource()
        rows = load_verified_trace(
            source, "normal", hashlib.sha256(raw).hexdigest()  # type: ignore[arg-type]
        )
        self.assertEqual(source.calls, 1)
        self.assertEqual(rows[0]["prompt"], "current turn")

    def test_hash_and_parse_errors_never_echo_prompt_text(self):
        sensitive = "SENSITIVE_SHAREGPT_CURRENT_TURN"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(sensitive, encoding="utf-8")
            with self.assertRaises(TraceIntegrityError) as mismatch:
                load_verified_trace(path, "normal", "0" * 64)
            self.assertNotIn(sensitive, str(mismatch.exception))

            malformed = f'{{"prompt_text":"{sensitive}",'.encode()
            path.write_bytes(malformed)
            with self.assertRaises(TraceIntegrityError) as parse_failure:
                load_verified_trace(
                    path, "normal", hashlib.sha256(malformed).hexdigest()
                )
            self.assertNotIn(sensitive, str(parse_failure.exception))


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
        sensitive = "private ShareGPT prompt and bearer token"
        class Boom(RecordingSender):
            async def __call__(self, endpoint, req, due):
                raise RuntimeError(sensitive)
        args = mk_args()
        results, admission, _ = run(args, mk_trace(4), Policy("all_local", 0, 0),
                                    sender=Boom())
        self.assertEqual(len(results), 4)
        self.assertTrue(all(not r["success"] for r in results))
        self.assertTrue(all(r["error"] == "sender exception; details omitted"
                            for r in results))
        self.assertNotIn(sensitive, json.dumps(results))
        self.assertEqual(admission.inflight, 0)           # nothing leaked


class TestCloudPath(unittest.TestCase):
    def test_token_alignment_exposes_scheduler_payload_mismatch(self):
        stats = token_alignment_stats([
            {"success": True, "routed_only": False, "endpoint": "local",
             "scheduler_prompt_tokens": 554, "prompt_tokens": 18,
             "scheduler_decode_tokens": 7, "completion_tokens": 7},
            {"success": True, "routed_only": False, "endpoint": "local",
             "scheduler_prompt_tokens": 13, "prompt_tokens": 21,
             "scheduler_decode_tokens": 9, "completion_tokens": 8},
            {"success": True, "routed_only": True,
             "scheduler_prompt_tokens": 99, "prompt_tokens": 1},
            {"success": True, "routed_only": False, "endpoint": "cloud",
             "scheduler_prompt_tokens": 1000, "prompt_tokens": 1},
        ])
        self.assertEqual(stats["measured_n"], 2)
        self.assertEqual(stats["absolute_error_max_tokens"], 536)
        self.assertGreater(stats["relative_error_p50"], 0)
        self.assertEqual(stats["decode_measured_n"], 2)
        self.assertEqual(stats["decode_cap_hit_n"], 1)

    def test_token_alignment_handles_completion_usage_without_prompt_usage(self):
        stats = token_alignment_stats([
            {"success": True, "routed_only": False, "endpoint": "local",
             "scheduler_prompt_tokens": 20, "prompt_tokens": None,
             "scheduler_decode_tokens": 8, "completion_tokens": 8},
        ])
        self.assertEqual(stats["measured_n"], 0)
        self.assertEqual(stats["missing_prompt_usage_n"], 1)
        self.assertIsNone(stats["absolute_error_max_tokens"])
        self.assertEqual(stats["decode_measured_n"], 1)
        self.assertEqual(stats["decode_cap_hit_n"], 1)

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
            self.assertEqual(r["pre_route_queue_ms"], 0.0)
            self.assertEqual(r["cloud_gate_wait_ms"], 0.0)
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
        for row in lines:
            self.assertEqual(row["scheduler_prompt_tokens"], 100)
            self.assertEqual(row["scheduler_uncached_prompt_tokens"], 100)
            self.assertEqual(row["scheduler_decode_tokens"], 50)
        stats = queue_stats(results, admission)
        self.assertIn("queue_delay_p50_ms", stats)
        self.assertGreaterEqual(stats["peak_inflight"], 1)


class TestCloudConcurrencyGate(unittest.TestCase):
    @staticmethod
    def _fixed_probe_args(*, concurrency: int = 2):
        return mk_args(
            policy="all_cloud",
            extra=[
                "--cloud", "real", "--cloud-url", "http://c",
                "--cloud-model", "m", "--cloud-api-key-env", "OPENROUTER_KEY",
                "--expected-trace-sha256", EXPECTED_TRACE_SHA256,
                "--cloud-provider", "deepinfra", "--cloud-no-fallbacks",
                "--cloud-stop-after-first-token", "--local-ignore-eos",
                "--cloud-max-concurrency", str(concurrency),
            ],
        )

    @staticmethod
    def _cloud_failure(req: dict, status: int) -> dict:
        return {
            "request_id": req["request_id"], "arrived_at": req["arrived_at"],
            "relative_arrival_s": req["relative_arrival_s"],
            "scheduled_lag_ms": 0.0, "endpoint": "cloud", "model": "m",
            "success": False,
            "error": f"HTTP {status}: provider response body omitted",
            "error_type": f"HTTP {status}", "http_status": status,
            "ttft_ms": None, "e2e_ms": 1.0, "tpot_ms": None, "chunks": 0,
            "prompt_tokens": None, "completion_tokens": None,
            "output_chars": 0, "cost_usd": 0.0,
        }

    @classmethod
    def _cloud_success(cls, req: dict) -> dict:
        row = cls._cloud_failure(req, 200)
        row.update({
            "success": True,
            "error": None,
            "error_type": None,
            "generation_id_sha256": "a" * 64,
            "provider": "deepinfra",
            "response_model": "m",
        })
        return row

    def test_fatal_401_stops_future_arrivals_and_preserves_trigger_row(self):
        calls: list[int] = []

        async def sender(req, due):
            calls.append(req["request_id"])
            await asyncio.sleep(0)
            return self._cloud_failure(req, 401)

        args = self._fixed_probe_args(concurrency=2)

        async def scenario():
            with self.assertRaises(CloudFatalError) as raised:
                await replay_queued(
                    args, mk_trace(5, gap_s=0.05), Policy("all_cloud", 1.0, 0),
                    None, CLOUD, send_cloud=sender,
                )
            self.assertEqual(
                str(raised.exception),
                "fatal cloud status gate: status=401 completed_n=1 "
                "cloud_success_n=0 non429_http_failure_n=1 http_400_n=0",
            )
            live = [
                task for task in asyncio.all_tasks()
                if task is not asyncio.current_task() and not task.done()
            ]
            self.assertEqual(live, [])

        asyncio.run(scenario())
        self.assertEqual(calls, [0])
        out = args.out_dir / "all_cloud_normal.jsonl"
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual([row["request_id"] for row in rows], [0])
        self.assertEqual(rows[0]["http_status"], 401)

    def test_forbidden_redirect_is_immediately_fatal(self):
        calls: list[int] = []

        async def sender(req, due):
            calls.append(req["request_id"])
            await asyncio.sleep(0)
            return self._cloud_failure(req, 307)

        args = self._fixed_probe_args(concurrency=1)
        with self.assertRaisesRegex(
            CloudFatalError,
            r"status=307 completed_n=1 .*http_400_n=0",
        ):
            asyncio.run(replay_queued(
                args, mk_trace(5), Policy("all_cloud", 1.0, 0),
                None, CLOUD, send_cloud=sender,
            ))
        self.assertEqual(calls, [0])

    def test_config_and_method_statuses_are_immediately_fatal(self):
        for status in (404, 405, 422):
            calls: list[int] = []

            async def sender(req, due, status=status):
                calls.append(req["request_id"])
                await asyncio.sleep(0)
                return self._cloud_failure(req, status)

            with self.subTest(status=status), self.assertRaisesRegex(
                CloudFatalError,
                rf"status={status} completed_n=1 .*non429_http_failure_n=1",
            ):
                asyncio.run(replay_queued(
                    self._fixed_probe_args(concurrency=1),
                    mk_trace(5), Policy("all_cloud", 1.0, 0),
                    None, CLOUD, send_cloud=sender,
                ))
            self.assertEqual(calls, [0])

    def test_fatal_gate_cancels_only_concurrency_window_and_cleans_tasks(self):
        calls: list[int] = []
        active = 0
        cancelled = 0

        async def sender(req, due):
            nonlocal active, cancelled
            calls.append(req["request_id"])
            active += 1
            try:
                if req["request_id"] == 0:
                    await asyncio.sleep(0)
                    return self._cloud_failure(req, 403)
                await asyncio.sleep(10)
                return self._cloud_failure(req, 429)
            except asyncio.CancelledError:
                cancelled += 1
                raise
            finally:
                active -= 1

        args = self._fixed_probe_args(concurrency=2)

        async def scenario():
            with self.assertRaises(CloudFatalError):
                await replay_queued(
                    args, mk_trace(10), Policy("all_cloud", 1.0, 0),
                    None, CLOUD, send_cloud=sender,
                )
            self.assertEqual(active, 0)
            live = [
                task for task in asyncio.all_tasks()
                if task is not asyncio.current_task() and not task.done()
            ]
            self.assertEqual(live, [])

        asyncio.run(scenario())
        self.assertLessEqual(len(calls), 2)
        self.assertIn(0, calls)
        self.assertGreaterEqual(cancelled, 1)
        out = args.out_dir / "all_cloud_normal.jsonl"
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["http_status"], 403)

    def test_http_400_requires_three_completed_and_eighty_percent(self):
        calls: list[int] = []
        statuses = [400, 400, 200, 400, 400]

        async def sender(req, due):
            calls.append(req["request_id"])
            await asyncio.sleep(0)
            status = statuses[req["request_id"]]
            return (
                self._cloud_success(req)
                if status == 200 else self._cloud_failure(req, status)
            )

        args = self._fixed_probe_args(concurrency=1)
        with self.assertRaisesRegex(
            CloudFatalError,
            r"status=400 completed_n=5 .*http_400_n=4",
        ):
            asyncio.run(replay_queued(
                args, mk_trace(8), Policy("all_cloud", 1.0, 0),
                None, CLOUD, send_cloud=sender,
            ))
        self.assertEqual(calls, [0, 1, 2, 3, 4])
        out = args.out_dir / "all_cloud_normal.jsonl"
        self.assertEqual(len(out.read_text().splitlines()), 5)

    def test_zero_success_non429_http_failures_stop_after_three(self):
        calls: list[int] = []

        async def sender(req, due):
            calls.append(req["request_id"])
            await asyncio.sleep(0)
            return self._cloud_failure(req, 503)

        with self.assertRaisesRegex(
            CloudFatalError,
            r"status=0 completed_n=3 cloud_success_n=0 "
            r"non429_http_failure_n=3",
        ):
            asyncio.run(replay_queued(
                self._fixed_probe_args(concurrency=1),
                mk_trace(8), Policy("all_cloud", 1.0, 0),
                None, CLOUD, send_cloud=sender,
            ))
        self.assertEqual(calls, [0, 1, 2])

    def test_isolated_non429_http_failure_remains_measured(self):
        calls: list[int] = []

        async def sender(req, due):
            calls.append(req["request_id"])
            await asyncio.sleep(0)
            if req["request_id"] == 0:
                return self._cloud_failure(req, 503)
            return self._cloud_success(req)

        results, _, _ = asyncio.run(replay_queued(
            self._fixed_probe_args(concurrency=1),
            mk_trace(6), Policy("all_cloud", 1.0, 0),
            None, CLOUD, send_cloud=sender,
        ))
        self.assertEqual(calls, list(range(6)))
        self.assertEqual(sum(not row["success"] for row in results), 1)

    def test_statusless_failures_do_not_count_as_dominant_http(self):
        calls: list[int] = []

        async def sender(req, due):
            calls.append(req["request_id"])
            await asyncio.sleep(0)
            if req["request_id"] == 3:
                return self._cloud_success(req)
            row = self._cloud_failure(req, 500)
            row.update({
                "http_status": None,
                "error_type": "TimeoutError",
                "error": "timeout after 600.0s",
            })
            return row

        results, _, _ = asyncio.run(replay_queued(
            self._fixed_probe_args(concurrency=1),
            mk_trace(4), Policy("all_cloud", 1.0, 0),
            None, CLOUD, send_cloud=sender,
        ))
        self.assertEqual(calls, list(range(4)))
        self.assertEqual(sum(not row["success"] for row in results), 3)

    def test_dominant_non429_http_failures_stop_at_ten(self):
        statuses = [200, 200] + [503] * 8
        calls: list[int] = []

        async def sender(req, due):
            calls.append(req["request_id"])
            await asyncio.sleep(0)
            status = statuses[req["request_id"]]
            return (
                self._cloud_success(req)
                if status == 200 else self._cloud_failure(req, status)
            )

        with self.assertRaisesRegex(
            CloudFatalError,
            r"status=0 completed_n=10 cloud_success_n=2 "
            r"non429_http_failure_n=8",
        ):
            asyncio.run(replay_queued(
                self._fixed_probe_args(concurrency=1),
                mk_trace(12), Policy("all_cloud", 1.0, 0),
                None, CLOUD, send_cloud=sender,
            ))
        self.assertEqual(calls, list(range(10)))

    def test_http_429_remains_a_normal_measured_failure(self):
        calls: list[int] = []

        async def sender(req, due):
            calls.append(req["request_id"])
            await asyncio.sleep(0)
            return self._cloud_failure(req, 429)

        args = self._fixed_probe_args(concurrency=2)
        results, _, _ = asyncio.run(replay_queued(
            args, mk_trace(5), Policy("all_cloud", 1.0, 0),
            None, CLOUD, send_cloud=sender,
        ))
        self.assertEqual(calls, list(range(5)))
        self.assertEqual(len(results), 5)
        self.assertTrue(all(row["http_status"] == 429 for row in results))

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
                              "--cloud-model", "m", "--cloud-max-concurrency", "2",
                              "--expected-trace-sha256", EXPECTED_TRACE_SHA256])
        results, _, _ = run(args, mk_trace(10), Policy("all_cloud", 1.0, 0),
                            cloud_sender=rec)
        self.assertEqual(rec.n, 10)
        self.assertEqual(len(results), 10)
        self.assertLessEqual(rec.peak, 2)                 # gate binds
        self.assertGreaterEqual(rec.peak, 2)              # and is actually exercised
        self.assertGreater(max(r["cloud_gate_wait_ms"] for r in results), 40.0)
        for r in results:
            self.assertEqual(r["pre_route_queue_ms"], 0.0)
            self.assertAlmostEqual(
                r["queue_delay_ms"], r["cloud_gate_wait_ms"], places=5
            )
            self.assertAlmostEqual(
                r["ttft_ms"],
                r["pre_route_queue_ms"] + r["cloud_gate_wait_ms"]
                + r["service_ttft_ms"],
                places=5,
            )

    def test_ttft_probe_row_is_successful_and_keeps_arrival_accounting(self):
        async def probe_sender(req, due):
            await asyncio.sleep(0.005)
            return {
                "request_id": req["request_id"], "arrived_at": req["arrived_at"],
                "relative_arrival_s": req["relative_arrival_s"],
                "scheduled_lag_ms": 0.0, "endpoint": "cloud", "model": "m",
                "success": True, "error": None, "error_type": None,
                "http_status": 200, "ttft_ms": 5.0, "e2e_ms": None,
                "tpot_ms": None, "chunks": 1, "prompt_tokens": None,
                "completion_tokens": None, "output_chars": 1, "cost_usd": None,
                "cost_pending": True, "response_completed": False,
                "stream_abort_requested": True,
                "generation_id_sha256": "a" * 64,
                "provider": "deepinfra",
                "response_model": "m",
            }

        args = mk_args(
            policy="all_cloud",
            extra=[
                "--cloud", "real", "--cloud-url", "http://c",
                "--cloud-model", "m", "--cloud-api-key-env", "OPENROUTER_KEY",
                "--expected-trace-sha256", EXPECTED_TRACE_SHA256,
                "--cloud-provider", "deepinfra", "--cloud-no-fallbacks",
                "--cloud-stop-after-first-token",
            ],
        )
        policy = Policy("all_cloud", 1.0, 0)
        results, _, _ = run(
            args, mk_trace(1), policy, cloud_sender=probe_sender,
        )
        row = results[0]
        self.assertTrue(row["success"])
        self.assertFalse(row["response_completed"])
        self.assertIsNone(row["e2e_ms"])
        self.assertEqual(row["pre_route_queue_ms"], 0.0)
        self.assertGreaterEqual(row["cloud_gate_wait_ms"], 0.0)
        self.assertAlmostEqual(
            row["ttft_ms"],
            row["pre_route_queue_ms"] + row["cloud_gate_wait_ms"]
            + row["service_ttft_ms"],
            places=5,
        )
        summary = summarize(results, policy, slo_s=5.0)
        self.assertEqual(summary["overall"]["slo_violations"], 0)
        self.assertEqual(summary["overall"]["slo_measured_n"], 1)
        self.assertIsNone(summary["overall"]["cost_usd"])
        self.assertEqual(summary["overall"]["cost_pending_n"], 1)

    def test_raw_writer_scrubs_untrusted_provider_metadata_and_stops(self):
        sensitive_id = "provider-generation-id-private-user-text"
        sensitive_provider = "DeepInfra private ShareGPT fragment"
        sensitive_model = "private/model/ShareGPT-fragment"

        async def sender(req, due):
            await asyncio.sleep(0)
            return {
                "request_id": req["request_id"],
                "arrived_at": req["arrived_at"],
                "relative_arrival_s": req["relative_arrival_s"],
                "scheduled_lag_ms": 0.0,
                "endpoint": "cloud",
                "model": "m",
                "success": True,
                "error": None,
                "error_type": None,
                "http_status": 200,
                "ttft_ms": 5.0,
                "e2e_ms": None,
                "tpot_ms": None,
                "chunks": 1,
                "prompt_tokens": None,
                "completion_tokens": None,
                "output_chars": 1,
                "cost_usd": None,
                "cost_pending": True,
                "response_completed": False,
                "stream_abort_requested": True,
                "generation_id": sensitive_id,
                "provider": sensitive_provider,
                "response_model": sensitive_model,
            }

        args = self._fixed_probe_args(concurrency=1)
        with self.assertRaisesRegex(
            CloudFatalError, r"fatal cloud protocol gate: completed_n=1"
        ):
            asyncio.run(replay_queued(
                args, mk_trace(3), Policy("all_cloud", 1.0, 0),
                None, CLOUD, send_cloud=sender,
            ))

        raw = (args.out_dir / "all_cloud_normal.jsonl").read_text()
        self.assertNotIn(sensitive_id, raw)
        self.assertNotIn(sensitive_provider, raw)
        self.assertNotIn(sensitive_model, raw)
        row = json.loads(raw)
        self.assertNotIn("generation_id", row)
        self.assertEqual(row["error_type"], "ProtocolMismatch")
        self.assertEqual(
            row["error"],
            "provider response metadata mismatch; details omitted",
        )
        self.assertIsNone(row["provider"])
        self.assertIsNone(row["response_model"])


class TestParseArgsQueued(unittest.TestCase):
    def test_ignore_eos_can_be_split_by_endpoint(self):
        base = [
            "--data", "t", "--scenario", "normal", "--policy", "random",
            "--local-url", "http://l", "--local-model", "m",
        ]
        default = parse_args(base)
        self.assertFalse(default.ignore_eos)
        self.assertFalse(default.local_ignore_eos)
        self.assertFalse(default.cloud_ignore_eos)

        legacy = parse_args(base + ["--ignore-eos"])
        self.assertTrue(legacy.ignore_eos)
        self.assertTrue(legacy.local_ignore_eos)
        self.assertTrue(legacy.cloud_ignore_eos)

        split = parse_args(base + ["--local-ignore-eos"])
        self.assertTrue(split.local_ignore_eos)
        self.assertFalse(split.cloud_ignore_eos)

        override = parse_args(base + ["--ignore-eos", "--no-cloud-ignore-eos"])
        self.assertTrue(override.local_ignore_eos)
        self.assertFalse(override.cloud_ignore_eos)

    def test_default_senders_receive_endpoint_ignore_eos(self):
        from unittest.mock import patch

        seen = []

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        class FakeAiohttp:
            class TCPConnector:
                def __init__(self, **kwargs):
                    pass

            @staticmethod
            def ClientSession(**kwargs):
                return FakeSession()

        async def fake_one_request(session, endpoint, req, due, **kwargs):
            seen.append((endpoint.name, kwargs["ignore_eos"]))
            return {
                "request_id": req["request_id"], "arrived_at": req["arrived_at"],
                "relative_arrival_s": req["relative_arrival_s"],
                "scheduled_lag_ms": 0.0, "endpoint": endpoint.name,
                "model": endpoint.model, "success": True, "error": None,
                "error_type": None, "http_status": 200, "ttft_ms": 5.0,
                "e2e_ms": 10.0, "tpot_ms": 1.0, "chunks": 2,
                "prompt_tokens": 10, "completion_tokens": 2,
                "output_chars": 2, "cost_usd": 0.0,
            }

        local_args = mk_args(extra=["--local-ignore-eos"])
        cloud_args = mk_args(
            policy="all_cloud",
            extra=[
                "--cloud", "real", "--cloud-url", "http://c",
                "--cloud-model", "m", "--local-ignore-eos",
                "--expected-trace-sha256", EXPECTED_TRACE_SHA256,
            ],
        )
        with (
            patch("router.run.one_request", new=fake_one_request),
            patch("router.run.aiohttp", new=FakeAiohttp),
        ):
            asyncio.run(replay_queued(
                local_args, mk_trace(1), Policy("all_local", 0.0, 0),
                LOCAL, None,
            ))
            asyncio.run(replay_queued(
                cloud_args, mk_trace(1), Policy("all_cloud", 1.0, 0),
                None, CLOUD,
            ))
        self.assertEqual(seen, [("local", True), ("cloud", False)])

    def test_random_default_cloud_is_null_sink(self):
        args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "random",
                           "--local-url", "http://l", "--local-model", "m"])
        self.assertEqual(args.cloud, "null")  # default: fake sink, no url/key needed
        self.assertIsNone(args.cloud_provider)
        self.assertFalse(args.cloud_no_fallbacks)
        self.assertFalse(args.cloud_stop_after_first_token)

    def test_real_cloud_requires_url_and_model(self):
        base = ["--data", "t", "--scenario", "normal", "--policy", "all_cloud",
                "--cloud", "real", "--expected-trace-sha256",
                EXPECTED_TRACE_SHA256]
        with self.assertRaises(SystemExit):
            parse_args(base)                                   # no url
        with self.assertRaises(SystemExit):
            parse_args(base + ["--cloud-url", "http://c"])     # url but no model
        args = parse_args(base + ["--cloud-url", "http://c",
                                  "--cloud-model", "qwen3-32b"])
        self.assertEqual(args.cloud_model, "qwen3-32b")

    def test_real_cloud_requires_exact_lowercase_trace_sha(self):
        base = [
            "--data", "t", "--scenario", "normal", "--policy", "all_cloud",
            "--cloud", "real", "--cloud-url", "http://c", "--cloud-model", "m",
        ]
        with self.assertRaises(SystemExit):
            parse_args(base)
        with self.assertRaises(SystemExit):
            parse_args(base + ["--expected-trace-sha256", "A" * 64])
        args = parse_args(
            base + ["--expected-trace-sha256", EXPECTED_TRACE_SHA256]
        )
        self.assertEqual(args.expected_trace_sha256, EXPECTED_TRACE_SHA256)

    def test_cloud_model_defaults_to_local_model(self):
        from router.common import build_endpoints
        args = parse_args(["--data", "t", "--scenario", "normal", "--policy", "random",
                           "--local-url", "http://l", "--local-model", "m",
                           "--cloud-url", "http://c"])
        _, cloud = build_endpoints(args)
        self.assertEqual(cloud.model, "m")

    def test_ttft_probe_requires_fixed_authenticated_real_cloud(self):
        base = [
            "--data", "t", "--scenario", "normal", "--policy", "all_cloud",
            "--cloud", "real", "--cloud-url", "http://c", "--cloud-model", "m",
            "--expected-trace-sha256", EXPECTED_TRACE_SHA256,
            "--cloud-stop-after-first-token",
        ]
        with self.assertRaises(SystemExit):
            parse_args(base)  # no provider / no-fallback / key env
        with self.assertRaises(SystemExit):
            parse_args(base + ["--cloud-provider", "deepinfra",
                               "--cloud-api-key-env", "OPENROUTER_KEY"])
        with self.assertRaises(SystemExit):
            parse_args(base + ["--cloud-provider", "deepinfra",
                               "--cloud-no-fallbacks"])
        with self.assertRaises(SystemExit):
            parse_args(base + ["--cloud-provider", "deepinfra",
                               "--cloud-provider", "groq", "--cloud-no-fallbacks",
                               "--cloud-api-key-env", "OPENROUTER_KEY"])
        with self.assertRaises(SystemExit):
            parse_args(base + ["--cloud-provider", "deepinfra",
                               "--cloud-no-fallbacks", "--cloud-api-key-env",
                               "OPENROUTER_KEY", "--ignore-eos"])
        with self.assertRaises(SystemExit):
            parse_args(base + ["--cloud-provider", "deepinfra",
                               "--cloud-no-fallbacks", "--cloud-api-key-env",
                               "OPENROUTER_KEY", "--cloud-ignore-eos"])

        args = parse_args(base + [
            "--cloud-provider", "deepinfra", "--cloud-no-fallbacks",
            "--cloud-api-key-env", "OPENROUTER_KEY", "--local-ignore-eos",
        ])
        self.assertEqual(args.cloud_provider, ["deepinfra"])
        self.assertTrue(args.cloud_no_fallbacks)
        self.assertTrue(args.cloud_stop_after_first_token)
        self.assertTrue(args.local_ignore_eos)
        self.assertFalse(args.cloud_ignore_eos)

    def test_openrouter_options_rejected_for_null_cloud(self):
        base = [
            "--data", "t", "--scenario", "normal", "--policy", "random",
            "--local-url", "http://l", "--local-model", "m",
        ]
        for option in (
            ["--cloud-provider", "deepinfra"],
            ["--cloud-no-fallbacks"],
            ["--cloud-stop-after-first-token"],
        ):
            with self.assertRaises(SystemExit):
                parse_args(base + option)

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
