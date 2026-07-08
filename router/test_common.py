"""Unit tests for router/common.py (shared library). No network/aiohttp: session stubbed.

Run from the repo root:  python3 -m unittest router.test_common -v
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from router.common import (
    SCENARIOS,
    Endpoint,
    Policy,
    compute_cost_usd,
    load_trace,
    make_payload,
    one_request,
    summarize,
)

LOCAL = Endpoint(name="local", url="http://x/v1/chat/completions", model="m")
CLOUD = Endpoint(name="cloud", url="http://c/v1/chat/completions", model="m",
                 input_price_per_mtok=0.15, output_price_per_mtok=1.20)

REQ = {"request_id": 0, "arrived_at": 123, "relative_arrival_s": 0.0,
       "prompt": "hi", "max_tokens": 64}


class FakeResp:
    def __init__(self, status: int, sse_lines: list[str], text: str = ""):
        self.status = status
        self._chunks = [line.encode() for line in sse_lines]
        self._text = text

    async def text(self) -> str:
        return self._text

    @property
    def content(self):
        return self

    async def iter_chunked(self, _n: int):
        for c in self._chunks:
            yield c

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, resp: FakeResp):
        self._resp = resp
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self._resp


def sse_ok() -> list[str]:
    return [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        'data: {"choices":[{"delta":{"content":" world"}}]}\n',
        'data: {"usage":{"prompt_tokens":10,"completion_tokens":2},"choices":[]}\n',
        "data: [DONE]\n",
    ]


class TestPolicy(unittest.TestCase):
    def test_all_local_never_outsources(self):
        p = Policy("all_local", 0.5, seed=1)
        self.assertFalse(any(p.outsource(REQ) for _ in range(200)))
        self.assertEqual(p.actual_fraction, 0.0)

    def test_all_cloud_always_outsources(self):
        p = Policy("all_cloud", 0.5, seed=1)
        self.assertTrue(all(p.outsource(REQ) for _ in range(200)))
        self.assertEqual(p.actual_fraction, 1.0)

    def test_random_fraction_and_determinism(self):
        a = Policy("random", 0.3, seed=42)
        b = Policy("random", 0.3, seed=42)
        da = [a.outsource(REQ) for _ in range(2000)]
        db = [b.outsource(REQ) for _ in range(2000)]
        self.assertEqual(da, db)                      # same seed -> same decisions
        self.assertAlmostEqual(a.actual_fraction, 0.3, delta=0.05)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            Policy("random", 1.5, seed=0)
        with self.assertRaises(ValueError):
            Policy("nimbus", 0.5, seed=0)


class TestLoadTrace(unittest.TestCase):
    def test_window_filter_sort_reindex(self):
        start, end = SCENARIOS["normal"]
        rows = [
            {"arrived_at": start + 5, "prompt_text": "b", "num_decode_tokens": 7},
            {"arrived_at": start - 1, "prompt_text": "out-of-window", "num_decode_tokens": 1},
            {"arrived_at": start + 1, "prompt_text": "a", "num_decode_tokens": 3},
            {"arrived_at": start + 2, "prompt_text": "", "num_decode_tokens": 9},  # empty prompt dropped
            {"arrived_at": end + 1, "prompt_text": "late", "num_decode_tokens": 1},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        trace = load_trace(Path(f.name), "normal")
        self.assertEqual([r["prompt"] for r in trace], ["a", "b"])       # sorted by arrival
        self.assertEqual([r["request_id"] for r in trace], [0, 1])       # reindexed
        self.assertEqual(trace[0]["relative_arrival_s"], 1)
        self.assertEqual(trace[1]["max_tokens"], 7)


class TestOneRequest(unittest.TestCase):
    def run_req(self, endpoint: Endpoint, resp: FakeResp) -> tuple[dict, FakeSession]:
        session = FakeSession(resp)
        res = asyncio.run(one_request(session, endpoint, REQ, time.perf_counter()))
        return res, session

    def test_success_parses_stream_and_usage(self):
        res, session = self.run_req(LOCAL, FakeResp(200, sse_ok()))
        self.assertTrue(res["success"])
        self.assertEqual(res["chunks"], 2)
        self.assertEqual(res["prompt_tokens"], 10)
        self.assertEqual(res["completion_tokens"], 2)
        self.assertIsNotNone(res["ttft_ms"])
        self.assertIsNotNone(res["tpot_ms"])          # gen_count=2 > 1
        self.assertEqual(res["endpoint"], "local")
        self.assertEqual(res["cost_usd"], 0.0)        # local is not billed
        self.assertEqual(session.calls[0]["json"]["max_tokens"], 64)

    def test_cloud_success_is_billed_from_usage(self):
        res, _ = self.run_req(CLOUD, FakeResp(200, sse_ok()))
        self.assertTrue(res["success"])
        expected = (10 * 0.15 + 2 * 1.20) / 1e6
        self.assertAlmostEqual(res["cost_usd"], expected)

    def test_http_error_records_and_bills_zero(self):
        res, _ = self.run_req(CLOUD, FakeResp(429, [], text="rate limited"))
        self.assertFalse(res["success"])
        self.assertEqual(res["http_status"], 429)
        self.assertEqual(res["error_type"], "HTTP 429")
        self.assertEqual(res["cost_usd"], 0.0)        # failed request: never bill

    def test_stream_error_records(self):
        lines = ['data: {"error":{"message":"boom"}}\n']
        res, _ = self.run_req(LOCAL, FakeResp(200, lines))
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "StreamError")
        self.assertEqual(res["error"], "boom")

    def test_max_tokens_override(self):
        session = FakeSession(FakeResp(200, sse_ok()))
        asyncio.run(one_request(session, LOCAL, REQ, time.perf_counter(),
                                max_tokens_override=512))
        self.assertEqual(session.calls[0]["json"]["max_tokens"], 512)

    def test_payload_shape_matches_baseline(self):
        payload = make_payload(LOCAL, REQ, None)
        self.assertEqual(payload, {
            "model": "m", "stream": True, "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "stream_options": {"include_usage": True},
        })


class TestCost(unittest.TestCase):
    def test_missing_usage_bills_zero(self):
        res = {"success": True, "prompt_tokens": None, "completion_tokens": None}
        self.assertEqual(compute_cost_usd(CLOUD, res), 0.0)


class TestSummarize(unittest.TestCase):
    def mk(self, endpoint: str, ttft: float | None, success: bool = True,
           cost: float = 0.0, error_type: str | None = None) -> dict:
        return {"endpoint": endpoint, "success": success, "ttft_ms": ttft,
                "tpot_ms": 20.0 if success else None, "cost_usd": cost,
                "error_type": error_type}

    def test_split_violations_cost(self):
        results = [
            self.mk("local", 100.0),
            self.mk("local", 9000.0),                                   # SLO violation
            self.mk("local", None, success=False, error_type="TimeoutError"),
            self.mk("cloud", 800.0, cost=0.001),
            self.mk("cloud", 900.0, cost=0.002),
        ]
        p = Policy("random", 0.4, seed=0)
        p.n_total, p.n_outsourced = 5, 2
        s = summarize(results, p, slo_s=5.0)
        self.assertEqual(s["overall"]["n"], 5)
        self.assertEqual(s["local"]["n"], 3)
        self.assertEqual(s["cloud"]["n"], 2)
        self.assertEqual(s["local"]["slo_violations"], 2)               # 9s + failure
        self.assertEqual(s["cloud"]["slo_violations"], 0)
        self.assertAlmostEqual(s["overall"]["cost_usd"], 0.003)
        self.assertEqual(s["actual_fraction"], 0.4)
        self.assertEqual(s["local"]["errors"], {"TimeoutError": 1})
        self.assertEqual(s["local"]["ttft_p50_ms"], 100.0)              # nearest-rank of [100, 9000]

    def test_empty_side(self):
        s = summarize([self.mk("local", 100.0)], Policy("all_local", 0, 0), 5.0)
        self.assertEqual(s["cloud"]["n"], 0)
        self.assertIsNone(s["cloud"]["ttft_p50_ms"])
        self.assertEqual(s["cloud"]["slo_violation_pct"], 0.0)


class TestNullCloud(unittest.TestCase):
    """The default architecture-proof sink: route & count, no latency claims."""

    def test_routes_counts_and_makes_no_latency_claim(self):
        from router.common import NullCloud
        req = dict(REQ, prompt_tokens=1000, max_tokens=200)
        r = NullCloud(CLOUD).serve(req, 0.0)
        self.assertTrue(r["success"])
        self.assertTrue(r["routed_only"])
        self.assertEqual(r["endpoint"], "cloud")
        self.assertIsNone(r["ttft_ms"])                    # nothing modeled
        self.assertEqual(r["prompt_tokens"], 1000)
        self.assertEqual(r["completion_tokens"], 200)
        self.assertAlmostEqual(r["cost_usd"], (1000 * 0.15 + 200 * 1.20) / 1e6)

    def test_max_tokens_override_replaces_like_payload(self):
        """Regression (PR #3 review): NullCloud must mirror make_payload —
        --max-tokens REPLACES the trace value in both directions."""
        from router.common import NullCloud
        req = dict(REQ, prompt_tokens=1000, max_tokens=200)
        r = NullCloud(CLOUD).serve(req, 0.0, max_tokens_override=16)
        self.assertEqual(r["completion_tokens"], 16)          # downward
        r2 = NullCloud(CLOUD).serve(req, 0.0, max_tokens_override=512)
        self.assertEqual(r2["completion_tokens"], 512)        # upward, = payload
        self.assertEqual(r2["completion_tokens"],
                         make_payload(CLOUD, req, 512)["max_tokens"])
        r3 = NullCloud(CLOUD).serve(req, 0.0)
        self.assertEqual(r3["completion_tokens"], 200)        # no override -> trace

    def test_routed_only_excluded_from_slo_stats(self):
        from router.common import NullCloud
        req = dict(REQ, prompt_tokens=10, max_tokens=5)
        cloud_rows = [NullCloud(CLOUD).serve(req, 0.0) for _ in range(3)]
        local_row = {"endpoint": "local", "success": True, "ttft_ms": 100.0,
                     "tpot_ms": 5.0, "cost_usd": 0.0, "error_type": None}
        s = summarize(cloud_rows + [local_row], Policy("random", 0.5, 0), slo_s=5.0)
        self.assertEqual(s["cloud"]["n"], 3)
        self.assertEqual(s["cloud"]["routed_only"], 3)
        self.assertEqual(s["cloud"]["slo_violations"], 0)      # no latency claim -> no viol
        self.assertEqual(s["cloud"]["slo_violation_pct"], 0.0)
        self.assertEqual(s["overall"]["slo_violations"], 0)
        self.assertGreater(s["cloud"]["cost_usd"], 0.0)        # but still counted & billed


if __name__ == "__main__":
    unittest.main()
