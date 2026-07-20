"""Unit tests for router/common.py (shared library). No network/aiohttp: session stubbed.

Run from the repo root:  python3 -m unittest router.test_common -v
"""
from __future__ import annotations

import asyncio
import hashlib
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
from tools.materialize_token_aligned_trace import sized_unique_prompt

LOCAL = Endpoint(name="local", url="http://x/v1/chat/completions", model="m")
CLOUD = Endpoint(name="cloud", url="http://c/v1/chat/completions", model="m",
                 input_price_per_mtok=0.15, output_price_per_mtok=1.20)

REQ = {"request_id": 0, "arrived_at": 123, "relative_arrival_s": 0.0,
       "prompt": "hi", "max_tokens": 64}


class FakeResp:
    def __init__(self, status: int, sse_lines: list[str], text: str = "",
                 headers: dict[str, str] | None = None):
        self.status = status
        self._chunks = [line.encode() for line in sse_lines]
        self._text = text
        self.headers = headers or {}
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

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

    def post(self, url, headers=None, json=None, timeout=None,
             allow_redirects=None):
        self.calls.append({
            "url": url,
            "headers": headers,
            "json": json,
            "allow_redirects": allow_redirects,
        })
        return self._resp


class RaisingSession:
    def __init__(self, message: str):
        self.message = message

    def post(self, url, headers=None, json=None, timeout=None,
             allow_redirects=None):
        raise RuntimeError(self.message)


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
            {"arrived_at": start + 1, "prompt_text": "a", "num_decode_tokens": 3,
             "num_prefill_tokens": 20, "uncached_prompt_tokens": 12,
             "num_cached_tokens": 8, "payload_mode": "token_aligned_unique",
             "cache_mode": "none", "trace_num_prefill_tokens": 19},
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
        self.assertEqual(trace[0]["prompt_tokens"], 20)
        self.assertEqual(trace[0]["uncached_prompt_tokens"], 12)
        self.assertEqual(trace[0]["num_cached_tokens"], 8)
        self.assertEqual(trace[0]["trace_prompt_tokens"], 19)
        self.assertEqual(trace[0]["payload_mode"], "token_aligned_unique")


class TestTokenAlignedMaterializer(unittest.TestCase):
    class WordTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            self.assertion = (tokenize, add_generation_prompt)
            return [0] * (8 + len(messages[0]["content"].split()))

    def test_sizes_to_chat_template_and_keeps_requests_unique(self):
        tokenizer = self.WordTokenizer()
        a, a_count = sized_unique_prompt(
            tokenizer, 50, salt="test", source_index=1
        )
        b, b_count = sized_unique_prompt(
            tokenizer, 50, salt="test", source_index=2
        )
        self.assertEqual((a_count, b_count), (50, 50))
        self.assertNotEqual(a, b)
        self.assertEqual(tokenizer.assertion, (True, True))


class TestOneRequest(unittest.TestCase):
    def run_req(self, endpoint: Endpoint, resp: FakeResp) -> tuple[dict, FakeSession]:
        session = FakeSession(resp)
        res = asyncio.run(one_request(session, endpoint, REQ, time.perf_counter()))
        return res, session

    def test_success_parses_stream_and_usage(self):
        resp = FakeResp(200, sse_ok())
        res, session = self.run_req(LOCAL, resp)
        self.assertTrue(res["success"])
        self.assertTrue(res["response_completed"])
        self.assertFalse(res["stream_abort_requested"])
        self.assertEqual(resp.close_count, 0)
        self.assertEqual(res["chunks"], 2)
        self.assertEqual(res["prompt_tokens"], 10)
        self.assertEqual(res["completion_tokens"], 2)
        self.assertIsNotNone(res["ttft_ms"])
        self.assertIsNotNone(res["tpot_ms"])          # gen_count=2 > 1
        self.assertEqual(res["endpoint"], "local")
        self.assertEqual(res["cost_usd"], 0.0)        # local is not billed
        self.assertEqual(session.calls[0]["json"]["max_tokens"], 64)
        self.assertEqual(
            session.calls[0]["json"]["stream_options"],
            {"include_usage": True},
        )
        self.assertIs(session.calls[0]["allow_redirects"], False)

    def test_cloud_success_is_billed_from_usage(self):
        res, _ = self.run_req(CLOUD, FakeResp(200, sse_ok()))
        self.assertTrue(res["success"])
        expected = (10 * 0.15 + 2 * 1.20) / 1e6
        self.assertAlmostEqual(res["cost_usd"], expected)

    def test_http_error_records_and_bills_zero(self):
        sensitive_body = "rejected prompt=private ShareGPT text"
        res, _ = self.run_req(CLOUD, FakeResp(429, [], text=sensitive_body))
        self.assertFalse(res["success"])
        self.assertEqual(res["http_status"], 429)
        self.assertEqual(res["error_type"], "HTTP 429")
        self.assertNotIn(sensitive_body, res["error"])
        self.assertEqual(res["error"], "HTTP 429: provider response body omitted")
        self.assertEqual(res["cost_usd"], 0.0)        # failed request: never bill

    def test_redirect_is_not_followed_and_is_an_explicit_failure(self):
        res, session = self.run_req(CLOUD, FakeResp(307, sse_ok()))
        self.assertFalse(res["success"])
        self.assertEqual(res["http_status"], 307)
        self.assertEqual(res["error_type"], "HTTP 307")
        self.assertIs(session.calls[0]["allow_redirects"], False)

    def test_stream_error_records(self):
        sensitive_message = "prompt contained private ShareGPT text"
        lines = [f'data: {{"error":{{"message":"{sensitive_message}"}}}}\n']
        res, _ = self.run_req(LOCAL, FakeResp(200, lines))
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "StreamError")
        self.assertNotIn(sensitive_message, res["error"])
        self.assertEqual(res["error"], "provider stream error message omitted")

    def test_stream_without_generated_token_is_explicit_failure(self):
        lines = [
            'data: {"id":"gen-empty","choices":[]}\n',
            "data: [DONE]\n",
        ]
        res, _ = self.run_req(CLOUD, FakeResp(200, lines))
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "NoToken")
        self.assertIsNone(res["ttft_ms"])

    def test_transport_exception_never_persists_exception_text(self):
        sensitive = "request payload contained private ShareGPT text"
        res = asyncio.run(
            one_request(
                RaisingSession(sensitive),
                CLOUD,
                REQ,
                time.perf_counter(),
            )
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "RuntimeError")
        self.assertEqual(
            res["error"], "request transport failure; details omitted"
        )
        self.assertNotIn(sensitive, json.dumps(res))

    def test_max_tokens_override(self):
        session = FakeSession(FakeResp(200, sse_ok()))
        asyncio.run(one_request(session, LOCAL, REQ, time.perf_counter(),
                                max_tokens_override=512))
        self.assertEqual(session.calls[0]["json"]["max_tokens"], 512)

    def test_controlled_residence_sampling_overrides(self):
        session = FakeSession(FakeResp(200, sse_ok()))
        asyncio.run(one_request(
            session, LOCAL, REQ, time.perf_counter(),
            temperature=0.0, ignore_eos=True,
        ))
        payload = session.calls[0]["json"]
        self.assertEqual(payload["temperature"], 0.0)
        self.assertIs(payload["ignore_eos"], True)

    def test_openrouter_provider_preferences_are_opt_in(self):
        payload = make_payload(
            CLOUD,
            REQ,
            None,
            provider_order=["deepinfra", "groq"],
            allow_fallbacks=False,
        )
        self.assertEqual(payload["provider"], {
            "order": ["deepinfra", "groq"],
            "allow_fallbacks": False,
        })

    def test_ttft_probe_aborts_after_first_generated_token(self):
        # Put content and usage after reasoning in the same transport chunk:
        # once TTFT is observed, cancellation must not consume the rest.
        stream = [
            'data: {"id":"gen-123","model":"m",'
            '"provider":"DeepInfra","choices":[{"delta":'
            '{"reasoning":"thinking"}}]}\n'
            'data: {"id":"gen-123","model":"m",'
            '"provider":"DeepInfra","choices":[{"delta":'
            '{"content":"Hello"}}]}\n'
            'data: {"usage":{"prompt_tokens":10,"completion_tokens":50},'
            '"choices":[]}\n'
            'data: [DONE]\n'
        ]
        resp = FakeResp(
            200,
            stream,
            headers={"X-Generation-Id": "gen-123"},
        )
        session = FakeSession(resp)
        result = asyncio.run(one_request(
            session,
            CLOUD,
            REQ,
            time.perf_counter(),
            provider_order=["deepinfra"],
            allow_fallbacks=False,
            stop_after_first_token=True,
        ))

        self.assertTrue(result["success"])
        self.assertIsNotNone(result["ttft_ms"])
        self.assertFalse(result["response_completed"])
        self.assertTrue(result["stream_abort_requested"])
        self.assertEqual(result["probe_mode"], "ttft_cancel")
        self.assertIsNone(result["e2e_ms"])
        self.assertIsNone(result["tpot_ms"])
        self.assertIsNone(result["cost_usd"])
        self.assertTrue(result["cost_pending"])
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["output_chars"], len("thinking"))
        self.assertIsNone(result["prompt_tokens"])
        self.assertIsNone(result["completion_tokens"])
        self.assertEqual(
            result["generation_id_sha256"],
            hashlib.sha256(b"gen-123").hexdigest(),
        )
        self.assertNotIn("generation_id", result)
        self.assertEqual(result["provider"], "deepinfra")
        self.assertEqual(result["response_model"], "m")
        self.assertEqual(result["first_token_kind"], "reasoning")
        self.assertIsNone(result["first_content_ttft_ms"])
        self.assertEqual(result["requested_provider_order"], ["deepinfra"])
        self.assertEqual(resp.close_count, 1)
        self.assertEqual(session.calls[0]["json"]["provider"], {
            "order": ["deepinfra"], "allow_fallbacks": False,
        })
        self.assertNotIn("gen-123", json.dumps(result))

    def test_provider_and_model_mismatch_never_persist_response_strings(self):
        sensitive_provider = "DeepInfra private ShareGPT prompt"
        sensitive_model = "private/model/with-user-text"
        sensitive_id = "generation-id-private-user-text"
        lines = [
            json.dumps({
                "id": sensitive_id,
                "provider": sensitive_provider,
                "model": sensitive_model,
                "choices": [{"delta": {"content": "Hello"}}],
            })
        ]
        resp = FakeResp(200, [f"data: {lines[0]}\n"])
        result = asyncio.run(one_request(
            FakeSession(resp),
            CLOUD,
            REQ,
            time.perf_counter(),
            provider_order=["deepinfra"],
            allow_fallbacks=False,
            stop_after_first_token=True,
        ))

        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "ProtocolMismatch")
        self.assertEqual(
            result["error"],
            "provider response metadata mismatch; details omitted",
        )
        self.assertIsNone(result["provider"])
        self.assertIsNone(result["response_model"])
        self.assertEqual(
            result["generation_id_sha256"],
            hashlib.sha256(sensitive_id.encode()).hexdigest(),
        )
        rendered = json.dumps(result)
        self.assertNotIn(sensitive_provider, rendered)
        self.assertNotIn(sensitive_model, rendered)
        self.assertNotIn(sensitive_id, rendered)
        self.assertEqual(resp.close_count, 1)

    def test_fixed_probe_requires_generation_identifier(self):
        lines = [
            'data: {"model":"m","provider":"DeepInfra",'
            '"choices":[{"delta":{"content":"Hello"}}]}\n'
        ]
        resp = FakeResp(200, lines)
        result = asyncio.run(one_request(
            FakeSession(resp),
            CLOUD,
            REQ,
            time.perf_counter(),
            provider_order=["deepinfra"],
            allow_fallbacks=False,
            stop_after_first_token=True,
        ))

        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "ProtocolMismatch")
        self.assertIsNone(result["generation_id_sha256"])
        self.assertIsNone(result["provider"])
        self.assertIsNone(result["response_model"])
        self.assertEqual(resp.close_count, 1)

    def test_model_mismatch_scrubs_previously_matched_provider(self):
        sensitive_model = "qwen/qwen3-32b private prompt fragment"
        lines = [
            "data: " + json.dumps({
                "provider": "DeepInfra",
                "model": sensitive_model,
                "choices": [{"delta": {"content": "Hello"}}],
            }) + "\n",
        ]
        result = asyncio.run(one_request(
            FakeSession(FakeResp(200, lines)),
            CLOUD,
            REQ,
            time.perf_counter(),
            provider_order=["deepinfra"],
            allow_fallbacks=False,
            stop_after_first_token=True,
        ))

        self.assertEqual(result["error_type"], "ProtocolMismatch")
        self.assertIsNone(result["provider"])
        self.assertIsNone(result["response_model"])
        self.assertNotIn(sensitive_model, json.dumps(result))

    def test_sse_generation_id_must_match_hashed_header(self):
        header_id = "header-secret-id"
        sse_id = "different-sse-secret-id"
        lines = [
            "data: " + json.dumps({
                "id": sse_id,
                "provider": "DeepInfra",
                "model": "m",
                "choices": [{"delta": {"content": "Hello"}}],
            }) + "\n",
        ]
        result = asyncio.run(one_request(
            FakeSession(FakeResp(
                200, lines, headers={"X-Generation-Id": header_id}
            )),
            CLOUD,
            REQ,
            time.perf_counter(),
            provider_order=["deepinfra"],
            stop_after_first_token=True,
        ))

        self.assertEqual(result["error_type"], "ProtocolMismatch")
        rendered = json.dumps(result)
        self.assertNotIn(header_id, rendered)
        self.assertNotIn(sse_id, rendered)

    def test_unpinned_provider_metadata_is_omitted(self):
        sensitive_provider = "provider-private-prompt-fragment"
        lines = [
            "data: " + json.dumps({
                "provider": sensitive_provider,
                "model": "m",
                "choices": [{"delta": {"content": "Hello"}}],
            }) + "\n",
            "data: [DONE]\n",
        ]
        result, _ = self.run_req(LOCAL, FakeResp(200, lines))

        self.assertTrue(result["success"])
        self.assertIsNone(result["provider"])
        self.assertEqual(result["response_model"], "m")
        self.assertNotIn(sensitive_provider, json.dumps(result))

    def test_output_progress_uses_exact_continuous_usage_not_chunk_count(self):
        mtp_stream = [
            'data: {"usage":{"prompt_tokens":10,"completion_tokens":3},'
            '"choices":[{"delta":{"content":"Hello"}}]}\n',
            'data: {"usage":{"prompt_tokens":10,"completion_tokens":6},'
            '"choices":[{"delta":{"content":" world"}}]}\n',
            'data: {"usage":{"prompt_tokens":10,"completion_tokens":6},'
            '"choices":[]}\n',
            "data: [DONE]\n",
        ]
        session = FakeSession(FakeResp(200, mtp_stream))
        progress = []
        result = asyncio.run(one_request(
            session,
            LOCAL,
            REQ,
            time.perf_counter(),
            on_output_progress=progress.append,
        ))
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(progress, [3, 6, 6])
        self.assertGreater(progress[1], result["chunks"])  # MTP: tokens != chunks
        self.assertTrue(
            session.calls[0]["json"]["stream_options"]["continuous_usage_stats"]
        )

    def test_output_progress_fails_if_endpoint_omits_continuous_usage(self):
        session = FakeSession(FakeResp(200, sse_ok()))
        result = asyncio.run(one_request(
            session,
            LOCAL,
            REQ,
            time.perf_counter(),
            on_output_progress=lambda _generated: None,
        ))
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "ProgressUnavailable")

    def test_payload_shape_matches_baseline(self):
        payload = make_payload(LOCAL, REQ, None)
        self.assertEqual(payload, {
            "model": "m", "stream": True, "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "stream_options": {"include_usage": True},
        })


class TestCost(unittest.TestCase):
    def test_missing_cloud_usage_is_unknown_but_local_is_free(self):
        res = {"success": True, "prompt_tokens": None, "completion_tokens": None}
        self.assertIsNone(compute_cost_usd(CLOUD, res))
        self.assertEqual(compute_cost_usd(LOCAL, res), 0.0)


class TestSummarize(unittest.TestCase):
    def mk(self, endpoint: str, ttft: float | None, success: bool = True,
           cost: float | None = 0.0, error_type: str | None = None) -> dict:
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
        self.assertEqual(s["pessimistic_combined"]["slo_violations"], 4)
        self.assertEqual(s["pessimistic_combined"]["slo_n"], 5)
        self.assertEqual(s["pessimistic_combined"]["slo_violation_pct"], 80.0)

    def test_empty_side(self):
        s = summarize([self.mk("local", 100.0)], Policy("all_local", 0, 0), 5.0)
        self.assertEqual(s["cloud"]["n"], 0)
        self.assertIsNone(s["cloud"]["ttft_p50_ms"])
        self.assertEqual(s["cloud"]["slo_violation_pct"], 0.0)

    def test_pending_probe_cost_never_looks_like_zero_total(self):
        results = [
            self.mk("local", 100.0, cost=0.0),
            self.mk("cloud", 200.0, cost=None),
        ]
        results[1]["cost_pending"] = True
        p = Policy("random", 0.5, seed=0)
        p.n_total, p.n_outsourced = 2, 1
        summary = summarize(results, p, slo_s=5.0)

        self.assertEqual(summary["overall"]["slo_violations"], 0)
        self.assertIsNone(summary["overall"]["cost_usd"])
        self.assertEqual(summary["overall"]["known_cost_usd"], 0.0)
        self.assertEqual(summary["overall"]["cost_measured_n"], 1)
        self.assertEqual(summary["overall"]["cost_pending_n"], 1)
        self.assertIsNone(summary["cloud"]["cost_usd"])
        self.assertEqual(summary["cloud"]["cost_pending_n"], 1)
        self.assertIsNone(summary["pessimistic_combined"]["cost_usd"])
        self.assertEqual(summary["pessimistic_combined"]["cost_pending_n"], 1)


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
        r4 = NullCloud(CLOUD).serve(dict(req, max_tokens=0), 0.0)
        self.assertEqual(r4["completion_tokens"], 0)          # trace 0 -> payload 0, bill 0
        self.assertAlmostEqual(r4["cost_usd"], (1000 * 0.15) / 1e6)

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
        self.assertEqual(s["cloud"]["slo_measured_n"], 0)      # denominator is explicit
        self.assertEqual(s["overall"]["slo_measured_n"], 1)    # only the local row measured
        self.assertEqual(s["overall"]["slo_violations"], 0)
        self.assertGreater(s["cloud"]["cost_usd"], 0.0)        # but still counted & billed
        self.assertEqual(s["pessimistic_combined"]["slo_violations"], 3)
        self.assertEqual(s["pessimistic_combined"]["slo_violation_pct"], 75.0)


if __name__ == "__main__":
    unittest.main()
