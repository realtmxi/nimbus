"""Tests for the synthetic OpenRouter TTFT-cancel canary."""
from __future__ import annotations

import asyncio
import hashlib
import json
import unittest
from unittest.mock import AsyncMock, patch

from tools.run_openrouter_ttft_canary import (
    OPENROUTER_MODEL,
    OPENROUTER_URL,
    CanaryError,
    assess_result,
    fetch_generation_is_byok,
    run_canary,
)


KEY = "TEST-OPENROUTER-CANARY-KEY"
KEY_FINGERPRINT = hashlib.sha256(KEY.encode()).hexdigest()


def valid_result() -> dict:
    return {
        "success": True,
        "http_status": 200,
        "ttft_ms": 321.5,
        "probe_mode": "ttft_cancel",
        "stream_abort_requested": True,
        "response_completed": False,
        "first_token_kind": "reasoning",
        "requested_provider_order": ["deepinfra"],
        "provider": "deepinfra",
        "response_model": "qwen/qwen3-32b",
        "cost_pending": True,
        "cost_usd": None,
        "generation_id_sha256": hashlib.sha256(
            b"gen-sensitive-provider-id"
        ).hexdigest(),
        "error": None,
        "prompt": "must never be copied",
    }


class TestOpenRouterTTFTCanary(unittest.TestCase):
    def test_generation_metadata_check_is_no_redirect_and_exact(self):
        class Content:
            async def read(self, limit):
                return b'{"data":{"is_byok":false}}'

        class Response:
            status = 200
            content = Content()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        class Session:
            def __init__(self):
                self.kwargs = None

            def get(self, url, **kwargs):
                self.url = url
                self.kwargs = kwargs
                return Response()

        session = Session()
        result = asyncio.run(fetch_generation_is_byok(
            session,
            api_key=KEY,
            generation_id="gen-sensitive-provider-id",
            attempts=1,
            poll_s=0,
        ))
        self.assertIs(result, False)
        self.assertIs(session.kwargs["allow_redirects"], False)
        self.assertEqual(
            session.kwargs["params"], {"id": "gen-sensitive-provider-id"}
        )

    def test_pass_output_is_whitelisted_and_hashes_generation_id(self):
        with patch(
            "tools.run_openrouter_ttft_canary._utc_now",
            return_value="2026-07-18T00:00:00Z",
        ):
            result = assess_result(
                valid_result(), key_fingerprint_sha256=KEY_FINGERPRINT,
                is_byok=False,
            )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["provider"], "deepinfra")
        self.assertEqual(result["ttft_ms"], 321.5)
        self.assertEqual(result["key_fingerprint_sha256"], KEY_FINGERPRINT)
        self.assertEqual(
            result["generation_id_sha256"],
            hashlib.sha256(b"gen-sensitive-provider-id").hexdigest(),
        )
        rendered = json.dumps(result)
        self.assertNotIn("gen-sensitive-provider-id", rendered)
        self.assertNotIn("must never be copied", rendered)
        self.assertNotIn("prompt", rendered)

    def test_rejects_invalid_transport_contract(self):
        mutations = {
            "success": False,
            "http_status": 401,
            "ttft_ms": None,
            "probe_mode": None,
            "stream_abort_requested": False,
            "response_completed": True,
            "first_token_kind": None,
            "requested_provider_order": ["other"],
            "provider": "Other",
            "response_model": "other/model",
            "cost_pending": False,
            "cost_usd": 0.0,
            "generation_id_sha256": "not-a-canonical-sha256",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                row = valid_result()
                row[field] = value
                with self.assertRaises(CanaryError):
                    assess_result(
                        row, key_fingerprint_sha256=KEY_FINGERPRINT,
                        is_byok=False,
                    )

    def test_rejects_nonfinite_or_negative_ttft(self):
        for value in (float("nan"), float("inf"), -1.0, True):
            with self.subTest(value=value):
                row = valid_result()
                row["ttft_ms"] = value
                with self.assertRaises(CanaryError):
                    assess_result(
                        row, key_fingerprint_sha256=KEY_FINGERPRINT,
                        is_byok=False,
                    )

    def test_run_canary_freezes_transport_and_payload_contract(self):
        class FakeConnector:
            def __init__(self, *, limit):
                self.limit = limit

        class FakeSession:
            def __init__(self, *, connector):
                self.connector = connector

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        class FakeAiohttp:
            TCPConnector = FakeConnector
            ClientSession = FakeSession

        async def send(*args, **kwargs):
            kwargs["on_generation_id"]("gen-sensitive-provider-id")
            return valid_result()

        sender = AsyncMock(side_effect=send)
        generation_metadata = AsyncMock(return_value=False)
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": KEY}, clear=False),
            patch("tools.run_openrouter_ttft_canary.common.aiohttp", FakeAiohttp),
            patch("tools.run_openrouter_ttft_canary.one_request", sender),
            patch(
                "tools.run_openrouter_ttft_canary.fetch_generation_is_byok",
                generation_metadata,
            ),
        ):
            result = asyncio.run(run_canary(
                api_key_env="OPENROUTER_API_KEY", timeout_s=120.0
            ))

        args, kwargs = sender.await_args
        endpoint = args[1]
        request = args[2]
        self.assertEqual(endpoint.url, OPENROUTER_URL)
        self.assertEqual(endpoint.model, OPENROUTER_MODEL)
        self.assertEqual(endpoint.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(request["max_tokens"], 1)
        self.assertEqual(request["prompt"], (
            "Nimbus E12 synthetic TTFT canary; no user or trace data."
        ))
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertIs(kwargs["ignore_eos"], False)
        self.assertEqual(kwargs["provider_order"], ["deepinfra"])
        self.assertIs(kwargs["allow_fallbacks"], False)
        self.assertIs(kwargs["stop_after_first_token"], True)
        self.assertEqual(result["key_fingerprint_sha256"], KEY_FINGERPRINT)
        self.assertIs(result["is_byok"], False)
        metadata_kwargs = generation_metadata.await_args.kwargs
        self.assertEqual(metadata_kwargs["generation_id"], "gen-sensitive-provider-id")

    def test_rejects_byok_generation(self):
        with self.assertRaisesRegex(CanaryError, "marketplace-billed"):
            assess_result(
                valid_result(),
                key_fingerprint_sha256=KEY_FINGERPRINT,
                is_byok=True,
            )


if __name__ == "__main__":
    unittest.main()
