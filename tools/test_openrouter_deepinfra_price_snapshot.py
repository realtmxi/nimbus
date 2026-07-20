"""Tests for the public OpenRouter/DeepInfra price snapshot."""
from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from tools.openrouter_deepinfra_price_snapshot import (
    PriceSnapshotError,
    build_price_snapshot,
)


MISSING = object()


def metadata(
    *,
    prompt: object = "0.00000008",
    completion: object = "0.00000028",
    request: object = MISSING,
):
    pricing = {
        "prompt": prompt,
        "completion": completion,
    }
    if request is not MISSING:
        pricing["request"] = request
    return {
        "id": "qwen/qwen3-32b",
        "private_note": "must-not-copy",
        "endpoints": [
            {
                "name": "DeepInfra private endpoint label",
                "provider_name": "DeepInfra",
                "context_length": 40_960,
                "pricing": pricing,
            }
        ],
    }


class TestOpenRouterDeepInfraPriceSnapshot(unittest.TestCase):
    def test_exact_price_is_converted_and_output_is_whitelisted(self):
        result = build_price_snapshot(
            metadata(),
            now=lambda: datetime(2026, 7, 18, tzinfo=timezone.utc),
        )
        self.assertEqual(result["prompt_price_per_million_usd"], "0.08")
        self.assertEqual(result["completion_price_per_million_usd"], "0.28")
        self.assertEqual(result["request_price_per_request_usd"], "0")
        self.assertEqual(result["request_price_source"], "absent_not_advertised")
        self.assertEqual(result["context_length"], 40_960)
        rendered = json.dumps(result)
        self.assertNotIn("private endpoint label", rendered)
        self.assertNotIn("must-not-copy", rendered)
        self.assertNotIn("private_note", rendered)

    def test_any_price_change_fails_closed(self):
        for field, value in (("prompt", "0.000000081"), ("completion", "0.00000029")):
            with self.subTest(field=field):
                kwargs = {field: value}
                with self.assertRaisesRegex(PriceSnapshotError, "price changed"):
                    build_price_snapshot(metadata(**kwargs))

    def test_request_price_may_be_absent_but_if_present_must_be_zero(self):
        for value in ("0.0001", 1, -1, None, True, "nan"):
            with self.subTest(value=value):
                with self.assertRaises(PriceSnapshotError):
                    build_price_snapshot(metadata(request=value))

        explicit = build_price_snapshot(metadata(request="0"))
        self.assertEqual(explicit["request_price_per_request_usd"], "0")
        self.assertEqual(explicit["request_price_source"], "explicit_zero")

    def test_rejects_nonzero_or_malformed_optional_text_pricing(self):
        for field in (
            "input_cache_read",
            "input_cache_write",
            "internal_reasoning",
        ):
            for value in ("0.000001", -1, None, True, "nan"):
                with self.subTest(field=field, value=value):
                    payload = metadata()
                    payload["endpoints"][0]["pricing"][field] = value
                    with self.assertRaises(PriceSnapshotError):
                        build_price_snapshot(payload)

            payload = metadata()
            payload["endpoints"][0]["pricing"][field] = "0"
            build_price_snapshot(payload)

    def test_requires_exact_model_single_provider_and_context(self):
        cases = []
        wrong_model = metadata()
        wrong_model["id"] = "other/model"
        cases.append(wrong_model)
        no_provider = metadata()
        no_provider["endpoints"] = []
        cases.append(no_provider)
        substring_only = metadata()
        substring_only["endpoints"][0]["provider_name"] = "NotDeepInfra"
        cases.append(substring_only)
        name_only = metadata()
        del name_only["endpoints"][0]["provider_name"]
        cases.append(name_only)
        short_context = metadata()
        short_context["endpoints"][0]["context_length"] = 32_768
        cases.append(short_context)
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(PriceSnapshotError):
                    build_price_snapshot(case)

        case_insensitive = metadata()
        case_insensitive["endpoints"][0]["provider_name"] = "DEEPINFRA"
        build_price_snapshot(case_insensitive)


if __name__ == "__main__":
    unittest.main()
