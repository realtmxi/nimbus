"""Tests for the E12 current-turn live budget guard."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from router.common import SCENARIOS
from tools.check_e12_live_budget import (
    BudgetCheckError,
    build_budget_attestation,
    main,
    validate_materializer_manifest_schema,
    write_json_atomic,
)
from tools.materialize_sharegpt_current_turn_trace import (
    materialize_current_turn_trace,
)


TRACE_SHA = "e838016a8e55660c565dadb1ad019770f6b88f878d8ca29f165c30887d2cb410"
NOW = datetime(2026, 7, 18, 1, 0, tzinfo=timezone.utc)


class FakeTokenizer:
    name_or_path = "/models/fake"
    model_max_length = 40_960
    chat_template = "fake-template-v1"
    init_kwargs = {"_commit_hash": "1" * 40}

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0, "b": 1}

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.assertions = (messages, tokenize, add_generation_prompt)
        return [0] * 10


def write_manifest(root: Path, **overrides: object) -> tuple[Path, str]:
    n = 11_604
    prompt_sum = 1_289_405
    original_prompt_sum = 2_000_000
    decode_sum = 3_038_796
    manifest: dict[str, object] = {
        "schema_version": 1,
        "tool_path": "tools/materialize_sharegpt_current_turn_trace.py",
        "tool_sha256": "a" * 64,
        "dependency_sha256": {
            "tools/materialize_token_aligned_trace.py": "b" * 64,
            "router/common.py": "c" * 64,
        },
        "command_argv": [
            "tools/materialize_sharegpt_current_turn_trace.py",
            "--scenario",
            "extreme_burst_1200",
            "--do-not-copy-arbitrary-manifest-text",
        ],
        "python_version": "3.11.9",
        "transformers_version": "4.57.6",
        "input": "/secret/source/path.jsonl",
        "input_sha256": "d" * 64,
        "output": "/secret/output/path.jsonl",
        "output_sha256": TRACE_SHA,
        "scenario": "extreme_burst_1200",
        "tokenizer": "/secret/model/path",
        "tokenizer_fingerprint": {
            "class": "Qwen2TokenizerFast",
            "name_or_path": "/secret/model/path",
            "vocab_size": 151_665,
            "vocab_sha256": "e" * 64,
            "chat_template_sha256": "f" * 64,
            "model_max_length": 131_072,
            "revision": "1" * 40,
        },
        "chat_template_source": {"kind": "tokenizer_default"},
        "payload_mode": "sharegpt_current_turn_retokenized",
        "cache_mode": "none",
        "semantic_scope": "verbatim_current_user_turn_only",
        "token_count_method": {
            "api": "tokenizer.apply_chat_template",
            "messages": [{"role": "user", "content": "<prompt_text>"}],
            "tokenize": True,
            "add_generation_prompt": True,
        },
        "source_fields_deliberately_not_copied": [
            "response_text",
            "block_hash_ids",
            "block_size",
        ],
        "limit": None,
        "max_decode_tokens": 1024,
        "max_context_tokens": 40_960,
        "overflow_policy": "drop",
        "input_rows_n": n,
        "scenario_rows_n": n,
        "empty_prompt_rows_n": 0,
        "selected_before_limit_n": n,
        "selected_n": n,
        "selected_source_indices_sha256": "2" * 64,
        "n": n,
        "context_overflow_affected_n": 0,
        "context_overflow_events": [],
        "unique_session_n": n,
        "source_session_id_missing_n": 0,
        "arrival": {"min": 1000, "max": 2200, "span_s": 1200},
        "actual_prompt_tokens": {
            "n": n, "sum": prompt_sum, "min": 1, "p50": 80.0,
            "p95": 300.0, "p99": 600.0, "max": 1200,
        },
        "original_trace_prompt_tokens": {
            "n": n, "sum": original_prompt_sum, "min": 1, "p50": 100.0,
            "p95": 600.0, "p99": 1200.0, "max": 5000,
        },
        "actual_minus_original_trace_tokens": {
            "n": n, "sum": prompt_sum - original_prompt_sum, "min": -4000,
            "p50": -20.0, "p95": 100.0, "p99": 300.0, "max": 700,
        },
        "decode_cap_affected_n": 0,
        "source_decode_tokens": {
            "n": n, "sum": decode_sum, "min": 1, "p50": 200.0,
            "p95": 700.0, "p99": 900.0, "max": 1024,
        },
        "output_decode_tokens": {
            "n": n, "sum": decode_sum, "min": 1, "p50": 200.0,
            "p95": 700.0, "p99": 900.0, "max": 1024,
        },
        "preservation_checks": {
            "prompt_text_exact_n": n,
            "arrived_at_exact_n": n,
            "decode_tokens_exact_n": n,
            "decode_source_provenance_n": n,
            "decode_cap_respected_n": n,
            "session_id_exact_n": n,
            "token_metadata_aligned_n": n,
        },
    }
    manifest.update(overrides)
    path = root / "trace.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def write_price_snapshot(
    root: Path, name: str = "price.json", **overrides: object
) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "captured_at_utc": "2026-07-18T00:00:00Z",
        "http_status": 200,
        "model": "qwen/qwen3-32b",
        "provider": "deepinfra",
        "matching_endpoint_count": 1,
        "context_length": 40_960,
        "prompt_price_per_token_usd": "0.00000008",
        "completion_price_per_token_usd": "0.00000028",
        "request_price_per_request_usd": "0",
        "request_price_source": "explicit_zero",
        "prompt_price_per_million_usd": "0.08",
        "completion_price_per_million_usd": "0.28",
    }
    payload.update(overrides)
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def check(root: Path, **kwargs: object) -> dict[str, object]:
    path, manifest_sha = write_manifest(root)
    price_path = write_price_snapshot(root)
    arguments: dict[str, object] = {
        "manifest_path": path,
        "price_snapshot_path": price_path,
        "expected_manifest_sha256": manifest_sha,
        "expected_trace_sha256": TRACE_SHA,
        "expected_n": 11_604,
        "expected_prompt_token_sum": 1_289_405,
        "expected_decode_token_sum": 3_038_796,
        "expected_payload_mode": "sharegpt_current_turn_retokenized",
        "expected_cache_mode": "none",
        "arm_count": 2,
        "input_price_per_million_usd": "0.08",
        "output_price_per_million_usd": "0.28",
        "budget_usd": "3",
        "now": NOW,
    }
    arguments.update(kwargs)
    return build_budget_attestation(**arguments)  # type: ignore[arg-type]


class TestE12LiveBudget(unittest.TestCase):
    def test_schema_accepts_current_materializer_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            start, _ = SCENARIOS["extreme_burst_1200"]
            source.write_text(json.dumps({
                "arrived_at": start,
                "num_prefill_tokens": 100,
                "num_decode_tokens": 20,
                "session_id": "test-session",
                "prompt_text": "synthetic test prompt",
            }) + "\n", encoding="utf-8")
            manifest = materialize_current_turn_trace(
                input_path=source,
                output_path=root / "output.jsonl",
                scenario="extreme_burst_1200",
                tokenizer=FakeTokenizer(),
                tokenizer_path="/models/fake",
                transformers_version="4.57.6",
                max_decode_tokens=1024,
                max_context_tokens=40_960,
                overflow_policy="drop",
                command_argv=[
                    "tools/materialize_sharegpt_current_turn_trace.py"
                ],
            )
            validate_materializer_manifest_schema(manifest)

    def test_exact_two_arm_cost_and_text_free_atomic_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = check(root)
            self.assertEqual(result["estimated_cost_usd"], "1.90803056")
            self.assertEqual(result["prompt_cost_usd"], "0.2063048")
            self.assertEqual(result["decode_cost_usd"], "1.70172576")
            self.assertEqual(result["request_price_per_request_usd"], "0")
            self.assertEqual(result["request_price_source"], "explicit_zero")
            self.assertEqual(result["budget_headroom_usd"], "1.09196944")

            output = root / "nested" / "budget.json"
            write_json_atomic(output, result)
            serialized = output.read_text(encoding="utf-8")
            self.assertEqual(json.loads(serialized), result)
            self.assertNotIn("/secret/source", serialized)
            self.assertNotIn("do-not-copy-arbitrary", serialized)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_rejects_budget_overrun_without_publishing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, manifest_sha = write_manifest(root)
            price = write_price_snapshot(root)
            output = root / "must-not-exist.json"
            exit_code = main([
                "--manifest", str(manifest),
                "--price-snapshot", str(price),
                "--expected-manifest-sha256", manifest_sha,
                "--expected-trace-sha256", TRACE_SHA,
                "--expected-n", "11604",
                "--expected-prompt-token-sum", "1289405",
                "--expected-decode-token-sum", "3038796",
                "--expected-payload-mode", "sharegpt_current_turn_retokenized",
                "--expected-cache-mode", "none",
                "--arm-count", "2",
                "--input-price-per-million-usd", "0.08",
                "--output-price-per-million-usd", "0.28",
                "--budget-usd", "1.90",
                "--output", str(output),
            ])
            self.assertEqual(exit_code, 2)
            self.assertFalse(output.exists())

    def test_rejects_sha_mode_population_and_sum_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mutations = [
                {"expected_manifest_sha256": "0" * 64},
                {"expected_trace_sha256": "0" * 64},
                {"expected_n": 11_603},
                {"expected_prompt_token_sum": 1_289_404},
                {"expected_decode_token_sum": 3_038_795},
                {"expected_payload_mode": "synthetic"},
                {"expected_cache_mode": "cache"},
            ]
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    with self.assertRaises(BudgetCheckError):
                        check(root, **mutation)

    def test_rejects_any_attempt_to_weaken_frozen_cost_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mutations = [
                {"arm_count": 1},
                {"arm_count": 3},
                {"input_price_per_million_usd": "0.07"},
                {"output_price_per_million_usd": "0.27"},
                {"budget_usd": "3.01"},
                {"budget_usd": "4"},
            ]
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    with self.assertRaisesRegex(BudgetCheckError, "frozen E12"):
                        check(root, **mutation)

    def test_rejects_malformed_manifest_aggregates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, manifest_sha = write_manifest(
                root,
                actual_prompt_tokens={
                    "n": 11_603,
                    "sum": 1_289_405,
                    "min": 1,
                    "p50": 80.0,
                    "p95": 300.0,
                    "p99": 600.0,
                    "max": 1200,
                },
            )
            price = write_price_snapshot(root)
            with self.assertRaisesRegex(BudgetCheckError, "population"):
                build_budget_attestation(
                    manifest_path=manifest,
                    price_snapshot_path=price,
                    expected_manifest_sha256=manifest_sha,
                    expected_trace_sha256=TRACE_SHA,
                    expected_n=11_604,
                    expected_prompt_token_sum=1_289_405,
                    expected_decode_token_sum=3_038_796,
                    expected_payload_mode="sharegpt_current_turn_retokenized",
                    expected_cache_mode="none",
                    arm_count=2,
                    input_price_per_million_usd="0.08",
                    output_price_per_million_usd="0.28",
                    budget_usd="3",
                    now=NOW,
                )

    def test_rejects_payload_field_without_reflecting_key_or_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret = "private ShareGPT payload that must never be echoed"
            manifest, manifest_sha = write_manifest(root, prompt_text=secret)
            price = write_price_snapshot(root)
            with self.assertRaises(BudgetCheckError) as caught:
                build_budget_attestation(
                    manifest_path=manifest,
                    price_snapshot_path=price,
                    expected_manifest_sha256=manifest_sha,
                    expected_trace_sha256=TRACE_SHA,
                    expected_n=11_604,
                    expected_prompt_token_sum=1_289_405,
                    expected_decode_token_sum=3_038_796,
                    expected_payload_mode="sharegpt_current_turn_retokenized",
                    expected_cache_mode="none",
                    arm_count=2,
                    input_price_per_million_usd="0.08",
                    output_price_per_million_usd="0.28",
                    budget_usd="3",
                    now=NOW,
                )
            rendered = str(caught.exception)
            self.assertNotIn("prompt_text", rendered)
            self.assertNotIn(secret, rendered)

            nested = json.loads(manifest.read_text(encoding="utf-8"))
            del nested["prompt_text"]
            nested["actual_prompt_tokens"]["prompt_text"] = secret
            with self.assertRaises(BudgetCheckError) as nested_caught:
                validate_materializer_manifest_schema(nested)
            nested_rendered = str(nested_caught.exception)
            self.assertNotIn("prompt_text", nested_rendered)
            self.assertNotIn(secret, nested_rendered)

    def test_rejects_price_snapshot_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            price = write_price_snapshot(
                root,
                "bad-price.json",
                prompt_price_per_million_usd="0.09",
            )
            with self.assertRaisesRegex(BudgetCheckError, "do not match"):
                check(root, price_snapshot_path=price)

    def test_rejects_missing_nonzero_malformed_or_noncanonical_request_price(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, value in enumerate(("0.01", 0, None, True, "nan")):
                with self.subTest(value=value):
                    price = write_price_snapshot(
                        root,
                        f"bad-request-{index}.json",
                        request_price_per_request_usd=value,
                    )
                    with self.assertRaises(BudgetCheckError):
                        check(root, price_snapshot_path=price)

            missing = write_price_snapshot(root, "missing-request.json")
            payload = json.loads(missing.read_text(encoding="utf-8"))
            del payload["request_price_per_request_usd"]
            missing.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BudgetCheckError):
                check(root, price_snapshot_path=missing)

            for source in (None, "missing", True):
                with self.subTest(source=source):
                    price = write_price_snapshot(
                        root,
                        f"bad-request-source-{source!s}.json",
                        request_price_source=source,
                    )
                    with self.assertRaises(BudgetCheckError):
                        check(root, price_snapshot_path=price)

    def test_accepts_absent_not_advertised_request_price_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            price = write_price_snapshot(
                root,
                "absent-request-price.json",
                request_price_source="absent_not_advertised",
            )
            result = check(root, price_snapshot_path=price)
            self.assertEqual(result["request_price_per_request_usd"], "0")
            self.assertEqual(
                result["request_price_source"], "absent_not_advertised"
            )

    def test_rejects_stale_future_or_noncanonical_price_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                {"captured_at_utc": "2026-07-17T23:59:59Z"},
                {"captured_at_utc": "2026-07-18T01:00:01Z"},
                {"http_status": 201},
                {"matching_endpoint_count": 2},
                {"prompt_price_per_token_usd": "0.00000007"},
                {"unexpected": "field"},
            )
            for index, overrides in enumerate(cases):
                with self.subTest(overrides=overrides):
                    price = write_price_snapshot(
                        root, f"bad-{index}.json", **overrides
                    )
                    with self.assertRaises(BudgetCheckError):
                        check(root, price_snapshot_path=price)


if __name__ == "__main__":
    unittest.main()
