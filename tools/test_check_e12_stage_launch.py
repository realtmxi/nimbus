"""Tests for the offline E12 per-stage launch authorization."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tools.check_e12_live_budget import build_budget_attestation
from tools.check_e12_stage_launch import (
    LIVE_CURRENT_USAGE_FILENAME,
    MAX_JSON_BYTES,
    MAX_PROFILE_JSON_BYTES,
    VERIFY_RECEIPT_FILENAME,
    LaunchCheckError,
    LaunchContext,
    create_launch_attestation,
    recompute_matrix_fingerprint,
    validate_verify_receipt,
    verify_launch_attestation,
    write_json_atomic,
)
from tools.check_openrouter_stage_budget import build_stage_gate_attestation
from tools.openrouter_key_contract import E12_MARKETPLACE_KEY_CONTRACT


NOW = datetime(2026, 7, 20, 12, 5, tzinfo=timezone.utc)
CONTRACT_ID = "e12-live-20260720"
TRACE_SHA = "e838016a8e55660c565dadb1ad019770f6b88f878d8ca29f165c30887d2cb410"
KEY_FINGERPRINT = "9" * 64
TOKENIZER_FINGERPRINT = {
    "class": "Qwen2TokenizerFast",
    "name_or_path": "/synthetic/model",
    "vocab_size": 151_665,
    "vocab_sha256": "1" * 64,
    "chat_template_sha256": "2" * 64,
    "model_max_length": 131_072,
    "revision": "3" * 40,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def usage(at: str, value: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "captured_at_utc": at,
        "key": {
            "http_status": 200,
            "usage": value,
            "limit": 3,
            "limit_remaining": 3 - value,
            "key_fingerprint_sha256": KEY_FINGERPRINT,
            "limit_reset": None,
            "include_byok_in_limit": True,
            "is_management_key": False,
            "is_provisioning_key": False,
            "is_free_tier": False,
            "expires_at_utc": None,
        },
        "account": {
            "http_status": 403,
            "total_usage": None,
            "total_credits": None,
            "remaining_credits": None,
        },
    }


class LaunchFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.context = LaunchContext(
            commit="a" * 40,
            server_pid=1234,
            server_log_prefix_sha256="b" * 64,
            endpoint_version_sha256="c" * 64,
            endpoint_models_identity_sha256="d" * 64,
            base_url="http://127.0.0.1:8010",
            chat_url="http://127.0.0.1:8010/v1/chat/completions",
            model="local-qwen3-32b",
        )
        self.trace_manifest = dump(root / "trace.manifest.json", {
            "schema_version": 1,
            "tool_path": "tools/materialize_sharegpt_current_turn_trace.py",
            "tool_sha256": "4" * 64,
            "dependency_sha256": {
                "tools/materialize_token_aligned_trace.py": "5" * 64,
                "router/common.py": "6" * 64,
            },
            "command_argv": [
                "tools/materialize_sharegpt_current_turn_trace.py",
                "--scenario",
                "extreme_burst_1200",
            ],
            "python_version": "3.11.9",
            "transformers_version": "4.57.6",
            "input": "/synthetic/source.jsonl",
            "input_sha256": "7" * 64,
            "output": "/synthetic/output.jsonl",
            "output_sha256": TRACE_SHA,
            "n": 11604,
            "scenario": "extreme_burst_1200",
            "tokenizer": "/synthetic/model",
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
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
                "response_text", "block_hash_ids", "block_size",
            ],
            "limit": None,
            "max_decode_tokens": 1024,
            "max_context_tokens": 40960,
            "overflow_policy": "drop",
            "input_rows_n": 11604,
            "scenario_rows_n": 11604,
            "empty_prompt_rows_n": 0,
            "selected_before_limit_n": 11604,
            "selected_n": 11604,
            "selected_source_indices_sha256": "8" * 64,
            "context_overflow_affected_n": 0,
            "context_overflow_events": [],
            "unique_session_n": 11604,
            "source_session_id_missing_n": 0,
            "arrival": {"min": 1000, "max": 2200, "span_s": 1200},
            "actual_prompt_tokens": {
                "n": 11604, "sum": 1289405, "min": 1, "p50": 80.0,
                "p95": 300.0, "p99": 600.0, "max": 1200,
            },
            "original_trace_prompt_tokens": {
                "n": 11604, "sum": 2000000, "min": 1, "p50": 100.0,
                "p95": 600.0, "p99": 1200.0, "max": 5000,
            },
            "actual_minus_original_trace_tokens": {
                "n": 11604, "sum": -710595, "min": -4000, "p50": -20.0,
                "p95": 100.0, "p99": 300.0, "max": 700,
            },
            "decode_cap_affected_n": 0,
            "source_decode_tokens": {
                "n": 11604, "sum": 3038796, "min": 1, "p50": 200.0,
                "p95": 700.0, "p99": 900.0, "max": 1024,
            },
            "output_decode_tokens": {
                "n": 11604, "sum": 3038796, "min": 1, "p50": 200.0,
                "p95": 700.0, "p99": 900.0, "max": 1024,
            },
            "preservation_checks": {
                "prompt_text_exact_n": 11604,
                "arrived_at_exact_n": 11604,
                "decode_tokens_exact_n": 11604,
                "decode_source_provenance_n": 11604,
                "decode_cap_respected_n": 11604,
                "session_id_exact_n": 11604,
                "token_metadata_aligned_n": 11604,
            },
        })
        self.profile = dump(root / "profile.json", {
            "schema_version": 2,
            "predictor_model": "seq_slots_shared_prefill_lane_v1",
            "valid": True,
            "cache_mode_required": "none",
            "model": self.context.model,
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "base_url": self.context.base_url,
            "server_pid": self.context.server_pid,
            "server_log_sha256_at_start": self.context.server_log_prefix_sha256,
            "server_proc_cmdline_sha256_at_start": "e" * 64,
            "sampling": {
                "temperature": 0,
                "ignore_eos": True,
                "continuous_usage_stats": True,
            },
            "profile_config": {"nimbus_tick_ms": 250, "slo_s": 5},
            "predictor_calibration": {
                "target_slo_s": 5,
                "heldout_n": 10,
                "heldout_violation_confusion": {"false_negative": 0},
            },
        })
        self.price = dump(root / "price.json", {
            "schema_version": 1,
            "status": "pass",
            "captured_at_utc": "2026-07-20T11:55:00Z",
            "http_status": 200,
            "model": "qwen/qwen3-32b",
            "provider": "deepinfra",
            "matching_endpoint_count": 1,
            "context_length": 40960,
            "prompt_price_per_token_usd": "0.00000008",
            "completion_price_per_token_usd": "0.00000028",
            "request_price_per_request_usd": "0",
            "request_price_source": "absent_not_advertised",
            "prompt_price_per_million_usd": "0.08",
            "completion_price_per_million_usd": "0.28",
        })
        budget_payload = build_budget_attestation(
            manifest_path=self.trace_manifest,
            price_snapshot_path=self.price,
            expected_manifest_sha256=sha(self.trace_manifest),
            expected_trace_sha256=TRACE_SHA,
            expected_n=11604,
            expected_prompt_token_sum=1289405,
            expected_decode_token_sum=3038796,
            expected_payload_mode="sharegpt_current_turn_retokenized",
            expected_cache_mode="none",
            arm_count=2,
            input_price_per_million_usd="0.08",
            output_price_per_million_usd="0.28",
            budget_usd="3",
            now=NOW,
        )
        self.budget = dump(root / "budget.json", budget_payload)
        self.canary = dump(root / "canary.json", {
            "schema_version": 1,
            "status": "pass",
            "captured_at_utc": "2026-07-20T12:00:00Z",
            "request_count": 1,
            "retry_count": 0,
            "trace_data_used": False,
            "http_status": 200,
            "ttft_ms": 321.5,
            "provider": "deepinfra",
            "response_model": "qwen/qwen3-32b",
            "first_token_kind": "reasoning",
            "stream_abort_requested": True,
            "response_completed": False,
            "cost_pending": True,
            "generation_id_sha256": "f" * 64,
            "key_fingerprint_sha256": KEY_FINGERPRINT,
        })
        self.baseline = dump(
            root / "baseline.json", usage("2026-07-20T11:58:00Z", 0.0)
        )
        self.previous = dump(
            root / "previous.json", usage("2026-07-20T12:03:00Z", 0.01)
        )
        self.current = dump(
            root / "current.json", usage("2026-07-20T12:04:00Z", 0.01)
        )
        gate_payload = build_stage_gate_attestation(
            baseline_path=self.baseline,
            current_path=self.current,
            settlement_previous_path=self.previous,
            next_stage_full_upper_bound_usd="1.90803056",
            now=NOW,
        )
        self.gate = dump(root / "gate.json", gate_payload)

    def create_a(self) -> dict[str, object]:
        return create_launch_attestation(
            stage="A",
            contract_id=CONTRACT_ID,
            trace_manifest_path=self.trace_manifest,
            profile_path=self.profile,
            price_snapshot_path=self.price,
            budget_attestation_path=self.budget,
            canary_path=self.canary,
            baseline_usage_path=self.baseline,
            settlement_previous_path=self.previous,
            settlement_current_path=self.current,
            stage_budget_gate_path=self.gate,
            context=self.context,
            now=NOW,
        )

    def verify_a_kwargs(
        self, attestation: Path, live: Path, *, now: datetime | None = None
    ) -> dict[str, object]:
        live.chmod(0o600)
        return {
            "attestation_path": attestation,
            "expected_sha256": sha(attestation),
            "stage": "A",
            "contract_id": CONTRACT_ID,
            "trace_manifest_sha256": sha(self.trace_manifest),
            "profile_sha256": sha(self.profile),
            "budget_attestation_sha256": sha(self.budget),
            "context": self.context,
            "trace_manifest_path": self.trace_manifest,
            "profile_path": self.profile,
            "price_snapshot_path": self.price,
            "budget_attestation_path": self.budget,
            "canary_path": self.canary,
            "baseline_usage_path": self.baseline,
            "settlement_previous_path": self.previous,
            "settlement_current_path": self.current,
            "stage_budget_gate_path": self.gate,
            "live_current_usage_path": live,
            "now": now or datetime(2026, 7, 20, 12, 5, 30, tzinfo=timezone.utc),
        }

    def configure_c_usage(self) -> None:
        dump(self.previous, usage("2026-07-20T12:07:00Z", 0.20))
        dump(self.current, usage("2026-07-20T12:08:00Z", 0.20))
        dump(
            self.gate,
            build_stage_gate_attestation(
                baseline_path=self.baseline,
                current_path=self.current,
                settlement_previous_path=self.previous,
                next_stage_full_upper_bound_usd="0.95401528",
                now=datetime(2026, 7, 20, 12, 9, tzinfo=timezone.utc),
            ),
        )

    def make_a_shell_evidence(
        self, launch_path: Path, *, receipt_now: datetime = NOW,
    ) -> tuple[Path, dict]:
        directory = self.root / "stage_a"
        directory.mkdir()
        launch_sha = sha(launch_path)
        live_path = dump(
            directory / LIVE_CURRENT_USAGE_FILENAME,
            usage("2026-07-20T12:05:00Z", 0.01),
        )
        receipt = verify_launch_attestation(
            **self.verify_a_kwargs(launch_path, live_path, now=receipt_now)
        )
        receipt_path = directory / VERIFY_RECEIPT_FILENAME
        write_json_atomic(receipt_path, receipt, overwrite=False)
        live_sha = sha(live_path)
        receipt_sha = sha(receipt_path)
        manifest: dict[str, object] = {
            "started_at": "2026-07-20T12:05:00Z",
            "commit": self.context.commit,
            "trace_sha256": TRACE_SHA,
            "trace_manifest_sha256": sha(self.trace_manifest),
            "trace_n": "11604",
            "profile_sha256": sha(self.profile),
            "cache_mode": "none",
            "server_pid": str(self.context.server_pid),
            "server_log": "/restricted/server.log",
            "server_log_prefix_sha256": self.context.server_log_prefix_sha256,
            "server_log_sha256_at_manifest": "1" * 64,
            "endpoint_version_sha256": self.context.endpoint_version_sha256,
            "endpoint_models_identity_sha256": self.context.endpoint_models_identity_sha256,
            "base_url": self.context.base_url,
            "chat_url": self.context.chat_url,
            "model": self.context.model,
            "python": "Python-3.12",
            "scenario": "extreme_burst_1200",
            "max_inflight": "128",
            "kv_cap": "112000",
            "prefill_tput": "3000",
            "tpot_ms": "150",
            "first_token_overhead_ms": "400",
            "slo_s": "5",
            "timeout_s": "600",
            "guard_ms": "1700",
            "tick_ms": "250",
            "temperature": "0",
            "ignore_eos": "1",
            "in_price": "0.08",
            "out_price": "0.28",
            "local_ignore_eos": "1",
            "cloud_ignore_eos": "0",
            "cloud": "real",
            "cloud_url": "https://openrouter.ai/api/v1/chat/completions",
            "cloud_model": "qwen/qwen3-32b",
            "cloud_api_key_env": "OPENROUTER_API_KEY",
            "cloud_max_concurrency": "16",
            "cloud_provider": "deepinfra",
            "cloud_no_fallbacks": "1",
            "cloud_stop_after_first_token": "1",
            "real_cloud_expected_trace_n": "11604",
            "secret_value_recorded": "false",
            "arm_order_mode": "e12_contract_stage",
            "arm_order_seed": "20260716",
            "arms": ("ttft_pred:cost_cachedisp_old:0",),
            "evidence_scope": "single-pass",
            "live_contract_id": CONTRACT_ID,
            "live_contract": "e12_current_turn_v1",
            "live_stage": "A",
            "live_expected_trace_n": "11604",
            "live_arm_order_seed": "20260716",
            "live_exact_arm": "ttft_pred:cost_cachedisp_old:0",
            "live_authorized_budget_usd": "3",
            "live_budget_attestation_sha256": sha(self.budget),
            "live_key_limit_max_usd": "3",
            "live_cooldown_s": "20",
            "live_stage_launch_attestation_sha256": launch_sha,
            "live_current_usage_sha256": live_sha,
            "live_stage_launch_verify_receipt_sha256": receipt_sha,
        }
        manifest["run_fingerprint"] = recompute_matrix_fingerprint(manifest)
        lines = []
        for key, value in manifest.items():
            if key == "arms":
                lines.append(f"arms={' '.join(value)} ")
            else:
                lines.append(f"{key}={value}")
        (directory / "matrix_manifest.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        (directory / "matrix_events.log").write_text(
            "arm_started_at=2026-07-20T12:05:01Z "
            "arm=ttft_pred:cost_cachedisp_old:0 "
            f"live_contract_id={CONTRACT_ID} live_stage=A "
            "live_authorized_budget_usd=3 "
            f"live_budget_attestation_sha256={sha(self.budget)} "
            "live_key_limit_max_usd=3 live_cooldown_s=20 "
            f"live_stage_launch_attestation_sha256={launch_sha} "
            f"live_current_usage_sha256={live_sha} "
            f"live_stage_launch_verify_receipt_sha256={receipt_sha} "
            "command=router.run_config_bound_by_fingerprint\n"
            "arm_finished_at=2026-07-20T12:05:10Z "
            "arm=ttft_pred:cost_cachedisp_old:0 "
            f"live_contract_id={CONTRACT_ID} live_stage=A "
            "live_authorized_budget_usd=3 "
            f"live_budget_attestation_sha256={sha(self.budget)} "
            "live_key_limit_max_usd=3 live_cooldown_s=20 "
            f"live_stage_launch_attestation_sha256={launch_sha} "
            f"live_current_usage_sha256={live_sha} "
            f"live_stage_launch_verify_receipt_sha256={receipt_sha}\n"
            "matrix_finished_at=2026-07-20T12:05:30Z "
            f"run_fingerprint={manifest['run_fingerprint']} "
            f"live_contract_id={CONTRACT_ID} live_stage=A "
            "live_authorized_budget_usd=3 "
            f"live_budget_attestation_sha256={sha(self.budget)} "
            "live_key_limit_max_usd=3 live_cooldown_s=20 "
            f"live_stage_launch_attestation_sha256={launch_sha} "
            f"live_current_usage_sha256={live_sha} "
            f"live_stage_launch_verify_receipt_sha256={receipt_sha}\n",
            encoding="utf-8",
        )
        marker_path = directory / "run_a.complete.json"
        dump(marker_path, {"schema_version": 2})
        (directory / "run_a.jsonl").write_text(
            json.dumps({
                "endpoint": "cloud", "http_status": 200, "success": True,
            }) + "\n",
            encoding="utf-8",
        )
        marker = {
            "schema_version": 2,
            "run_fingerprint": manifest["run_fingerprint"],
            "arm": "ttft_pred:cost_cachedisp_old:0",
            "artifacts": {
                "raw": {"sha256": "2" * 64, "nonempty_line_n": 11604},
                "summary": {"sha256": "3" * 64, "nonempty_line_n": 1},
                "decisions": {"sha256": "4" * 64, "nonempty_line_n": 1},
            },
        }
        return directory, marker


class TestE12StageLaunch(unittest.TestCase):
    def test_profile_has_a_narrow_larger_json_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            original = fixture.profile.read_bytes()
            real_profile_size = 3_090_188
            fixture.profile.write_bytes(
                original + b" " * (real_profile_size - len(original))
            )
            self.assertEqual(fixture.profile.stat().st_size, real_profile_size)
            self.assertLessEqual(
                fixture.profile.stat().st_size, MAX_PROFILE_JSON_BYTES,
            )
            self.assertEqual(fixture.create_a()["stage"], "A")

            fixture.profile.write_bytes(
                fixture.profile.read_bytes()
                + b" " * (
                    MAX_PROFILE_JSON_BYTES + 1 - fixture.profile.stat().st_size
                )
            )
            with self.assertRaisesRegex(LaunchCheckError, "profile is too large"):
                fixture.create_a()

        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            original = fixture.price.read_bytes()
            fixture.price.write_bytes(
                original + b" " * (MAX_JSON_BYTES + 1 - len(original))
            )
            with self.assertRaisesRegex(
                LaunchCheckError, "price snapshot is too large",
            ):
                fixture.create_a()

    def test_marketplace_contract_binds_non_byok_usage_and_exact_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            canary = json.loads(fixture.canary.read_text())
            canary["is_byok"] = False
            dump(fixture.canary, canary)
            for path in (fixture.baseline, fixture.previous, fixture.current):
                payload = json.loads(path.read_text())
                key = payload["key"]
                key["limit"] = 5
                key["limit_remaining"] = 5 - key["usage"]
                key["include_byok_in_limit"] = False
                key["byok_usage"] = 0.4
                dump(path, payload)
            dump(fixture.gate, build_stage_gate_attestation(
                baseline_path=fixture.baseline,
                current_path=fixture.current,
                settlement_previous_path=fixture.previous,
                next_stage_full_upper_bound_usd="1.90803056",
                now=NOW,
                key_contract_mode=E12_MARKETPLACE_KEY_CONTRACT,
            ))
            result = create_launch_attestation(
                stage="A",
                contract_id=CONTRACT_ID,
                trace_manifest_path=fixture.trace_manifest,
                profile_path=fixture.profile,
                price_snapshot_path=fixture.price,
                budget_attestation_path=fixture.budget,
                canary_path=fixture.canary,
                baseline_usage_path=fixture.baseline,
                settlement_previous_path=fixture.previous,
                settlement_current_path=fixture.current,
                stage_budget_gate_path=fixture.gate,
                context=fixture.context,
                now=NOW,
                key_contract_mode=E12_MARKETPLACE_KEY_CONTRACT,
            )
            self.assertEqual(result["settled_key"]["limit_usd"], "5")
            self.assertEqual(result["settled_key"]["byok_usage_usd"], "0.4")
            self.assertEqual(
                result["frozen_contract"]["openrouter_billing_route"],
                "marketplace",
            )
            self.assertEqual(
                result["frozen_contract"]["static_full_pair_bound_usd"],
                "1.90803056",
            )

    def test_create_a_and_verify_live_key_are_text_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            result = fixture.create_a()
            self.assertEqual(result["stage"], "A")
            self.assertEqual(result["prior_a"], None)
            self.assertEqual(
                result["frozen_contract"]["stage_future_bound_usd"],
                "1.90803056",
            )
            rendered = json.dumps(result, sort_keys=True)
            for forbidden in (
                CONTRACT_ID,
                "OPENROUTER_API_KEY",
                "qwen/qwen3-32b",
                "prompt_text",
                str(fixture.root),
            ):
                self.assertNotIn(forbidden, rendered)

            attestation = fixture.root / "launch.json"
            write_json_atomic(attestation, result)
            live = dump(
                fixture.root / "live.json",
                usage("2026-07-20T12:05:15Z", 0.01),
            )
            receipt = verify_launch_attestation(
                **fixture.verify_a_kwargs(attestation, live)
            )
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["key_usage_usd"], "0.01")
            self.assertEqual(receipt["attestation_sha256"], sha(attestation))
            self.assertEqual(receipt["live_current_usage_sha256"], sha(live))
            receipt_path = fixture.root / VERIFY_RECEIPT_FILENAME
            write_json_atomic(receipt_path, receipt, overwrite=False)
            self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                validate_verify_receipt(
                    receipt_path=receipt_path,
                    expected_sha256=sha(receipt_path),
                    launch_attestation_path=attestation,
                    live_current_usage_path=live,
                ),
                receipt,
            )
            with self.assertRaisesRegex(LaunchCheckError, "unable to publish"):
                write_json_atomic(receipt_path, receipt, overwrite=False)
            changed = json.loads(live.read_text())
            changed["captured_at_utc"] = "2026-07-20T12:05:16Z"
            dump(live, changed)
            with self.assertRaisesRegex(LaunchCheckError, "usage hash"):
                validate_verify_receipt(
                    receipt_path=receipt_path,
                    expected_sha256=sha(receipt_path),
                    launch_attestation_path=attestation,
                    live_current_usage_path=live,
                )

    def test_rejects_wrong_stage_bound_and_unsettled_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            wrong_gate = build_stage_gate_attestation(
                baseline_path=fixture.baseline,
                current_path=fixture.current,
                settlement_previous_path=fixture.previous,
                next_stage_full_upper_bound_usd="0.95401528",
                now=NOW,
            )
            dump(fixture.gate, wrong_gate)
            with self.assertRaisesRegex(LaunchCheckError, "exact settled"):
                fixture.create_a()

        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            payload = json.loads(fixture.previous.read_text())
            payload["key"]["usage"] = 0.02
            payload["key"]["limit_remaining"] = 2.98
            dump(fixture.previous, payload)
            with self.assertRaises(LaunchCheckError):
                fixture.create_a()

    def test_rejects_live_key_swap_counter_change_and_sha_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            attestation = fixture.root / "launch.json"
            write_json_atomic(attestation, fixture.create_a())
            changed = dump(
                fixture.root / "changed.json",
                usage("2026-07-20T12:05:15Z", 0.02),
            )
            kwargs = fixture.verify_a_kwargs(attestation, changed)
            with self.assertRaisesRegex(LaunchCheckError, "live key counters"):
                verify_launch_attestation(**kwargs)
            kwargs["expected_sha256"] = "0" * 64
            with self.assertRaisesRegex(LaunchCheckError, "SHA256 differs"):
                verify_launch_attestation(**kwargs)

    def test_verify_revalidates_sources_instead_of_trusting_pass_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            attestation = fixture.root / "launch.json"
            write_json_atomic(attestation, fixture.create_a())
            live = dump(
                fixture.root / "live.json",
                usage("2026-07-20T12:05:15Z", 0.01),
            )
            canary = json.loads(fixture.canary.read_text())
            canary["http_status"] = 401
            dump(fixture.canary, canary)
            with self.assertRaisesRegex(LaunchCheckError, "canary"):
                verify_launch_attestation(
                    **fixture.verify_a_kwargs(attestation, live)
                )

    def test_c_fails_closed_without_completed_a(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            # Rebuild the gate with C's exact one-arm upper bound; absence of
            # A evidence must still stop authorization.
            gate = build_stage_gate_attestation(
                baseline_path=fixture.baseline,
                current_path=fixture.current,
                settlement_previous_path=fixture.previous,
                next_stage_full_upper_bound_usd="0.95401528",
                now=NOW,
            )
            dump(fixture.gate, gate)
            with self.assertRaisesRegex(LaunchCheckError, "prior-A"):
                create_launch_attestation(
                    stage="C",
                    contract_id=CONTRACT_ID,
                    trace_manifest_path=fixture.trace_manifest,
                    profile_path=fixture.profile,
                    price_snapshot_path=fixture.price,
                    budget_attestation_path=fixture.budget,
                    canary_path=fixture.canary,
                    baseline_usage_path=fixture.baseline,
                    settlement_previous_path=fixture.previous,
                    settlement_current_path=fixture.current,
                    stage_budget_gate_path=fixture.gate,
                    context=fixture.context,
                    now=NOW,
                )

    def test_c_orders_fractional_receipt_against_whole_second_events(self) -> None:
        cases = (
            (datetime(2026, 7, 20, 12, 5, 0, 728166, tzinfo=timezone.utc), True),
            (datetime(2026, 7, 20, 12, 5, 1, 1, tzinfo=timezone.utc), False),
        )
        for receipt_now, should_pass in cases:
            with self.subTest(receipt_now=receipt_now, should_pass=should_pass):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = LaunchFixture(Path(directory))
                    a_launch = fixture.root / "a-launch.json"
                    write_json_atomic(a_launch, fixture.create_a())
                    a_dir, marker = fixture.make_a_shell_evidence(
                        a_launch, receipt_now=receipt_now,
                    )
                    fixture.configure_c_usage()

                    def create_c() -> dict[str, object]:
                        return create_launch_attestation(
                            stage="C",
                            contract_id=CONTRACT_ID,
                            trace_manifest_path=fixture.trace_manifest,
                            profile_path=fixture.profile,
                            price_snapshot_path=fixture.price,
                            budget_attestation_path=fixture.budget,
                            canary_path=fixture.canary,
                            baseline_usage_path=fixture.baseline,
                            settlement_previous_path=fixture.previous,
                            settlement_current_path=fixture.current,
                            stage_budget_gate_path=fixture.gate,
                            context=fixture.context,
                            a_dir=a_dir,
                            a_launch_attestation_path=a_launch,
                            now=datetime(2026, 7, 20, 12, 9, tzinfo=timezone.utc),
                        )

                    with patch(
                        "tools.check_e12_stage_launch.validate_or_write_marker",
                        return_value=marker,
                    ):
                        if should_pass:
                            self.assertEqual(create_c()["stage"], "C")
                        else:
                            with self.assertRaisesRegex(
                                LaunchCheckError,
                                "stage A launch/event times are not ordered",
                            ):
                                create_c()

    def test_c_binds_exact_completed_a_and_rejects_lifecycle_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            a_launch = fixture.root / "a-launch.json"
            write_json_atomic(a_launch, fixture.create_a())
            a_dir, marker = fixture.make_a_shell_evidence(a_launch)
            fixture.configure_c_usage()
            now = datetime(2026, 7, 20, 12, 9, tzinfo=timezone.utc)

            def create_c() -> dict[str, object]:
                return create_launch_attestation(
                    stage="C",
                    contract_id=CONTRACT_ID,
                    trace_manifest_path=fixture.trace_manifest,
                    profile_path=fixture.profile,
                    price_snapshot_path=fixture.price,
                    budget_attestation_path=fixture.budget,
                    canary_path=fixture.canary,
                    baseline_usage_path=fixture.baseline,
                    settlement_previous_path=fixture.previous,
                    settlement_current_path=fixture.current,
                    stage_budget_gate_path=fixture.gate,
                    context=fixture.context,
                    a_dir=a_dir,
                    a_launch_attestation_path=a_launch,
                    now=now,
                )

            with patch(
                "tools.check_e12_stage_launch.validate_or_write_marker",
                return_value=marker,
            ) as validate_marker:
                result = create_c()
            self.assertEqual(result["stage"], "C")
            self.assertEqual(result["prior_a"]["raw_n"], 11604)
            self.assertEqual(
                result["prior_a"]["matrix_finished_at_utc"],
                "2026-07-20T12:05:30Z",
            )
            self.assertEqual(result["prior_a"]["authorization_interval_s"], "210")
            self.assertEqual(
                validate_marker.call_args.kwargs["arm_text"],
                "ttft_pred:cost_cachedisp_old:0",
            )
            self.assertFalse(validate_marker.call_args.kwargs["write_marker"])

            raw_path = a_dir / "run_a.jsonl"
            for status in (401, 404, 405, 422):
                raw_path.write_text(
                    json.dumps({
                        "endpoint": "cloud", "http_status": status,
                        "success": False,
                    }) + "\n",
                    encoding="utf-8",
                )
                with self.subTest(status=status), patch(
                    "tools.check_e12_stage_launch.validate_or_write_marker",
                    return_value=marker,
                ):
                    with self.assertRaisesRegex(LaunchCheckError, "systemic cloud"):
                        create_c()

            raw_path.write_text(
                json.dumps({
                    "endpoint": "cloud", "http_status": 429,
                    "success": False,
                }) + "\n",
                encoding="utf-8",
            )
            with patch(
                "tools.check_e12_stage_launch.validate_or_write_marker",
                return_value=marker,
            ):
                with self.assertRaisesRegex(LaunchCheckError, "systemic cloud"):
                    create_c()

            dominant_rows = [
                {"endpoint": "cloud", "http_status": 200, "success": True}
                for _ in range(2)
            ] + [
                {"endpoint": "cloud", "http_status": 503, "success": False}
                for _ in range(8)
            ]
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in dominant_rows),
                encoding="utf-8",
            )
            with patch(
                "tools.check_e12_stage_launch.validate_or_write_marker",
                return_value=marker,
            ):
                with self.assertRaisesRegex(LaunchCheckError, "systemic cloud"):
                    create_c()
            raw_path.write_text(
                json.dumps({
                    "endpoint": "cloud", "http_status": 200,
                    "success": True,
                }) + "\n",
                encoding="utf-8",
            )

            manifest_path = a_dir / "matrix_manifest.txt"
            text = manifest_path.read_text()
            manifest_path.write_text(
                text.replace("server_pid=1234", "server_pid=9999"),
                encoding="utf-8",
            )
            with patch(
                "tools.check_e12_stage_launch.validate_or_write_marker",
                return_value=marker,
            ):
                with self.assertRaisesRegex(LaunchCheckError, "server_pid"):
                    create_c()

    def test_rejects_resettable_or_special_live_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            attestation = fixture.root / "launch.json"
            write_json_atomic(attestation, fixture.create_a())
            payload = usage("2026-07-20T12:05:15Z", 0.01)
            payload["key"]["limit_reset"] = "daily"
            live = dump(fixture.root / "live.json", payload)
            with self.assertRaisesRegex(LaunchCheckError, "key contract"):
                verify_launch_attestation(
                    **fixture.verify_a_kwargs(attestation, live)
                )

    def test_rejects_text_or_extra_fields_in_live_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            attestation = fixture.root / "launch.json"
            write_json_atomic(attestation, fixture.create_a())
            payload = usage("2026-07-20T12:05:15Z", 0.01)
            payload["prompt_text"] = "sensitive-current-turn"
            live = dump(fixture.root / "live.json", payload)
            with self.assertRaisesRegex(LaunchCheckError, "schema"):
                verify_launch_attestation(
                    **fixture.verify_a_kwargs(attestation, live)
                )

    def test_rejects_canary_from_a_different_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = LaunchFixture(Path(directory))
            payload = json.loads(fixture.canary.read_text())
            payload["key_fingerprint_sha256"] = "8" * 64
            dump(fixture.canary, payload)
            with self.assertRaisesRegex(LaunchCheckError, "different keys"):
                fixture.create_a()


if __name__ == "__main__":
    unittest.main()
