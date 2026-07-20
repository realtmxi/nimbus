"""Focused end-to-end synthetic tests for the E12 live final auditor."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest import mock

from tools.audit_e12_live import (
    CACHE_MODE,
    CLOUD_MODEL,
    EvidenceError,
    LIVE_CONTRACT,
    PAYLOAD_MODE,
    TraceIdentity,
    _parse_manifest,
    audit,
    main,
    recompute_run_fingerprint,
    render_json,
)
from tools.check_e12_live_budget import build_budget_attestation
from tools.check_e12_stage_launch import (
    LIVE_CURRENT_USAGE_FILENAME,
    VERIFY_RECEIPT_FILENAME,
    LaunchContext,
    create_launch_attestation,
    verify_launch_attestation,
)
from tools.check_openrouter_stage_budget import build_stage_gate_attestation
from tools.openrouter_deepinfra_price_snapshot import build_price_snapshot
from tools.openrouter_usage_snapshot import capture_usage_snapshot
from tools.run_openrouter_ttft_canary import assess_result


N = 4
TRACE_SHA = "1" * 64
# Keep the real E12 aggregate sums so the stage-launch validator's exact
# $1.90803056 bound is exercised while request-row fixtures remain tiny.
PROMPT_SUM = 1_289_405
DECODE_SUM = 3_038_796
ORIGINAL_PROMPT_SUM = 2_000_000
COMMIT = "a" * 40
API_KEY = "synthetic-dedicated-openrouter-key"
KEY_FINGERPRINT = hashlib.sha256(API_KEY.encode()).hexdigest()
EXPIRY = "2026-07-19T00:00:00Z"
ARMS = {
    "A": "ttft_pred:cost_cachedisp_old:0",
    "C": "ttft_pred:cost_disp_current:0",
}
SELECTORS = {"A": "cost_cachedisp_old", "C": "cost_disp_current"}
ROUTES = {"A": {0, 1}, "C": {1, 2}}
TOKENIZER_FINGERPRINT = {
    "class": "Qwen2TokenizerFast",
    "name_or_path": "/synthetic/model",
    "vocab_size": 151_665,
    "vocab_sha256": "2" * 64,
    "chat_template_sha256": "3" * 64,
    "model_max_length": 131_072,
    "revision": "4" * 40,
}
TIMES = {
    "baseline": "2026-07-18T00:00:00Z",
    "canary": "2026-07-18T00:00:01Z",
    "pre_a_previous": "2026-07-18T00:01:10Z",
    "pre_a_current": "2026-07-18T00:02:10Z",
    "a_authorized": "2026-07-18T00:02:11Z",
    "a_live": "2026-07-18T00:02:11Z",
    "a_verified": "2026-07-18T00:02:11Z",
    "a_started": "2026-07-18T00:02:12Z",
    "a_finished": "2026-07-18T00:02:20Z",
    "a_matrix": "2026-07-18T00:02:21Z",
    "post_a_previous": "2026-07-18T00:03:30Z",
    "post_a_current": "2026-07-18T00:04:30Z",
    "c_authorized": "2026-07-18T00:04:31Z",
    "c_live": "2026-07-18T00:04:31Z",
    "c_verified": "2026-07-18T00:04:31Z",
    "c_started": "2026-07-18T00:04:32Z",
    "c_finished": "2026-07-18T00:04:40Z",
    "c_matrix": "2026-07-18T00:04:41Z",
    "final_previous": "2026-07-18T00:05:50Z",
    "final_current": "2026-07-18T00:06:50Z",
}


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def artifact(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "nonempty_line_n": sum(bool(line.strip()) for line in raw.splitlines()),
    }


class JsonResponse:
    def __init__(self, payload: dict[str, object], url: str) -> None:
        self.status = 200
        self.raw = json.dumps({"data": payload}).encode()
        self.url = url

    def __enter__(self) -> "JsonResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self.url

    def read(self, _: int = -1) -> bytes:
        return self.raw


class LiveFixture(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.context = LaunchContext(
            commit=COMMIT,
            server_pid=123,
            server_log_prefix_sha256="4" * 64,
            endpoint_version_sha256="6" * 64,
            endpoint_models_identity_sha256="7" * 64,
            base_url="http://127.0.0.1:8010",
            chat_url="http://127.0.0.1:8010/v1/chat/completions",
            model="local-qwen3-32b",
        )
        self.trace_manifest = self.make_trace_manifest()
        trace_manifest_sha = hashlib.sha256(
            self.trace_manifest.read_bytes()
        ).hexdigest()
        self.identity = TraceIdentity(
            n=N,
            trace_sha256=TRACE_SHA,
            manifest_sha256=trace_manifest_sha,
            prompt_token_sum=PROMPT_SUM,
            decode_token_sum=DECODE_SUM,
        )
        self.profile = self.make_profile()
        self.profile_sha = hashlib.sha256(self.profile.read_bytes()).hexdigest()
        self.price = self.make_price()
        with self.launch_patch():
            self.budget = self.make_budget()
        self.budget_sha = hashlib.sha256(self.budget.read_bytes()).hexdigest()
        self.canary = self.make_canary()

        self.usage_baseline = self.make_usage(
            "baseline", TIMES["baseline"], Decimal("0.10"), Decimal("1.00")
        )
        self.usage_pre_a_previous = self.make_usage(
            "pre_a_previous", TIMES["pre_a_previous"],
            Decimal("0.11"), Decimal("1.01"),
        )
        self.usage_pre_a_current = self.make_usage(
            "pre_a_current", TIMES["pre_a_current"],
            Decimal("0.11"), Decimal("1.01"),
        )
        self.pre_a_gate = self.make_gate(
            "pre_a", self.usage_pre_a_previous, self.usage_pre_a_current,
            future_bound=Decimal("1.90803056"), now=TIMES["a_authorized"],
        )
        self.a_launch = self.make_launch(
            "A", TIMES["a_authorized"], self.usage_pre_a_previous,
            self.usage_pre_a_current, self.pre_a_gate,
        )
        self.a_launch_sha = hashlib.sha256(self.a_launch.read_bytes()).hexdigest()
        a_live_source = self.make_usage(
            "a_live", TIMES["a_live"], Decimal("0.11"), Decimal("1.01")
        )
        a_receipt_source = self.make_launch_receipt(
            "A", self.a_launch, a_live_source,
            self.usage_pre_a_previous, self.usage_pre_a_current,
            self.pre_a_gate, TIMES["a_verified"],
        )
        self.a_dir = self.make_stage(
            "A", self.a_launch_sha, a_live_source, a_receipt_source
        )
        self.a_live_current_usage = self.a_dir / LIVE_CURRENT_USAGE_FILENAME
        self.a_launch_verify_receipt = self.a_dir / VERIFY_RECEIPT_FILENAME

        self.usage_post_a_previous = self.make_usage(
            "post_a_previous", TIMES["post_a_previous"],
            Decimal("0.21"), Decimal("1.11"),
        )
        self.usage_post_a_current = self.make_usage(
            "post_a_current", TIMES["post_a_current"],
            Decimal("0.21"), Decimal("1.11"),
        )
        self.pre_c_gate = self.make_gate(
            "pre_c", self.usage_post_a_previous, self.usage_post_a_current,
            future_bound=Decimal("0.95401528"), now=TIMES["c_authorized"],
        )
        self.c_launch = self.make_launch(
            "C", TIMES["c_authorized"], self.usage_post_a_previous,
            self.usage_post_a_current, self.pre_c_gate,
        )
        self.c_launch_sha = hashlib.sha256(self.c_launch.read_bytes()).hexdigest()
        c_live_source = self.make_usage(
            "c_live", TIMES["c_live"], Decimal("0.21"), Decimal("1.11")
        )
        c_receipt_source = self.make_launch_receipt(
            "C", self.c_launch, c_live_source,
            self.usage_post_a_previous, self.usage_post_a_current,
            self.pre_c_gate, TIMES["c_verified"],
        )
        self.c_dir = self.make_stage(
            "C", self.c_launch_sha, c_live_source, c_receipt_source,
            failure_ids={2},
        )
        self.c_live_current_usage = self.c_dir / LIVE_CURRENT_USAGE_FILENAME
        self.c_launch_verify_receipt = self.c_dir / VERIFY_RECEIPT_FILENAME

        self.usage_final_previous = self.make_usage(
            "final_previous", TIMES["final_previous"],
            Decimal("0.31"), Decimal("1.21"),
        )
        self.usage_final_current = self.make_usage(
            "final_current", TIMES["final_current"],
            Decimal("0.31"), Decimal("1.21"),
        )
        self.final_gate = self.make_gate(
            "final", self.usage_final_previous, self.usage_final_current,
            future_bound=Decimal("0.95401528"), now=TIMES["final_current"],
            final=True,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    @contextmanager
    def launch_patch():
        with mock.patch.multiple(
            "tools.check_e12_stage_launch", TRACE_N=N, TRACE_SHA256=TRACE_SHA
        ), mock.patch.multiple(
            "tools.check_e12_live_budget", TRACE_N=N, TRACE_SHA256=TRACE_SHA
        ):
            yield

    def write_json(self, name: str, payload: dict[str, object]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    def make_trace_manifest(self) -> Path:
        return self.write_json("trace.manifest.json", {
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
            ],
            "python_version": "3.11.9",
            "transformers_version": "4.57.6",
            "input": "/synthetic/source.jsonl",
            "input_sha256": "d" * 64,
            "output": "/synthetic/output.jsonl",
            "output_sha256": TRACE_SHA,
            "n": N,
            "scenario": "extreme_burst_1200",
            "tokenizer": "/synthetic/model",
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "chat_template_source": {"kind": "tokenizer_default"},
            "payload_mode": PAYLOAD_MODE,
            "cache_mode": CACHE_MODE,
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
            "max_context_tokens": 40_960,
            "overflow_policy": "drop",
            "input_rows_n": N,
            "scenario_rows_n": N,
            "empty_prompt_rows_n": 0,
            "selected_before_limit_n": N,
            "selected_n": N,
            "selected_source_indices_sha256": "e" * 64,
            "context_overflow_affected_n": 0,
            "context_overflow_events": [],
            "unique_session_n": N,
            "source_session_id_missing_n": 0,
            "arrival": {"min": 1000, "max": 2200, "span_s": 1200},
            "actual_prompt_tokens": {
                "n": N, "sum": PROMPT_SUM, "min": 1, "p50": 80.0,
                "p95": 300.0, "p99": 600.0, "max": 1_200_000,
            },
            "original_trace_prompt_tokens": {
                "n": N, "sum": ORIGINAL_PROMPT_SUM, "min": 1, "p50": 100.0,
                "p95": 600.0, "p99": 1200.0, "max": 1_900_000,
            },
            "actual_minus_original_trace_tokens": {
                "n": N, "sum": PROMPT_SUM - ORIGINAL_PROMPT_SUM,
                "min": -1_800_000, "p50": -20.0, "p95": 100.0,
                "p99": 300.0, "max": 700,
            },
            "decode_cap_affected_n": 0,
            "source_decode_tokens": {
                "n": N, "sum": DECODE_SUM, "min": 1, "p50": 200.0,
                "p95": 700.0, "p99": 900.0, "max": 3_000_000,
            },
            "output_decode_tokens": {
                "n": N, "sum": DECODE_SUM, "min": 1, "p50": 200.0,
                "p95": 700.0, "p99": 900.0, "max": 3_000_000,
            },
            "preservation_checks": {
                "prompt_text_exact_n": N,
                "arrived_at_exact_n": N,
                "decode_tokens_exact_n": N,
                "decode_source_provenance_n": N,
                "decode_cap_respected_n": N,
                "session_id_exact_n": N,
                "token_metadata_aligned_n": N,
            },
        })

    def make_profile(self) -> Path:
        return self.write_json("profile.json", {
            "schema_version": 2,
            "predictor_model": "seq_slots_shared_prefill_lane_v1",
            "valid": True,
            "cache_mode_required": "none",
            "model": self.context.model,
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "base_url": self.context.base_url,
            "server_pid": self.context.server_pid,
            "server_log_sha256_at_start": self.context.server_log_prefix_sha256,
            "server_proc_cmdline_sha256_at_start": "5" * 64,
            "kv_capacity_tokens": 112000,
            "sampling": {
                "temperature": 0.0,
                "ignore_eos": True,
                "continuous_usage_stats": True,
            },
            "profile_config": {"nimbus_tick_ms": 250, "slo_s": 5},
            "predictor_calibration": {
                "recommended_prefill_tput_tokens_per_s": 3000,
                "recommended_tpot_ms": 150,
                "recommended_first_token_overhead_ms": 400,
                "recommended_ttft_guard_ms": 1700,
                "target_slo_s": 5,
                "heldout_n": 5,
                "heldout_violation_confusion": {"false_negative": 0},
            },
        })

    def make_price(self) -> Path:
        payload = build_price_snapshot(
            {
                "id": CLOUD_MODEL,
                "endpoints": [{
                    "provider_name": "deepinfra",
                    "context_length": 40960,
                    "pricing": {
                        "prompt": "0.00000008",
                        "completion": "0.00000028",
                        "input_cache_read": "0",
                        "input_cache_write": "0",
                        "internal_reasoning": "0",
                    },
                }],
            },
            now=lambda: utc(TIMES["baseline"]),
        )
        return self.write_json("price.json", payload)

    def make_budget(self) -> Path:
        payload = build_budget_attestation(
            manifest_path=self.trace_manifest,
            price_snapshot_path=self.price,
            expected_manifest_sha256=self.identity.manifest_sha256,
            expected_trace_sha256=TRACE_SHA,
            expected_n=N,
            expected_prompt_token_sum=PROMPT_SUM,
            expected_decode_token_sum=DECODE_SUM,
            expected_payload_mode=PAYLOAD_MODE,
            expected_cache_mode=CACHE_MODE,
            arm_count=2,
            input_price_per_million_usd=Decimal("0.08"),
            output_price_per_million_usd=Decimal("0.28"),
            budget_usd=Decimal("3"),
            now=utc(TIMES["baseline"]),
        )
        return self.write_json("budget.json", payload)

    def make_canary(self) -> Path:
        raw = {
            "success": True,
            "http_status": 200,
            "probe_mode": "ttft_cancel",
            "stream_abort_requested": True,
            "response_completed": False,
            "first_token_kind": "reasoning",
            "requested_provider_order": ["deepinfra"],
            "provider": "deepinfra",
            "response_model": CLOUD_MODEL,
            "cost_pending": True,
            "cost_usd": None,
            "ttft_ms": 25.0,
            "generation_id_sha256": hashlib.sha256(
                b"synthetic-generation"
            ).hexdigest(),
        }
        with mock.patch(
            "tools.run_openrouter_ttft_canary._utc_now",
            return_value=TIMES["canary"],
        ):
            payload = assess_result(
                raw, key_fingerprint_sha256=KEY_FINGERPRINT
            )
        return self.write_json("canary.json", payload)

    def make_usage(
        self, name: str, captured_at: str, usage: Decimal,
        account_usage: Decimal,
    ) -> Path:
        key_payload = {
            "usage": float(usage),
            "limit": 3.0,
            "limit_remaining": float(Decimal("3") - usage),
            "limit_reset": None,
            "include_byok_in_limit": True,
            "is_management_key": False,
            "is_provisioning_key": False,
            "is_free_tier": False,
            "expires_at": EXPIRY,
        }
        account_payload = {
            "total_usage": float(account_usage),
            "total_credits": 10.0,
        }

        def opener(request: object, *, timeout: float) -> JsonResponse:
            del timeout
            url = getattr(request, "full_url")
            return JsonResponse(
                key_payload if str(url).endswith("/api/v1/key") else account_payload,
                str(url),
            )

        payload = capture_usage_snapshot(
            api_key=API_KEY,
            opener=opener,
            now=lambda: utc(captured_at),
        )
        return self.write_json(f"usage_{name}.json", payload)

    def make_gate(
        self, name: str, previous: Path, current: Path, *,
        future_bound: Decimal, now: str, final: bool = False,
    ) -> Path:
        payload = build_stage_gate_attestation(
            baseline_path=self.usage_baseline,
            current_path=current,
            settlement_previous_path=previous,
            next_stage_full_upper_bound_usd=future_bound,
            final=final,
            now=utc(now),
        )
        return self.write_json(f"gate_{name}.json", payload)

    def make_launch(
        self, label: str, authorized_at: str, previous: Path,
        current: Path, gate: Path,
    ) -> Path:
        kwargs: dict[str, object] = {}
        if label == "C":
            kwargs.update(
                a_dir=self.a_dir,
                a_launch_attestation_path=self.a_launch,
            )
        with self.launch_patch():
            payload = create_launch_attestation(
                stage=label,
                contract_id="synthetic-contract",
                trace_manifest_path=self.trace_manifest,
                profile_path=self.profile,
                price_snapshot_path=self.price,
                budget_attestation_path=self.budget,
                canary_path=self.canary,
                baseline_usage_path=self.usage_baseline,
                settlement_previous_path=previous,
                settlement_current_path=current,
                stage_budget_gate_path=gate,
                context=self.context,
                now=utc(authorized_at),
                **kwargs,
            )
        return self.write_json(f"launch_{label}.json", payload)

    def make_launch_receipt(
        self, label: str, launch: Path, live_current_usage: Path,
        previous: Path, current: Path, gate: Path, verified_at: str,
    ) -> Path:
        live_current_usage.chmod(0o600)
        kwargs: dict[str, object] = {}
        if label == "C":
            kwargs.update(
                a_dir=self.a_dir,
                a_launch_attestation_path=self.a_launch,
            )
        with self.launch_patch():
            payload = verify_launch_attestation(
                attestation_path=launch,
                expected_sha256=hashlib.sha256(launch.read_bytes()).hexdigest(),
                stage=label,
                contract_id="synthetic-contract",
                trace_manifest_sha256=self.identity.manifest_sha256,
                profile_sha256=self.profile_sha,
                budget_attestation_sha256=self.budget_sha,
                context=self.context,
                trace_manifest_path=self.trace_manifest,
                profile_path=self.profile,
                price_snapshot_path=self.price,
                budget_attestation_path=self.budget,
                canary_path=self.canary,
                baseline_usage_path=self.usage_baseline,
                settlement_previous_path=previous,
                settlement_current_path=current,
                stage_budget_gate_path=gate,
                live_current_usage_path=live_current_usage,
                now=utc(verified_at),
                **kwargs,
            )
        return self.write_json(f"launch_{label}_verify_receipt.json", payload)

    @staticmethod
    def side_stats(rows: list[dict], slo_ms: float = 5000.0) -> dict:
        violations = sum(
            not row["success"]
            or row.get("ttft_ms") is None
            or row["ttft_ms"] > slo_ms
            for row in rows
        )
        errors: dict[str, int] = {}
        for row in rows:
            if row.get("error_type"):
                errors[row["error_type"]] = errors.get(row["error_type"], 0) + 1
        return {
            "n": len(rows),
            "success": sum(bool(row["success"]) for row in rows),
            "routed_only": 0,
            "errors": errors,
            "ttft_p50_ms": None,
            "ttft_p95_ms": None,
            "ttft_p99_ms": None,
            "tpot_p50_ms": None,
            "slo_violations": violations,
            "slo_measured_n": len(rows),
            "slo_violation_pct": 100.0 * violations / max(len(rows), 1),
            "cost_usd": None,
            "known_cost_usd": 0.0,
            "cost_measured_n": 0,
            "cost_pending_n": len(rows),
        }

    def make_stage(
        self, label: str, launch_sha: str, live_current_usage: Path,
        launch_verify_receipt: Path,
        *, failure_ids: set[int] | None = None,
    ) -> Path:
        failure_ids = set(failure_ids or set())
        directory = self.root / label
        directory.mkdir()
        persisted_live = directory / LIVE_CURRENT_USAGE_FILENAME
        persisted_receipt = directory / VERIFY_RECEIPT_FILENAME
        persisted_live.write_bytes(live_current_usage.read_bytes())
        persisted_receipt.write_bytes(launch_verify_receipt.read_bytes())
        persisted_live.chmod(0o600)
        persisted_receipt.chmod(0o600)
        live_current_usage_sha = hashlib.sha256(
            persisted_live.read_bytes()
        ).hexdigest()
        launch_verify_receipt_sha = hashlib.sha256(
            persisted_receipt.read_bytes()
        ).hexdigest()
        stem = f"run_{label}"
        raw_path = directory / f"{stem}.jsonl"
        summary_path = directory / f"{stem}.summary.json"
        decisions_path = directory / f"{stem}.decisions.jsonl"
        rows: list[dict] = []
        for request_id in range(N):
            prompt = 100
            decode = 10
            if request_id not in ROUTES[label]:
                row = {
                    "request_id": request_id,
                    "endpoint": "local",
                    "model": self.context.model,
                    "routed_only": False,
                    "success": True,
                    "error": None,
                    "error_type": None,
                    "http_status": 200,
                    "ttft_ms": 1000.0 + request_id,
                    "prompt_tokens": prompt,
                    "completion_tokens": decode,
                    "scheduler_prompt_tokens": prompt,
                    "scheduler_uncached_prompt_tokens": prompt,
                    "scheduler_decode_tokens": decode,
                    "payload_mode": PAYLOAD_MODE,
                    "cache_mode": CACHE_MODE,
                    "output_chars": 1,
                    "cost_usd": 0.0,
                }
            elif request_id in failure_ids:
                row = {
                    "request_id": request_id,
                    "endpoint": "cloud",
                    "model": CLOUD_MODEL,
                    "routed_only": False,
                    "success": False,
                    "error": "HTTP 429: provider response body omitted",
                    "error_type": "HTTP 429",
                    "http_status": 429,
                    "ttft_ms": None,
                    "scheduler_prompt_tokens": prompt,
                    "scheduler_uncached_prompt_tokens": prompt,
                    "scheduler_decode_tokens": decode,
                    "payload_mode": PAYLOAD_MODE,
                    "cache_mode": CACHE_MODE,
                    "requested_provider_order": ["deepinfra"],
                    "probe_mode": "ttft_cancel",
                    "provider": None,
                    "response_model": None,
                    "stream_abort_requested": False,
                    "response_completed": False,
                    "cost_usd": 0.0,
                    "cost_pending": False,
                }
            else:
                row = {
                    "request_id": request_id,
                    "endpoint": "cloud",
                    "model": CLOUD_MODEL,
                    "routed_only": False,
                    "success": True,
                    "error": None,
                    "error_type": None,
                    "http_status": 200,
                    "pre_route_queue_ms": 100.0,
                    "cloud_gate_wait_ms": 20.0,
                    "service_ttft_ms": 300.0,
                    "ttft_ms": 420.0,
                    "scheduler_prompt_tokens": prompt,
                    "scheduler_uncached_prompt_tokens": prompt,
                    "scheduler_decode_tokens": decode,
                    "payload_mode": PAYLOAD_MODE,
                    "cache_mode": CACHE_MODE,
                    "requested_provider_order": ["deepinfra"],
                    "probe_mode": "ttft_cancel",
                    "provider": "deepinfra",
                    "response_model": CLOUD_MODEL,
                    "first_token_kind": "reasoning",
                    "generation_id_sha256": hashlib.sha256(
                        f"gen-{label}-{request_id}".encode()
                    ).hexdigest(),
                    "stream_abort_requested": True,
                    "response_completed": False,
                    "cost_usd": None,
                    "cost_pending": True,
                }
            rows.append(row)
        raw_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        decisions = [{
            "decision_id": 1,
            "at_s": 1.0,
            "status": "applied",
            "trigger": "ttft_pred",
            "selector": SELECTORS[label],
            "snapshot_n": N,
            "snapshot_hash": "a" * 16,
            "inflight_n": 0,
            "prediction_scope": "waiting_only",
            "prediction_model": "seq_slots_shared_prefill_lane_v1",
            "waiting_predicted_max_ttft_s": 7.0,
            "waiting_post_kick_max_ttft_s": 3.0,
            "proposed_victim_ids": sorted(ROUTES[label]),
            "applied_victim_ids": sorted(ROUTES[label]),
            "decision_ms": 0.1,
        }]
        decisions_path.write_text(
            "".join(json.dumps(row) + "\n" for row in decisions),
            encoding="utf-8",
        )
        local_rows = [row for row in rows if row["endpoint"] == "local"]
        cloud_rows = [row for row in rows if row["endpoint"] == "cloud"]
        config = {
            "scenario": "extreme_burst_1200",
            "seed": 0,
            "time_scale": 1.0,
            "max_inflight": 128,
            "max_tokens_override": None,
            "temperature": 0.0,
            "ignore_eos": True,
            "local_ignore_eos": True,
            "cloud_ignore_eos": False,
            "nimbus_trigger": "ttft_pred",
            "nimbus_selector": SELECTORS[label],
            "prefill_tput": 3000.0,
            "tpot_ms": 150.0,
            "first_token_overhead_ms": 400.0,
            "slo_s": 5.0,
            "timeout_s": 600.0,
            "ttft_guard_ms": 1700.0,
            "nimbus_tick_ms": 250.0,
            "kv_capacity_tokens": 112000.0,
            "kv_hysteresis_fraction": 0.05,
            "cloud": "real",
            "cloud_url": "https://openrouter.ai/api/v1/chat/completions",
            "cloud_model": CLOUD_MODEL,
            "cloud_api_key_env": "OPENROUTER_API_KEY",
            "cloud_max_concurrency": 16,
            "cloud_provider_order": ["deepinfra"],
            "cloud_no_fallbacks": True,
            "cloud_stop_after_first_token": True,
            "local_url": self.context.chat_url,
            "local_model": self.context.model,
            "in_price": 0.08,
            "out_price": 0.28,
        }
        summary = {
            "policy": "nimbus",
            "target_fraction": 0.0,
            "actual_fraction": len(cloud_rows) / len(rows),
            "slo_s": 5.0,
            "overall": self.side_stats(rows),
            "local": self.side_stats(local_rows),
            "cloud": self.side_stats(cloud_rows),
            "pessimistic_combined": {
                "slo_violations": len(cloud_rows),
                "slo_n": len(rows),
                "slo_violation_pct": 100.0 * len(cloud_rows) / len(rows),
                "cloud_assumed_violations": len(cloud_rows),
                "cost_usd": None,
                "known_cost_usd": 0.0,
                "cost_measured_n": 0,
                "cost_pending_n": len(rows),
            },
            "token_alignment": {
                "local_success_n": len(local_rows),
                "measured_n": len(local_rows),
                "missing_prompt_usage_n": 0,
                "prompt_exact_n": len(local_rows),
                "prompt_exact_fraction": 1.0,
                "absolute_error_p50_tokens": 0,
                "absolute_error_p95_tokens": 0,
                "absolute_error_max_tokens": 0,
                "relative_error_p50": 0.0,
                "relative_error_p95": 0.0,
                "actual_over_scheduler_p50": 1.0,
                "actual_over_scheduler_p95": 1.0,
                "decode_measured_n": len(local_rows),
                "missing_completion_usage_n": 0,
                "decode_cap_hit_n": len(local_rows),
                "decode_cap_hit_fraction": 1.0,
                "completion_over_scheduler_p50": 1.0,
                "completion_over_scheduler_p05": 1.0,
            },
            "config": config,
            "queue": {
                "queue_delay_p50_ms": 0.0,
                "queue_delay_p99_ms": 0.0,
                "queue_delay_max_ms": 0.0,
                "peak_inflight": 2,
                "peak_waiting": 2,
                "nimbus_ticks": 1,
                "nimbus_kick_rounds": 1,
                "nimbus_kicked": len(cloud_rows),
                "nimbus_trigger": "ttft_pred",
                "nimbus_selector": SELECTORS[label],
                "nimbus_applied_kick_rounds": 1,
                "nimbus_stale_decisions": 0,
                "nimbus_decision_calls": 1,
                "nimbus_decision_mean_ms": 0.1,
                "nimbus_decision_max_ms": 0.1,
                "nimbus_prediction_scope": "waiting_only",
                "nimbus_prediction_model": "seq_slots_shared_prefill_lane_v1",
                "nimbus_max_waiting_predicted_ttft_s": 7.0,
                "nimbus_max_waiting_post_kick_ttft_s": 3.0,
            },
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        manifest_path = directory / "matrix_manifest.txt"
        manifest_started = (
            TIMES["a_authorized"] if label == "A" else TIMES["c_authorized"]
        )
        lines = [
            "run_fingerprint=FINGERPRINT_PENDING",
            f"started_at={manifest_started}",
            f"commit={COMMIT}",
            f"trace_sha256={TRACE_SHA} trace_manifest_sha256={self.identity.manifest_sha256} trace_n={N}",
            f"profile_sha256={self.profile_sha}",
            "cache_mode=none server_pid=123 server_log=server.log server_log_prefix_sha256=" + "4" * 64,
            "server_log_sha256_at_manifest=" + ("a" if label == "A" else "c") * 64,
            "endpoint_version_sha256=" + "6" * 64 + " endpoint_models_identity_sha256=" + "7" * 64,
            f"base_url={self.context.base_url} chat_url={self.context.chat_url} model={self.context.model}",
            "python=Python 3.12.0",
            "scenario=extreme_burst_1200 max_inflight=128 kv_cap=112000",
            "prefill_tput=3000 tpot_ms=150 first_token_overhead_ms=400 slo_s=5 timeout_s=600 guard_ms=1700 tick_ms=250",
            "temperature=0 ignore_eos=1 in_price=0.08 out_price=0.28",
            "local_ignore_eos=1 cloud_ignore_eos=0",
            "cloud=real cloud_url=https://openrouter.ai/api/v1/chat/completions cloud_model=qwen/qwen3-32b cloud_api_key_env=OPENROUTER_API_KEY",
            "cloud_max_concurrency=16 cloud_provider=deepinfra cloud_no_fallbacks=1 cloud_stop_after_first_token=1",
            f"real_cloud_expected_trace_n={N} secret_value_recorded=false",
            f"live_contract_id=synthetic-contract live_contract={LIVE_CONTRACT} live_stage={label} live_expected_trace_n={N} live_arm_order_seed=20260716 live_exact_arm={ARMS[label]} live_authorized_budget_usd=3 live_budget_attestation_sha256={self.budget_sha} live_key_limit_max_usd=3 live_cooldown_s=20 live_stage_launch_attestation_sha256={launch_sha} live_current_usage_sha256={live_current_usage_sha} live_stage_launch_verify_receipt_sha256={launch_verify_receipt_sha}",
            "arm_order_mode=e12_contract_stage arm_order_seed=20260716",
            f"arms={ARMS[label]} ",
            "evidence_scope=single-pass synthetic live fixture",
        ]
        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        fingerprint = recompute_run_fingerprint(
            _parse_manifest(manifest_path, f"{label} fixture")
        )
        manifest_path.write_text(
            manifest_path.read_text().replace("FINGERPRINT_PENDING", fingerprint),
            encoding="utf-8",
        )
        event_times = (
            (TIMES["a_started"], TIMES["a_finished"], TIMES["a_matrix"])
            if label == "A"
            else (TIMES["c_started"], TIMES["c_finished"], TIMES["c_matrix"])
        )
        live = (
            f"live_contract_id=synthetic-contract live_stage={label} "
            f"live_authorized_budget_usd=3 live_budget_attestation_sha256={self.budget_sha} "
            f"live_key_limit_max_usd=3 live_cooldown_s=20 "
            f"live_stage_launch_attestation_sha256={launch_sha} "
            f"live_current_usage_sha256={live_current_usage_sha} "
            f"live_stage_launch_verify_receipt_sha256={launch_verify_receipt_sha}"
        )
        (directory / "matrix_events.log").write_text(
            f"arm_started_at={event_times[0]} arm={ARMS[label]} {live} command=router.run_config_bound_by_fingerprint\n"
            f"arm_finished_at={event_times[1]} arm={ARMS[label]} {live}\n"
            f"matrix_finished_at={event_times[2]} run_fingerprint={fingerprint} {live}\n",
            encoding="utf-8",
        )
        self.rewrite_marker(directory, label)
        return directory

    def rewrite_marker(self, directory: Path, label: str) -> None:
        stem = f"run_{label}"
        manifest = _parse_manifest(directory / "matrix_manifest.txt", "fixture")
        marker = {
            "schema_version": 2,
            "run_fingerprint": manifest["run_fingerprint"],
            "arm": ARMS[label],
            "artifacts": {
                "raw": artifact(directory / f"{stem}.jsonl"),
                "summary": artifact(directory / f"{stem}.summary.json"),
                "decisions": artifact(directory / f"{stem}.decisions.jsonl"),
            },
        }
        (directory / f"{stem}.complete.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )

    def rebind_stage_receipt_hash(
        self, directory: Path, label: str, old_sha: str, new_sha: str,
    ) -> None:
        manifest_path = directory / "matrix_manifest.txt"
        old_manifest = _parse_manifest(manifest_path, "fixture")
        old_fingerprint = old_manifest["run_fingerprint"]
        manifest_path.write_text(
            manifest_path.read_text().replace(old_sha, new_sha),
            encoding="utf-8",
        )
        new_fingerprint = recompute_run_fingerprint(
            _parse_manifest(manifest_path, "fixture")
        )
        manifest_path.write_text(
            manifest_path.read_text().replace(
                old_fingerprint, new_fingerprint
            ),
            encoding="utf-8",
        )
        events_path = directory / "matrix_events.log"
        events_path.write_text(
            events_path.read_text().replace(old_sha, new_sha).replace(
                old_fingerprint, new_fingerprint
            ),
            encoding="utf-8",
        )
        self.rewrite_marker(directory, label)

    def audit_kwargs(self) -> dict[str, object]:
        return {
            "a_dir": self.a_dir,
            "c_dir": self.c_dir,
            "trace_manifest": self.trace_manifest,
            "profile": self.profile,
            "price_snapshot": self.price,
            "budget_attestation": self.budget,
            "canary_attestation": self.canary,
            "a_launch_attestation": self.a_launch,
            "c_launch_attestation": self.c_launch,
            "a_live_current_usage": self.a_live_current_usage,
            "a_launch_verify_receipt": self.a_launch_verify_receipt,
            "c_live_current_usage": self.c_live_current_usage,
            "c_launch_verify_receipt": self.c_launch_verify_receipt,
            "usage_baseline": self.usage_baseline,
            "usage_pre_a_previous": self.usage_pre_a_previous,
            "usage_pre_a_current": self.usage_pre_a_current,
            "pre_a_gate": self.pre_a_gate,
            "usage_post_a_previous": self.usage_post_a_previous,
            "usage_post_a_current": self.usage_post_a_current,
            "pre_c_gate": self.pre_c_gate,
            "usage_final_previous": self.usage_final_previous,
            "usage_final_current": self.usage_final_current,
            "final_gate": self.final_gate,
            "trace_identity": self.identity,
        }

    def run_audit(self) -> dict:
        with self.launch_patch():
            return audit(**self.audit_kwargs())

    def test_full_capture_gate_launch_and_final_audit_chain_passes(self) -> None:
        result = self.run_audit()
        self.assertEqual(result["verdict"], "pass")
        self.assertTrue(result["evidence"]["launch_chain_revalidated"])
        self.assertTrue(
            result["evidence"]["launch_time_usage_receipts_revalidated"]
        )
        self.assertEqual(
            result["evidence"]["request_price_source"],
            "absent_not_advertised",
        )
        self.assertEqual(result["stages"]["C"]["cloud_failure_n"], 1)
        self.assertEqual(
            result["usage"]["deltas_usd"]["key"],
            {"canary": "0.01", "A": "0.1", "C": "0.1", "total": "0.21"},
        )
        self.assertTrue(result["usage"]["same_key_fingerprint_verified"])
        self.assertTrue(result["usage"]["all_settled_pairs_verified"])
        self.assertTrue(result["usage"]["launch_time_usage_verified"])
        rendered = render_json(result)
        for forbidden in (
            API_KEY, KEY_FINGERPRINT, "provider response body omitted",
            "prompt_text", "synthetic-generation", "gen-A-0", "gen-C-0",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_rejects_missing_canary(self) -> None:
        self.canary.unlink()
        with self.assertRaisesRegex(EvidenceError, "launch evidence is invalid"):
            self.run_audit()

    def test_rejects_stale_or_wrong_pre_c_gate(self) -> None:
        gate = json.loads(self.pre_c_gate.read_text())
        gate["current_snapshot_sha256"] = "f" * 64
        self.pre_c_gate.write_text(json.dumps(gate))
        with self.assertRaisesRegex(EvidenceError, "launch evidence is invalid"):
            self.run_audit()

    def test_rejects_c_without_completed_a(self) -> None:
        (self.a_dir / "run_A.complete.json").unlink()
        with self.assertRaisesRegex(EvidenceError, "completion marker"):
            self.run_audit()

    def test_rejects_a_to_c_cooldown_below_twenty_seconds(self) -> None:
        manifest = self.c_dir / "matrix_manifest.txt"
        manifest.write_text(
            manifest.read_text().replace(
                f"started_at={TIMES['c_authorized']}",
                "started_at=2026-07-18T00:02:30Z",
            )
        )
        events = self.c_dir / "matrix_events.log"
        events.write_text(
            events.read_text().replace(
                f"arm_started_at={TIMES['c_started']}",
                "arm_started_at=2026-07-18T00:02:31Z",
            )
        )
        with self.assertRaisesRegex(EvidenceError, "cooldown is below 20"):
            self.run_audit()

    def test_rejects_unsettled_final_pair(self) -> None:
        payload = json.loads(self.usage_final_previous.read_text())
        payload["key"]["usage"] = 0.30
        payload["key"]["limit_remaining"] = 2.70
        payload["account"]["total_usage"] = 1.20
        payload["account"]["remaining_credits"] = 8.80
        self.usage_final_previous.write_text(json.dumps(payload))
        with self.assertRaisesRegex(EvidenceError, "settled pair counters differ"):
            self.run_audit()

    def test_rejects_wrong_launch_hash_bound_by_manifest(self) -> None:
        payload = json.loads(self.a_launch.read_text())
        payload["status"] = "tampered"
        self.a_launch.write_text(json.dumps(payload))
        with self.assertRaisesRegex(EvidenceError, "launch attestation SHA256"):
            self.run_audit()

    def test_rejects_bundle_without_persisted_launch_verify_receipt(self) -> None:
        self.a_launch_verify_receipt.unlink()
        with self.assertRaisesRegex(EvidenceError, "launch verify receipt: unreadable"):
            self.run_audit()

    def test_rejects_tampered_persisted_launch_verify_receipt(self) -> None:
        old_sha = hashlib.sha256(
            self.c_launch_verify_receipt.read_bytes()
        ).hexdigest()
        payload = json.loads(self.c_launch_verify_receipt.read_text())
        payload["key_usage_usd"] = "0.22"
        self.c_launch_verify_receipt.write_text(json.dumps(payload))
        new_sha = hashlib.sha256(
            self.c_launch_verify_receipt.read_bytes()
        ).hexdigest()
        self.rebind_stage_receipt_hash(
            self.c_dir, "C", old_sha, new_sha
        )
        with self.assertRaisesRegex(
            EvidenceError, "persisted launch verification is invalid"
        ):
            self.run_audit()

    def test_failure_message_does_not_echo_tampered_text(self) -> None:
        sensitive = "sensitive-current-turn-string-must-not-leak"
        manifest = self.a_dir / "matrix_manifest.txt"
        manifest.write_text(
            manifest.read_text().replace(
                "cloud_model=qwen/qwen3-32b", f"cloud_model={sensitive}"
            )
        )
        with self.assertRaises(EvidenceError) as caught:
            self.run_audit()
        self.assertNotIn(sensitive, str(caught.exception))

    def test_rejects_key_change_even_with_valid_snapshot_shape(self) -> None:
        payload = json.loads(self.usage_final_current.read_text())
        payload["key"]["key_fingerprint_sha256"] = "e" * 64
        self.usage_final_current.write_text(json.dumps(payload))
        with self.assertRaisesRegex(EvidenceError, "API key fingerprint changed"):
            self.run_audit()

    def test_rejects_payload_field_even_when_marker_hash_is_updated(self) -> None:
        raw = self.a_dir / "run_A.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        rows[0]["prompt_text"] = "must-not-leak"
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.rewrite_marker(self.a_dir, "A")
        with self.assertRaisesRegex(EvidenceError, "payload-bearing field"):
            self.run_audit()

    def test_rejects_raw_generation_id_even_after_marker_rewrite(self) -> None:
        sensitive = "provider-generation-id-with-private-user-text"
        raw = self.a_dir / "run_A.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        cloud = next(row for row in rows if row["endpoint"] == "cloud")
        cloud["generation_id"] = sensitive
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.rewrite_marker(self.a_dir, "A")
        with self.assertRaises(EvidenceError) as caught:
            self.run_audit()
        self.assertIn("payload-bearing field", str(caught.exception))
        self.assertNotIn(sensitive, str(caught.exception))

    def test_rejects_noncanonical_generation_id_hash(self) -> None:
        raw = self.a_dir / "run_A.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        cloud = next(row for row in rows if row["endpoint"] == "cloud")
        cloud["generation_id_sha256"] = "A" * 64
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.rewrite_marker(self.a_dir, "A")
        with self.assertRaisesRegex(EvidenceError, "identifier hash"):
            self.run_audit()

    def test_rejects_success_without_generation_id_hash(self) -> None:
        raw = self.a_dir / "run_A.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        cloud = next(
            row for row in rows
            if row["endpoint"] == "cloud" and row["success"]
        )
        cloud["generation_id_sha256"] = None
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.rewrite_marker(self.a_dir, "A")
        with self.assertRaisesRegex(EvidenceError, "lacks generation identifier hash"):
            self.run_audit()

    def test_rejects_non_allowlisted_provider_without_echoing_it(self) -> None:
        sensitive = "DeepInfra-private-current-turn-fragment"
        raw = self.a_dir / "run_A.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        cloud = next(row for row in rows if row["endpoint"] == "cloud")
        cloud["provider"] = sensitive
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.rewrite_marker(self.a_dir, "A")
        with self.assertRaises(EvidenceError) as caught:
            self.run_audit()
        self.assertIn("provider is not exactly", str(caught.exception))
        self.assertNotIn(sensitive, str(caught.exception))

    def test_protocol_mismatch_may_not_retain_response_metadata(self) -> None:
        raw = self.c_dir / "run_C.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        failed = next(row for row in rows if not row["success"])
        failed.update({
            "http_status": 200,
            "error_type": "ProtocolMismatch",
            "error": "provider response metadata mismatch; details omitted",
            "provider": "deepinfra",
            "response_model": None,
        })
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.rewrite_marker(self.c_dir, "C")
        with self.assertRaisesRegex(EvidenceError, "retained response metadata"):
            self.run_audit()

    def test_protocol_mismatch_invalidates_live_evidence_when_scrubbed(self) -> None:
        raw = self.c_dir / "run_C.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        failed = next(row for row in rows if not row["success"])
        failed.update({
            "http_status": 200,
            "error_type": "ProtocolMismatch",
            "error": "provider response metadata mismatch; details omitted",
            "provider": None,
            "response_model": None,
        })
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.rewrite_marker(self.c_dir, "C")
        with self.assertRaisesRegex(EvidenceError, "metadata did not conform"):
            self.run_audit()

    def test_rejects_payload_field_in_summary_after_marker_rewrite(self) -> None:
        sensitive = "SENSITIVE_CURRENT_TURN_MUST_NOT_LEAK"
        summary_path = self.c_dir / "run_C.summary.json"
        summary = json.loads(summary_path.read_text())
        summary["prompt_text"] = sensitive
        summary_path.write_text(json.dumps(summary))
        self.rewrite_marker(self.c_dir, "C")
        with self.assertRaises(EvidenceError) as caught:
            self.run_audit()
        self.assertIn("payload-bearing field", str(caught.exception))
        self.assertNotIn(sensitive, str(caught.exception))

    def test_rejects_payload_field_in_trace_manifest_without_echoing_it(self) -> None:
        sensitive = "PRIVATE_SHAREGPT_CURRENT_TURN_MUST_NOT_LEAK"
        payload = json.loads(self.trace_manifest.read_text())
        payload["prompt_text"] = sensitive
        self.trace_manifest.write_text(json.dumps(payload, sort_keys=True))
        self.identity = TraceIdentity(
            n=self.identity.n,
            trace_sha256=self.identity.trace_sha256,
            manifest_sha256=hashlib.sha256(
                self.trace_manifest.read_bytes()
            ).hexdigest(),
            prompt_token_sum=self.identity.prompt_token_sum,
            decode_token_sum=self.identity.decode_token_sum,
        )
        with self.assertRaises(EvidenceError) as caught:
            self.run_audit()
        rendered = str(caught.exception)
        self.assertIn("text-free materializer schema", rendered)
        self.assertNotIn("prompt_text", rendered)
        self.assertNotIn(sensitive, rendered)

    def test_rejects_sensitive_free_text_in_start_event(self) -> None:
        sensitive = "SENSITIVE_CURRENT_TURN_MUST_NOT_LEAK"
        events = self.a_dir / "matrix_events.log"
        events.write_text(
            events.read_text().replace(
                "command=router.run_config_bound_by_fingerprint",
                f"command={sensitive}",
            )
        )
        with self.assertRaises(EvidenceError) as caught:
            self.run_audit()
        self.assertIn("command attestation is invalid", str(caught.exception))
        self.assertNotIn(sensitive, str(caught.exception))

    def test_rejects_immediate_fixed_endpoint_statuses(self) -> None:
        raw = self.c_dir / "run_C.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        failed = next(row for row in rows if not row["success"])
        summary_path = self.c_dir / "run_C.summary.json"
        for status in (401, 404, 405, 422):
            failed.update({
                "http_status": status,
                "error_type": f"HTTP {status}",
                "error": f"HTTP {status}: provider response body omitted",
            })
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = json.loads(summary_path.read_text())
            for side in ("overall", "cloud"):
                summary[side]["errors"] = {f"HTTP {status}": 1}
            summary_path.write_text(json.dumps(summary))
            self.rewrite_marker(self.c_dir, "C")
            with self.subTest(status=status), self.assertRaisesRegex(
                EvidenceError, "systematic cloud"
            ):
                self.run_audit()

    def test_rejects_complete_arm_with_zero_cloud_successes(self) -> None:
        raw = self.c_dir / "run_C.jsonl"
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        for row in rows:
            if row["endpoint"] != "cloud":
                continue
            row.update({
                "success": False,
                "http_status": 429,
                "error_type": "HTTP 429",
                "error": "HTTP 429: provider response body omitted",
                "ttft_ms": None,
            })
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary_path = self.c_dir / "run_C.summary.json"
        summary = json.loads(summary_path.read_text())
        local_rows = [row for row in rows if row["endpoint"] == "local"]
        cloud_rows = [row for row in rows if row["endpoint"] == "cloud"]
        summary["overall"] = self.side_stats(rows)
        summary["local"] = self.side_stats(local_rows)
        summary["cloud"] = self.side_stats(cloud_rows)
        summary_path.write_text(json.dumps(summary))
        self.rewrite_marker(self.c_dir, "C")
        with self.assertRaisesRegex(EvidenceError, "systematic cloud"):
            self.run_audit()

    def test_cli_requires_and_binds_current_manifest_sha(self) -> None:
        values = self.audit_kwargs()
        values.pop("trace_identity")
        flag_names = {
            "a_dir": "a-dir",
            "c_dir": "c-dir",
            "trace_manifest": "trace-manifest",
            "profile": "profile",
            "price_snapshot": "price-snapshot",
            "budget_attestation": "budget-attestation",
            "canary_attestation": "canary-attestation",
            "a_launch_attestation": "a-launch-attestation",
            "c_launch_attestation": "c-launch-attestation",
            "a_live_current_usage": "a-live-current-usage",
            "a_launch_verify_receipt": "a-launch-verify-receipt",
            "c_live_current_usage": "c-live-current-usage",
            "c_launch_verify_receipt": "c-launch-verify-receipt",
            "usage_baseline": "usage-baseline",
            "usage_pre_a_previous": "usage-pre-a-previous",
            "usage_pre_a_current": "usage-pre-a-current",
            "pre_a_gate": "pre-a-gate",
            "usage_post_a_previous": "usage-post-a-previous",
            "usage_post_a_current": "usage-post-a-current",
            "pre_c_gate": "pre-c-gate",
            "usage_final_previous": "usage-final-previous",
            "usage_final_current": "usage-final-current",
            "final_gate": "final-gate",
        }
        argv: list[str] = []
        for key, flag in flag_names.items():
            argv.extend((f"--{flag}", str(values[key])))
        argv.extend(("--expected-trace-manifest-sha256", "f" * 64))
        frozen = {
            "TRACE_N": N,
            "TRACE_SHA256": TRACE_SHA,
            "PROMPT_TOKEN_SUM": PROMPT_SUM,
            "DECODE_TOKEN_SUM": DECODE_SUM,
        }
        with self.launch_patch(), mock.patch.multiple(
            "tools.audit_e12_live", **frozen
        ), mock.patch("sys.stderr"), mock.patch("sys.stdout"):
            self.assertEqual(main(argv), 2)
            argv[-1] = self.identity.manifest_sha256
            self.assertEqual(main(argv), 0)


if __name__ == "__main__":
    unittest.main()
