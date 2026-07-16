"""Focused stdlib tests for the TTFT matrix arm/evidence contract."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from tools.ttft_matrix_evidence import (
    ANCHOR_ARM,
    MatrixEvidenceError,
    MatrixExpectations,
    main,
    parse_arm,
    validate_trace_materializer_manifest,
    validate_or_write_marker,
)


class TestTTFTMatrixEvidence(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.expected = MatrixExpectations(
            trace_n=2,
            prefill_tput=3255.0,
            tpot_ms=152.3,
            first_token_overhead_ms=440.4,
            ttft_guard_ms=1712.0,
            in_price=0.15,
            out_price=1.20,
            slo_s=5.0,
            timeout_s=600.0,
            nimbus_tick_ms=250.0,
            max_inflight=128,
            temperature=0.0,
            ignore_eos=True,
            kv_capacity_tokens=112656.0,
            model="qwen3-32b",
            chat_url="http://127.0.0.1:8010/v1/chat/completions",
            scenario="extreme_burst_1200",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _repo_sha(relative_path: str) -> str:
        root = Path(__file__).resolve().parents[1]
        return hashlib.sha256((root / relative_path).read_bytes()).hexdigest()

    def test_trace_preflight_accepts_legacy_synthetic_and_current_turn(self) -> None:
        root = Path(__file__).resolve().parents[1]
        synthetic_path = "tools/materialize_token_aligned_trace.py"
        current_path = "tools/materialize_sharegpt_current_turn_trace.py"
        common_path = "router/common.py"
        synthetic_sha = self._repo_sha(synthetic_path)
        current_sha = self._repo_sha(current_path)
        common_sha = self._repo_sha(common_path)

        # The historical manifest has no tool_path or top-level payload_mode.
        validate_trace_materializer_manifest({
            "tool_sha256": synthetic_sha,
        }, root)
        validate_trace_materializer_manifest({
            "payload_mode": "sharegpt_current_turn_retokenized",
            "tool_path": current_path,
            "tool_sha256": current_sha,
            "dependency_sha256": {
                synthetic_path: synthetic_sha,
                common_path: common_sha,
            },
        }, root)

        # The embedded shell preflight calls this helper with ``os.curdir``.
        # Exercise that path-like string contract, not only direct Path calls.
        validate_trace_materializer_manifest({
            "tool_sha256": synthetic_sha,
        }, str(root))

    def test_current_turn_trace_preflight_rejects_wrong_tool_binding(self) -> None:
        root = Path(__file__).resolve().parents[1]
        synthetic_path = "tools/materialize_token_aligned_trace.py"
        current_path = "tools/materialize_sharegpt_current_turn_trace.py"
        common_path = "router/common.py"
        base = {
            "payload_mode": "sharegpt_current_turn_retokenized",
            "tool_path": current_path,
            "tool_sha256": self._repo_sha(current_path),
            "dependency_sha256": {
                synthetic_path: self._repo_sha(synthetic_path),
                common_path: self._repo_sha(common_path),
            },
        }
        mutations = {
            "tool_path": {**base, "tool_path": "/tmp/untrusted.py"},
            "current checkout tool": {**base, "tool_sha256": "0" * 64},
            "dependency differs": {
                **base,
                "dependency_sha256": {
                    synthetic_path: "f" * 64,
                    common_path: self._repo_sha(common_path),
                },
            },
            "router/common.py": {
                **base,
                "dependency_sha256": {
                    synthetic_path: self._repo_sha(synthetic_path),
                    common_path: "e" * 64,
                },
            },
            "lacks dependency": {**base, "dependency_sha256": None},
        }
        for message, manifest in mutations.items():
            with self.subTest(message=message), self.assertRaisesRegex(
                MatrixEvidenceError, message
            ):
                validate_trace_materializer_manifest(manifest, root)

    def test_legacy_synthetic_preflight_keeps_original_tool_hash_check(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(
            MatrixEvidenceError, "trace was not materialized"
        ):
            validate_trace_materializer_manifest({
                "tool_sha256": "0" * 64,
            }, root)

    def _summary(self, policy: str, *, trigger: str = "ttft_pred",
                 selector: str = "cost_cachedisp_old") -> dict:
        return {
            "policy": policy,
            "overall": {"n": 2, "success": 2},
            "local": {"n": 2, "success": 2},
            "cloud": {"n": 0, "success": 0},
            "token_alignment": {
                "local_success_n": 2,
                "measured_n": 2,
                "missing_prompt_usage_n": 0,
                "prompt_exact_n": 2,
                "decode_measured_n": 2,
                "missing_completion_usage_n": 0,
                "decode_cap_hit_n": 2,
            },
            "config": {
                "scenario": self.expected.scenario,
                "seed": 0,
                "time_scale": 1.0,
                "max_inflight": self.expected.max_inflight,
                "max_tokens_override": None,
                "temperature": self.expected.temperature,
                "ignore_eos": self.expected.ignore_eos,
                "local_ignore_eos": True,
                "cloud_ignore_eos": True,
                "nimbus_trigger": trigger,
                "nimbus_selector": selector,
                "prefill_tput": self.expected.prefill_tput,
                "tpot_ms": self.expected.tpot_ms,
                "first_token_overhead_ms": self.expected.first_token_overhead_ms,
                "slo_s": self.expected.slo_s,
                "timeout_s": self.expected.timeout_s,
                "ttft_guard_ms": self.expected.ttft_guard_ms,
                "nimbus_tick_ms": self.expected.nimbus_tick_ms,
                "kv_capacity_tokens": self.expected.kv_capacity_tokens,
                "kv_hysteresis_fraction": 0.05,
                "cloud": "null",
                "cloud_url": None,
                "cloud_model": self.expected.model,
                "cloud_api_key_env": None,
                "cloud_max_concurrency": 32,
                "cloud_provider_order": None,
                "cloud_no_fallbacks": False,
                "cloud_stop_after_first_token": False,
                "local_url": self.expected.chat_url,
                "local_model": self.expected.model,
                "in_price": self.expected.in_price,
                "out_price": self.expected.out_price,
            },
            "queue": {
                "nimbus_trigger": trigger,
                "nimbus_selector": selector,
            },
        }

    def _real_summary(self, *, cloud_success: bool = True) -> dict:
        payload = self._summary("nimbus")
        payload["overall"] = {
            "n": 2,
            "success": 2 if cloud_success else 1,
            "slo_measured_n": 2,
        }
        payload["local"] = {"n": 1, "success": 1, "routed_only": 0}
        payload["cloud"] = {
            "n": 1,
            "success": 1 if cloud_success else 0,
            "routed_only": 0,
            "cost_usd": None,
            "known_cost_usd": 0.0,
            "cost_measured_n": 0,
            "cost_pending_n": 1,
        }
        payload["token_alignment"].update({
            "local_success_n": 1,
            "measured_n": 1,
            "prompt_exact_n": 1,
            "decode_measured_n": 1,
            "decode_cap_hit_n": 1,
        })
        payload["config"].update({
            "cloud": "real",
            "cloud_url": "https://openrouter.ai/api/v1/chat/completions",
            "cloud_model": "qwen/qwen3-32b",
            "cloud_api_key_env": "OPENROUTER_API_KEY",
            "cloud_provider_order": ["deepinfra"],
            "cloud_no_fallbacks": True,
            "cloud_stop_after_first_token": True,
            "local_ignore_eos": True,
            "cloud_ignore_eos": False,
        })
        return payload

    def _real_expected(self) -> MatrixExpectations:
        return replace(
            self.expected,
            cloud="real",
            cloud_url="https://openrouter.ai/api/v1/chat/completions",
            cloud_model="qwen/qwen3-32b",
            cloud_api_key_env="OPENROUTER_API_KEY",
            cloud_provider_order=("deepinfra",),
            cloud_no_fallbacks=True,
            cloud_stop_after_first_token=True,
            local_ignore_eos=True,
            cloud_ignore_eos=False,
        )

    @staticmethod
    def _real_raw(*, cloud_success: bool = True) -> str:
        local = {
            "request_id": "local-1",
            "endpoint": "local",
            "success": True,
            "scheduler_prompt_tokens": 10,
            "prompt_tokens": 10,
            "scheduler_decode_tokens": 5,
            "completion_tokens": 5,
        }
        cloud = {
            "request_id": "cloud-1",
            "endpoint": "cloud",
            "success": cloud_success,
            "ttft_ms": 467.0 if cloud_success else None,
            "pre_route_queue_ms": 100.0,
            "cloud_gate_wait_ms": 17.0,
            "service_ttft_ms": 350.0,
            "routed_only": False,
            "response_completed": False,
            "stream_abort_requested": cloud_success,
            "probe_mode": "ttft_cancel",
            "cost_usd": None,
            "cost_pending": True,
        }
        return json.dumps(local) + "\n" + json.dumps(cloud) + "\n"

    @staticmethod
    def _real_decisions(victim: str = "cloud-1") -> str:
        return (
            json.dumps({"status": "stale_retry"}) + "\n"
            + json.dumps({"status": "applied", "applied_victim_ids": [victim]})
            + "\n"
        )

    def _paths(self, summary: dict, decisions: str = "") -> dict[str, Path]:
        paths = {
            name: self.root / name
            for name in ("raw", "summary", "decisions", "marker")
        }
        paths["raw"].write_text('{"request_id": 1}\n{"request_id": 2}\n',
                                encoding="utf-8")
        paths["summary"].write_text(json.dumps(summary), encoding="utf-8")
        paths["decisions"].write_text(decisions, encoding="utf-8")
        return paths

    def _validate(self, paths: dict[str, Path], arm: str, *, write: bool) -> dict:
        return validate_or_write_marker(
            raw_path=paths["raw"],
            summary_path=paths["summary"],
            decisions_path=paths["decisions"],
            marker_path=paths["marker"],
            fingerprint="fingerprint",
            arm_text=arm,
            expected=self.expected,
            write_marker=write,
        )

    def test_anchor_policy_cli_never_contains_nimbus_flags(self) -> None:
        anchor = parse_arm(ANCHOR_ARM)
        self.assertEqual(anchor.policy_cli, ("--policy", "all_local"))
        nimbus = parse_arm("ttft_pred:cost_cachedisp_old:0")
        self.assertEqual(nimbus.policy_cli, (
            "--policy", "nimbus",
            "--nimbus-trigger", "ttft_pred",
            "--nimbus-selector", "cost_cachedisp_old",
        ))
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["policy-cli", ANCHOR_ARM]), 0)
        self.assertEqual(output.getvalue().splitlines(), ["--policy", "all_local"])

    def test_unsupported_anchor_forms_are_rejected(self) -> None:
        for arm in (
            "anchor:all_local:1",
            "anchor:newest:0",
            "ttft_pred:all_local:0",
            "all_local:cost_disp_current:0",
        ):
            with self.subTest(arm=arm), self.assertRaises(MatrixEvidenceError):
                parse_arm(arm)

    def test_anchor_writes_and_revalidates_marker_with_empty_decisions(self) -> None:
        # Baseline argparse defaults still appear in config, but they are not
        # interpreted as anchor trigger/selector evidence.
        paths = self._paths(self._summary("all_local"))
        marker = self._validate(paths, ANCHOR_ARM, write=True)
        self.assertEqual(marker["artifacts"]["decisions"]["nonempty_line_n"], 0)
        self.assertEqual(self._validate(paths, ANCHOR_ARM, write=False), marker)

    def test_marker_binds_request_timeout(self) -> None:
        payload = self._summary("all_local")
        payload["config"]["timeout_s"] = self.expected.timeout_s + 1
        paths = self._paths(payload)
        with self.assertRaisesRegex(MatrixEvidenceError, "timeout_s"):
            self._validate(paths, ANCHOR_ARM, write=True)

    def test_anchor_requires_every_trace_row_to_align_and_empty_decisions(self) -> None:
        partial = self._summary("all_local")
        partial["token_alignment"].update({
            "local_success_n": 1,
            "measured_n": 1,
            "prompt_exact_n": 1,
            "decode_measured_n": 1,
            "decode_cap_hit_n": 1,
        })
        paths = self._paths(partial)
        with self.assertRaisesRegex(MatrixEvidenceError, "anchor measured row count"):
            self._validate(paths, ANCHOR_ARM, write=True)

        paths = self._paths(self._summary("all_local"), '{"decision": 1}\n')
        with self.assertRaisesRegex(MatrixEvidenceError, "decision log is not empty"):
            self._validate(paths, ANCHOR_ARM, write=True)

    def test_nimbus_still_requires_nonempty_decisions_and_exact_selector(self) -> None:
        arm = "ttft_pred:cost_cachedisp_old:0"
        paths = self._paths(self._summary("nimbus"))
        with self.assertRaisesRegex(MatrixEvidenceError, "Nimbus decision log is empty"):
            self._validate(paths, arm, write=True)

        paths["decisions"].write_text('{"trigger": "ttft_pred"}\n', encoding="utf-8")
        self._validate(paths, arm, write=True)
        wrong = self._summary("nimbus", selector="newest")
        paths["summary"].write_text(json.dumps(wrong), encoding="utf-8")
        with self.assertRaisesRegex(MatrixEvidenceError, "nimbus_selector"):
            self._validate(paths, arm, write=True)

    def test_kv_gap_rejects_any_metrics_read_failure(self) -> None:
        arm = "kv_gap:cost_disp_current:0"
        payload = self._summary(
            "nimbus", trigger="kv_gap", selector="cost_disp_current"
        )
        payload["queue"]["kv_read_failures"] = 1
        paths = self._paths(payload, '{"trigger": "kv_gap"}\n')
        with self.assertRaisesRegex(MatrixEvidenceError, "kv_read_failures"):
            self._validate(paths, arm, write=True)

        payload["queue"]["kv_read_failures"] = 0
        paths["summary"].write_text(json.dumps(payload), encoding="utf-8")
        self._validate(paths, arm, write=True)

    def test_real_cloud_marker_accepts_pending_cost_and_endpoint_failure(self) -> None:
        arm = "ttft_pred:cost_cachedisp_old:0"
        for cloud_success in (True, False):
            with self.subTest(cloud_success=cloud_success):
                paths = self._paths(
                    self._real_summary(cloud_success=cloud_success),
                    self._real_decisions(),
                )
                paths["raw"].write_text(
                    self._real_raw(cloud_success=cloud_success), encoding="utf-8",
                )
                marker = validate_or_write_marker(
                    raw_path=paths["raw"],
                    summary_path=paths["summary"],
                    decisions_path=paths["decisions"],
                    marker_path=paths["marker"],
                    fingerprint="fingerprint",
                    arm_text=arm,
                    expected=self._real_expected(),
                    write_marker=True,
                )
                self.assertEqual(marker["artifacts"]["raw"]["nonempty_line_n"], 2)

    def test_real_cloud_requires_all_rows_measured_and_no_routed_only(self) -> None:
        arm = "ttft_pred:cost_cachedisp_old:0"
        summary = self._real_summary()
        summary["overall"]["slo_measured_n"] = 1
        paths = self._paths(summary, self._real_decisions())
        paths["raw"].write_text(self._real_raw(), encoding="utf-8")
        with self.assertRaisesRegex(MatrixEvidenceError, "measured SLO row count"):
            validate_or_write_marker(
                raw_path=paths["raw"], summary_path=paths["summary"],
                decisions_path=paths["decisions"], marker_path=paths["marker"],
                fingerprint="fingerprint", arm_text=arm,
                expected=self._real_expected(), write_marker=True,
            )

        summary = self._real_summary()
        summary["cloud"]["routed_only"] = 1
        paths = self._paths(summary, self._real_decisions())
        paths["raw"].write_text(self._real_raw(), encoding="utf-8")
        with self.assertRaisesRegex(MatrixEvidenceError, "cloud routed_only count"):
            validate_or_write_marker(
                raw_path=paths["raw"], summary_path=paths["summary"],
                decisions_path=paths["decisions"], marker_path=paths["marker"],
                fingerprint="fingerprint", arm_text=arm,
                expected=self._real_expected(), write_marker=True,
            )

    def test_real_cloud_requires_success_ttft_abort_and_local_token_exact(self) -> None:
        arm = "ttft_pred:cost_cachedisp_old:0"
        mutations = {
            "finite nonnegative TTFT": lambda rows: rows[1].update(ttft_ms=None),
            "response_completed": lambda rows: rows[1].update(response_completed=True),
            "stream_abort_requested": lambda rows: rows[1].update(
                stream_abort_requested=False
            ),
            "prompt_tokens": lambda rows: rows[0].update(prompt_tokens=9),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message):
                rows = [json.loads(line) for line in self._real_raw().splitlines()]
                mutate(rows)
                paths = self._paths(
                    self._real_summary(), self._real_decisions(),
                )
                paths["raw"].write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(MatrixEvidenceError, message):
                    validate_or_write_marker(
                        raw_path=paths["raw"], summary_path=paths["summary"],
                        decisions_path=paths["decisions"],
                        marker_path=paths["marker"], fingerprint="fingerprint",
                        arm_text=arm, expected=self._real_expected(),
                        write_marker=True,
                    )

    def test_real_cloud_requires_exact_unique_applied_victim_mapping(self) -> None:
        arm = "ttft_pred:cost_cachedisp_old:0"
        bad_decisions = {
            "duplicate request IDs": (
                json.dumps({"applied_victim_ids": ["cloud-1"]}) + "\n"
                + json.dumps({"applied_victim_ids": ["cloud-1"]}) + "\n"
            ),
            "do not match raw cloud request IDs": self._real_decisions("other"),
            "applied_victim_ids is not a list": json.dumps({
                "applied_victim_ids": "cloud-1",
            }) + "\n",
        }
        for message, decisions in bad_decisions.items():
            with self.subTest(message=message):
                paths = self._paths(self._real_summary(), decisions)
                paths["raw"].write_text(self._real_raw(), encoding="utf-8")
                with self.assertRaisesRegex(MatrixEvidenceError, message):
                    validate_or_write_marker(
                        raw_path=paths["raw"], summary_path=paths["summary"],
                        decisions_path=paths["decisions"],
                        marker_path=paths["marker"], fingerprint="fingerprint",
                        arm_text=arm, expected=self._real_expected(),
                        write_marker=True,
                    )

    def test_real_cloud_requires_finite_nonnegative_additive_ttft_parts(self) -> None:
        arm = "ttft_pred:cost_cachedisp_old:0"
        mutations = {
            "pre_route_queue_ms": lambda row: row.update(pre_route_queue_ms=None),
            "cloud_gate_wait_ms": lambda row: row.update(cloud_gate_wait_ms=-1.0),
            "service_ttft_ms": lambda row: row.update(service_ttft_ms=float("inf")),
            "latency component sum": lambda row: row.update(ttft_ms=468.0),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message):
                rows = [json.loads(line) for line in self._real_raw().splitlines()]
                mutate(rows[1])
                paths = self._paths(
                    self._real_summary(), self._real_decisions(),
                )
                paths["raw"].write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(MatrixEvidenceError, message):
                    validate_or_write_marker(
                        raw_path=paths["raw"], summary_path=paths["summary"],
                        decisions_path=paths["decisions"],
                        marker_path=paths["marker"], fingerprint="fingerprint",
                        arm_text=arm, expected=self._real_expected(),
                        write_marker=True,
                    )


if __name__ == "__main__":
    unittest.main()
