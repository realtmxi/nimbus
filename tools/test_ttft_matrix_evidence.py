"""Focused stdlib tests for the TTFT matrix arm/evidence contract."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.ttft_matrix_evidence import (
    ANCHOR_ARM,
    MatrixEvidenceError,
    MatrixExpectations,
    main,
    parse_arm,
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
                "nimbus_trigger": trigger,
                "nimbus_selector": selector,
                "prefill_tput": self.expected.prefill_tput,
                "tpot_ms": self.expected.tpot_ms,
                "first_token_overhead_ms": self.expected.first_token_overhead_ms,
                "slo_s": self.expected.slo_s,
                "ttft_guard_ms": self.expected.ttft_guard_ms,
                "nimbus_tick_ms": self.expected.nimbus_tick_ms,
                "kv_capacity_tokens": self.expected.kv_capacity_tokens,
                "kv_hysteresis_fraction": 0.05,
                "cloud": "null",
                "cloud_max_concurrency": 32,
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


if __name__ == "__main__":
    unittest.main()
