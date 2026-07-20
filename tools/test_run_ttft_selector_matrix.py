"""Static and embedded-guard tests for the TTFT matrix shell runner."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_ttft_selector_matrix.sh"
ATTESTATION_SHA = "a" * 64
LAUNCH_SHA = "b" * 64
LAUNCH_PATH = "/tmp/e12-launch.json"
EVIDENCE_ENV = {
    "E12_LIVE_PRICE_SNAPSHOT": "/tmp/price.json",
    "E12_LIVE_BUDGET_ATTESTATION": "/tmp/budget.json",
    "E12_LIVE_CANARY_ATTESTATION": "/tmp/canary.json",
    "E12_LIVE_BASELINE_USAGE": "/tmp/baseline.json",
    "E12_LIVE_SETTLEMENT_PREVIOUS_USAGE": "/tmp/previous.json",
    "E12_LIVE_SETTLEMENT_CURRENT_USAGE": "/tmp/current.json",
    "E12_LIVE_STAGE_BUDGET_GATE": "/tmp/gate.json",
}


class TestTTFTSelectorMatrixRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")
        opening = "<<'PY_SYSTEMIC_CLOUD_CHECK'\n"
        start = cls.source.index(opening) + len(opening)
        end = cls.source.index("\nPY_SYSTEMIC_CLOUD_CHECK", start)
        cls.cloud_guard = cls.source[start:end] + "\n"
        contract_start = cls.source.index("REAL_CLOUD_EXPECT_N_WAS_SET=")
        contract_end = cls.source.index("\nfor bit_name", contract_start)
        cls.contract_setup = cls.source[contract_start:contract_end] + "\n"

    def run_cloud_guard(
        self, rows: list[dict[str, object]]
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "arm.jsonl"
            raw.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            return subprocess.run(
                [sys.executable, "-", str(raw), "test-arm", "unit-test"],
                input=self.cloud_guard,
                text=True,
                capture_output=True,
                check=False,
            )

    def run_contract_setup(
        self, **environment: str
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for name in (
            "CLOUD",
            "REAL_CLOUD_EXPECT_N",
            "E12_LIVE_CONTRACT_ID",
            "E12_LIVE_STAGE",
            "E12_LIVE_EXPECT_N",
            "E12_LIVE_ARM_ORDER_SEED",
            "E12_LIVE_AUTHORIZED_BUDGET_USD",
            "E12_LIVE_BUDGET_ATTESTATION_SHA256",
            "E12_LIVE_KEY_LIMIT_MAX_USD",
            "E12_LIVE_KEY_CONTRACT_MODE",
            "E12_LIVE_STAGE_LAUNCH_ATTESTATION",
            "E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256",
            "E12_LIVE_PRICE_SNAPSHOT",
            "E12_LIVE_BUDGET_ATTESTATION",
            "E12_LIVE_CANARY_ATTESTATION",
            "E12_LIVE_BASELINE_USAGE",
            "E12_LIVE_SETTLEMENT_PREVIOUS_USAGE",
            "E12_LIVE_SETTLEMENT_CURRENT_USAGE",
            "E12_LIVE_STAGE_BUDGET_GATE",
            "E12_LIVE_A_DIR",
            "E12_LIVE_A_STAGE_LAUNCH_ATTESTATION",
        ):
            env.pop(name, None)
        env.update(environment)
        program = self.contract_setup + (
            "printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\tEND\\n' "
            '"$E12_LIVE_ACTIVE" "${E12_LIVE_EXACT_ARM-}" '
            '"$REAL_CLOUD_EXPECT_N" "$E12_LIVE_CONTRACT" '
            '"$E12_LIVE_EXPECT_N" "$E12_LIVE_ARM_ORDER_SEED" '
            '"$E12_LIVE_AUTHORIZED_BUDGET_USD" '
            '"$E12_LIVE_BUDGET_ATTESTATION_SHA256" '
            '"$E12_LIVE_KEY_LIMIT_MAX_USD"\n'
        )
        return subprocess.run(
            ["bash", "-c", program],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def cloud_row(status: int | None, *, success: bool = False) -> dict[str, object]:
        return {
            "request_id": f"cloud-{status}",
            "endpoint": "cloud",
            "success": success,
            "http_status": status,
        }

    def test_runner_has_valid_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_e11_default_and_e12_single_stage_contract_are_both_present(self) -> None:
        # The legacy real-cloud path remains the exact ordered A/C pair.
        self.assertIn("real-cloud E11 requires exact A then C arms", self.source)
        default_start = self.source.index("default_arms=(")
        default_end = self.source.index(")", default_start)
        default_block = self.source[default_start:default_end]
        self.assertLess(
            default_block.index("ttft_pred:cost_cachedisp_old:0"),
            default_block.index("ttft_pred:cost_disp_current:0"),
        )

        # E12 is activated only through explicit contract variables and maps
        # each stage to one immutable arm.
        self.assertIn("E12_LIVE_CONTRACT_ID_WAS_SET", self.source)
        self.assertIn("E12_LIVE_STAGE_WAS_SET", self.source)
        self.assertIn("E12_LIVE_EXPECT_N=${E12_LIVE_EXPECT_N:-11604}", self.source)
        self.assertIn(
            "E12_LIVE_ARM_ORDER_SEED=${E12_LIVE_ARM_ORDER_SEED:-20260716}",
            self.source,
        )
        self.assertIn(
            "E12_LIVE_EXACT_ARM=ttft_pred:cost_cachedisp_old:0",
            self.source,
        )
        self.assertIn(
            "E12_LIVE_EXACT_ARM=ttft_pred:cost_disp_current:0",
            self.source,
        )
        self.assertIn('set -- "$E12_LIVE_EXACT_ARM"', self.source)
        self.assertIn("ARM_ORDER_MODE=e12_contract_stage", self.source)
        self.assertIn('[[ "$IN_PRICE" != 0.08 ]]', self.source)
        self.assertIn('[[ "$OUT_PRICE" != 0.28 ]]', self.source)
        self.assertIn('[[ "$TIMEOUT_S" != 600 ]]', self.source)
        self.assertIn('[[ "$COOLDOWN_S" != 20 ]]', self.source)
        self.assertIn(
            '[[ "$CLOUD_API_KEY_ENV" != OPENROUTER_API_KEY ]]', self.source
        )

    def test_contract_setup_maps_each_e12_stage_and_preserves_e11_default(self) -> None:
        e11 = self.run_contract_setup(CLOUD="real")
        self.assertEqual(e11.returncode, 0, e11.stderr)
        self.assertEqual(
            e11.stdout.strip().split("\t"),
            [
                "0",
                "",
                "11605",
                "e12_current_turn_v1",
                "11604",
                "20260716",
                "",
                "",
                "",
                "END",
            ],
        )

        expected_arms = {
            "A": "ttft_pred:cost_cachedisp_old:0",
            "C": "ttft_pred:cost_disp_current:0",
        }
        for stage, arm in expected_arms.items():
            with self.subTest(stage=stage):
                result = self.run_contract_setup(
                    CLOUD="real",
                    E12_LIVE_CONTRACT_ID="e12-live-001",
                    E12_LIVE_STAGE=stage,
                    E12_LIVE_AUTHORIZED_BUDGET_USD="3",
                    E12_LIVE_BUDGET_ATTESTATION_SHA256=ATTESTATION_SHA,
                    E12_LIVE_KEY_LIMIT_MAX_USD="3",
                    E12_LIVE_STAGE_LAUNCH_ATTESTATION=LAUNCH_PATH,
                    E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256=LAUNCH_SHA,
                    **EVIDENCE_ENV,
                    **(
                        {
                            "E12_LIVE_A_DIR": "/tmp/a",
                            "E12_LIVE_A_STAGE_LAUNCH_ATTESTATION": "/tmp/a-launch.json",
                        }
                        if stage == "C" else {}
                    ),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    result.stdout.strip().split("\t"),
                    [
                        "1",
                        arm,
                        "11604",
                        "e12_current_turn_v1",
                        "11604",
                        "20260716",
                        "3",
                        ATTESTATION_SHA,
                        "3",
                        "END",
                    ],
                )

        marketplace = self.run_contract_setup(
            CLOUD="real",
            E12_LIVE_CONTRACT_ID="e12-live-marketplace-001",
            E12_LIVE_STAGE="A",
            E12_LIVE_AUTHORIZED_BUDGET_USD="3",
            E12_LIVE_BUDGET_ATTESTATION_SHA256=ATTESTATION_SHA,
            E12_LIVE_KEY_LIMIT_MAX_USD="5",
            E12_LIVE_KEY_CONTRACT_MODE="e12_marketplace_deepinfra_no_byok_v1",
            E12_LIVE_STAGE_LAUNCH_ATTESTATION=LAUNCH_PATH,
            E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256=LAUNCH_SHA,
            **EVIDENCE_ENV,
        )
        self.assertEqual(marketplace.returncode, 0, marketplace.stderr)
        fields = marketplace.stdout.strip().split("\t")
        self.assertEqual(fields[3], "e12_current_turn_marketplace_v2")
        self.assertEqual(fields[8], "5")

    def test_contract_setup_rejects_partial_or_mutated_e12_contracts(self) -> None:
        authorized = {
            **EVIDENCE_ENV,
            "CLOUD": "real",
            "E12_LIVE_CONTRACT_ID": "e12-live-001",
            "E12_LIVE_STAGE": "A",
            "E12_LIVE_AUTHORIZED_BUDGET_USD": "3",
            "E12_LIVE_BUDGET_ATTESTATION_SHA256": ATTESTATION_SHA,
            "E12_LIVE_KEY_LIMIT_MAX_USD": "3",
            "E12_LIVE_STAGE_LAUNCH_ATTESTATION": LAUNCH_PATH,
            "E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256": LAUNCH_SHA,
        }
        cases = (
            {"CLOUD": "real", "E12_LIVE_STAGE": "A"},
            {
                **authorized,
                "E12_LIVE_CONTRACT_ID": "unsafe id",
            },
            {
                **authorized,
                "E12_LIVE_STAGE": "B",
            },
            {
                **authorized,
                "E12_LIVE_EXPECT_N": "11605",
            },
            {
                **authorized,
                "REAL_CLOUD_EXPECT_N": "11605",
            },
            {**authorized, "E12_LIVE_AUTHORIZED_BUDGET_USD": "3.0"},
            {**authorized, "E12_LIVE_BUDGET_ATTESTATION_SHA256": "f" * 63},
            {**authorized, "E12_LIVE_BUDGET_ATTESTATION_SHA256": "A" * 64},
            {**authorized, "E12_LIVE_KEY_LIMIT_MAX_USD": "4"},
            {**authorized, "E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256": "f" * 63},
            {**authorized, "E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256": "B" * 64},
        )
        for environment in cases:
            with self.subTest(environment=environment):
                self.assertEqual(
                    self.run_contract_setup(**environment).returncode,
                    4,
                )

    def test_e12_contract_is_bound_to_fingerprint_manifest_and_events(self) -> None:
        ordered_fingerprint_fields = """LIVE_FINGERPRINT_ARGS=(
    "$E12_LIVE_CONTRACT_ID"
    "$E12_LIVE_CONTRACT"
    "$E12_LIVE_STAGE"
    "$E12_LIVE_EXPECT_N"
    "$E12_LIVE_ARM_ORDER_SEED"
    "$E12_LIVE_EXACT_ARM"
    "$E12_LIVE_AUTHORIZED_BUDGET_USD"
    "$E12_LIVE_BUDGET_ATTESTATION_SHA256"
    "$E12_LIVE_KEY_LIMIT_MAX_USD"
    "$E12_LIVE_KEY_CONTRACT_MODE"
    "$E12_LIVE_COOLDOWN_S"
    "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256"
    "$E12_LIVE_CURRENT_USAGE_SHA256"
    "$E12_LIVE_VERIFY_RECEIPT_SHA256"
  )"""
        self.assertIn(ordered_fingerprint_fields, self.source)
        self.assertIn(
            "live_contract_id=%s live_contract=%s live_stage=%s "
            "live_expected_trace_n=%s live_arm_order_seed=%s live_exact_arm=%s "
            "live_authorized_budget_usd=%s "
            "live_budget_attestation_sha256=%s live_key_limit_max_usd=%s "
            "live_key_contract_mode=%s "
            "live_cooldown_s=%s live_stage_launch_attestation_sha256=%s "
            "live_current_usage_sha256=%s "
            "live_stage_launch_verify_receipt_sha256=%s",
            self.source,
        )
        self.assertIn(
            "arm_started_at=%s arm=%s live_contract_id=%s live_stage=%s "
            "live_authorized_budget_usd=%s "
            "live_budget_attestation_sha256=%s live_key_limit_max_usd=%s "
            "live_key_contract_mode=%s "
            "live_cooldown_s=%s live_stage_launch_attestation_sha256=%s "
            "live_current_usage_sha256=%s "
            "live_stage_launch_verify_receipt_sha256=%s "
            "command=router.run_config_bound_by_fingerprint",
            self.source,
        )
        self.assertIn(
            "matrix_finished_at=%s run_fingerprint=%s live_contract_id=%s "
            "live_stage=%s live_authorized_budget_usd=%s "
            "live_budget_attestation_sha256=%s live_key_limit_max_usd=%s "
            "live_key_contract_mode=%s "
            "live_cooldown_s=%s live_stage_launch_attestation_sha256=%s "
            "live_current_usage_sha256=%s "
            "live_stage_launch_verify_receipt_sha256=%s",
            self.source,
        )
        fixed_command = "command=router.run_config_bound_by_fingerprint\\n'"
        fixed_at = self.source.index(fixed_command)
        else_at = self.source.index("    else\n", fixed_at)
        argv_at = self.source.index("printf '%q ' \"${cmd[@]}\"", else_at)
        self.assertLess(fixed_at, else_at)
        self.assertLess(else_at, argv_at)

    def test_e12_launch_verification_precedes_every_router_run(self) -> None:
        capture = '"$PYBIN" -m tools.openrouter_usage_snapshot'
        verify = '"$PYBIN" -m tools.check_e12_stage_launch verify'
        launch = '"${cmd[@]}"'
        self.assertIn(capture, self.source)
        self.assertIn(verify, self.source)
        self.assertLess(self.source.index(capture), self.source.index(verify))
        self.assertLess(self.source.index(verify), self.source.index(launch))
        self.assertIn('--expected-sha256 "$E12_LIVE_STAGE_LAUNCH_ATTESTATION_SHA256"', self.source)
        self.assertIn(
            'E12_LIVE_CURRENT_USAGE_ARTIFACT="$OUT_DIR/e12_live_current_usage.json"',
            self.source,
        )
        self.assertIn(
            'E12_LIVE_VERIFY_RECEIPT_ARTIFACT="$OUT_DIR/e12_stage_launch_verify_receipt.json"',
            self.source,
        )
        self.assertIn(
            '--live-current-usage "$E12_LIVE_CURRENT_USAGE_ARTIFACT"',
            self.source,
        )
        self.assertIn('--output "$E12_LIVE_VERIFY_RECEIPT_ARTIFACT"', self.source)
        self.assertIn("--no-overwrite", self.source)
        self.assertIn('chmod 600 "$E12_LIVE_CURRENT_USAGE_ARTIFACT"', self.source)
        self.assertIn('chmod 600 "$E12_LIVE_VERIFY_RECEIPT_ARTIFACT"', self.source)
        self.assertNotIn("E12_LIVE_CURRENT_USAGE_TMP", self.source)
        self.assertIn(
            '--data "$DATA" --expected-trace-sha256 "$TRACE_SHA"',
            self.source,
        )

    def test_openrouter_secret_is_unset_before_children_and_scoped(self) -> None:
        capture = (
            "MATRIX_OPENROUTER_API_KEY=${OPENROUTER_API_KEY-}\n"
            "unset OPENROUTER_API_KEY"
        )
        first_external = 'REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"'
        self.assertIn("set +x", self.source)
        self.assertIn(capture, self.source)
        self.assertLess(self.source.index(capture), self.source.index(first_external))
        self.assertIn(
            'export "$CLOUD_API_KEY_ENV=$MATRIX_CLOUD_API_KEY_SECRET"\n'
            '    "$PYBIN" -m tools.openrouter_usage_snapshot',
            self.source,
        )
        self.assertIn(
            'export "$CLOUD_API_KEY_ENV=$MATRIX_CLOUD_API_KEY_SECRET"\n'
            '    "${cmd[@]}"',
            self.source,
        )
        self.assertEqual(
            self.source.count(
                'export "$CLOUD_API_KEY_ENV=$MATRIX_CLOUD_API_KEY_SECRET"'
            ),
            2,
        )

    def test_e12_payload_and_expected_n_cannot_bypass_contract(self) -> None:
        bypass = self.run_contract_setup(
            CLOUD="real", REAL_CLOUD_EXPECT_N="11604"
        )
        self.assertEqual(bypass.returncode, 4)
        self.assertIn("reserved for the explicit E12", bypass.stderr)
        self.assertIn(
            '[[ "$TRACE_PAYLOAD_MODE" == sharegpt_current_turn_retokenized ]]',
            self.source,
        )

    def test_cloud_guard_rejects_all_or_dominant_400(self) -> None:
        all_bad = self.run_cloud_guard([self.cloud_row(400), self.cloud_row(400)])
        self.assertEqual(all_bad.returncode, 5)
        self.assertIn("systemic cloud authentication/configuration", all_bad.stderr)

        dominant = self.run_cloud_guard(
            [self.cloud_row(400) for _ in range(4)]
            + [self.cloud_row(200, success=True)]
        )
        self.assertEqual(dominant.returncode, 5)

    def test_cloud_guard_rejects_credential_config_or_method_status(self) -> None:
        for status in (401, 402, 403, 404, 405, 422):
            with self.subTest(status=status):
                result = self.run_cloud_guard(
                    [self.cloud_row(200, success=True), self.cloud_row(status)]
                )
                self.assertEqual(result.returncode, 5)
                self.assertIn(f"{status}:1", result.stderr)

    def test_cloud_guard_allows_an_isolated_400(self) -> None:
        result = self.run_cloud_guard(
            [self.cloud_row(400)]
            + [self.cloud_row(200, success=True) for _ in range(4)]
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cloud_guard_rejects_zero_success_and_dominant_non429_http(self) -> None:
        no_success = self.run_cloud_guard([self.cloud_row(429)])
        self.assertEqual(no_success.returncode, 5)
        self.assertIn("cloud_success_n=0", no_success.stderr)

        dominant = self.run_cloud_guard(
            [self.cloud_row(200, success=True) for _ in range(2)]
            + [self.cloud_row(503) for _ in range(8)]
        )
        self.assertEqual(dominant.returncode, 5)
        self.assertIn("non429_http_failure_n=8", dominant.stderr)

    def test_cloud_guard_measures_isolated_http_and_statusless_failures(self) -> None:
        isolated = self.run_cloud_guard(
            [self.cloud_row(503)]
            + [self.cloud_row(200, success=True) for _ in range(9)]
        )
        self.assertEqual(isolated.returncode, 0, isolated.stderr)

        statusless = self.run_cloud_guard(
            [self.cloud_row(None) for _ in range(9)]
            + [self.cloud_row(200, success=True)]
        )
        self.assertEqual(statusless.returncode, 0, statusless.stderr)

    def test_cloud_guard_surrounds_marker_creation_and_resume(self) -> None:
        resume = 'check_systemic_cloud_errors "$raw" "$arm" resume_marker'
        before = 'check_systemic_cloud_errors "$raw" "$arm" pre_marker'
        marker = '"${evidence_cmd[@]}" --write-marker'
        after = 'check_systemic_cloud_errors "$raw" "$arm" post_marker'
        self.assertIn(resume, self.source)
        self.assertLess(self.source.index(before), self.source.index(marker))
        self.assertLess(self.source.index(marker), self.source.index(after))


if __name__ == "__main__":
    unittest.main()
