"""Tests for the offline OpenRouter usage/budget stage gate."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tools.check_openrouter_stage_budget import (
    StageGateError,
    build_stage_gate_attestation,
    main,
    write_json_atomic,
)
from tools.openrouter_key_contract import E12_MARKETPLACE_KEY_CONTRACT


SECRET_LABEL = "provider-label-must-not-leak"
SECRET_BODY = "provider-body-must-not-leak"
SECRET_PATH = "snapshot-path-must-not-leak"
KEY_FINGERPRINT = "a" * 64
BASELINE_AT = "2026-07-18T00:00:00Z"
SETTLEMENT_AT = "2026-07-18T00:01:00Z"
CURRENT_AT = "2026-07-18T00:02:00Z"
NOW = datetime(2026, 7, 18, 0, 3, tzinfo=timezone.utc)
AUTO_SETTLEMENT = object()


def snapshot(
    *,
    usage: object,
    limit: object = 3,
    limit_remaining: object = 3,
    account_status: int = 200,
    total_usage: object = 1,
    total_credits: object = 11,
    remaining_credits: object = 10,
    captured_at_utc: object = BASELINE_AT,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "captured_at_utc": captured_at_utc,
        "key": {
            "http_status": 200,
            "usage": usage,
            "limit": limit,
            "limit_remaining": limit_remaining,
            "key_fingerprint_sha256": KEY_FINGERPRINT,
            "limit_reset": None,
            "include_byok_in_limit": True,
            "is_management_key": False,
            "is_provisioning_key": False,
            "is_free_tier": False,
            "expires_at_utc": None,
        },
        "account": {
            "http_status": account_status,
            "total_usage": total_usage,
            "total_credits": total_credits,
            "remaining_credits": remaining_credits,
            "body": SECRET_BODY,
        },
        "input_path": f"/private/{SECRET_PATH}.json",
    }


def unavailable_snapshot(**kwargs: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "account_status": 403,
        "total_usage": None,
        "total_credits": None,
        "remaining_credits": None,
    }
    arguments.update(kwargs)
    return snapshot(
        **arguments,
    )


def write_snapshot(root: Path, name: str, payload: object) -> Path:
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def gate(
    root: Path,
    baseline: dict[str, object],
    current: dict[str, object],
    settlement_previous: dict[str, object] | None | object = AUTO_SETTLEMENT,
    preserve_timestamps: bool = False,
    **kwargs: object,
) -> dict[str, object]:
    baseline = copy.deepcopy(baseline)
    current = copy.deepcopy(current)
    if not preserve_timestamps:
        baseline["captured_at_utc"] = BASELINE_AT
        current["captured_at_utc"] = CURRENT_AT
    baseline_path = write_snapshot(root, "baseline.json", baseline)
    current_path = write_snapshot(root, "current.json", current)
    if settlement_previous is AUTO_SETTLEMENT:
        settlement_previous = copy.deepcopy(current)
    if settlement_previous is not None:
        assert isinstance(settlement_previous, dict)
        settlement_previous = copy.deepcopy(settlement_previous)
        if not preserve_timestamps:
            settlement_previous["captured_at_utc"] = SETTLEMENT_AT
        kwargs["settlement_previous_path"] = write_snapshot(
            root, "settlement_previous.json", settlement_previous
        )
    kwargs.setdefault("now", NOW)
    return build_stage_gate_attestation(
        baseline_path=baseline_path,
        current_path=current_path,
        **kwargs,
    )


class TestOpenRouterStageBudget(unittest.TestCase):
    def test_explicit_marketplace_mode_keeps_three_dollar_gate_and_no_byok(self):
        def marketplace(value: float) -> dict[str, object]:
            result = unavailable_snapshot(
                usage=value, limit=5, limit_remaining=5 - value
            )
            key = result["key"]
            assert isinstance(key, dict)
            key["include_byok_in_limit"] = False
            key["byok_usage"] = 0.4
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = gate(
                root,
                marketplace(1.9),
                marketplace(2.0),
                key_contract_mode=E12_MARKETPLACE_KEY_CONTRACT,
            )
            self.assertEqual(result["authorized_budget_usd"], "3")
            self.assertEqual(result["baseline_key_limit_usd"], "5")
            self.assertEqual(result["key_contract_mode"], E12_MARKETPLACE_KEY_CONTRACT)
            self.assertIs(result["marketplace_route"]["byok_allowed"], False)

            changed = marketplace(2.0)
            changed_key = changed["key"]
            assert isinstance(changed_key, dict)
            changed_key["byok_usage"] = 0.5
            with self.assertRaisesRegex(StageGateError, "BYOK usage changed"):
                gate(
                    root,
                    marketplace(1.9),
                    changed,
                    key_contract_mode=E12_MARKETPLACE_KEY_CONTRACT,
                )
            with self.assertRaises(StageGateError):
                gate(root, marketplace(1.9), marketplace(2.0))

    def test_before_next_uses_exact_decimal_delta_and_whitelist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = gate(
                root,
                snapshot(
                    usage=0.1,
                    limit_remaining=2.9,
                    total_usage=1.1,
                    total_credits=11.1,
                ),
                snapshot(
                    usage=0.3,
                    limit_remaining=2.7,
                    total_usage=1.3,
                    total_credits=11.3,
                ),
            )
            self.assertEqual(result["mode"], "before_next")
            self.assertEqual(result["usage_delta_usd"], "0.2")
            self.assertEqual(result["required_from_baseline_usd"], "1.15401528")
            self.assertEqual(result["authorized_budget_headroom_usd"], "1.84598472")
            self.assertEqual(
                result["current_key_limit_remaining_headroom_usd"], "1.74598472"
            )

            expected_keys = {
                "schema_version",
                "status",
                "mode",
                "baseline_snapshot_sha256",
                "current_snapshot_sha256",
                "key_fingerprint_sha256",
                "captured_at_utc",
                "settlement",
                "authorized_budget_usd",
                "next_stage_full_upper_bound_usd",
                "baseline_key_limit_usd",
                "current_key_limit_usd",
                "baseline_key_usage_usd",
                "current_key_usage_usd",
                "usage_delta_usd",
                "required_from_baseline_usd",
                "authorized_budget_headroom_usd",
                "baseline_key_limit_remaining_usd",
                "baseline_key_limit_remaining_headroom_usd",
                "current_key_limit_remaining_usd",
                "current_key_limit_remaining_headroom_usd",
                "account_credit_checks",
            }
            self.assertEqual(set(result), expected_keys)
            rendered = json.dumps(result)
            for forbidden in (SECRET_LABEL, SECRET_BODY, SECRET_PATH):
                self.assertNotIn(forbidden, rendered)
            self.assertEqual(
                result["captured_at_utc"],
                {
                    "baseline": BASELINE_AT,
                    "settlement_previous": SETTLEMENT_AT,
                    "current": CURRENT_AT,
                },
            )
            self.assertTrue(result["settlement"]["verified"])
            self.assertEqual(result["settlement"]["observed_interval_s"], "60")

    def test_before_next_accepts_safe_unavailable_account_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            result = gate(
                Path(directory),
                unavailable_snapshot(usage=0, limit_remaining=3),
                unavailable_snapshot(usage=0.5, limit_remaining=2.5),
            )
            checks = result["account_credit_checks"]
            self.assertEqual(
                checks["baseline"],
                {
                    "availability": "unavailable_http_403",
                    "remaining_credits_usd": None,
                    "required_credits_usd": None,
                    "headroom_usd": None,
                },
            )
            self.assertEqual(checks["current"], checks["baseline"])

    def test_final_checks_delta_but_not_future_stage_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            result = gate(
                Path(directory),
                snapshot(
                    usage=0,
                    limit_remaining=3,
                    total_usage=8,
                    total_credits=11,
                    remaining_credits=3,
                ),
                snapshot(
                    usage=2.9,
                    limit_remaining=0.1,
                    total_usage=10.9,
                    total_credits=11,
                    remaining_credits=0.1,
                ),
                settlement_previous=snapshot(
                    usage=2.9,
                    limit_remaining=0.1,
                    total_usage=10.9,
                    total_credits=11,
                    remaining_credits=0.1,
                ),
                final=True,
            )
            self.assertEqual(result["mode"], "final")
            self.assertEqual(result["usage_delta_usd"], "2.9")
            self.assertEqual(result["required_from_baseline_usd"], "2.9")
            self.assertIsNone(result["current_key_limit_remaining_headroom_usd"])
            current_account = result["account_credit_checks"]["current"]
            self.assertIsNone(current_account["required_credits_usd"])
            self.assertIsNone(current_account["headroom_usd"])
            self.assertTrue(result["settlement"]["verified"])
            self.assertEqual(result["settlement"]["observed_interval_s"], "60")

    def test_rejects_invalid_key_limits_usage_and_snapshot_identity(self):
        mutations = [
            (snapshot(usage=0, limit=None), snapshot(usage=0)),
            (snapshot(usage=0, limit=0), snapshot(usage=0, limit=0)),
            (snapshot(usage=0, limit=3.01), snapshot(usage=0, limit=3.01)),
            (snapshot(usage=None), snapshot(usage=0)),
            (snapshot(usage=True), snapshot(usage=0)),
            (snapshot(usage="0"), snapshot(usage=0)),
            (snapshot(usage=0, limit=2.5), snapshot(usage=0, limit=3)),
            (snapshot(usage=0.2), snapshot(usage=0.1)),
        ]
        for field, value in (
            ("limit_reset", "daily"),
            ("include_byok_in_limit", False),
            ("is_management_key", True),
            ("is_provisioning_key", True),
            ("is_free_tier", True),
        ):
            bad = snapshot(usage=0)
            bad_key = bad["key"]
            assert isinstance(bad_key, dict)
            bad_key[field] = value
            mutations.append((bad, snapshot(usage=0)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for baseline, current in mutations:
                with self.subTest(baseline=baseline, current=current):
                    with self.assertRaises(StageGateError):
                        gate(root, baseline, current)

    def test_rejects_each_before_next_budget_shortfall(self):
        cases = [
            (
                snapshot(usage=0, limit_remaining=1.4),
                snapshot(usage=0.5, limit_remaining=1.5),
                {},
            ),
            (
                snapshot(usage=0, limit_remaining=3),
                snapshot(usage=2.1, limit_remaining=0.9),
                {},
            ),
            (
                snapshot(usage=0, limit_remaining=3),
                snapshot(usage=0.5, limit_remaining=0.9),
                {},
            ),
            (
                snapshot(
                    usage=0,
                    limit_remaining=3,
                    total_usage=9.6,
                    total_credits=11,
                    remaining_credits=1.4,
                ),
                snapshot(usage=0.5, limit_remaining=2.5),
                {},
            ),
            (
                snapshot(usage=0, limit_remaining=3),
                snapshot(
                    usage=0.5,
                    limit_remaining=2.5,
                    total_usage=10.1,
                    total_credits=11,
                    remaining_credits=0.9,
                ),
                {},
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for baseline, current, kwargs in cases:
                with self.subTest(baseline=baseline, current=current):
                    with self.assertRaises(StageGateError):
                        gate(root, baseline, current, **kwargs)

    def test_rejects_malformed_account_availability(self):
        cases = [
            unavailable_snapshot(usage=0, total_usage=1),
            snapshot(usage=0, remaining_credits=None),
            snapshot(usage=0, total_credits=10, remaining_credits=10),
            snapshot(usage=0, account_status=500),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for baseline in cases:
                with self.subTest(baseline=baseline):
                    with self.assertRaises(StageGateError):
                        gate(root, baseline, snapshot(usage=0))

    def test_final_rejects_delta_above_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(StageGateError):
                gate(
                    Path(directory),
                    snapshot(usage=0, limit_remaining=3),
                    snapshot(usage=3.01, limit_remaining=0),
                    settlement_previous=snapshot(
                        usage=3.01,
                        limit_remaining=0,
                    ),
                    final=True,
                )

    def test_every_gate_requires_a_settled_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(StageGateError, "settled snapshot pair"):
                gate(
                    root,
                    snapshot(usage=0, limit_remaining=3),
                    snapshot(usage=0, limit_remaining=3),
                    settlement_previous=None,
                )

    def test_rejects_unsettled_usage_or_short_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(StageGateError, "did not remain equal"):
                gate(
                    root,
                    snapshot(usage=0, limit_remaining=3),
                    snapshot(usage=0.2, limit_remaining=2.8),
                    settlement_previous=snapshot(
                        usage=0.1,
                        limit_remaining=2.9,
                    ),
                )

            short_previous = snapshot(
                usage=0.2,
                limit_remaining=2.8,
                captured_at_utc="2026-07-18T00:01:30Z",
            )
            current = snapshot(
                usage=0.2,
                limit_remaining=2.8,
                captured_at_utc=CURRENT_AT,
            )
            with self.assertRaisesRegex(StageGateError, "at least 60 seconds"):
                gate(
                    root,
                    snapshot(
                        usage=0,
                        limit_remaining=3,
                        captured_at_utc=BASELINE_AT,
                    ),
                    current,
                    settlement_previous=short_previous,
                    preserve_timestamps=True,
                )

    def test_caller_cannot_lower_the_frozen_next_stage_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(StageGateError, "cannot be lower"):
                gate(
                    Path(directory),
                    snapshot(usage=0, limit_remaining=3),
                    snapshot(usage=0, limit_remaining=3),
                    next_stage_full_upper_bound_usd="0.01",
                )

    def test_cli_failure_is_nonzero_sanitized_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = write_snapshot(
                root, f"{SECRET_PATH}-baseline.json", {"body": SECRET_BODY}
            )
            current = write_snapshot(root, "current.json", snapshot(usage=0))
            output = root / "must-not-exist.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = main([
                    "--baseline", str(baseline),
                    "--current", str(current),
                    "--output", str(output),
                ])
            self.assertEqual(exit_code, 2)
            self.assertFalse(output.exists())
            rendered_error = stderr.getvalue()
            for forbidden in (SECRET_PATH, SECRET_BODY, SECRET_LABEL):
                self.assertNotIn(forbidden, rendered_error)

    def test_atomic_output_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = gate(
                root,
                unavailable_snapshot(usage=0, limit_remaining=3),
                unavailable_snapshot(usage=0.5, limit_remaining=2.5),
            )
            output = root / "nested" / "gate.json"
            write_json_atomic(output, result)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
