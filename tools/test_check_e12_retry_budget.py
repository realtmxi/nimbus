"""Tests for the text-free cumulative E12 retry-budget guard."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from tools.check_e12_retry_budget import (
    RetryBudgetError,
    build_retry_budget_attestation,
    main,
    write_json_atomic_no_overwrite,
)
from tools.openrouter_key_contract import E12_MARKETPLACE_KEY_CONTRACT


KEY_FINGERPRINT = "a" * 64
SECRET = "sensitive-provider-body-must-not-leak"
GLOBAL_AT = "2026-07-19T00:00:00Z"
RETRY_AT = "2026-07-20T11:00:00Z"
SETTLEMENT_AT = "2026-07-20T11:58:00Z"
CURRENT_AT = "2026-07-20T11:59:00Z"
NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def snapshot(
    *,
    at: str,
    usage: object,
    limit: object = 3,
    remaining: object | None = None,
    fingerprint: str = KEY_FINGERPRINT,
    expires: str | None = None,
    byok: object | None = None,
    account_usage: object = 1,
    account_credits: object = 5,
    account_remaining: object = 4,
) -> dict[str, object]:
    if remaining is None:
        remaining = Decimal(str(limit)) - Decimal(str(usage))
    key: dict[str, object] = {
        "http_status": 200,
        "usage": usage,
        "limit": limit,
        "limit_remaining": remaining,
        "key_fingerprint_sha256": fingerprint,
        "limit_reset": None,
        "include_byok_in_limit": byok is None,
        "is_management_key": False,
        "is_provisioning_key": False,
        "is_free_tier": False,
        "expires_at_utc": expires,
    }
    if byok is not None:
        key["byok_usage"] = byok
    return {
        "schema_version": 1,
        "captured_at_utc": at,
        "key": key,
        "account": {
            "http_status": 200,
            "total_usage": account_usage,
            "total_credits": account_credits,
            "remaining_credits": account_remaining,
            "body": SECRET,
        },
        "input_path": f"/private/{SECRET}.json",
    }


def write(root: Path, name: str, payload: object) -> Path:
    path = root / name
    # Usage snapshots carry JSON numbers.  Decimal inputs make the expected
    # arithmetic explicit in tests; render them as numeric JSON, never strings.
    path.write_text(json.dumps(payload, default=float), encoding="utf-8")
    return path


def guard(
    root: Path,
    global_snapshot: dict[str, object],
    retry_snapshot: dict[str, object],
    settlement_previous: dict[str, object],
    current_snapshot: dict[str, object],
    **kwargs: object,
) -> dict[str, object]:
    return build_retry_budget_attestation(
        global_baseline_path=write(root, "global.json", global_snapshot),
        retry_baseline_path=write(root, "retry.json", retry_snapshot),
        settlement_previous_path=write(
            root, "settlement_previous.json", settlement_previous
        ),
        current_path=write(root, "current.json", current_snapshot),
        now=kwargs.pop("now", NOW),
        **kwargs,
    )


class TestE12RetryBudget(unittest.TestCase):
    def test_old_global_baseline_passes_with_exact_whitelist_and_decimal_math(self):
        with tempfile.TemporaryDirectory() as directory:
            result = guard(
                Path(directory),
                snapshot(at=GLOBAL_AT, usage=Decimal("0.1")),
                snapshot(at=RETRY_AT, usage=Decimal("0.2")),
                snapshot(at=SETTLEMENT_AT, usage=Decimal("0.3")),
                snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
            )
        self.assertEqual(
            set(result),
            {
                "schema_version", "status", "mode", "key_contract_mode",
                "snapshot_sha256", "captured_at_utc", "freshness", "key",
                "budget",
            },
        )
        self.assertEqual(
            set(result["key"]),
            {
                "fingerprint_sha256", "limit_usd", "expires_at_utc",
                "global_baseline_usage_usd", "retry_baseline_usage_usd",
                "settlement_previous_usage_usd", "current_usage_usd",
                "current_limit_remaining_usd", "fresh_pair_settled",
            },
        )
        self.assertEqual(
            set(result["budget"]),
            {
                "authorized_budget_usd", "next_stage_full_upper_bound_usd",
                "global_usage_delta_usd", "retry_usage_delta_usd",
                "projected_global_spend_usd", "projected_headroom_usd",
                "current_limit_remaining_after_future_usd",
            },
        )
        self.assertEqual(
            set(result["snapshot_sha256"]),
            {"global_baseline", "retry_baseline", "settlement_previous", "current"},
        )
        self.assertEqual(result["captured_at_utc"], {
            "global_baseline": GLOBAL_AT,
            "retry_baseline": RETRY_AT,
            "settlement_previous": SETTLEMENT_AT,
            "current": CURRENT_AT,
        })
        self.assertEqual(
            set(result["freshness"]),
            {
                "global_baseline_max_age_enforced",
                "retry_baseline_max_age_s", "current_max_age_s",
                "minimum_settlement_interval_s",
                "observed_settlement_interval_s",
            },
        )
        self.assertEqual(result["key"]["global_baseline_usage_usd"], "0.1")
        self.assertEqual(result["key"]["retry_baseline_usage_usd"], "0.2")
        self.assertEqual(result["key"]["settlement_previous_usage_usd"], "0.3")
        self.assertEqual(result["budget"]["global_usage_delta_usd"], "0.2")
        self.assertEqual(result["budget"]["retry_usage_delta_usd"], "0.1")
        self.assertEqual(
            result["budget"]["projected_global_spend_usd"], "1.15401528"
        )
        self.assertEqual(result["budget"]["projected_headroom_usd"], "1.84598472")
        self.assertFalse(result["freshness"]["global_baseline_max_age_enforced"])
        self.assertEqual(result["freshness"]["observed_settlement_interval_s"], "60")
        self.assertNotIn(SECRET, json.dumps(result, sort_keys=True))

    def test_global_may_equal_retry_at_six_hour_freshness_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            result = guard(
                Path(directory),
                snapshot(at="2026-07-20T06:00:00Z", usage=Decimal("0.1")),
                snapshot(at="2026-07-20T06:00:00Z", usage=Decimal("0.1")),
                snapshot(at=SETTLEMENT_AT, usage=Decimal("0.3")),
                snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
            )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["freshness"]["retry_baseline_max_age_s"], "21600")
        self.assertEqual(result["freshness"]["current_max_age_s"], "600")

    def test_global_cumulative_cap_and_current_capacity_fail_closed(self):
        cases = (
            (
                snapshot(at=GLOBAL_AT, usage=Decimal("0.1")),
                snapshot(at=RETRY_AT, usage=Decimal("2.1")),
                snapshot(at=SETTLEMENT_AT, usage=Decimal("2.2")),
                snapshot(at=CURRENT_AT, usage=Decimal("2.2")),
                "authorized budget",
            ),
            (
                snapshot(at=GLOBAL_AT, usage=Decimal("2.0")),
                snapshot(at=RETRY_AT, usage=Decimal("2.05")),
                snapshot(at=SETTLEMENT_AT, usage=Decimal("2.1")),
                snapshot(at=CURRENT_AT, usage=Decimal("2.1")),
                "limit_remaining",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for (
                global_snapshot, retry_snapshot, settlement_previous,
                current_snapshot, message,
            ) in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    RetryBudgetError, message
                ):
                    guard(
                        root, global_snapshot, retry_snapshot,
                        settlement_previous, current_snapshot,
                    )

    def test_exact_three_dollar_boundary_uses_decimal_arithmetic(self):
        exact_usage = Decimal("2.04598472")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = guard(
                root,
                snapshot(at=GLOBAL_AT, usage=Decimal("0")),
                snapshot(at=RETRY_AT, usage=Decimal("2")),
                snapshot(at=SETTLEMENT_AT, usage=exact_usage),
                snapshot(at=CURRENT_AT, usage=exact_usage),
            )
            self.assertEqual(result["budget"]["projected_global_spend_usd"], "3")
            self.assertEqual(result["budget"]["projected_headroom_usd"], "0")

            over = Decimal("2.04598473")
            with self.assertRaisesRegex(RetryBudgetError, "authorized budget"):
                guard(
                    root,
                    snapshot(at=GLOBAL_AT, usage=Decimal("0")),
                    snapshot(at=RETRY_AT, usage=Decimal("2")),
                    snapshot(at=SETTLEMENT_AT, usage=over),
                    snapshot(at=CURRENT_AT, usage=over),
                )

            with self.assertRaisesRegex(RetryBudgetError, "cannot be lower"):
                guard(
                    root,
                    snapshot(at=GLOBAL_AT, usage=Decimal("0")),
                    snapshot(at=RETRY_AT, usage=Decimal("0.1")),
                    snapshot(at=SETTLEMENT_AT, usage=Decimal("0.2")),
                    snapshot(at=CURRENT_AT, usage=Decimal("0.2")),
                    next_stage_full_upper_bound_usd="0.1",
                )

    def test_final_mode_uses_zero_future_and_hard_global_cap(self):
        def marketplace(at: str, usage: Decimal) -> dict[str, object]:
            return snapshot(
                at=at,
                usage=usage,
                limit=5,
                remaining=Decimal("5") - usage,
                byok=Decimal("0"),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = guard(
                root,
                marketplace(GLOBAL_AT, Decimal("1.9")),
                marketplace(RETRY_AT, Decimal("3")),
                marketplace(SETTLEMENT_AT, Decimal("4.9")),
                marketplace(CURRENT_AT, Decimal("4.9")),
                final=True,
                key_contract_mode=E12_MARKETPLACE_KEY_CONTRACT,
            )
            self.assertEqual(result["mode"], "retry_final")
            self.assertEqual(result["budget"]["next_stage_full_upper_bound_usd"], "0")
            self.assertEqual(result["budget"]["global_usage_delta_usd"], "3")
            self.assertEqual(result["budget"]["projected_global_spend_usd"], "3")
            self.assertEqual(result["budget"]["projected_headroom_usd"], "0")
            self.assertIsNone(result["freshness"]["retry_baseline_max_age_s"])

            over = Decimal("4.90000001")
            with self.assertRaisesRegex(RetryBudgetError, "authorized budget"):
                guard(
                    root,
                    marketplace(GLOBAL_AT, Decimal("1.9")),
                    marketplace(RETRY_AT, Decimal("3")),
                    marketplace(SETTLEMENT_AT, over),
                    marketplace(CURRENT_AT, over),
                    final=True,
                    key_contract_mode=E12_MARKETPLACE_KEY_CONTRACT,
                )
            with self.assertRaisesRegex(RetryBudgetError, "must not include"):
                guard(
                    root,
                    marketplace(GLOBAL_AT, Decimal("1.9")),
                    marketplace(RETRY_AT, Decimal("3")),
                    marketplace(SETTLEMENT_AT, Decimal("4")),
                    marketplace(CURRENT_AT, Decimal("4")),
                    final=True,
                    next_stage_full_upper_bound_usd="0.95401528",
                    key_contract_mode=E12_MARKETPLACE_KEY_CONTRACT,
                )

    def test_only_final_accepts_a_stale_retry_accounting_baseline(self):
        stale_retry_at = "2026-07-20T05:59:59Z"
        snapshots = (
            snapshot(at=GLOBAL_AT, usage=Decimal("0.1")),
            snapshot(at=stale_retry_at, usage=Decimal("0.2")),
            snapshot(at=SETTLEMENT_AT, usage=Decimal("0.3")),
            snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = guard(root, *copy.deepcopy(snapshots), final=True)
            self.assertEqual(result["mode"], "retry_final")
            self.assertIsNone(result["freshness"]["retry_baseline_max_age_s"])
            with self.assertRaisesRegex(
                RetryBudgetError, "retry-baseline snapshot is stale"
            ):
                guard(root, *copy.deepcopy(snapshots))
            with self.assertRaisesRegex(
                RetryBudgetError, "current snapshot is stale"
            ):
                guard(
                    root,
                    copy.deepcopy(snapshots[0]),
                    copy.deepcopy(snapshots[1]),
                    snapshot(
                        at="2026-07-20T11:48:00Z",
                        usage=Decimal("0.3"),
                    ),
                    snapshot(
                        at="2026-07-20T11:49:59Z",
                        usage=Decimal("0.3"),
                    ),
                    final=True,
                )
            with self.assertRaisesRegex(RetryBudgetError, "not settled"):
                guard(
                    root,
                    copy.deepcopy(snapshots[0]),
                    copy.deepcopy(snapshots[1]),
                    snapshot(at=SETTLEMENT_AT, usage=Decimal("0.2")),
                    snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
                    final=True,
                )

    def test_account_capacity_is_required_when_available_but_403_is_safe(self):
        snapshots = [
            snapshot(at=GLOBAL_AT, usage=Decimal("0.1")),
            snapshot(at=RETRY_AT, usage=Decimal("0.2")),
            snapshot(at=SETTLEMENT_AT, usage=Decimal("0.3")),
            snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
        ]
        for payload in snapshots:
            account = payload["account"]
            assert isinstance(account, dict)
            account.update({
                "total_usage": 5,
                "total_credits": 5,
                "remaining_credits": 0,
            })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                RetryBudgetError, "account remaining credits"
            ):
                guard(root, *copy.deepcopy(snapshots))

            unavailable = copy.deepcopy(snapshots)
            for payload in unavailable:
                account = payload["account"]
                assert isinstance(account, dict)
                account.update({
                    "http_status": 403,
                    "total_usage": None,
                    "total_credits": None,
                    "remaining_credits": None,
                })
            self.assertEqual(guard(root, *unavailable)["status"], "pass")

    def test_requires_order_freshness_interval_and_settlement(self):
        base = snapshot(at=GLOBAL_AT, usage=Decimal("0.1"))
        cases = (
            (
                snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
                snapshot(at=RETRY_AT, usage=Decimal("0.3")),
                snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
                NOW,
                "not ordered",
            ),
            (
                snapshot(at="2026-07-20T05:59:59Z", usage=Decimal("0.3")),
                snapshot(at=SETTLEMENT_AT, usage=Decimal("0.3")),
                snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
                NOW,
                "retry-baseline snapshot is stale",
            ),
            (
                snapshot(at=RETRY_AT, usage=Decimal("0.3")),
                snapshot(at="2026-07-20T11:48:00Z", usage=Decimal("0.3")),
                snapshot(at="2026-07-20T11:49:59Z", usage=Decimal("0.3")),
                NOW,
                "current snapshot is stale",
            ),
            (
                snapshot(at=RETRY_AT, usage=Decimal("0.3")),
                snapshot(at="2026-07-20T11:49:59Z", usage=Decimal("0.3")),
                snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
                NOW,
                "settlement-previous snapshot is stale",
            ),
            (
                snapshot(at=RETRY_AT, usage=Decimal("0.2")),
                snapshot(at="2026-07-20T11:58:30Z", usage=Decimal("0.3")),
                snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
                NOW,
                "at least 60 seconds",
            ),
            (
                snapshot(at=RETRY_AT, usage=Decimal("0.2")),
                snapshot(at=SETTLEMENT_AT, usage=Decimal("0.2")),
                snapshot(at=CURRENT_AT, usage=Decimal("0.3")),
                NOW,
                "not settled",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for (
                retry_snapshot, settlement_previous, current_snapshot,
                now, message,
            ) in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    RetryBudgetError, message
                ):
                    guard(
                        root, copy.deepcopy(base), retry_snapshot,
                        settlement_previous, current_snapshot, now=now,
                    )

    def test_rejects_identity_expiry_usage_and_account_changes(self):
        global_snapshot = snapshot(at=GLOBAL_AT, usage=Decimal("0.1"))
        retry = snapshot(at=RETRY_AT, usage=Decimal("0.2"))
        settled = snapshot(at=SETTLEMENT_AT, usage=Decimal("0.3"))
        mutations: list[tuple[str, dict[str, object], dict[str, object]]] = []
        for field, value, message in (
            ("key_fingerprint_sha256", "b" * 64, "API key changed"),
            ("limit", 2, "key limit changed"),
            ("expires_at_utc", "2026-08-20T00:00:00Z", "expiration changed"),
        ):
            current = snapshot(at=CURRENT_AT, usage=Decimal("0.3"))
            current_key = current["key"]
            assert isinstance(current_key, dict)
            current_key[field] = value
            if field == "limit":
                current_key["limit_remaining"] = Decimal("1.7")
            mutations.append((message, copy.deepcopy(retry), current))
        decreasing_global = snapshot(at=GLOBAL_AT, usage=Decimal("0.4"))
        mutations.append(("usage decreased", copy.deepcopy(retry), snapshot(
            at=CURRENT_AT, usage=Decimal("0.3")
        )))
        changed_account = snapshot(
            at=CURRENT_AT, usage=Decimal("0.3"), account_usage=Decimal("1.1"),
            account_credits=Decimal("5.1"), account_remaining=Decimal("4"),
        )
        mutations.append(("account counters are not settled", copy.deepcopy(retry), changed_account))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for message, retry_snapshot, current_snapshot in mutations:
                selected_global = (
                    decreasing_global if message == "usage decreased"
                    else global_snapshot
                )
                with self.subTest(message=message), self.assertRaisesRegex(
                    RetryBudgetError, message
                ):
                    guard(
                        root, selected_global, retry_snapshot,
                        copy.deepcopy(settled), current_snapshot,
                    )

    def test_marketplace_requires_constant_byok_and_preserves_binary64_dust(self):
        def marketplace(at: str, usage: object, byok: object = 0) -> dict[str, object]:
            return snapshot(
                at=at,
                usage=usage,
                limit=5,
                remaining=Decimal("5") - Decimal(str(usage)),
                byok=byok,
                account_usage=1.91640375,
                account_credits=5,
                account_remaining=3.0835962500000003,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = guard(
                root,
                marketplace(GLOBAL_AT, Decimal("1.9")),
                marketplace(RETRY_AT, Decimal("1.95")),
                marketplace(SETTLEMENT_AT, Decimal("2.0")),
                marketplace(CURRENT_AT, Decimal("2.0")),
                key_contract_mode=E12_MARKETPLACE_KEY_CONTRACT,
            )
            self.assertEqual(result["marketplace"], {
                "byok_usage_usd": "0",
                "byok_usage_unchanged": True,
            })

            changed = marketplace(CURRENT_AT, Decimal("2.0"), byok=0.1)
            with self.assertRaisesRegex(RetryBudgetError, "BYOK usage changed"):
                guard(
                    root,
                    marketplace(GLOBAL_AT, Decimal("1.9")),
                    marketplace(RETRY_AT, Decimal("1.95")),
                    marketplace(SETTLEMENT_AT, Decimal("2.0")),
                    changed,
                    key_contract_mode=E12_MARKETPLACE_KEY_CONTRACT,
                )

    def test_cli_publishes_mode_0600_once_and_sanitizes_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_path = write(
                root, f"{SECRET}-global.json",
                snapshot(at=GLOBAL_AT, usage=Decimal("0.1")),
            )
            retry_path = write(
                root, "retry.json", snapshot(at=RETRY_AT, usage=Decimal("0.2"))
            )
            settlement_path = write(
                root, "settlement.json",
                snapshot(at=SETTLEMENT_AT, usage=Decimal("0.3")),
            )
            current_path = write(
                root, "current.json", snapshot(at=CURRENT_AT, usage=Decimal("0.3"))
            )
            output = root / "nested" / "guard.json"
            argv = [
                "--global-baseline", str(global_path),
                "--retry-baseline", str(retry_path),
                "--settlement-previous", str(settlement_path),
                "--current", str(current_path),
                "--output", str(output),
            ]
            with patch(
                "tools.check_e12_retry_budget._current_time", return_value=NOW
            ):
                self.assertEqual(main(argv), 0)
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)
                original = output.read_bytes()
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(argv), 2)
            self.assertEqual(output.read_bytes(), original)
            self.assertIn("refusing overwrite", stderr.getvalue())
            self.assertNotIn(SECRET, stderr.getvalue())
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

            final_output = root / "nested" / "final.json"
            final_argv = [*argv[:-1], str(final_output), "--final"]
            with patch(
                "tools.check_e12_retry_budget._current_time", return_value=NOW
            ):
                self.assertEqual(main(final_argv), 0)
            final_payload = json.loads(final_output.read_text(encoding="utf-8"))
            self.assertEqual(final_payload["mode"], "retry_final")
            self.assertEqual(
                final_payload["budget"]["next_stage_full_upper_bound_usd"], "0"
            )

    def test_writer_refuses_existing_file_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "guard.json"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(RetryBudgetError, "refusing overwrite"):
                write_json_atomic_no_overwrite(output, {"status": "pass"})
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
