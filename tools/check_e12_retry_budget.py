#!/usr/bin/env python3
"""Fail closed on cumulative E12 spend before retrying a live stage.

The original experiment baseline may be older than the operational six-hour
window.  Two fresh, settled snapshots still prove the current counter, while
the budget calculation remains anchored to that original baseline.  This tool
is offline, emits only fixed text-free fields, and never overwrites evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from tools.check_openrouter_stage_budget import (
    AUTHORIZED_BUDGET_USD,
    CURRENT_SNAPSHOT_MAX_AGE,
    DEFAULT_NEXT_STAGE_FULL_UPPER_BOUND_USD,
    SNAPSHOT_MAX_AGE,
    SETTLEMENT_MIN_INTERVAL,
    StageGateError,
    _account_values,
    _current_time,
    _decimal_argument,
    _decimal_text,
    _key_values,
    _load_snapshot,
    _timestamp,
    _timestamp_text,
)
from tools.openrouter_key_contract import (
    E12_MARKETPLACE_KEY_CONTRACT,
    KEY_CONTRACT_MODES,
    STRICT_KEY_CONTRACT,
    validate_key_contract_mode,
)


class RetryBudgetError(ValueError):
    """A deliberately sanitized retry-budget failure."""


def _stage_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Reuse the stage gate's strict snapshot parser with our error type."""
    try:
        return function(*args, **kwargs)
    except StageGateError as exc:
        raise RetryBudgetError(str(exc)) from None


def _validate_times(
    *, global_at: datetime, retry_at: datetime,
    settlement_previous_at: datetime, current_at: datetime, now: datetime,
) -> Decimal:
    if not global_at <= retry_at < settlement_previous_at < current_at:
        raise RetryBudgetError(
            "global, retry-baseline, settlement, and current timestamps are not ordered"
        )
    if current_at > now:
        raise RetryBudgetError("current snapshot timestamp is in the future")
    if now - retry_at > SNAPSHOT_MAX_AGE:
        raise RetryBudgetError("retry-baseline snapshot is stale")
    if now - current_at > CURRENT_SNAPSHOT_MAX_AGE:
        raise RetryBudgetError("current snapshot is stale")
    if now - settlement_previous_at > CURRENT_SNAPSHOT_MAX_AGE:
        raise RetryBudgetError("settlement-previous snapshot is stale")
    interval = current_at - settlement_previous_at
    if interval < SETTLEMENT_MIN_INTERVAL:
        raise RetryBudgetError(
            "retry settlement snapshots must be at least 60 seconds apart"
        )
    return Decimal(str(interval.total_seconds()))


def build_retry_budget_attestation(
    *,
    global_baseline_path: Path,
    retry_baseline_path: Path,
    settlement_previous_path: Path,
    current_path: Path,
    next_stage_full_upper_bound_usd: str | Decimal | None = None,
    final: bool = False,
    now: datetime | None = None,
    key_contract_mode: str = STRICT_KEY_CONTRACT,
) -> dict[str, Any]:
    """Validate cumulative and fresh-settlement evidence for an E12 retry."""
    if not isinstance(final, bool):
        raise RetryBudgetError("final must be boolean")
    try:
        validate_key_contract_mode(key_contract_mode)
    except ValueError:
        raise RetryBudgetError("unknown OpenRouter key contract mode") from None

    if final:
        if next_stage_full_upper_bound_usd is not None:
            raise RetryBudgetError(
                "final mode must not include a next-stage upper bound"
            )
        next_upper = Decimal("0")
    else:
        supplied_upper = (
            DEFAULT_NEXT_STAGE_FULL_UPPER_BOUND_USD
            if next_stage_full_upper_bound_usd is None
            else next_stage_full_upper_bound_usd
        )
        next_upper = _stage_call(
            _decimal_argument,
            supplied_upper,
            "next_stage_full_upper_bound_usd",
            positive=True,
        )
        if next_upper < DEFAULT_NEXT_STAGE_FULL_UPPER_BOUND_USD:
            raise RetryBudgetError(
                "next-stage upper bound cannot be lower than the frozen E12 bound"
            )

    global_snapshot, global_sha = _stage_call(
        _load_snapshot, global_baseline_path, "global-baseline"
    )
    retry_snapshot, retry_sha = _stage_call(
        _load_snapshot, retry_baseline_path, "retry-baseline"
    )
    settlement_previous, settlement_previous_sha = _stage_call(
        _load_snapshot, settlement_previous_path, "settlement-previous"
    )
    current_snapshot, current_sha = _stage_call(
        _load_snapshot, current_path, "current"
    )
    global_at = _stage_call(
        _timestamp,
        global_snapshot.get("captured_at_utc"),
        "global-baseline captured_at_utc",
    )
    retry_at = _stage_call(
        _timestamp,
        retry_snapshot.get("captured_at_utc"),
        "retry-baseline captured_at_utc",
    )
    settlement_previous_at = _stage_call(
        _timestamp,
        settlement_previous.get("captured_at_utc"),
        "settlement-previous captured_at_utc",
    )
    current_at = _stage_call(
        _timestamp,
        current_snapshot.get("captured_at_utc"),
        "current captured_at_utc",
    )
    current_time = _stage_call(_current_time, now)
    settlement_interval = _validate_times(
        global_at=global_at,
        retry_at=retry_at,
        settlement_previous_at=settlement_previous_at,
        current_at=current_at,
        now=current_time,
    )

    global_key = _stage_call(
        _key_values,
        global_snapshot,
        "global-baseline",
        key_contract_mode=key_contract_mode,
    )
    retry_key = _stage_call(
        _key_values,
        retry_snapshot,
        "retry-baseline",
        key_contract_mode=key_contract_mode,
    )
    settlement_previous_key = _stage_call(
        _key_values,
        settlement_previous,
        "settlement-previous",
        key_contract_mode=key_contract_mode,
    )
    current_key = _stage_call(
        _key_values,
        current_snapshot,
        "current",
        key_contract_mode=key_contract_mode,
    )

    for field, label in (
        ("fingerprint", "API key"),
        ("limit", "key limit"),
        ("expires_at_utc", "key expiration"),
    ):
        if not (
            global_key[field]
            == retry_key[field]
            == settlement_previous_key[field]
            == current_key[field]
        ):
            raise RetryBudgetError(f"{label} changed between retry snapshots")
    if not (
        global_key["usage"]
        <= retry_key["usage"]
        <= settlement_previous_key["usage"]
        <= current_key["usage"]
    ):
        raise RetryBudgetError("key usage decreased between retry snapshots")
    if key_contract_mode == E12_MARKETPLACE_KEY_CONTRACT and not (
        global_key["byok_usage"]
        == retry_key["byok_usage"]
        == settlement_previous_key["byok_usage"]
        == current_key["byok_usage"]
    ):
        raise RetryBudgetError("BYOK usage changed during marketplace experiment")
    if settlement_previous_key != current_key:
        raise RetryBudgetError(
            "settlement-previous and current key counters are not settled"
        )

    # Validate account shapes and require the fresh pair to be settled, but do
    # not copy any account value or provider body into the attestation.
    _stage_call(_account_values, global_snapshot, "global-baseline")
    _stage_call(_account_values, retry_snapshot, "retry-baseline")
    settlement_previous_account = _stage_call(
        _account_values, settlement_previous, "settlement-previous"
    )
    current_account = _stage_call(_account_values, current_snapshot, "current")
    if settlement_previous_account != current_account:
        raise RetryBudgetError(
            "settlement-previous and current account counters are not settled"
        )

    global_delta = current_key["usage"] - global_key["usage"]
    retry_delta = current_key["usage"] - retry_key["usage"]
    projected = global_delta + next_upper
    if projected > AUTHORIZED_BUDGET_USD:
        raise RetryBudgetError(
            "global usage delta plus next-stage upper bound exceeds the authorized budget"
        )
    if current_key["remaining"] < next_upper:
        raise RetryBudgetError("current key limit_remaining is insufficient")
    current_account_remaining = current_account["remaining"]
    if (
        current_account_remaining is not None
        and current_account_remaining < next_upper
    ):
        raise RetryBudgetError("current account remaining credits are insufficient")

    # Explicit whitelist: no source paths, account fields, response bodies,
    # provider labels, arbitrary input text, or credentials are copied here.
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "mode": "retry_final" if final else "retry_before_next",
        "key_contract_mode": key_contract_mode,
        "snapshot_sha256": {
            "global_baseline": global_sha,
            "retry_baseline": retry_sha,
            "settlement_previous": settlement_previous_sha,
            "current": current_sha,
        },
        "captured_at_utc": {
            "global_baseline": _timestamp_text(global_at),
            "retry_baseline": _timestamp_text(retry_at),
            "settlement_previous": _timestamp_text(settlement_previous_at),
            "current": _timestamp_text(current_at),
        },
        "freshness": {
            "global_baseline_max_age_enforced": False,
            "retry_baseline_max_age_s": _decimal_text(
                Decimal(str(SNAPSHOT_MAX_AGE.total_seconds()))
            ),
            "current_max_age_s": _decimal_text(
                Decimal(str(CURRENT_SNAPSHOT_MAX_AGE.total_seconds()))
            ),
            "minimum_settlement_interval_s": _decimal_text(
                Decimal(str(SETTLEMENT_MIN_INTERVAL.total_seconds()))
            ),
            "observed_settlement_interval_s": _decimal_text(
                settlement_interval
            ),
        },
        "key": {
            "fingerprint_sha256": global_key["fingerprint"],
            "limit_usd": _decimal_text(global_key["limit"]),
            "expires_at_utc": global_key["expires_at_utc"],
            "global_baseline_usage_usd": _decimal_text(global_key["usage"]),
            "retry_baseline_usage_usd": _decimal_text(retry_key["usage"]),
            "settlement_previous_usage_usd": _decimal_text(
                settlement_previous_key["usage"]
            ),
            "current_usage_usd": _decimal_text(current_key["usage"]),
            "current_limit_remaining_usd": _decimal_text(
                current_key["remaining"]
            ),
            "fresh_pair_settled": True,
        },
        "budget": {
            "authorized_budget_usd": _decimal_text(AUTHORIZED_BUDGET_USD),
            "next_stage_full_upper_bound_usd": _decimal_text(next_upper),
            "global_usage_delta_usd": _decimal_text(global_delta),
            "retry_usage_delta_usd": _decimal_text(retry_delta),
            "projected_global_spend_usd": _decimal_text(projected),
            "projected_headroom_usd": _decimal_text(
                AUTHORIZED_BUDGET_USD - projected
            ),
            "current_limit_remaining_after_future_usd": _decimal_text(
                current_key["remaining"] - next_upper
            ),
        },
    }
    if key_contract_mode == E12_MARKETPLACE_KEY_CONTRACT:
        result["marketplace"] = {
            "byok_usage_usd": _decimal_text(global_key["byok_usage"]),
            "byok_usage_unchanged": True,
        }
    return result


def write_json_atomic_no_overwrite(
    path: Path, payload: dict[str, Any]
) -> None:
    """Atomically publish mode-0600 JSON without replacing an existing path."""
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            os.fchmod(target.fileno(), 0o600)
            json.dump(payload, target, sort_keys=True, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise RetryBudgetError(
                "retry-budget attestation already exists; refusing overwrite"
            ) from None
    except RetryBudgetError:
        raise
    except (OSError, TypeError, ValueError):
        raise RetryBudgetError(
            "unable to publish retry-budget attestation"
        ) from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--global-baseline", type=Path, required=True)
    parser.add_argument("--retry-baseline", type=Path, required=True)
    parser.add_argument("--settlement-previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--next-stage-full-upper-bound-usd",
    )
    mode.add_argument("--final", action="store_true")
    parser.add_argument(
        "--key-contract-mode",
        choices=sorted(KEY_CONTRACT_MODES),
        default=STRICT_KEY_CONTRACT,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_retry_budget_attestation(
            global_baseline_path=args.global_baseline,
            retry_baseline_path=args.retry_baseline,
            settlement_previous_path=args.settlement_previous,
            current_path=args.current,
            next_stage_full_upper_bound_usd=(
                args.next_stage_full_upper_bound_usd
            ),
            final=args.final,
            key_contract_mode=args.key_contract_mode,
        )
        write_json_atomic_no_overwrite(args.output, result)
    except RetryBudgetError as exc:
        print(f"E12 retry budget guard failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
