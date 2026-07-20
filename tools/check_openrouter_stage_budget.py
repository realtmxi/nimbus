#!/usr/bin/env python3
"""Gate an OpenRouter stage using two offline, secret-free usage snapshots.

The authorized key budget is fixed at three dollars.  The default mode is a
before-next-stage check; ``--final`` instead checks only the usage already
incurred.  This tool performs no network access and never emits input paths or
arbitrary snapshot content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tools.openrouter_key_contract import (
    E12_MARKETPLACE_KEY_CONTRACT,
    KEY_CONTRACT_MODES,
    STRICT_KEY_CONTRACT,
    include_byok_in_limit_required,
    key_limit_max_usd,
    validate_key_contract_mode,
)


AUTHORIZED_BUDGET_USD = Decimal("3")
DEFAULT_NEXT_STAGE_FULL_UPPER_BOUND_USD = Decimal("0.95401528")
MAX_SNAPSHOT_BYTES = 1_000_000
SNAPSHOT_MAX_AGE = timedelta(hours=6)
CURRENT_SNAPSHOT_MAX_AGE = timedelta(minutes=10)
SETTLEMENT_MIN_INTERVAL = timedelta(seconds=60)
ARITHMETIC_TOLERANCE = Decimal("0.0000001")
KEY_FIELDS = frozenset({
    "http_status",
    "usage",
    "limit",
    "limit_remaining",
    "key_fingerprint_sha256",
    "limit_reset",
    "include_byok_in_limit",
    "is_management_key",
    "is_provisioning_key",
    "is_free_tier",
    "expires_at_utc",
})
E12_MARKETPLACE_KEY_FIELDS = KEY_FIELDS | {"byok_usage"}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class StageGateError(ValueError):
    """A deliberately sanitized snapshot or budget-gate failure."""


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _decimal_argument(value: str | Decimal, field: str, *, positive: bool) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise StageGateError(f"{field} must be a finite decimal") from None
    if not result.is_finite():
        raise StageGateError(f"{field} must be a finite decimal")
    if result < 0 or (positive and result == 0):
        comparator = "> 0" if positive else ">= 0"
        raise StageGateError(f"{field} must be {comparator}")
    return result


def _number(value: Any, field: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise StageGateError(f"{field} must be numeric")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise StageGateError(f"{field} must be numeric") from None
    if not result.is_finite():
        raise StageGateError(f"{field} must be finite")
    if result < 0 or (positive and result == 0):
        comparator = "> 0" if positive else ">= 0"
        raise StageGateError(f"{field} must be {comparator}")
    return result


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StageGateError(f"{field} must be an object")
    return value


def _http_status(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StageGateError(f"{field} must be an integer")
    return value


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _load_snapshot(path: Path, which: str) -> tuple[dict[str, Any], str]:
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_SNAPSHOT_BYTES + 1)
    except OSError:
        raise StageGateError(f"unable to read {which} snapshot") from None
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise StageGateError(f"{which} snapshot is too large")
    try:
        parsed = json.loads(
            raw,
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise StageGateError(f"{which} snapshot is not valid UTF-8 JSON") from None
    if not isinstance(parsed, dict):
        raise StageGateError(f"{which} snapshot root must be an object")
    return parsed, hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise StageGateError(f"{field} must be a UTC ISO-8601 Z timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise StageGateError(f"{field} must be a UTC ISO-8601 Z timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise StageGateError(f"{field} must be a UTC ISO-8601 Z timestamp")
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _current_time(now: datetime | None) -> datetime:
    current = datetime.now(timezone.utc) if now is None else now
    if current.tzinfo is None or current.utcoffset() is None:
        raise StageGateError("current time must be timezone-aware")
    return current.astimezone(timezone.utc)


def _validate_snapshot_times(
    *,
    baseline: datetime,
    current: datetime,
    now: datetime,
    settlement_previous: datetime | None,
) -> None:
    if not baseline < current:
        raise StageGateError("snapshot timestamps must be strictly increasing")
    if current > now or baseline > now:
        raise StageGateError("snapshot timestamp is in the future")
    if now - baseline > SNAPSHOT_MAX_AGE:
        raise StageGateError("baseline snapshot is stale")
    if now - current > CURRENT_SNAPSHOT_MAX_AGE:
        raise StageGateError("current snapshot is stale")
    if settlement_previous is None:
        return
    if not baseline < settlement_previous < current:
        raise StageGateError(
            "settlement snapshots must be strictly ordered after baseline"
        )
    if settlement_previous > now:
        raise StageGateError("settlement snapshot timestamp is in the future")
    if now - settlement_previous > CURRENT_SNAPSHOT_MAX_AGE:
        raise StageGateError("settlement snapshot is stale")
    if current - settlement_previous < SETTLEMENT_MIN_INTERVAL:
        raise StageGateError("settlement snapshots must be at least 60 seconds apart")


def _key_values(
    snapshot: dict[str, Any], which: str, *, key_contract_mode: str
) -> dict[str, Any]:
    schema_version = snapshot.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise StageGateError(f"{which} snapshot schema_version must be 1")
    key = _mapping(snapshot.get("key"), f"{which} key")
    expected_fields = (
        E12_MARKETPLACE_KEY_FIELDS
        if key_contract_mode == E12_MARKETPLACE_KEY_CONTRACT
        else KEY_FIELDS
    )
    if set(key) != expected_fields:
        raise StageGateError(f"{which} key snapshot fields do not match E12 schema")
    if _http_status(key.get("http_status"), f"{which} key http_status") != 200:
        raise StageGateError(f"{which} key snapshot did not return HTTP 200")
    if key.get("limit_reset") is not None:
        raise StageGateError(f"{which} key limit_reset must be null")
    fingerprint = key.get("key_fingerprint_sha256")
    if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
        raise StageGateError(f"{which} key fingerprint is invalid")
    required_include_byok = include_byok_in_limit_required(key_contract_mode)
    if key.get("include_byok_in_limit") is not required_include_byok:
        raise StageGateError(f"{which} key does not match the selected contract mode")
    for field in ("is_management_key", "is_provisioning_key", "is_free_tier"):
        value = key.get(field)
        if not isinstance(value, bool):
            raise StageGateError(f"{which} key {field} must be boolean")
        if value:
            raise StageGateError(f"{which} key {field} must be false")

    usage = _number(key.get("usage"), f"{which} key usage")
    limit = _number(key.get("limit"), f"{which} key limit", positive=True)
    remaining = _number(
        key.get("limit_remaining"), f"{which} key limit_remaining"
    )
    limit_max = key_limit_max_usd(key_contract_mode)
    if key_contract_mode == E12_MARKETPLACE_KEY_CONTRACT:
        if limit != limit_max:
            raise StageGateError("key limit is not the exact E12 marketplace value")
    elif limit > limit_max:
        raise StageGateError("key limit exceeds the authorized budget")
    if remaining > limit:
        raise StageGateError(f"{which} key limit_remaining exceeds key limit")
    if abs((usage + remaining) - limit) > ARITHMETIC_TOLERANCE:
        raise StageGateError(f"{which} key usage/remaining arithmetic is inconsistent")
    expires = key.get("expires_at_utc")
    if expires is not None:
        _timestamp(expires, f"{which} key expires_at_utc")
    byok_usage = None
    if key_contract_mode == E12_MARKETPLACE_KEY_CONTRACT:
        byok_usage = _number(key.get("byok_usage"), f"{which} key byok_usage")
    return {
        "usage": usage,
        "limit": limit,
        "remaining": remaining,
        "fingerprint": fingerprint,
        "expires_at_utc": expires,
        "byok_usage": byok_usage,
    }


def _account_values(
    snapshot: dict[str, Any], which: str
) -> dict[str, int | Decimal | None]:
    account = _mapping(snapshot.get("account"), f"{which} account")
    status = _http_status(
        account.get("http_status"), f"{which} account http_status"
    )
    field_names = ("total_usage", "total_credits", "remaining_credits")

    if status == 403:
        if any(account.get(name) is not None for name in field_names):
            raise StageGateError(
                f"{which} unavailable account snapshot must contain null numbers"
            )
        return {
            "status": status,
            "usage": None,
            "credits": None,
            "remaining": None,
        }
    if status != 200:
        raise StageGateError(
            f"{which} account snapshot must return HTTP 200 or safe HTTP 403"
        )

    total_usage = _number(
        account.get("total_usage"), f"{which} account total_usage"
    )
    total_credits = _number(
        account.get("total_credits"), f"{which} account total_credits"
    )
    remaining = _number(
        account.get("remaining_credits"), f"{which} account remaining_credits"
    )
    # OpenRouter serializes these values as JSON binary64 numbers and derives
    # remaining_credits by subtraction. Decimal(str(float)) can retain tiny
    # representation dust, so use the same tight tolerance as key arithmetic.
    if abs((total_usage + remaining) - total_credits) > ARITHMETIC_TOLERANCE:
        raise StageGateError(f"{which} account credit arithmetic is inconsistent")
    return {
        "status": status,
        "usage": total_usage,
        "credits": total_credits,
        "remaining": remaining,
    }


def _account_result(
    remaining: Decimal | None,
    required: Decimal | None,
) -> dict[str, str | None]:
    if remaining is None:
        return {
            "availability": "unavailable_http_403",
            "remaining_credits_usd": None,
            "required_credits_usd": None,
            "headroom_usd": None,
        }
    return {
        "availability": "available",
        "remaining_credits_usd": _decimal_text(remaining),
        "required_credits_usd": (
            _decimal_text(required) if required is not None else None
        ),
        "headroom_usd": (
            _decimal_text(remaining - required) if required is not None else None
        ),
    }


def build_stage_gate_attestation(
    *,
    baseline_path: Path,
    current_path: Path,
    next_stage_full_upper_bound_usd: str | Decimal = (
        DEFAULT_NEXT_STAGE_FULL_UPPER_BOUND_USD
    ),
    final: bool = False,
    settlement_previous_path: Path | None = None,
    now: datetime | None = None,
    key_contract_mode: str = STRICT_KEY_CONTRACT,
) -> dict[str, Any]:
    """Validate two snapshots and return only fixed, text-free fields."""
    if not isinstance(final, bool):
        raise StageGateError("final must be boolean")
    try:
        validate_key_contract_mode(key_contract_mode)
    except ValueError:
        raise StageGateError("unknown OpenRouter key contract mode") from None
    next_upper = _decimal_argument(
        next_stage_full_upper_bound_usd,
        "next_stage_full_upper_bound_usd",
        positive=True,
    )
    if next_upper < DEFAULT_NEXT_STAGE_FULL_UPPER_BOUND_USD:
        raise StageGateError(
            "next-stage upper bound cannot be lower than the frozen E12 bound"
        )
    if settlement_previous_path is None:
        raise StageGateError("every stage gate requires a settled snapshot pair")

    baseline, baseline_sha256 = _load_snapshot(baseline_path, "baseline")
    current, current_sha256 = _load_snapshot(current_path, "current")
    baseline_at = _timestamp(
        baseline.get("captured_at_utc"), "baseline captured_at_utc"
    )
    current_at = _timestamp(
        current.get("captured_at_utc"), "current captured_at_utc"
    )
    settlement_previous: dict[str, Any] | None = None
    settlement_previous_sha256: str | None = None
    settlement_previous_at: datetime | None = None
    if settlement_previous_path is not None:
        settlement_previous, settlement_previous_sha256 = _load_snapshot(
            settlement_previous_path, "settlement-previous"
        )
        settlement_previous_at = _timestamp(
            settlement_previous.get("captured_at_utc"),
            "settlement-previous captured_at_utc",
        )
    current_time = _current_time(now)
    _validate_snapshot_times(
        baseline=baseline_at,
        current=current_at,
        now=current_time,
        settlement_previous=settlement_previous_at,
    )

    baseline_key = _key_values(
        baseline, "baseline", key_contract_mode=key_contract_mode
    )
    current_key = _key_values(
        current, "current", key_contract_mode=key_contract_mode
    )
    if baseline_key["limit"] != current_key["limit"]:
        raise StageGateError("key limit changed between snapshots")
    if baseline_key["fingerprint"] != current_key["fingerprint"]:
        raise StageGateError("API key changed between snapshots")
    if baseline_key["expires_at_utc"] != current_key["expires_at_utc"]:
        raise StageGateError("key expiration changed between snapshots")
    if current_key["usage"] < baseline_key["usage"]:
        raise StageGateError("key usage decreased between snapshots")
    if key_contract_mode == E12_MARKETPLACE_KEY_CONTRACT:
        if current_key["usage"] <= baseline_key["usage"]:
            raise StageGateError("marketplace key usage did not increase")
        if current_key["byok_usage"] != baseline_key["byok_usage"]:
            raise StageGateError("BYOK usage changed during marketplace experiment")

    settlement_interval_s: Decimal | None = None
    if settlement_previous is not None:
        previous_key = _key_values(
            settlement_previous,
            "settlement-previous",
            key_contract_mode=key_contract_mode,
        )
        if previous_key != current_key:
            raise StageGateError(
                "settlement key usage/limit/remaining did not remain equal"
            )
        assert settlement_previous_at is not None
        settlement_interval_s = Decimal(
            str((current_at - settlement_previous_at).total_seconds())
        )

    delta = current_key["usage"] - baseline_key["usage"]
    required_from_baseline = delta if final else delta + next_upper
    if required_from_baseline > AUTHORIZED_BUDGET_USD:
        if final:
            raise StageGateError("usage delta exceeds the authorized budget")
        raise StageGateError(
            "usage delta plus next-stage upper bound exceeds the authorized budget"
        )
    if baseline_key["remaining"] < required_from_baseline:
        raise StageGateError("baseline key limit_remaining is insufficient")
    if not final and current_key["remaining"] < next_upper:
        raise StageGateError("current key limit_remaining is insufficient")

    baseline_account = _account_values(baseline, "baseline")
    current_account = _account_values(current, "current")
    if settlement_previous is not None:
        previous_account = _account_values(
            settlement_previous, "settlement-previous"
        )
        if previous_account != current_account:
            raise StageGateError("settlement account usage/remaining did not remain equal")
    baseline_account_remaining = baseline_account["remaining"]
    current_account_remaining = current_account["remaining"]
    assert baseline_account_remaining is None or isinstance(
        baseline_account_remaining, Decimal
    )
    assert current_account_remaining is None or isinstance(
        current_account_remaining, Decimal
    )
    if (
        baseline_account_remaining is not None
        and baseline_account_remaining < required_from_baseline
    ):
        raise StageGateError("baseline account remaining credits are insufficient")
    current_account_required = None if final else next_upper
    if (
        current_account_remaining is not None
        and current_account_required is not None
        and current_account_remaining < current_account_required
    ):
        raise StageGateError("current account remaining credits are insufficient")

    # This is an explicit whitelist.  Timestamps are parsed and re-rendered in
    # canonical UTC form because stage ordering/settlement is safety evidence;
    # labels, paths, provider bodies, and arbitrary snapshot fields stay out.
    result = {
        "schema_version": 1,
        "status": "pass",
        "mode": "final" if final else "before_next",
        "baseline_snapshot_sha256": baseline_sha256,
        "current_snapshot_sha256": current_sha256,
        "key_fingerprint_sha256": baseline_key["fingerprint"],
        "captured_at_utc": {
            "baseline": _timestamp_text(baseline_at),
            "settlement_previous": (
                None
                if settlement_previous_at is None
                else _timestamp_text(settlement_previous_at)
            ),
            "current": _timestamp_text(current_at),
        },
        "settlement": {
            "verified": settlement_previous is not None,
            "minimum_interval_s": _decimal_text(
                Decimal(str(SETTLEMENT_MIN_INTERVAL.total_seconds()))
            ),
            "observed_interval_s": (
                None
                if settlement_interval_s is None
                else _decimal_text(settlement_interval_s)
            ),
            "previous_snapshot_sha256": settlement_previous_sha256,
            "key_usage_and_remaining_equal": (
                None if settlement_previous is None else True
            ),
            "account_usage_and_remaining_equal": (
                None if settlement_previous is None else True
            ),
        },
        "authorized_budget_usd": _decimal_text(AUTHORIZED_BUDGET_USD),
        "next_stage_full_upper_bound_usd": _decimal_text(next_upper),
        "baseline_key_limit_usd": _decimal_text(baseline_key["limit"]),
        "current_key_limit_usd": _decimal_text(current_key["limit"]),
        "baseline_key_usage_usd": _decimal_text(baseline_key["usage"]),
        "current_key_usage_usd": _decimal_text(current_key["usage"]),
        "usage_delta_usd": _decimal_text(delta),
        "required_from_baseline_usd": _decimal_text(required_from_baseline),
        "authorized_budget_headroom_usd": _decimal_text(
            AUTHORIZED_BUDGET_USD - required_from_baseline
        ),
        "baseline_key_limit_remaining_usd": _decimal_text(
            baseline_key["remaining"]
        ),
        "baseline_key_limit_remaining_headroom_usd": _decimal_text(
            baseline_key["remaining"] - required_from_baseline
        ),
        "current_key_limit_remaining_usd": _decimal_text(current_key["remaining"]),
        "current_key_limit_remaining_headroom_usd": (
            None
            if final
            else _decimal_text(current_key["remaining"] - next_upper)
        ),
        "account_credit_checks": {
            "baseline": _account_result(
                baseline_account_remaining, required_from_baseline
            ),
            "current": _account_result(
                current_account_remaining, current_account_required
            ),
        },
    }
    if key_contract_mode != STRICT_KEY_CONTRACT:
        result["key_contract_mode"] = key_contract_mode
        result["marketplace_route"] = {
            "origin": "openrouter",
            "provider": "deepinfra",
            "byok_allowed": False,
            "fallbacks_allowed": False,
        }
        result["baseline_byok_usage_usd"] = _decimal_text(
            baseline_key["byok_usage"]
        )
        result["current_byok_usage_usd"] = _decimal_text(
            current_key["byok_usage"]
        )
    return result


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
            json.dump(payload, target, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError:
        raise StageGateError("unable to publish stage-gate attestation") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument(
        "--next-stage-full-upper-bound-usd",
        default=_decimal_text(DEFAULT_NEXT_STAGE_FULL_UPPER_BOUND_USD),
    )
    parser.add_argument(
        "--settlement-previous",
        type=Path,
        help=(
            "the first of two equal usage snapshots at least 60 seconds apart; "
            "supply for pre-C and required for --final"
        ),
    )
    parser.add_argument("--final", action="store_true")
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
        attestation = build_stage_gate_attestation(
            baseline_path=args.baseline,
            current_path=args.current,
            next_stage_full_upper_bound_usd=(
                args.next_stage_full_upper_bound_usd
            ),
            final=args.final,
            settlement_previous_path=args.settlement_previous,
            key_contract_mode=args.key_contract_mode,
        )
        write_json_atomic(args.output, attestation)
    except StageGateError as exc:
        print(f"stage budget gate failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
