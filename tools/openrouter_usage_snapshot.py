#!/usr/bin/env python3
"""Capture a plaintext-secret-free OpenRouter key/account usage snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from tools.openrouter_key_contract import (
    E12_MARKETPLACE_KEY_CONTRACT,
    KEY_CONTRACT_MODES,
    STRICT_KEY_CONTRACT,
    include_byok_in_limit_required,
    key_limit_max_usd,
    validate_key_contract_mode,
)


DEFAULT_BASE_URL = "https://openrouter.ai"
MAX_RESPONSE_BYTES = 1_000_000
MIN_KEY_LIFETIME = timedelta(hours=6)


class SnapshotError(ValueError):
    """A deliberately sanitized snapshot failure."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward a bearer credential through an HTTP redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _open_no_redirect(request: urllib.request.Request, *, timeout: float):
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


def _numeric_or_none(value: Any, field: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotError(f"OpenRouter {field} is not numeric or null")
    if isinstance(value, float) and not math.isfinite(value):
        raise SnapshotError(f"OpenRouter {field} is not finite")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise SnapshotError(f"OpenRouter {field} is not boolean")
    return value


def _expiry(value: Any, *, captured: datetime) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SnapshotError("OpenRouter key expires_at is not a timestamp or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise SnapshotError("OpenRouter key expires_at is invalid") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SnapshotError("OpenRouter key expires_at is not timezone-aware")
    expires = parsed.astimezone(timezone.utc)
    if expires - captured < MIN_KEY_LIFETIME:
        raise SnapshotError("OpenRouter key expires too soon for the staged run")
    return expires.isoformat().replace("+00:00", "Z")


def _fetch_json(
    *,
    base_url: str,
    endpoint: str,
    api_key: str,
    timeout_s: float,
    opener: Callable[..., Any],
    allowed_error_statuses: frozenset[int] = frozenset(),
) -> tuple[dict[str, Any] | None, int]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{endpoint}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with opener(request, timeout=timeout_s) as response:
            status = int(getattr(response, "status", response.getcode()))
            if response.geturl() != request.full_url:
                # Check before reading any bytes.  A test/injected opener must
                # not be able to turn a redirect response into trusted data.
                raise SnapshotError("OpenRouter metadata request was redirected")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        # Never read or stringify the response: provider error bodies can echo
        # request metadata, labels, or credentials.
        status = int(exc.code)
        try:
            exc.close()
        except OSError:
            pass
        if status in allowed_error_statuses:
            return None, status
        raise SnapshotError(f"OpenRouter metadata request returned HTTP {status}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise SnapshotError("OpenRouter metadata request failed") from None
    if not 200 <= status < 300:
        raise SnapshotError(f"OpenRouter metadata request returned HTTP {status}")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SnapshotError("OpenRouter metadata response is too large")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SnapshotError("OpenRouter metadata response is not valid JSON") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise SnapshotError("OpenRouter metadata response has an invalid schema")
    return payload["data"], status


def capture_usage_snapshot(
    *,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 15.0,
    opener: Callable[..., Any] | None = None,
    now: Callable[[], datetime] | None = None,
    key_contract_mode: str = STRICT_KEY_CONTRACT,
) -> dict[str, Any]:
    """Fetch both endpoints and return only approved numeric metadata."""
    if not api_key:
        raise SnapshotError("API key is unset or empty")
    if base_url != DEFAULT_BASE_URL:
        # This process carries a bearer credential.  E12 metadata may only be
        # sent to the frozen HTTPS OpenRouter origin; do not follow a typo,
        # userinfo URL, plaintext origin, proxy, or test endpoint from the CLI.
        raise SnapshotError("OpenRouter metadata base URL is not the frozen HTTPS origin")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise SnapshotError("timeout_s must be finite and > 0")
    try:
        validate_key_contract_mode(key_contract_mode)
    except ValueError:
        raise SnapshotError("unknown OpenRouter key contract mode") from None

    captured = now() if now is not None else datetime.now(timezone.utc)
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise SnapshotError("snapshot time must be timezone-aware")
    captured = captured.astimezone(timezone.utc)
    opener = _open_no_redirect if opener is None else opener

    key_data, key_status = _fetch_json(
        base_url=base_url,
        endpoint="/api/v1/key",
        api_key=api_key,
        timeout_s=timeout_s,
        opener=opener,
    )
    credits_data, credits_status = _fetch_json(
        base_url=base_url,
        endpoint="/api/v1/credits",
        api_key=api_key,
        timeout_s=timeout_s,
        opener=opener,
        # OpenRouter currently documents /api/v1/credits as management-key
        # only.  A normal inference key can still provide the hard experiment
        # cap through /api/v1/key; retain a text-free 403 instead of making an
        # otherwise valid limited key unusable.
        allowed_error_statuses=frozenset({403}),
    )

    if key_data is None:  # not possible with the required endpoint contract
        raise SnapshotError("OpenRouter key metadata is unavailable")
    credits_data = credits_data or {}

    usage = _numeric_or_none(key_data.get("usage"), "key usage")
    limit = _numeric_or_none(key_data.get("limit"), "key limit")
    limit_remaining = _numeric_or_none(
        key_data.get("limit_remaining"), "key limit_remaining"
    )
    if "limit_reset" not in key_data:
        raise SnapshotError("OpenRouter key limit_reset is missing")
    limit_reset = key_data.get("limit_reset")
    if limit_reset is not None:
        # A reset can replenish the key during a staged run and therefore
        # defeat a total-experiment dollar cap.
        raise SnapshotError("OpenRouter key limit_reset must be null")
    is_management_key = _boolean(
        key_data.get("is_management_key"), "key is_management_key"
    )
    is_provisioning_key = _boolean(
        key_data.get("is_provisioning_key"), "key is_provisioning_key"
    )
    is_free_tier = _boolean(key_data.get("is_free_tier"), "key is_free_tier")
    include_byok_in_limit = _boolean(
        key_data.get("include_byok_in_limit"), "key include_byok_in_limit"
    )
    if is_management_key or is_provisioning_key:
        raise SnapshotError("OpenRouter key is not a dedicated inference key")
    if is_free_tier:
        raise SnapshotError("OpenRouter key is attached to a free-tier account")
    required_include_byok = include_byok_in_limit_required(key_contract_mode)
    if include_byok_in_limit is not required_include_byok:
        raise SnapshotError("OpenRouter key does not match the selected contract mode")
    if limit is None:
        raise SnapshotError("OpenRouter key limit must be numeric")
    numeric_limit = Decimal(str(limit))
    limit_max = key_limit_max_usd(key_contract_mode)
    if numeric_limit <= 0:
        raise SnapshotError("OpenRouter key limit must be positive")
    if key_contract_mode == E12_MARKETPLACE_KEY_CONTRACT:
        if numeric_limit != limit_max:
            raise SnapshotError("OpenRouter key limit is not the exact E12 value")
    elif numeric_limit > limit_max:
        raise SnapshotError("OpenRouter key limit exceeds the authorized budget")
    byok_usage: int | float | None = None
    if key_contract_mode == E12_MARKETPLACE_KEY_CONTRACT:
        if "byok_usage" not in key_data:
            raise SnapshotError("OpenRouter key byok_usage is missing")
        byok_usage = _numeric_or_none(
            key_data.get("byok_usage"), "key byok_usage"
        )
        if byok_usage is None or byok_usage < 0:
            raise SnapshotError("OpenRouter key byok_usage must be nonnegative")
    if "expires_at" not in key_data:
        raise SnapshotError("OpenRouter key expires_at is missing")
    expires_at_utc = _expiry(key_data.get("expires_at"), captured=captured)
    total_usage = _numeric_or_none(
        credits_data.get("total_usage"), "account total_usage"
    )
    total_credits = _numeric_or_none(
        credits_data.get("total_credits"), "account total_credits"
    )

    remaining_credits: int | float | None = None
    if total_usage is not None and total_credits is not None:
        remaining_credits = total_credits - total_usage

    captured_utc = captured.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    # Do not include the API key, its env-var name, provider key label, URLs,
    # raw responses, or lengths.  The only derived identifier is a SHA-256 of
    # the high-entropy key, used solely to prove that all stages used the same
    # credential without persisting the credential itself.
    key_result = {
        "http_status": key_status,
        "usage": usage,
        "limit": limit,
        "limit_remaining": limit_remaining,
        "key_fingerprint_sha256": hashlib.sha256(api_key.encode()).hexdigest(),
        "limit_reset": None,
        "include_byok_in_limit": include_byok_in_limit,
        "is_management_key": False,
        "is_provisioning_key": False,
        "is_free_tier": False,
        "expires_at_utc": expires_at_utc,
    }
    if key_contract_mode == E12_MARKETPLACE_KEY_CONTRACT:
        key_result["byok_usage"] = byok_usage

    return {
        "schema_version": 1,
        "captured_at_utc": captured_utc,
        "key": key_result,
        "account": {
            "http_status": credits_status,
            "total_usage": total_usage,
            "total_credits": total_credits,
            "remaining_credits": remaining_credits,
        },
    }


def snapshot_from_environment(
    *,
    api_key_env: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 15.0,
    environ: dict[str, str] | os._Environ[str] | None = None,
    opener: Callable[..., Any] | None = None,
    now: Callable[[], datetime] | None = None,
    key_contract_mode: str = STRICT_KEY_CONTRACT,
) -> dict[str, Any]:
    if not api_key_env:
        raise SnapshotError("API key environment variable name is empty")
    source = os.environ if environ is None else environ
    api_key = source.get(api_key_env)
    if not api_key:
        # Deliberately omit the environment-variable name.
        raise SnapshotError("API key environment variable is unset or empty")
    return capture_usage_snapshot(
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
        opener=opener,
        now=now,
        key_contract_mode=key_contract_mode,
    )


def write_json_atomic(
    path: Path, payload: dict[str, Any], *, overwrite: bool = True
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
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
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise SnapshotError("usage snapshot output already exists") from None
            temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument(
        "--key-contract-mode",
        choices=sorted(KEY_CONTRACT_MODES),
        default=STRICT_KEY_CONTRACT,
        help=(
            "strict_server_cap_v1 is the default; the E12 marketplace mode is "
            "an explicit experiment-only exception for fixed DeepInfra/no-BYOK"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        snapshot = snapshot_from_environment(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            timeout_s=args.timeout_s,
            key_contract_mode=args.key_contract_mode,
        )
        write_json_atomic(args.output, snapshot, overwrite=not args.no_overwrite)
    except SnapshotError as exc:
        print(f"usage snapshot failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
