#!/usr/bin/env python3
"""Verify and snapshot the frozen OpenRouter/DeepInfra Qwen3-32B price."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable


ENDPOINTS_URL = (
    "https://openrouter.ai/api/v1/models/qwen/qwen3-32b/endpoints"
)
MODEL = "qwen/qwen3-32b"
PROVIDER = "deepinfra"
EXPECTED_PROMPT_PER_TOKEN_USD = Decimal("0.00000008")
EXPECTED_COMPLETION_PER_TOKEN_USD = Decimal("0.00000028")
EXPECTED_REQUEST_PER_REQUEST_USD = Decimal("0")
REQUEST_PRICE_SOURCE_ABSENT = "absent_not_advertised"
REQUEST_PRICE_SOURCE_EXPLICIT = "explicit_zero"
# These dimensions can affect an otherwise text-only request even when Nimbus
# does not opt into tools or multimodal inputs.  Absence means the endpoint does
# not advertise that charge; if advertised, E12 requires an exact zero so the
# token-only full-response bound remains conservative.
OPTIONAL_ZERO_TEXT_PRICING_FIELDS = (
    "input_cache_read",
    "input_cache_write",
    "internal_reasoning",
)
MILLION = Decimal(1_000_000)
MAX_RESPONSE_BYTES = 1_000_000


class PriceSnapshotError(ValueError):
    """A sanitized public-metadata or price-contract failure."""


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise PriceSnapshotError(f"{field} is not a decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise PriceSnapshotError(f"{field} is not a decimal") from None
    if not result.is_finite() or result < 0:
        raise PriceSnapshotError(f"{field} is not a finite nonnegative decimal")
    return result


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _fetch_public_json(
    *,
    timeout_s: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[dict[str, Any], int]:
    request = urllib.request.Request(
        ENDPOINTS_URL,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with opener(request, timeout=timeout_s) as response:
            status = int(getattr(response, "status", response.getcode()))
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            exc.close()
        except OSError:
            pass
        raise PriceSnapshotError(
            f"OpenRouter public metadata returned HTTP {status}"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise PriceSnapshotError("OpenRouter public metadata request failed") from None
    if status != 200:
        raise PriceSnapshotError(
            f"OpenRouter public metadata returned HTTP {status}"
        )
    if len(raw) > MAX_RESPONSE_BYTES:
        raise PriceSnapshotError("OpenRouter public metadata response is too large")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise PriceSnapshotError(
            "OpenRouter public metadata response is not valid JSON"
        ) from None
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise PriceSnapshotError("OpenRouter public metadata schema is invalid")
    return payload["data"], status


def build_price_snapshot(
    data: dict[str, Any],
    *,
    http_status: int = 200,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    if data.get("id") != MODEL:
        raise PriceSnapshotError("OpenRouter metadata model does not match")
    endpoints = data.get("endpoints")
    if not isinstance(endpoints, list):
        raise PriceSnapshotError("OpenRouter endpoints metadata is invalid")
    matches = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        provider_name = endpoint.get("provider_name")
        if (
            isinstance(provider_name, str)
            and provider_name.casefold() == PROVIDER
        ):
            matches.append(endpoint)
    if len(matches) != 1:
        raise PriceSnapshotError("expected exactly one DeepInfra endpoint")

    endpoint = matches[0]
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, dict):
        raise PriceSnapshotError("DeepInfra pricing metadata is invalid")
    prompt = _decimal(pricing.get("prompt"), "DeepInfra prompt price")
    completion = _decimal(
        pricing.get("completion"), "DeepInfra completion price"
    )
    if "request" in pricing:
        request_price = _decimal(
            pricing.get("request"), "DeepInfra per-request price"
        )
        request_price_source = REQUEST_PRICE_SOURCE_EXPLICIT
    else:
        # OpenRouter documents per-request pricing as provider-dependent.  Its
        # live Qwen3-32B/DeepInfra endpoint currently omits this dimension,
        # meaning no such fee is advertised.  Preserve that distinction in the
        # snapshot instead of pretending the API returned an explicit zero.
        request_price = EXPECTED_REQUEST_PER_REQUEST_USD
        request_price_source = REQUEST_PRICE_SOURCE_ABSENT
    if prompt != EXPECTED_PROMPT_PER_TOKEN_USD:
        raise PriceSnapshotError("DeepInfra prompt price changed from frozen value")
    if completion != EXPECTED_COMPLETION_PER_TOKEN_USD:
        raise PriceSnapshotError(
            "DeepInfra completion price changed from frozen value"
        )
    if request_price != EXPECTED_REQUEST_PER_REQUEST_USD:
        raise PriceSnapshotError(
            "DeepInfra per-request price changed from frozen zero value"
        )
    for field in OPTIONAL_ZERO_TEXT_PRICING_FIELDS:
        if field not in pricing:
            continue
        value = _decimal(pricing[field], f"DeepInfra {field} price")
        if value != 0:
            raise PriceSnapshotError(
                f"DeepInfra {field} price must be zero for E12"
            )
    context = endpoint.get("context_length")
    if isinstance(context, bool) or not isinstance(context, int) or context < 40_960:
        raise PriceSnapshotError("DeepInfra context length is below 40960")

    captured = now() if now is not None else datetime.now(timezone.utc)
    if captured.tzinfo is None:
        raise PriceSnapshotError("snapshot time must be timezone-aware")
    captured_at = captured.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    # Whitelist only the public fields required by the E12 budget contract.
    return {
        "schema_version": 1,
        "status": "pass",
        "captured_at_utc": captured_at,
        "http_status": http_status,
        "model": MODEL,
        "provider": PROVIDER,
        "matching_endpoint_count": 1,
        "context_length": context,
        "prompt_price_per_token_usd": _decimal_text(prompt),
        "completion_price_per_token_usd": _decimal_text(completion),
        "request_price_per_request_usd": _decimal_text(request_price),
        "request_price_source": request_price_source,
        "prompt_price_per_million_usd": _decimal_text(prompt * MILLION),
        "completion_price_per_million_usd": _decimal_text(completion * MILLION),
    }


def capture_price_snapshot(
    *,
    timeout_s: float = 20.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise PriceSnapshotError("timeout must be finite and positive")
    data, status = _fetch_public_json(timeout_s=timeout_s, opener=opener)
    return build_price_snapshot(data, http_status=status, now=now)


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
        raise PriceSnapshotError("unable to publish price snapshot") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        snapshot = capture_price_snapshot(timeout_s=args.timeout_s)
        write_json_atomic(args.output, snapshot)
    except PriceSnapshotError as exc:
        print(f"price snapshot failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
