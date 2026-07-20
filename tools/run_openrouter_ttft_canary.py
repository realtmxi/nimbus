#!/usr/bin/env python3
"""Run one synthetic, fixed-provider OpenRouter TTFT-cancel canary.

The canary never reads the experiment trace.  Its prompt is a fixed public
string, it performs exactly one request with no retry, and its output is a
whitelist of text-free transport facts.  It is intended to catch auth/model/
provider configuration errors before any ShareGPT current-turn text is sent.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import common
from router.common import Endpoint, one_request
from tools.openrouter_usage_snapshot import write_json_atomic


CANARY_PROMPT = "Nimbus E12 synthetic TTFT canary; no user or trace data."
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "qwen/qwen3-32b"
OPENROUTER_PROVIDER = "deepinfra"
ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class CanaryError(ValueError):
    """A prompt-free canary configuration or validation failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def assess_result(
    result: dict[str, Any], *, key_fingerprint_sha256: str
) -> dict[str, Any]:
    """Validate a raw request result and return only approved fields."""
    if not isinstance(key_fingerprint_sha256, str) or not SHA256_RE.fullmatch(
        key_fingerprint_sha256
    ):
        raise CanaryError("canary key fingerprint is invalid")
    ttft = result.get("ttft_ms")
    if isinstance(ttft, bool) or not isinstance(ttft, (int, float)):
        raise CanaryError("canary did not return a numeric TTFT")
    if not math.isfinite(float(ttft)) or float(ttft) < 0:
        raise CanaryError("canary TTFT is invalid")
    if result.get("success") is not True:
        raise CanaryError("canary request failed")
    if result.get("http_status") != 200:
        raise CanaryError("canary did not return HTTP 200")
    if result.get("probe_mode") != "ttft_cancel":
        raise CanaryError("canary probe mode is invalid")
    if result.get("stream_abort_requested") is not True:
        raise CanaryError("canary did not abort after the first token")
    if result.get("response_completed") is not False:
        raise CanaryError("canary unexpectedly completed the response")
    if result.get("first_token_kind") not in {"content", "reasoning"}:
        raise CanaryError("canary first-token kind is invalid")
    requested = result.get("requested_provider_order")
    if requested != [OPENROUTER_PROVIDER]:
        raise CanaryError("canary provider request was not pinned")
    provider = result.get("provider")
    if provider != OPENROUTER_PROVIDER:
        raise CanaryError("canary was not served by the pinned provider")
    if result.get("response_model") != OPENROUTER_MODEL:
        raise CanaryError("canary response model does not match the frozen model")
    if result.get("cost_pending") is not True or result.get("cost_usd") is not None:
        raise CanaryError("canary cancel cost was not kept pending")

    # ``one_request`` hashes the provider-controlled header/SSE id at the
    # transport boundary.  Validate and forward that digest; never accept a
    # raw identifier here and never double-hash the digest.
    generation_id_sha256 = result.get("generation_id_sha256")
    if generation_id_sha256 is not None and (
        not isinstance(generation_id_sha256, str)
        or not SHA256_RE.fullmatch(generation_id_sha256)
    ):
        raise CanaryError("canary generation id hash is invalid")

    return {
        "schema_version": 1,
        "status": "pass",
        "captured_at_utc": _utc_now(),
        "request_count": 1,
        "retry_count": 0,
        "trace_data_used": False,
        "http_status": 200,
        "ttft_ms": float(ttft),
        "provider": OPENROUTER_PROVIDER,
        "response_model": OPENROUTER_MODEL,
        "first_token_kind": result["first_token_kind"],
        "stream_abort_requested": True,
        "response_completed": False,
        "cost_pending": True,
        "key_fingerprint_sha256": key_fingerprint_sha256,
        "generation_id_sha256": generation_id_sha256,
    }


async def run_canary(*, api_key_env: str, timeout_s: float) -> dict[str, Any]:
    if not ENV_NAME_RE.fullmatch(api_key_env):
        raise CanaryError("API key environment-variable name is invalid")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise CanaryError("API key environment variable is unset or empty")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise CanaryError("timeout must be finite and positive")
    if common.aiohttp is None:
        raise CanaryError("aiohttp is unavailable")

    endpoint = Endpoint(
        name="cloud",
        url=OPENROUTER_URL,
        model=OPENROUTER_MODEL,
        api_key_env=api_key_env,
        input_price_per_mtok=0.08,
        output_price_per_mtok=0.28,
    )
    request = {
        "request_id": 0,
        "arrived_at": 0,
        "relative_arrival_s": 0.0,
        "prompt": CANARY_PROMPT,
        "max_tokens": 1,
    }
    async with common.aiohttp.ClientSession(
        connector=common.aiohttp.TCPConnector(limit=1)
    ) as session:
        raw = await one_request(
            session,
            endpoint,
            request,
            time.perf_counter(),
            timeout_s=timeout_s,
            temperature=0.0,
            ignore_eos=False,
            provider_order=[OPENROUTER_PROVIDER],
            allow_fallbacks=False,
            stop_after_first_token=True,
        )
    return assess_result(
        raw,
        key_fingerprint_sha256=hashlib.sha256(api_key.encode()).hexdigest(),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        attestation = asyncio.run(
            run_canary(api_key_env=args.api_key_env, timeout_s=args.timeout_s)
        )
        write_json_atomic(args.output, attestation)
    except CanaryError as exc:
        print(f"canary failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
