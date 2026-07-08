"""Shared building blocks for the hybrid router (see router/run.py for the entry point).

Contents: SCENARIOS / load_trace (line-for-line from the verified vllm/run.py),
Endpoint, Policy (all_local / all_cloud / random), one_request (the verified
streaming request primitive), NullCloud (fake cloud sink), billing, summarize.

This module is a LIBRARY — it has no main. The single runnable entry point is
router/run.py (external queue + work-conserving dispatcher). The open-loop
baseline lives outside this package: vllm/run.py (Jialu's verified load
generator) is the reference to compare against; an in-schema open-loop runner
existed here through PR #3 review, established the parity anchor
(paired TTFT p50 ratio 1.003 vs vllm/run.py), and was then removed.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # aiohttp is only needed to actually send requests; unit tests stub the session
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

# BurstGPT arrived_at windows; byte-identical to vllm/run.py SCENARIOS.
SCENARIOS = {
    "normal": (1477007, 1478206),               # ~0.18 req/s  (low)
    "burst": (1698942, 1702541),
    "burst_1200": (1700032, 1701231),           # ~2.3 req/s   (mid)
    "burst_300": (1700593, 1700892),
    "extreme_burst_1200": (1260532, 1261731),   # ~9.7 req/s   (high)
    "extreme_burst": (1255679, 1266478),
}

DEFAULT_TIMEOUT_S: float = float(os.environ.get("TIMEOUT_S", "600"))


@dataclass(frozen=True)
class Endpoint:
    """One OpenAI-compatible chat-completions endpoint."""

    name: str                      # "local" | "cloud"
    url: str
    model: str
    api_key_env: str | None = None
    # $ per 1M tokens; local endpoint keeps 0.0 (its cost is the GPU, not billed here)
    input_price_per_mtok: float = 0.0
    output_price_per_mtok: float = 0.0

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            h["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        return h


class Policy:
    """Arrival-time routing decision. Returns True to outsource to cloud.

    all_local / all_cloud / random are all the same rule with p = 0 / 1 / fraction:
    rng.random() is uniform on [0, 1), so p=0 never outsources and p=1 always does.
    """

    def __init__(self, name: str, fraction: float, seed: int):
        if name == "all_local":
            p = 0.0
        elif name == "all_cloud":
            p = 1.0
        elif name == "random":
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"--fraction must be in [0,1], got {fraction}")
            p = fraction
        else:
            raise ValueError(f"unknown policy {name!r}")
        self.name = name
        self.p = p
        self.rng = random.Random(seed)
        self.n_total = 0
        self.n_outsourced = 0

    def outsource(self, req: dict[str, Any]) -> bool:
        self.n_total += 1
        out = self.rng.random() < self.p
        if out:
            self.n_outsourced += 1
        return out

    @property
    def actual_fraction(self) -> float:
        return self.n_outsourced / max(self.n_total, 1)


def load_trace(path: Path, scenario: str) -> list[dict[str, Any]]:
    """Load the BurstGPT JSONL slice for a scenario; identical to vllm/run.py."""
    start, end = SCENARIOS[scenario]
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            arrived_at = int(obj["arrived_at"])
            prompt = obj.get("prompt_text", "")

            if start <= arrived_at <= end and prompt:
                rows.append({
                    "request_id": len(rows),
                    "arrived_at": arrived_at,
                    "relative_arrival_s": arrived_at - start,
                    "prompt": prompt,
                    "max_tokens": int(obj["num_decode_tokens"]),
                    # additive vs vllm/run.py: exact token counts from the trace,
                    # used by cloud-sink billing and the queued dispatcher's KV math
                    "prompt_tokens": int(obj.get("num_prefill_tokens") or 0),
                    "session_id": obj.get("session_id"),
                })

    rows.sort(key=lambda x: (x["arrived_at"], x["request_id"]))

    for i, row in enumerate(rows):
        row["request_id"] = i

    return rows


def make_payload(endpoint: Endpoint, req: dict[str, Any], max_tokens_override: int | None) -> dict[str, Any]:
    return {
        "model": endpoint.model,
        "stream": True,
        "max_tokens": max_tokens_override if max_tokens_override is not None else req["max_tokens"],
        "messages": [{"role": "user", "content": req["prompt"]}],
        "stream_options": {"include_usage": True},
    }


class NullCloud:
    """Fake cloud sink: the request is ROUTED and recorded, nothing is modeled.

    The most basic architecture-proof mode (default): no network, no latency
    model, no SLO claims — just "this request went to the cloud", plus the
    deterministic token counts from the trace. Switch to --cloud real when the
    experiment needs actual cloud latency numbers.
    """

    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint

    def serve(self, req: dict[str, Any], due_time: float, *,
              max_tokens_override: int | None = None) -> dict[str, Any]:
        # same semantics as make_payload: --max-tokens REPLACES the trace value.
        # completion_tokens therefore reflects the payload cap (an upper bound
        # on what a real call would bill; the fake sink has no model to EOS early).
        effective = (int(max_tokens_override) if max_tokens_override is not None
                     else int(req["max_tokens"]))
        result = {
            "request_id": req["request_id"],
            "arrived_at": req["arrived_at"],
            "relative_arrival_s": req["relative_arrival_s"],
            "scheduled_lag_ms": max(0, (time.perf_counter() - due_time) * 1000),
            "endpoint": self.endpoint.name,
            "model": self.endpoint.model,
            "routed_only": True,          # no latency modeled -> excluded from SLO stats
            "success": True,
            "error": None, "error_type": None, "http_status": None,
            "ttft_ms": None, "e2e_ms": None, "tpot_ms": None,
            "chunks": 0,
            "prompt_tokens": int(req.get("prompt_tokens") or 0),
            "completion_tokens": max(1, effective),
            "output_chars": 0,
            "cost_usd": 0.0,
        }
        result["cost_usd"] = compute_cost_usd(self.endpoint, result)
        return result


def compute_cost_usd(endpoint: Endpoint, result: dict[str, Any]) -> float:
    """Bill a completed request from its returned usage. Failed requests cost $0."""
    if not result["success"]:
        return 0.0
    prompt_tokens = result["prompt_tokens"] or 0
    completion_tokens = result["completion_tokens"] or 0
    return (prompt_tokens * endpoint.input_price_per_mtok
            + completion_tokens * endpoint.output_price_per_mtok) / 1e6


async def one_request(
    session,
    endpoint: Endpoint,
    req: dict[str, Any],
    due_time: float,
    *,
    max_tokens_override: int | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Send one streaming chat-completion and measure TTFT/TPOT/e2e.

    SSE parsing, timing and error handling are line-for-line from vllm/run.py
    one_request; only the endpoint parameterization and billing are new.
    """
    start = time.perf_counter()
    first_token_time: float | None = None
    end_time: float | None = None
    chunks = 0
    usage: dict[str, Any] = {}
    output_chars = 0

    result = {
        "request_id": req["request_id"],
        "arrived_at": req["arrived_at"],
        "relative_arrival_s": req["relative_arrival_s"],
        "scheduled_lag_ms": max(0, (start - due_time) * 1000),
        "endpoint": endpoint.name,
        "model": endpoint.model,
        "success": False,
        "error": None,
        "error_type": None,
        "http_status": None,
        "ttft_ms": None,
        "e2e_ms": None,
        "tpot_ms": None,
        "chunks": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "output_chars": 0,
        "cost_usd": 0.0,
    }

    timeout = aiohttp.ClientTimeout(total=timeout_s) if aiohttp is not None else None

    def record_error(error_type: str, message: str) -> None:
        result["error_type"] = error_type
        result["error"] = message
        result["e2e_ms"] = (time.perf_counter() - start) * 1000
        result["chunks"] = chunks

        if first_token_time is not None:
            result["ttft_ms"] = (first_token_time - start) * 1000

    try:
        async with session.post(
            endpoint.url,
            headers=endpoint.headers(),
            json=make_payload(endpoint, req, max_tokens_override),
            timeout=timeout,
        ) as resp:
            result["http_status"] = resp.status

            if resp.status >= 400:
                error_text = (await resp.text())[:500]
                record_error(f"HTTP {resp.status}", f"HTTP {resp.status}: {error_text}")
                return result

            buffer = ""
            done = False

            async for raw in resp.content.iter_chunked(8192):
                buffer += raw.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    if data == "[DONE]":
                        end_time = time.perf_counter()
                        done = True
                        break

                    obj = json.loads(data)

                    if obj.get("error"):
                        error = obj["error"]
                        message = error.get("message") if isinstance(error, dict) else str(error)
                        record_error("StreamError", message)
                        return result

                    if obj.get("usage"):
                        usage = obj["usage"]

                    choices = obj.get("choices") or []
                    delta = choices[0].get("delta") if choices else {}
                    token = delta.get("content") if isinstance(delta, dict) else ""

                    if token:
                        now = time.perf_counter()
                        first_token_time = first_token_time or now
                        chunks += 1
                        output_chars += len(token)

                if done:
                    break

        end_time = end_time or time.perf_counter()

        result["success"] = True
        result["e2e_ms"] = (end_time - start) * 1000
        result["chunks"] = chunks

        if first_token_time is not None:
            result["ttft_ms"] = (first_token_time - start) * 1000

        result["prompt_tokens"] = usage.get("prompt_tokens")
        result["completion_tokens"] = usage.get("completion_tokens")
        result["output_chars"] = output_chars

        gen_count = result["completion_tokens"] or chunks
        if gen_count and gen_count > 1 and result["ttft_ms"] is not None:
            result["tpot_ms"] = (result["e2e_ms"] - result["ttft_ms"]) / (gen_count - 1)

    except asyncio.TimeoutError:
        record_error("TimeoutError", f"timeout after {timeout_s}s")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        record_error(type(exc).__name__, str(exc))
    except Exception as exc:
        record_error(type(exc).__name__, str(exc) or repr(exc))

    result["cost_usd"] = compute_cost_usd(endpoint, result)
    return result


def resolve_output_path(args: argparse.Namespace) -> Path:
    """Resolved raw-results path; the summary lives next to it (same directory)."""
    out = args.output or Path(f"{args.policy}_{args.scenario}.jsonl")
    if not out.is_absolute():
        out = args.out_dir / out
    return out


@contextlib.asynccontextmanager
async def _null_session():
    """Stands in for an aiohttp session when no real endpoint is used."""
    yield None


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    """Nearest-rank percentile on a pre-sorted list; None on empty."""
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def summarize(results: list[dict[str, Any]], policy: Policy, slo_s: float) -> dict[str, Any]:
    """Aggregate per-side and overall stats. A request violates the SLO when it
    failed or its TTFT exceeds the threshold (failures cannot meet an SLO)."""
    slo_ms = slo_s * 1000.0

    def side_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [r for r in rows if r["success"]]
        ttfts = sorted(r["ttft_ms"] for r in ok if r["ttft_ms"] is not None)
        tpots = sorted(r["tpot_ms"] for r in ok if r["tpot_ms"] is not None)
        # routed_only rows (NullCloud) carry no latency claim -> not in SLO stats
        measured = [r for r in rows if not r.get("routed_only")]
        viol = sum(1 for r in measured
                   if not r["success"] or r["ttft_ms"] is None or r["ttft_ms"] > slo_ms)
        errors: dict[str, int] = {}
        for r in rows:
            if r["error_type"]:
                errors[r["error_type"]] = errors.get(r["error_type"], 0) + 1
        return {
            "n": len(rows),
            "success": len(ok),
            "routed_only": sum(1 for r in rows if r.get("routed_only")),
            "errors": errors,
            "ttft_p50_ms": _percentile(ttfts, 50),
            "ttft_p95_ms": _percentile(ttfts, 95),
            "ttft_p99_ms": _percentile(ttfts, 99),
            "tpot_p50_ms": _percentile(tpots, 50),
            "slo_violations": viol,
            "slo_violation_pct": 100.0 * viol / max(len(measured), 1),
            "cost_usd": sum(r["cost_usd"] for r in rows),
        }

    local_rows = [r for r in results if r["endpoint"] == "local"]
    cloud_rows = [r for r in results if r["endpoint"] == "cloud"]

    return {
        "policy": policy.name,
        "target_fraction": policy.p,
        "actual_fraction": policy.actual_fraction,
        "slo_s": slo_s,
        "overall": side_stats(results),
        "local": side_stats(local_rows),
        "cloud": side_stats(cloud_rows),
    }


def build_endpoints(args: argparse.Namespace) -> tuple[Endpoint | None, Endpoint | None]:
    local = None
    if args.local_url:
        local = Endpoint(name="local", url=args.local_url, model=args.local_model)
    cloud = None
    needs_cloud = args.policy in ("all_cloud", "random")
    if args.cloud_url or (needs_cloud and args.cloud == "null"):
        model = args.cloud_model or args.local_model or "fake-cloud"
        cloud = Endpoint(
            name="cloud",
            url=args.cloud_url or "fake://",
            model=model,
            api_key_env=args.cloud_api_key_env,
            input_price_per_mtok=args.in_price,
            output_price_per_mtok=args.out_price,
        )
    return local, cloud


def build_cloud_sink(
    args: argparse.Namespace, needs_cloud: bool,
) -> tuple[Endpoint | None, Endpoint | None, NullCloud | None]:
    """Endpoints + the fake cloud sink for --cloud null (None for --cloud real:
    those requests go through a real streaming call)."""
    local, cloud = build_endpoints(args)
    sink = None
    if needs_cloud and args.cloud == "null":
        sink = NullCloud(cloud)
        print("cloud: null sink (route & count only — no latency modeled)")
    return local, cloud, sink
