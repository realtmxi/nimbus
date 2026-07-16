"""Shared building blocks for the hybrid router (see router/run.py for the entry point).

Contents: SCENARIOS / load_trace / one_request, line-for-line from the verified
baseline at its mtp revision (vllm/run.py @ dff1a81, branch Jialu — the version
she actually runs, md5-matched on the GPU host; main's older copy differs:
temperature/top_p, no stream_options),
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
from typing import Any, Callable

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
DEFAULT_KV_HYSTERESIS_FRACTION: float = 0.05


def effective_decode(
    req: dict[str, Any],
    max_tokens_override: int | None = None,
) -> int:
    """Decode length requested by both real and fake endpoints."""
    return (
        int(max_tokens_override)
        if max_tokens_override is not None
        else int(req["max_tokens"])
    )


def local_prompt_tokens(req: dict[str, Any]) -> int:
    """Marginal prompt KV added locally, falling back to the full prompt."""
    uncached = req.get("uncached_prompt_tokens")
    if uncached is not None:
        return int(uncached)
    return int(req.get("prompt_tokens") or 0)


def token_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
) -> float:
    """Shared token-price kernel for estimated and measured cloud cost."""
    return (
        int(prompt_tokens) * input_price_per_mtok
        + int(completion_tokens) * output_price_per_mtok
    ) / 1e6


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
    """Load the BurstGPT JSONL slice for a scenario; identical to vllm/run.py.

    scenario="full" disables window filtering: the whole file replays with
    arrivals relative to its first timestamp (for non-BurstGPT traces, e.g.
    the rednote long-prompt slices, whose arrived_at starts at 0).
    """
    if scenario == "full":
        start, end = float("-inf"), float("inf")
    else:
        start, end = SCENARIOS[scenario]
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            arrived_at = int(obj["arrived_at"])
            prompt = obj.get("prompt_text", "")

            if start <= arrived_at <= end and prompt:
                row = {
                    "request_id": len(rows),
                    "arrived_at": arrived_at,
                    "relative_arrival_s": arrived_at - start,
                    "prompt": prompt,
                    "max_tokens": int(obj["num_decode_tokens"]),
                    # additive vs vllm/run.py: exact token counts from the trace,
                    # used by cloud-sink billing and the queued dispatcher's KV math
                    "prompt_tokens": int(obj.get("num_prefill_tokens") or 0),
                    "session_id": obj.get("session_id"),
                }
                # Optional fields emitted by token-aligned/cache-aware trace
                # materializers.  Keeping them separate prevents a cumulative
                # full prompt from being confused with the suffix actually
                # prefetched on this deployment.
                for source_key, target_key in (
                    ("uncached_prompt_tokens", "uncached_prompt_tokens"),
                    ("num_cached_tokens", "num_cached_tokens"),
                    ("trace_num_prefill_tokens", "trace_prompt_tokens"),
                    ("trace_num_decode_tokens", "trace_decode_tokens"),
                    ("payload_mode", "payload_mode"),
                    ("cache_mode", "cache_mode"),
                    ("source_request_index", "source_request_index"),
                ):
                    if obj.get(source_key) is not None:
                        row[target_key] = obj[source_key]
                rows.append(row)

    rows.sort(key=lambda x: (x["arrived_at"], x["request_id"]))

    if rows and scenario == "full":
        first = rows[0]["arrived_at"]
        for row in rows:
            row["relative_arrival_s"] = row["arrived_at"] - first

    for i, row in enumerate(rows):
        row["request_id"] = i

    return rows


def make_payload(
    endpoint: Endpoint,
    req: dict[str, Any],
    max_tokens_override: int | None,
    *,
    continuous_usage: bool = False,
    temperature: float | None = None,
    ignore_eos: bool = False,
    provider_order: list[str] | None = None,
    allow_fallbacks: bool | None = None,
) -> dict[str, Any]:
    payload = {
        "model": endpoint.model,
        "stream": True,
        "max_tokens": effective_decode(req, max_tokens_override),
        "messages": [{"role": "user", "content": req["prompt"]}],
        "stream_options": {"include_usage": True},
    }
    if continuous_usage:
        # vLLM reports exact cumulative accepted-token counts on every stream
        # chunk, including when MTP emits several tokens in one content delta.
        payload["stream_options"]["continuous_usage_stats"] = True
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if ignore_eos:
        payload["ignore_eos"] = True
    if provider_order or allow_fallbacks is not None:
        # OpenRouter-specific preferences are opt-in so the default payload
        # remains byte-for-byte compatible with ordinary OpenAI endpoints.
        provider: dict[str, Any] = {}
        if provider_order:
            provider["order"] = list(provider_order)
        if allow_fallbacks is not None:
            provider["allow_fallbacks"] = bool(allow_fallbacks)
        payload["provider"] = provider
    return payload


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
        # exact mirror of make_payload: --max-tokens REPLACES the trace value,
        # and completion_tokens == the payload's max_tokens (an upper bound on
        # what a real call would bill; the fake sink has no model to EOS early).
        effective = effective_decode(req, max_tokens_override)
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
            "completion_tokens": effective,
            "output_chars": 0,
            "cost_usd": 0.0,
        }
        result["cost_usd"] = compute_cost_usd(self.endpoint, result)
        return result


def compute_cost_usd(endpoint: Endpoint,
                     result: dict[str, Any]) -> float | None:
    """Bill a request from returned usage, preserving unknown cloud costs.

    A TTFT-only probe deliberately aborts before the final usage SSE event, so
    treating absent usage as zero would under-report spend.  Local requests are
    still exactly free even when their endpoint omits usage.
    """
    if not result["success"]:
        return 0.0
    if endpoint.input_price_per_mtok == endpoint.output_price_per_mtok == 0.0:
        return 0.0
    prompt_tokens = result.get("prompt_tokens")
    completion_tokens = result.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    return token_cost_usd(
        prompt_tokens,
        completion_tokens,
        endpoint.input_price_per_mtok,
        endpoint.output_price_per_mtok,
    )


async def one_request(
    session,
    endpoint: Endpoint,
    req: dict[str, Any],
    due_time: float,
    *,
    max_tokens_override: int | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    on_output_progress: Callable[[int], None] | None = None,
    temperature: float | None = None,
    ignore_eos: bool = False,
    provider_order: list[str] | None = None,
    allow_fallbacks: bool | None = None,
    stop_after_first_token: bool = False,
) -> dict[str, Any]:
    """Send one streaming chat-completion and measure TTFT/TPOT/e2e.

    SSE parsing, timing and error handling are line-for-line from vllm/run.py
    one_request; endpoint parameterization, billing, and the optional
    exact cumulative output-progress hook are additive. When the hook is
    present, the request enables vLLM's continuous usage stats; content chunks
    are never treated as tokens because MTP may place several tokens in one.
    """
    start = time.perf_counter()
    first_token_time: float | None = None
    end_time: float | None = None
    chunks = 0
    usage: dict[str, Any] = {}
    output_chars = 0
    probe_stopped = False

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
        "cost_pending": False,
        "response_completed": False,
        "stream_abort_requested": False,
        "probe_mode": "ttft_cancel" if stop_after_first_token else None,
        "first_token_kind": None,
        "first_content_ttft_ms": None,
        "generation_id": None,
        "provider": None,
        "response_model": None,
        "requested_provider_order": list(provider_order or []),
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
            json=make_payload(
                endpoint,
                req,
                max_tokens_override,
                continuous_usage=on_output_progress is not None,
                temperature=temperature,
                ignore_eos=ignore_eos,
                provider_order=provider_order,
                allow_fallbacks=allow_fallbacks,
            ),
            timeout=timeout,
        ) as resp:
            result["http_status"] = resp.status
            headers = getattr(resp, "headers", None)
            if headers is not None:
                result["generation_id"] = headers.get("X-Generation-Id")

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

                    # OpenRouter exposes the generation id in a response
                    # header and may repeat id/provider/model on SSE chunks.
                    # Keep both paths: the header survives a first-token abort.
                    if obj.get("id"):
                        result["generation_id"] = obj["id"]
                    if obj.get("provider"):
                        result["provider"] = obj["provider"]
                    if obj.get("model"):
                        result["response_model"] = obj["model"]

                    if obj.get("error"):
                        error = obj["error"]
                        message = error.get("message") if isinstance(error, dict) else str(error)
                        record_error("StreamError", message)
                        return result

                    completion_progress = None
                    if obj.get("usage"):
                        usage = obj["usage"]
                        completion_progress = usage.get("completion_tokens")

                    choices = obj.get("choices") or []
                    delta = choices[0].get("delta") if choices else {}
                    if isinstance(delta, dict):
                        content = delta.get("content") or ""
                        # OpenRouter separates Qwen reasoning from visible
                        # content, while our vLLM deployment (no reasoning
                        # parser) streams the same <think> tokens in content.
                        # Count either as the first generated token so local
                        # and cloud TTFT use the same semantic boundary.
                        reasoning = (
                            delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or ""
                        )
                    else:
                        content = ""
                        reasoning = ""
                    token = content or reasoning

                    if (
                        on_output_progress is not None
                        and choices
                        and completion_progress is None
                    ):
                        record_error(
                            "ProgressUnavailable",
                            "endpoint did not return continuous completion-token "
                            "usage requested by Nimbus",
                        )
                        return result
                    if token:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                            result["first_token_kind"] = (
                                "content" if content else "reasoning"
                            )
                        if content and result["first_content_ttft_ms"] is None:
                            result["first_content_ttft_ms"] = (now - start) * 1000
                        chunks += 1
                        output_chars += len(token)
                        if stop_after_first_token:
                            # Abort the HTTP stream, not the asyncio task.  The
                            # latter can escape callers as CancelledError and
                            # lose the result row/semaphore release entirely.
                            result["stream_abort_requested"] = True
                            probe_stopped = True
                            close = getattr(resp, "close", None)
                            if close is not None:
                                close()
                            break
                    if (
                        on_output_progress is not None
                        and completion_progress is not None
                    ):
                        on_output_progress(int(completion_progress))

                if done or probe_stopped:
                    break

        result["success"] = True
        result["chunks"] = chunks

        if first_token_time is not None:
            result["ttft_ms"] = (first_token_time - start) * 1000

        result["prompt_tokens"] = usage.get("prompt_tokens")
        result["completion_tokens"] = usage.get("completion_tokens")
        result["output_chars"] = output_chars
        if probe_stopped:
            # TTFT was measured successfully, but completion/cost must be
            # enriched later through OpenRouter's generation endpoint.
            result["e2e_ms"] = None
            result["tpot_ms"] = None
            result["cost_usd"] = None
            result["cost_pending"] = True
        else:
            end_time = end_time or time.perf_counter()
            result["response_completed"] = done
            result["e2e_ms"] = (end_time - start) * 1000
            gen_count = result["completion_tokens"] or chunks
            if gen_count and gen_count > 1 and result["ttft_ms"] is not None:
                result["tpot_ms"] = (
                    (result["e2e_ms"] - result["ttft_ms"]) / (gen_count - 1)
                )

    except asyncio.TimeoutError:
        record_error("TimeoutError", f"timeout after {timeout_s}s")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        record_error(type(exc).__name__, str(exc))
    except Exception as exc:
        record_error(type(exc).__name__, str(exc) or repr(exc))

    if not probe_stopped:
        result["cost_usd"] = compute_cost_usd(endpoint, result)
        result["cost_pending"] = result["cost_usd"] is None
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
        measured_costs = [
            float(r["cost_usd"])
            for r in rows
            if isinstance(r.get("cost_usd"), (int, float))
            and not isinstance(r.get("cost_usd"), bool)
        ]
        pending_cost_n = len(rows) - len(measured_costs)
        known_cost = sum(measured_costs)
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
            "slo_measured_n": len(measured),   # SLO denominator (routed_only excluded)
            "slo_violation_pct": 100.0 * viol / max(len(measured), 1),
            # Never present a known subtotal as the total when probe rows await
            # generation-metadata enrichment.
            "cost_usd": known_cost if pending_cost_n == 0 else None,
            "known_cost_usd": known_cost,
            "cost_measured_n": len(measured_costs),
            "cost_pending_n": pending_cost_n,
        }

    local_rows = [r for r in results if r["endpoint"] == "local"]
    cloud_rows = [r for r in results if r["endpoint"] == "cloud"]
    overall = side_stats(results)
    local = side_stats(local_rows)
    cloud = side_stats(cloud_rows)
    # Paper-facing conservative bound: every cloud-routed request misses the
    # local TTFT SLO.  This keeps NullCloud runs comparable without inventing a
    # cloud latency and prevents an aggressive shedder from looking good merely
    # because routed-only rows are absent from the measured denominator.
    pessimistic_violations = local["slo_violations"] + len(cloud_rows)

    return {
        "policy": policy.name,
        "target_fraction": policy.p,
        "actual_fraction": policy.actual_fraction,
        "slo_s": slo_s,
        "overall": overall,
        "local": local,
        "cloud": cloud,
        "pessimistic_combined": {
            "slo_violations": pessimistic_violations,
            "slo_n": len(results),
            "slo_violation_pct": (
                100.0 * pessimistic_violations / max(len(results), 1)
            ),
            "cloud_assumed_violations": len(cloud_rows),
            "cost_usd": overall["cost_usd"],
            "known_cost_usd": overall["known_cost_usd"],
            "cost_measured_n": overall["cost_measured_n"],
            "cost_pending_n": overall["cost_pending_n"],
        },
    }


def build_endpoints(args: argparse.Namespace,
                    needs_cloud: bool | None = None) -> tuple[Endpoint | None, Endpoint | None]:
    local = None
    if args.local_url:
        local = Endpoint(name="local", url=args.local_url, model=args.local_model)
    cloud = None
    if needs_cloud is None:
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
    local, cloud = build_endpoints(args, needs_cloud)
    sink = None
    if needs_cloud and args.cloud == "null":
        sink = NullCloud(cloud)
        print("cloud: null sink (route & count only — no latency modeled)")
    return local, cloud, sink
