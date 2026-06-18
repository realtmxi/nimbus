#!/usr/bin/env python3
"""Fixed-fraction offload strategy replay.

This script is for baseline and oracle-style fixed-fraction comparisons against
live SGLang. The online Nimbus algorithm lives in ``experiments/run_engine.py``
and the core package ``nimbus/decision.py``.

Usage:
    # Compare selected strategies at one fraction
    python experiments/run_offload_strategies.py --sglang-url http://localhost:8200 \
        --mode compare --fraction 0.25 --strategies cache_disp

    # Sweep fractions for several strategies
    python experiments/run_offload_strategies.py --sglang-url http://localhost:8200 \
        --mode knee --fractions 0.0 0.15 0.20 0.25 0.30 \
        --strategies cache_disp
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TRACE = (
    _PROJECT_ROOT / "data" / "sharegpt_burstgpt" / "sharegpt_prompts_burstgpt_timestamps.jsonl"
)


class MissingExperimentDependency(RuntimeError):
    """Raised when a live experiment dependency is not installed."""


def _require_aiohttp():
    """Import aiohttp only for live SGLang paths.

    Strategy construction, trace loading, and ``--help`` should work on a
    lightweight local Python. Real replay/probing still requires project deps.
    """
    try:
        import aiohttp
    except ModuleNotFoundError as exc:
        raise MissingExperimentDependency(
            "aiohttp is required for live SGLang replay. Install project deps with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    return aiohttp


def _require_collect_loop():
    try:
        from metrics_collector import collect_loop
    except ModuleNotFoundError as exc:
        raise MissingExperimentDependency(
            "metrics collection dependencies are missing. Install project deps with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    return collect_loop


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
def load_trace(
    trace_file: str,
    max_requests: int | None,
    duration_hours: float | None,
    start_hours: float = 0.0,
) -> list[dict]:
    """Load JSONL trace, optionally skipping initial quiet period."""
    requests: list[dict] = []
    first_ts: float | None = None
    skipped = 0

    with open(trace_file) as f:
        for i, line in enumerate(f):
            if max_requests and len(requests) >= max_requests:
                break
            row = json.loads(line)
            ts = float(row["arrived_at"])
            if first_ts is None:
                first_ts = ts
            elapsed_h = (ts - first_ts) / 3600
            if elapsed_h < start_hours:
                skipped += 1
                continue
            if duration_hours is not None and elapsed_h > start_hours + duration_hours:
                break
            session_id = row.get("session_id", 0)
            if session_id is None or session_id == "":
                session_id = 0
            requests.append({
                "request_id": i,
                "arrived_at": ts,
                "ts_sec": ts,
                "session_id": session_id,
                "prompt_text": row.get("prompt_text", ""),
                "num_prefill_tokens": int(row.get("num_prefill_tokens", 0)),
                "num_decode_tokens": int(row.get("num_decode_tokens", 256)),
            })
    if skipped:
        print(f"  Skipped {skipped} requests before {start_hours}h")
    return requests


# ---------------------------------------------------------------------------
# Request sender (same as cascade verification)
# ---------------------------------------------------------------------------
def stream_chunk_has_content(chunk: dict) -> bool:
    """Return True when an OpenAI streaming chunk carries non-empty content."""
    choices = chunk.get("choices") or []
    if not choices:
        return False
    delta = choices[0].get("delta") or {}
    return isinstance(delta.get("content"), str) and bool(delta["content"])


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    req_id: str,
    messages: list[dict] | None = None,
) -> dict:
    """Send one streaming chat-completion and measure TTFT + latency.

    If `messages` is provided, sends it directly as the messages array.
    Otherwise wraps `prompt` as a single user message.
    """
    if messages is None:
        # Try parsing prompt_text as a JSON messages array
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                messages = parsed
        except (json.JSONDecodeError, TypeError):
            pass
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    # Sanitize messages for SGLang compatibility:
    # - Convert 'tool' role to 'user' (Qwen3 template doesn't support tool role)
    # - Remove empty assistant messages
    # - Merge consecutive same-role messages
    sanitized: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "tool":
            role = "user"
        if role == "assistant" and not content:
            continue
        if sanitized and sanitized[-1]["role"] == role:
            sanitized[-1]["content"] += "\n" + content
        else:
            sanitized.append({"role": role, "content": content})
    messages = sanitized

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": min(max_tokens, 512),
        "stream": True,
    }
    headers = {"Content-Type": "application/json", "X-Request-ID": req_id}

    start = time.time()
    ttft_ms = None
    first_choice_ms = None
    chunk_count = 0
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    cj = json.loads(data)
                    chunk_count += 1
                    choices = cj.get("choices") or []
                    if choices and first_choice_ms is None:
                        first_choice_ms = (time.time() - start) * 1000
                    if ttft_ms is None and choices:
                        # Some OpenAI-compatible servers emit an initial role-only
                        # or empty delta. Prefer first non-empty content, but fall
                        # back to the first choices chunk before returning.
                        if stream_chunk_has_content(cj):
                            ttft_ms = (time.time() - start) * 1000
                except Exception:
                    pass
            latency_ms = (time.time() - start) * 1000
            if ttft_ms is None:
                ttft_ms = first_choice_ms
            return {
                "success": resp.status == 200,
                "ttft_ms": ttft_ms,
                "latency_ms": latency_ms,
                "status": resp.status,
                "chunks": chunk_count,
            }
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        return {
            "success": False,
            "ttft_ms": ttft_ms,
            "latency_ms": latency_ms,
            "status": 0,
            "chunks": chunk_count,
            "error": str(e)[:200],
        }


# ---------------------------------------------------------------------------
# KV pressure probe (for pressure-gated strategy)
# ---------------------------------------------------------------------------
_last_kv_pressure: float = 0.0


async def probe_kv_pressure(sglang_url: str) -> float:
    """Quick probe of current KV utilization (0-1).

    On timeout/error, returns last known value instead of 0 to avoid
    false negatives under heavy load.
    """
    aiohttp = _require_aiohttp()
    global _last_kv_pressure
    url = f"{sglang_url.rstrip('/')}/metrics"
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                text = await resp.text()
        used = max_tok = 0
        for line in text.splitlines():
            if line.startswith("sglang:num_used_tokens"):
                used = float(line.split()[-1])
            elif line.startswith("sglang:max_total_num_tokens"):
                max_tok = float(line.split()[-1])
        if max_tok > 0:
            _last_kv_pressure = used / max_tok
        return _last_kv_pressure
    except Exception:
        return _last_kv_pressure


# ---------------------------------------------------------------------------
# Offload decision strategies
# ---------------------------------------------------------------------------
class OffloadStrategy:
    """Base class for offload decision."""

    def __init__(self, fraction: float, seed: int):
        self.fraction = fraction
        self.rng = random.Random(seed)
        self.n_outsourced = 0
        self.n_total = 0

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        raise NotImplementedError

    @property
    def actual_fraction(self) -> float:
        return self.n_outsourced / max(self.n_total, 1)

    def set_fraction(self, fraction: float) -> None:
        """Dynamically update the outsource fraction (for hysteresis ramp)."""
        self.fraction = fraction


class AllLocalStrategy(OffloadStrategy):
    """No outsourcing: every request runs locally.

    Cost lower bound but worst latency under burst (will trigger
    bistability collapse for any non-trivial trace).  The `fraction`
    argument is ignored.
    """

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        self.n_total += 1
        return False


class AllCloudStrategy(OffloadStrategy):
    """Full outsourcing: every request goes to the cloud API.

    Cost upper bound but no local resource pressure.  Useful as a
    reference for "what if we never used the local GPU".  The
    `fraction` argument is ignored.
    """

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        self.n_total += 1
        self.n_outsourced += 1
        return True



class RandomRequestStrategy(OffloadStrategy):
    """Random per-request outsourcing at a fixed fraction (baseline).

    Each request is outsourced i.i.d. with probability `fraction`,
    independent of features or system state.
    """

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        self.n_total += 1
        if self.rng.random() < self.fraction:
            self.n_outsourced += 1
            return True
        return False






class CacheDispStrategy(OffloadStrategy):
    """Outsource requests with highest cache displacement (prefill * decode).

    Cache displacement = KV memory footprint (prefill tokens) * time occupied
    in decode batch (decode tokens). This is the memory-time product analogous
    to Denning's page residence time.

    Pre-computes a percentile threshold on the full trace.
    """

    def __init__(self, fraction: float, seed: int, trace: list[dict]):
        super().__init__(fraction, seed)
        scores = [self._score(r) for r in trace]
        scores_sorted = sorted(scores)
        if fraction <= 0:
            self.threshold = float("inf")
        else:
            idx = max(0, int(len(scores_sorted) * (1 - fraction)))
            self.threshold = scores_sorted[min(idx, len(scores_sorted) - 1)]

    def _score(self, req: dict) -> float:
        return req["num_prefill_tokens"] * req["num_decode_tokens"]

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        self.n_total += 1
        s = self._score(req)
        if s > self.threshold:
            self.n_outsourced += 1
            return True
        if s == self.threshold and self.rng.random() < self.fraction:
            self.n_outsourced += 1
            return True
        return False




DEFAULT_COMPARE_STRATEGIES = [
    "all_local",
    "all_cloud",
    "random_request",
    "cache_disp",
]

DEFAULT_KNEE_STRATEGIES = [
    "cache_disp",
]


def make_strategy_factories(frac: float, args: argparse.Namespace, trace: list[dict]):
    """Return strategy factories keyed by CLI strategy name.

    """
    return {
        # Intuitive baselines (oblivious / extremes).
        "all_local": lambda: AllLocalStrategy(frac, args.seed),
        "all_cloud": lambda: AllCloudStrategy(frac, args.seed),
        "random_request": lambda: RandomRequestStrategy(frac, args.seed),
        # Current fixed-fraction heuristic.
        "cache_disp": lambda: CacheDispStrategy(frac, args.seed, trace),
    }


def build_strategies(
    names: list[str],
    frac: float,
    args: argparse.Namespace,
    trace: list[dict],
) -> list[tuple[str, OffloadStrategy]]:
    factories = make_strategy_factories(frac, args, trace)
    unknown = [name for name in names if name not in factories]
    if unknown:
        raise ValueError(
            f"Unknown strategies: {unknown}. Choose from: {sorted(factories)}"
        )
    return [(name, factories[name]()) for name in names]


# ---------------------------------------------------------------------------
# Replay with strategy
# ---------------------------------------------------------------------------
async def replay_with_strategy(
    sglang_url: str,
    model: str,
    trace: list[dict],
    time_scale: float,
    strategy: OffloadStrategy,
    output_path: Path,
    probe_interval: float = 2.0,
) -> list[dict]:
    """Replay trace using the given offload strategy."""
    aiohttp = _require_aiohttp()
    url = f"{sglang_url.rstrip('/')}/v1/chat/completions"
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None)

    results: list[dict] = []
    tasks: list[tuple[int, asyncio.Task | None, dict, bool]] = []

    first_ts = trace[0]["ts_sec"]
    clock_start = time.time()

    # Initial probe so pressure-gated strategies don't start blind
    kv_pressure = await probe_kv_pressure(sglang_url)
    last_probe = 0.0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i, req in enumerate(trace):
            target = (req["ts_sec"] - first_ts) / time_scale
            elapsed = time.time() - clock_start
            if target > elapsed:
                await asyncio.sleep(target - elapsed)

            send_time = time.time() - clock_start

            # Probe KV pressure periodically
            if send_time - last_probe > probe_interval:
                kv_pressure = await probe_kv_pressure(sglang_url)
                last_probe = send_time

            outsourced = strategy.should_outsource(req, kv_pressure)

            if outsourced:
                tasks.append((i, None, {**req, "send_elapsed_s": send_time}, True))
            else:
                task = asyncio.create_task(
                    send_request(
                        session, url, model, req["prompt_text"],
                        req["num_decode_tokens"], f"strat_{i:06d}",
                    )
                )
                tasks.append((i, task, {**req, "send_elapsed_s": send_time}, False))

            if (i + 1) % 500 == 0:
                e = time.time() - clock_start
                print(
                    f"  [{e:.0f}s] Dispatched {i + 1}/{len(trace)} "
                    f"(outsourced {strategy.n_outsourced}, "
                    f"kv={kv_pressure:.1%})"
                )

        print(
            f"  All {len(trace)} dispatched "
            f"(outsourced {strategy.n_outsourced}, "
            f"local {len(trace) - strategy.n_outsourced}), "
            f"waiting..."
        )

        for idx, task, meta, outsourced in tasks:
            if outsourced:
                results.append({
                    "idx": idx,
                    "send_elapsed_s": f"{meta['send_elapsed_s']:.3f}",
                    "session_id": meta["session_id"],
                    "prefill_tokens": meta["num_prefill_tokens"],
                    "decode_tokens": meta["num_decode_tokens"],
                    "success": True,
                    "ttft_ms": "",
                    "latency_ms": "",
                    "status": "outsourced",
                    "chunks": 0,
                    "outsourced": True,
                })
            else:
                res = await task
                results.append({
                    "idx": idx,
                    "send_elapsed_s": f"{meta['send_elapsed_s']:.3f}",
                    "session_id": meta["session_id"],
                    "prefill_tokens": meta["num_prefill_tokens"],
                    "decode_tokens": meta["num_decode_tokens"],
                    "success": res["success"],
                    "ttft_ms": f"{res['ttft_ms']:.1f}" if res["ttft_ms"] is not None else "",
                    "latency_ms": f"{res['latency_ms']:.1f}",
                    "status": res["status"],
                    "chunks": res["chunks"],
                    "outsourced": False,
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "idx", "send_elapsed_s", "session_id", "prefill_tokens", "decode_tokens",
        "success", "ttft_ms", "latency_ms", "status", "chunks", "outsourced",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    return results


# ---------------------------------------------------------------------------
# Statistics (reused from cascade verification)
# ---------------------------------------------------------------------------
def compute_stats(results: list[dict], label: str) -> dict:
    """Compute summary statistics for a single run."""
    local = [r for r in results if not r["outsourced"]]
    ok = [r for r in local if r["success"]]
    ttfts = [float(r["ttft_ms"]) for r in ok if r["ttft_ms"]]
    lats = [float(r["latency_ms"]) for r in ok if r["latency_ms"]]

    def pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(int(len(s) * p), len(s) - 1)]

    stats = {
        "label": label,
        "total_requests": len(results),
        "local_requests": len(local),
        "outsourced_requests": len(results) - len(local),
        "actual_outsource_pct": (len(results) - len(local)) / max(len(results), 1),
        "local_success": len(ok),
        "local_success_rate": len(ok) / max(1, len(local)),
        "ttft_p50": pct(ttfts, 0.50),
        "ttft_p90": pct(ttfts, 0.90),
        "ttft_p95": pct(ttfts, 0.95),
        "ttft_p99": pct(ttfts, 0.99),
        "ttft_max": max(ttfts) if ttfts else 0,
        "latency_p50": pct(lats, 0.50),
        "latency_p99": pct(lats, 0.99),
    }

    print(f"\n{'=' * 60}")
    print(f"Strategy: {label}")
    print(f"  Total: {stats['total_requests']}, "
          f"Local: {stats['local_requests']}, "
          f"Outsourced: {stats['outsourced_requests']} "
          f"({stats['actual_outsource_pct']:.1%})")
    print(f"  Local success: {stats['local_success']} "
          f"({stats['local_success_rate']:.1%})")
    if ttfts:
        print(f"  TTFT (ms):  p50={stats['ttft_p50']:.0f}  "
              f"p90={stats['ttft_p90']:.0f}  "
              f"p95={stats['ttft_p95']:.0f}  "
              f"p99={stats['ttft_p99']:.0f}  "
              f"max={stats['ttft_max']:.0f}")
    return stats


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------
async def wait_for_cooldown(sglang_url: str, max_wait: float = 120) -> None:
    """Wait until SGLang has 0 running and 0 queued requests."""
    aiohttp = _require_aiohttp()
    url = f"{sglang_url.rstrip('/')}/metrics"
    timeout = aiohttp.ClientTimeout(total=5)
    start = time.time()
    print("  Cooling down...", end="", flush=True)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while time.time() - start < max_wait:
            try:
                async with session.get(url) as resp:
                    text = await resp.text()
                running = queued = 0
                for line in text.splitlines():
                    if line.startswith("sglang:num_running_reqs"):
                        running = int(float(line.split()[-1]))
                    elif line.startswith("sglang:num_queue_reqs"):
                        queued = int(float(line.split()[-1]))
                if running == 0 and queued == 0:
                    print(f" done ({time.time() - start:.0f}s)")
                    return
            except Exception:
                pass
            await asyncio.sleep(2)
    print(f" timeout after {max_wait}s")


# ---------------------------------------------------------------------------
# Hysteresis replay: ramp fraction during a single continuous replay
# ---------------------------------------------------------------------------
async def replay_with_ramp(
    sglang_url: str,
    model: str,
    trace: list[dict],
    time_scale: float,
    schedule: list[tuple[int, float]],
    seed: int,
    output_path: Path,
    probe_interval: float = 2.0,
) -> list[dict]:
    """Replay trace with dynamically changing outsource fraction.

    Args:
        schedule: list of (request_index, fraction) pairs. The fraction
                  switches at the given dispatch index. E.g.:
                  [(0, 0.0), (4000, 0.30), (8000, 0.40)]
                  means 0% for first 4000 reqs, then 30%, then 40%.
    """
    aiohttp = _require_aiohttp()
    url = f"{sglang_url.rstrip('/')}/v1/chat/completions"
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None)

    strategy = RandomRequestStrategy(schedule[0][1], seed)
    schedule_idx = 0

    results: list[dict] = []
    tasks: list[tuple[int, asyncio.Task | None, dict, bool]] = []

    first_ts = trace[0]["ts_sec"]
    clock_start = time.time()
    kv_pressure = 0.0
    last_probe = 0.0

    # Track per-phase stats
    phase_boundaries = [s[0] for s in schedule]
    phase_fractions = [s[1] for s in schedule]

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i, req in enumerate(trace):
            # Check if we should switch fraction
            if schedule_idx + 1 < len(schedule) and i >= schedule[schedule_idx + 1][0]:
                schedule_idx += 1
                new_frac = schedule[schedule_idx][1]
                strategy.set_fraction(new_frac)
                elapsed = time.time() - clock_start
                print(f"\n  >>> [{elapsed:.0f}s] FRACTION SWITCH at req {i}: "
                      f"{new_frac:.0%} <<<\n")

            target = (req["ts_sec"] - first_ts) / time_scale
            elapsed = time.time() - clock_start
            if target > elapsed:
                await asyncio.sleep(target - elapsed)

            send_time = time.time() - clock_start

            if send_time - last_probe > probe_interval:
                kv_pressure = await probe_kv_pressure(sglang_url)
                last_probe = send_time

            outsourced = strategy.should_outsource(req, kv_pressure)

            if outsourced:
                tasks.append((i, None, {**req, "send_elapsed_s": send_time}, True))
            else:
                task = asyncio.create_task(
                    send_request(
                        session, url, model, req["prompt_text"],
                        req["num_decode_tokens"], f"ramp_{i:06d}",
                    )
                )
                tasks.append((i, task, {**req, "send_elapsed_s": send_time}, False))

            if (i + 1) % 500 == 0:
                e = time.time() - clock_start
                cur_frac = phase_fractions[schedule_idx]
                print(
                    f"  [{e:.0f}s] Dispatched {i + 1}/{len(trace)} "
                    f"(frac={cur_frac:.0%}, outsourced={strategy.n_outsourced}, "
                    f"kv={kv_pressure:.1%})"
                )

        print(
            f"  All {len(trace)} dispatched "
            f"(outsourced {strategy.n_outsourced}), waiting..."
        )

        for idx, task, meta, outsourced in tasks:
            # Determine which phase this request belongs to
            phase = 0
            for pi, boundary in enumerate(phase_boundaries):
                if idx >= boundary:
                    phase = pi

            if outsourced:
                results.append({
                    "idx": idx,
                    "send_elapsed_s": f"{meta['send_elapsed_s']:.3f}",
                    "session_id": meta["session_id"],
                    "prefill_tokens": meta["num_prefill_tokens"],
                    "decode_tokens": meta["num_decode_tokens"],
                    "success": True,
                    "ttft_ms": "",
                    "latency_ms": "",
                    "status": "outsourced",
                    "chunks": 0,
                    "outsourced": True,
                    "phase": phase,
                    "phase_fraction": phase_fractions[phase],
                })
            else:
                res = await task
                results.append({
                    "idx": idx,
                    "send_elapsed_s": f"{meta['send_elapsed_s']:.3f}",
                    "session_id": meta["session_id"],
                    "prefill_tokens": meta["num_prefill_tokens"],
                    "decode_tokens": meta["num_decode_tokens"],
                    "success": res["success"],
                    "ttft_ms": f"{res['ttft_ms']:.1f}" if res["ttft_ms"] is not None else "",
                    "latency_ms": f"{res['latency_ms']:.1f}",
                    "status": res["status"],
                    "chunks": res["chunks"],
                    "outsourced": False,
                    "phase": phase,
                    "phase_fraction": phase_fractions[phase],
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "idx", "send_elapsed_s", "session_id", "prefill_tokens", "decode_tokens",
        "success", "ttft_ms", "latency_ms", "status", "chunks", "outsourced",
        "phase", "phase_fraction",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    return results


def compute_phase_stats(results: list[dict]) -> list[dict]:
    """Compute per-phase summary statistics from ramp results."""
    phases = defaultdict(list)
    for r in results:
        phases[r["phase"]].append(r)

    all_stats = []
    for phase_num in sorted(phases.keys()):
        phase_results = phases[phase_num]
        frac = phase_results[0]["phase_fraction"]

        local = [r for r in phase_results if not r["outsourced"]]
        ok = [r for r in local if r["success"]]
        ttfts = [float(r["ttft_ms"]) for r in ok if r["ttft_ms"]]

        def pct(vals, p):
            if not vals:
                return 0.0
            s = sorted(vals)
            return s[min(int(len(s) * p), len(s) - 1)]

        stats = {
            "phase": phase_num,
            "fraction": frac,
            "total": len(phase_results),
            "local": len(local),
            "outsourced": len(phase_results) - len(local),
            "success": len(ok),
            "success_rate": len(ok) / max(1, len(local)),
            "ttft_p50": pct(ttfts, 0.50),
            "ttft_p90": pct(ttfts, 0.90),
            "ttft_p99": pct(ttfts, 0.99),
        }
        all_stats.append(stats)
    return all_stats


def _coefficient_of_variation(values: list[float]) -> float:
    vals = [float(v) for v in values if float(v) > 0]
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return (var ** 0.5) / mean


def _steady_state_reached(
    segment_stats: list[dict],
    min_windows: int,
    cv_threshold: float,
) -> tuple[bool, dict]:
    """Return whether recent hold segments look steady enough to compare."""
    if len(segment_stats) < min_windows:
        return False, {
            "steady_windows": len(segment_stats),
            "ttft_p50_cv": "",
            "success_rate_cv": "",
        }

    recent = segment_stats[-min_windows:]
    ttft_cv = _coefficient_of_variation([s.get("ttft_p50", 0.0) for s in recent])
    success_cv = _coefficient_of_variation(
        [max(float(s.get("local_success_rate", 0.0)), 1e-9) for s in recent]
    )
    steady = ttft_cv <= cv_threshold and success_cv <= cv_threshold
    return steady, {
        "steady_windows": min_windows,
        "ttft_p50_cv": ttft_cv,
        "success_rate_cv": success_cv,
    }


def _trace_chunk(trace: list[dict], start: int, count: int) -> list[dict]:
    if not trace or count <= 0:
        return []
    if start + count <= len(trace):
        return trace[start:start + count]
    # Wrap around for long hold protocols. Each replay chunk re-bases timestamps
    # to its own first request, so wrapping only reuses request shapes.
    out = trace[start:]
    remaining = count - len(out)
    while remaining > 0:
        take = min(remaining, len(trace))
        out.extend(trace[:take])
        remaining -= take
    return out


async def _flush_cache_best_effort(sglang_url: str) -> None:
    aiohttp = _require_aiohttp()
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{sglang_url.rstrip('/')}/flush_cache") as resp:
                print(f"  Flush response: {resp.status}")
    except Exception as exc:
        print(f"  Flush failed: {exc}")


async def _run_hold_until_steady(
    *,
    args: argparse.Namespace,
    trace: list[dict],
    trace_start: int,
    fraction: float,
    label: str,
    out_dir: Path,
) -> tuple[int, list[dict], dict]:
    """Run repeated fixed-fraction hold segments until steady or max segments."""
    segment_stats: list[dict] = []
    trace_pos = trace_start
    out_dir.mkdir(parents=True, exist_ok=True)

    for segment_idx in range(args.bistability_max_holds):
        chunk = _trace_chunk(trace, trace_pos % len(trace), args.bistability_hold_requests)
        trace_pos += args.bistability_hold_requests
        strategy = RandomRequestStrategy(fraction, args.seed + segment_idx)
        requests_csv = out_dir / f"{label}_segment_{segment_idx:02d}.csv"
        print(
            f"  Hold {label} segment={segment_idx} fraction={fraction:.0%} "
            f"requests={len(chunk)}"
        )
        results = await replay_with_strategy(
            args.sglang_url,
            args.model,
            chunk,
            args.time_scale,
            strategy,
            requests_csv,
        )
        stats = compute_stats(results, f"{label}_segment_{segment_idx:02d}")
        stats.update(
            {
                "stage": label,
                "segment": segment_idx,
                "target_fraction": fraction,
                "actual_outsource_pct": strategy.actual_fraction,
                "requests_csv": str(requests_csv),
            }
        )
        segment_stats.append(stats)

        steady, steady_metrics = _steady_state_reached(
            segment_stats,
            args.bistability_steady_windows,
            args.bistability_steady_cv,
        )
        stats.update(steady_metrics)
        stats["steady"] = steady
        if steady:
            break

    final = segment_stats[-1] if segment_stats else {}
    return trace_pos, segment_stats, final


# ---------------------------------------------------------------------------
# Main: hysteresis verification mode
# ---------------------------------------------------------------------------
async def run_hysteresis(args: argparse.Namespace) -> None:
    """Test for hysteresis by ramping fraction UP then DOWN without cache flush."""
    aiohttp = _require_aiohttp()
    collect_loop = _require_collect_loop()
    fractions = args.fractions  # e.g. [0.0, 0.30, 0.35, 0.40, 0.45, 0.50]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"hysteresis_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=== Hysteresis Verification Experiment ===")
    print(f"SGLang:       {args.sglang_url}")
    print(f"Fractions:    {[f'{f:.0%}' for f in fractions]}")
    print(f"Output:       {sweep_dir}")
    print()

    print("Loading trace...")
    trace = load_trace(
        args.trace_file, args.max_requests, args.duration_hours, args.start_hours,
    )
    if not trace:
        print("No requests loaded!")
        return
    print(f"  {len(trace)} requests loaded")

    n_phases = len(fractions)
    reqs_per_phase = len(trace) // n_phases
    print(f"  {reqs_per_phase} requests per phase ({n_phases} phases)")

    config = {
        "sglang_url": args.sglang_url,
        "model": args.model,
        "trace_file": args.trace_file,
        "start_hours": args.start_hours,
        "duration_hours": args.duration_hours,
        "time_scale": args.time_scale,
        "fractions": fractions,
        "reqs_per_phase": reqs_per_phase,
        "seed": args.seed,
        "timestamp": ts,
    }
    with open(sweep_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_direction_stats = {}

    for direction, frac_order in [
        ("ramp_up", fractions),           # 0% → 50% (start saturated)
        ("ramp_down", list(reversed(fractions))),  # 50% → 0% (start healthy)
    ]:
        run_dir = sweep_dir / direction
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"Direction: {direction}")
        print(f"  Fractions: {[f'{f:.0%}' for f in frac_order]}")
        print(f"{'#' * 60}")

        # Flush cache before each direction to start from a known state
        print("  Flushing SGLang cache...")
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    f"{args.sglang_url.rstrip('/')}/flush_cache"
                ) as resp:
                    print(f"  Flush response: {resp.status}")
        except Exception as e:
            print(f"  Flush failed: {e}")

        await wait_for_cooldown(args.sglang_url)
        await asyncio.sleep(5)

        # Build schedule: [(req_idx, fraction), ...]
        schedule = [(i * reqs_per_phase, f) for i, f in enumerate(frac_order)]

        metrics_csv = run_dir / "metrics.csv"
        requests_csv = run_dir / "requests.csv"

        metrics_url = f"{args.sglang_url.rstrip('/')}/metrics"
        stop_event = asyncio.Event()
        collector_task = asyncio.create_task(
            collect_loop(metrics_url, metrics_csv, interval_s=0.1, stop_event=stop_event)
        )

        try:
            results = await replay_with_ramp(
                args.sglang_url, args.model, trace, args.time_scale,
                schedule, args.seed, requests_csv,
            )
        finally:
            stop_event.set()
            await asyncio.sleep(0.2)
            collector_task.cancel()
            try:
                await collector_task
            except asyncio.CancelledError:
                pass

        phase_stats = compute_phase_stats(results)
        all_direction_stats[direction] = phase_stats

        print(f"\n  --- {direction} per-phase results ---")
        print(f"  {'Phase':>5} {'Frac':>6} {'Local':>6} {'Success%':>9} "
              f"{'TTFT_p50':>10} {'TTFT_p90':>10}")
        for s in phase_stats:
            print(f"  {s['phase']:>5} {s['fraction']:>5.0%} "
                  f"{s['local']:>6} {s['success_rate']:>8.1%} "
                  f"{s['ttft_p50']:>9.0f}ms {s['ttft_p90']:>9.0f}ms")

    # Write combined summary
    summary_path = sweep_dir / "hysteresis_summary.csv"
    cols = ["direction", "phase", "fraction", "total", "local", "outsourced",
            "success", "success_rate", "ttft_p50", "ttft_p90", "ttft_p99"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for direction, stats_list in all_direction_stats.items():
            for s in stats_list:
                row = {"direction": direction, **s}
                w.writerow(row)

    # Print comparison
    print(f"\n{'=' * 70}")
    print("HYSTERESIS COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Frac':>6} | {'Ramp UP TTFT_p50':>16} {'Success%':>9} | "
          f"{'Ramp DOWN TTFT_p50':>18} {'Success%':>9} | {'Gap':>8}")
    print("-" * 80)

    up_stats = {s["fraction"]: s for s in all_direction_stats["ramp_up"]}
    down_stats = {s["fraction"]: s for s in all_direction_stats["ramp_down"]}

    for frac in fractions:
        u = up_stats.get(frac, {})
        d = down_stats.get(frac, {})
        u_ttft = u.get("ttft_p50", 0)
        d_ttft = d.get("ttft_p50", 0)
        u_succ = u.get("success_rate", 0)
        d_succ = d.get("success_rate", 0)
        gap = u_ttft - d_ttft if u_ttft and d_ttft else 0
        print(f"{frac:>5.0%} | {u_ttft:>12.0f}ms {u_succ:>8.1%} | "
              f"{d_ttft:>14.0f}ms {d_succ:>8.1%} | {gap:>+7.0f}ms")

    print(f"\nSummary: {summary_path}")
    print("\nIf ramp_up TTFT >> ramp_down TTFT at the same fraction,")
    print("hysteresis is confirmed: the system 'remembers' its previous state.")


async def run_bistability(args: argparse.Namespace) -> None:
    """Validate bistability with steady-state hold plus trigger-removal.

    For each target outsource fraction:
      1. Clean path: flush cache, hold directly at the target fraction until
         recent segment metrics stabilize.
      2. Post-trigger path: flush cache, hold at a trigger fraction until stable,
         then remove the trigger by switching to the same target fraction and
         hold again until stable.

    If the two final steady states differ at the same target fraction, the
    experiment supports hysteresis/bistability. The old ramp mode is kept only
    as a quick exploratory smoke.
    """
    collect_loop = _require_collect_loop()
    fractions = args.fractions
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"bistability_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=== Bistability Hold + Trigger-Removal Experiment ===")
    print(f"SGLang:          {args.sglang_url}")
    print(f"Target fractions:{[f'{f:.0%}' for f in fractions]}")
    print(f"Trigger fraction:{args.bistability_trigger_fraction:.0%}")
    print(f"Hold requests:   {args.bistability_hold_requests}")
    print(f"Max holds:       {args.bistability_max_holds}")
    print(f"Steady windows:  {args.bistability_steady_windows}")
    print(f"Steady CV:       {args.bistability_steady_cv}")
    print(f"Output:          {sweep_dir}")
    print()

    trace = load_trace(
        args.trace_file,
        args.max_requests,
        args.duration_hours,
        args.start_hours,
    )
    if not trace:
        print("No requests loaded!")
        return
    print(f"  {len(trace)} requests loaded")

    config = {
        "sglang_url": args.sglang_url,
        "model": args.model,
        "trace_file": args.trace_file,
        "start_hours": args.start_hours,
        "duration_hours": args.duration_hours,
        "time_scale": args.time_scale,
        "fractions": fractions,
        "trigger_fraction": args.bistability_trigger_fraction,
        "hold_requests": args.bistability_hold_requests,
        "max_holds": args.bistability_max_holds,
        "steady_windows": args.bistability_steady_windows,
        "steady_cv": args.bistability_steady_cv,
        "seed": args.seed,
        "timestamp": ts,
    }
    with open(sweep_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_csv = sweep_dir / "metrics.csv"
    stop_event = asyncio.Event()
    collector_task = asyncio.create_task(
        collect_loop(
            f"{args.sglang_url.rstrip('/')}/metrics",
            metrics_csv,
            interval_s=0.1,
            stop_event=stop_event,
        )
    )

    all_segments: list[dict] = []
    summary_rows: list[dict] = []
    trace_pos = 0

    try:
        for target_fraction in fractions:
            print(f"\n{'#' * 72}")
            print(f"Target fraction: {target_fraction:.0%}")
            print(f"{'#' * 72}")

            print("\n[CLEAN PATH] flush -> hold target")
            await _flush_cache_best_effort(args.sglang_url)
            await wait_for_cooldown(args.sglang_url)
            clean_dir = sweep_dir / f"target_{int(target_fraction * 100):03d}" / "clean"
            trace_pos, clean_segments, clean_final = await _run_hold_until_steady(
                args=args,
                trace=trace,
                trace_start=trace_pos,
                fraction=target_fraction,
                label=f"clean_target_{int(target_fraction * 100):03d}",
                out_dir=clean_dir,
            )
            for row in clean_segments:
                row["path"] = "clean"
                row["target_fraction"] = target_fraction
            all_segments.extend(clean_segments)

            print("\n[POST-TRIGGER PATH] flush -> hold trigger -> hold target")
            await _flush_cache_best_effort(args.sglang_url)
            await wait_for_cooldown(args.sglang_url)
            trigger_dir = sweep_dir / f"target_{int(target_fraction * 100):03d}" / "trigger"
            trace_pos, trigger_segments, trigger_final = await _run_hold_until_steady(
                args=args,
                trace=trace,
                trace_start=trace_pos,
                fraction=args.bistability_trigger_fraction,
                label=f"trigger_{int(target_fraction * 100):03d}",
                out_dir=trigger_dir,
            )
            for row in trigger_segments:
                row["path"] = "trigger"
                row["target_fraction"] = target_fraction
            all_segments.extend(trigger_segments)

            post_dir = sweep_dir / f"target_{int(target_fraction * 100):03d}" / "post_trigger"
            trace_pos, post_segments, post_final = await _run_hold_until_steady(
                args=args,
                trace=trace,
                trace_start=trace_pos,
                fraction=target_fraction,
                label=f"post_trigger_target_{int(target_fraction * 100):03d}",
                out_dir=post_dir,
            )
            for row in post_segments:
                row["path"] = "post_trigger"
                row["target_fraction"] = target_fraction
            all_segments.extend(post_segments)

            clean_ttft = float(clean_final.get("ttft_p50", 0.0) or 0.0)
            post_ttft = float(post_final.get("ttft_p50", 0.0) or 0.0)
            ratio = post_ttft / clean_ttft if clean_ttft > 0 else 0.0
            gap_ms = post_ttft - clean_ttft
            summary_rows.append(
                {
                    "target_fraction": target_fraction,
                    "trigger_fraction": args.bistability_trigger_fraction,
                    "clean_steady": bool(clean_final.get("steady", False)),
                    "post_trigger_steady": bool(post_final.get("steady", False)),
                    "clean_segments": len(clean_segments),
                    "trigger_segments": len(trigger_segments),
                    "post_trigger_segments": len(post_segments),
                    "clean_ttft_p50": clean_ttft,
                    "post_trigger_ttft_p50": post_ttft,
                    "gap_ms": gap_ms,
                    "ratio": ratio,
                    "clean_success_rate": clean_final.get("local_success_rate", 0.0),
                    "post_trigger_success_rate": post_final.get("local_success_rate", 0.0),
                }
            )

            print(
                f"\nTarget {target_fraction:.0%}: clean={clean_ttft:.0f}ms "
                f"post_trigger={post_ttft:.0f}ms gap={gap_ms:+.0f}ms ratio={ratio:.2f}x"
            )
    finally:
        stop_event.set()
        await asyncio.sleep(0.2)
        collector_task.cancel()
        try:
            await collector_task
        except asyncio.CancelledError:
            pass

    segment_cols = [
        "path", "stage", "segment", "target_fraction", "actual_outsource_pct",
        "steady", "steady_windows", "ttft_p50_cv", "success_rate_cv",
        "label", "total_requests", "local_requests", "outsourced_requests",
        "local_success", "local_success_rate",
        "ttft_p50", "ttft_p90", "ttft_p95", "ttft_p99", "ttft_max",
        "latency_p50", "latency_p99", "requests_csv",
    ]
    with open(sweep_dir / "bistability_segments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=segment_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_segments)

    summary_cols = [
        "target_fraction", "trigger_fraction",
        "clean_steady", "post_trigger_steady",
        "clean_segments", "trigger_segments", "post_trigger_segments",
        "clean_ttft_p50", "post_trigger_ttft_p50",
        "gap_ms", "ratio",
        "clean_success_rate", "post_trigger_success_rate",
    ]
    summary_path = sweep_dir / "bistability_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nSummary: {summary_path}")
    print("Use only rows where clean_steady and post_trigger_steady are both true.")


# ---------------------------------------------------------------------------
# Main: strategy comparison mode
# ---------------------------------------------------------------------------
async def run_compare(args: argparse.Namespace) -> None:
    """Compare offload strategies at matched budget."""
    aiohttp = _require_aiohttp()
    collect_loop = _require_collect_loop()
    frac = args.fraction
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"strategies_{ts}_f{int(frac*100):02d}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2.6: Offload Strategy Comparison ===")
    print(f"SGLang:       {args.sglang_url}")
    print(f"Window:       {args.start_hours}h–{args.start_hours + args.duration_hours}h "
          f"at {args.time_scale}x")
    print(f"Budget:       {frac:.0%} outsource")
    strat_names = args.strategies or DEFAULT_COMPARE_STRATEGIES
    print(f"Strategies:   {', '.join(strat_names)}")
    print(f"Output:       {sweep_dir}")
    print()

    print("Loading trace...")
    trace = load_trace(
        args.trace_file, args.max_requests, args.duration_hours, args.start_hours,
    )
    if not trace:
        print("No requests loaded!")
        return
    print(f"  {len(trace)} requests loaded")

    n_sessions = len({r["session_id"] for r in trace})
    print(f"  {n_sessions} unique sessions")

    # Save config
    config = {
        "sglang_url": args.sglang_url,
        "model": args.model,
        "trace_file": args.trace_file,
        "start_hours": args.start_hours,
        "duration_hours": args.duration_hours,
        "time_scale": args.time_scale,
        "fraction": frac,
        "num_requests": len(trace),
        "num_sessions": n_sessions,
        "seed": args.seed,
        "timestamp": ts,
    }
    with open(sweep_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    strategies = build_strategies(strat_names, frac, args, trace)

    all_stats: list[dict] = []

    for si, (name, strategy) in enumerate(strategies):
        run_dir = sweep_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"Run {si + 1}/{len(strategies)}: {name} (target {frac:.0%})")
        print(f"{'#' * 60}")

        # Flush cache before every strategy to ensure fair comparison
        print("  Flushing SGLang cache...")
        try:
            async with aiohttp.ClientSession() as flush_sess:
                async with flush_sess.post(
                    f"{args.sglang_url.rstrip('/')}/flush_cache"
                ) as resp:
                    print(f"  Flush response: {resp.status}")
        except Exception as e:
            print(f"  Flush failed: {e}")

        await wait_for_cooldown(args.sglang_url)
        if si > 0:
            await asyncio.sleep(args.cooldown)

        metrics_csv = run_dir / "metrics.csv"
        requests_csv = run_dir / "requests.csv"

        metrics_url = f"{args.sglang_url.rstrip('/')}/metrics"
        stop_event = asyncio.Event()
        collector_task = asyncio.create_task(
            collect_loop(metrics_url, metrics_csv, interval_s=0.1, stop_event=stop_event)
        )

        try:
            results = await replay_with_strategy(
                args.sglang_url, args.model, trace, args.time_scale,
                strategy, requests_csv,
            )
        finally:
            stop_event.set()
            await asyncio.sleep(0.2)
            collector_task.cancel()
            try:
                await collector_task
            except asyncio.CancelledError:
                pass

        stats = compute_stats(results, name)
        stats["target_fraction"] = frac
        all_stats.append(stats)

        print(f"  Actual outsource rate: {strategy.actual_fraction:.1%}")

    # Write summary
    summary_path = sweep_dir / "strategy_summary.csv"
    cols = [
        "label", "target_fraction", "actual_outsource_pct",
        "total_requests", "local_requests", "outsourced_requests",
        "local_success", "local_success_rate",
        "ttft_p50", "ttft_p90", "ttft_p95", "ttft_p99", "ttft_max",
        "latency_p50", "latency_p99",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_stats)

    print(f"\n{'=' * 70}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<20} {'Outsourced%':>11} {'Success%':>9} "
          f"{'TTFT_p50':>10} {'TTFT_p99':>10} {'CacheHit':>9}")
    print("-" * 70)
    for s in all_stats:
        print(f"{s['label']:<20} {s['actual_outsource_pct']:>10.1%} "
              f"{s['local_success_rate']:>8.1%} "
              f"{s['ttft_p50']:>9.0f}ms {s['ttft_p99']:>9.0f}ms")

    print(f"\nSummary: {summary_path}")


# ---------------------------------------------------------------------------
# Main: knee-finding sweep mode
# ---------------------------------------------------------------------------
async def run_knee(args: argparse.Namespace) -> None:
    """Sweep fractions and strategies to find capacity knees."""
    aiohttp = _require_aiohttp()
    collect_loop = _require_collect_loop()
    fractions = args.fractions
    strat_names = args.strategies or DEFAULT_KNEE_STRATEGIES
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"knee_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2.6: Capacity Knee Finding ===")
    print(f"SGLang:       {args.sglang_url}")
    print(f"Window:       {args.start_hours}h–{args.start_hours + args.duration_hours}h "
          f"at {args.time_scale}x")
    print(f"Fractions:    {[f'{f:.0%}' for f in fractions]}")
    print(f"Strategies:   {', '.join(strat_names)}")
    print(f"Output:       {sweep_dir}")
    print()

    print("Loading trace...")
    trace = load_trace(
        args.trace_file, args.max_requests, args.duration_hours, args.start_hours,
    )
    if not trace:
        print("No requests loaded!")
        return
    print(f"  {len(trace)} requests loaded")

    config = {
        "sglang_url": args.sglang_url,
        "model": args.model,
        "trace_file": args.trace_file,
        "start_hours": args.start_hours,
        "duration_hours": args.duration_hours,
        "time_scale": args.time_scale,
        "fractions": fractions,
        "strategies": strat_names,
        "seed": args.seed,
        "timestamp": ts,
    }
    with open(sweep_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_stats: list[dict] = []
    run_count = 0
    total_runs = len(fractions) * len(strat_names)

    for frac in fractions:
        frac_tag = f"f{int(frac * 100):02d}"
        frac_dir = sweep_dir / f"strategies_{ts}_{frac_tag}"
        frac_dir.mkdir(parents=True, exist_ok=True)
        frac_config = {
            **config,
            "fraction": frac,
            "num_requests": len(trace),
            "num_sessions": len({r["session_id"] for r in trace}),
        }
        with open(frac_dir / "config.json", "w") as f:
            json.dump(frac_config, f, indent=2)

        frac_stats: list[dict] = []
        strategies = build_strategies(strat_names, frac, args, trace)

        for name, strategy in strategies:
            run_count += 1
            run_dir = frac_dir / name
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'#' * 60}")
            print(
                f"Run {run_count}/{total_runs}: {name} "
                f"(target {frac:.0%})"
            )
            print(f"{'#' * 60}")

            print("  Flushing SGLang cache...")
            try:
                async with aiohttp.ClientSession() as flush_sess:
                    async with flush_sess.post(
                        f"{args.sglang_url.rstrip('/')}/flush_cache"
                    ) as resp:
                        print(f"  Flush response: {resp.status}")
            except Exception as e:
                print(f"  Flush failed: {e}")

            await wait_for_cooldown(args.sglang_url)
            if run_count > 1:
                await asyncio.sleep(args.cooldown)

            metrics_csv = run_dir / "metrics.csv"
            requests_csv = run_dir / "requests.csv"

            metrics_url = f"{args.sglang_url.rstrip('/')}/metrics"
            stop_event = asyncio.Event()
            collector_task = asyncio.create_task(
                collect_loop(metrics_url, metrics_csv, interval_s=0.1, stop_event=stop_event)
            )

            try:
                results = await replay_with_strategy(
                    args.sglang_url, args.model, trace, args.time_scale,
                    strategy, requests_csv,
                )
            finally:
                stop_event.set()
                await asyncio.sleep(0.2)
                collector_task.cancel()
                try:
                    await collector_task
                except asyncio.CancelledError:
                    pass

            stats = compute_stats(results, name)
            stats["target_fraction"] = frac
            stats["actual_outsource_pct"] = strategy.actual_fraction
            frac_stats.append(stats)
            all_stats.append(stats)
            print(f"  Actual outsource rate: {strategy.actual_fraction:.1%}")

        frac_summary = frac_dir / "strategy_summary.csv"
        cols = [
            "label", "target_fraction", "actual_outsource_pct",
            "total_requests", "local_requests", "outsourced_requests",
            "local_success", "local_success_rate",
            "ttft_p50", "ttft_p90", "ttft_p95", "ttft_p99", "ttft_max",
            "latency_p50", "latency_p99",
        ]
        with open(frac_summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(frac_stats)

    summary_path = sweep_dir / "knee_summary.csv"
    cols = [
        "label", "target_fraction", "actual_outsource_pct",
        "total_requests", "local_requests", "outsourced_requests",
        "local_success", "local_success_rate",
        "ttft_p50", "ttft_p90", "ttft_p95", "ttft_p99", "ttft_max",
        "latency_p50", "latency_p99",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_stats)

    print(f"\n{'=' * 70}")
    print("KNEE-FINDING SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Strategy':<16} {'Frac':>6} {'Outsrc%':>8} {'Success%':>9} "
          f"{'TTFT_p50':>10} {'TTFT_p99':>10}")
    print("-" * 70)
    for s in all_stats:
        print(f"{s['label']:<16} {s['target_fraction']:>5.0%} "
              f"{s['actual_outsource_pct']:>7.1%} "
              f"{s['local_success_rate']:>8.1%} "
              f"{s['ttft_p50']:>9.0f}ms {s['ttft_p99']:>9.0f}ms")

    print(f"\nSummary: {summary_path}")


# ---------------------------------------------------------------------------
# Main: policy invariance verification (Idea 5)
# ---------------------------------------------------------------------------
async def run_policy_invariance(args: argparse.Namespace) -> None:
    """Validate Idea 5: scheduling policy doesn't matter.

    Sweeps multiple fractions × multiple strategies to show that under
    saturation AND under healthy regime, all policies give the same result.
    The only variable that matters is the outsource fraction (admission rate).
    """
    aiohttp = _require_aiohttp()
    collect_loop = _require_collect_loop()
    fractions = args.fractions
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"policy_invariance_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=== Idea 5 Validation: Policy Invariance ===")
    print(f"SGLang:       {args.sglang_url}")
    print(f"Fractions:    {[f'{f:.0%}' for f in fractions]}")
    print(f"Output:       {sweep_dir}")
    print()

    print("Loading trace...")
    trace = load_trace(
        args.trace_file, args.max_requests, args.duration_hours, args.start_hours,
    )
    if not trace:
        print("No requests loaded!")
        return
    print(f"  {len(trace)} requests loaded")

    prefills = [r["num_prefill_tokens"] for r in trace]
    prefills_sorted = sorted(prefills)
    print(f"  Prefill tokens: p10={prefills_sorted[len(prefills)//10]}, "
          f"p50={prefills_sorted[len(prefills)//2]}, "
          f"p90={prefills_sorted[len(prefills)*9//10]}")

    config = {
        "sglang_url": args.sglang_url,
        "model": args.model,
        "trace_file": args.trace_file,
        "start_hours": args.start_hours,
        "duration_hours": args.duration_hours,
        "time_scale": args.time_scale,
        "fractions": fractions,
        "strategies": ["random_request", "outsource_long", "outsource_short"],
        "seed": args.seed,
        "timestamp": ts,
        "hypothesis": "All strategies give same TTFT at each fraction",
    }
    with open(sweep_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_stats: list[dict] = []
    run_count = 0
    total_runs = len(fractions) * 3

    for frac in fractions:
        strategy_defs = [
            ("random_request", lambda f=frac: RandomRequestStrategy(f, args.seed)),
        ]

        for strat_name, make_strat in strategy_defs:
            run_count += 1
            tag = f"f{int(frac * 100):03d}_{strat_name}"
            run_dir = sweep_dir / tag
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'#' * 60}")
            print(f"Run {run_count}/{total_runs}: {strat_name} @ {frac:.0%}")
            print(f"{'#' * 60}")

            # Flush cache before each run
            print("  Flushing SGLang cache...")
            try:
                async with aiohttp.ClientSession() as flush_sess:
                    async with flush_sess.post(
                        f"{args.sglang_url.rstrip('/')}/flush_cache"
                    ) as resp:
                        print(f"  Flush response: {resp.status}")
            except Exception as e:
                print(f"  Flush failed: {e}")

            await wait_for_cooldown(args.sglang_url)
            if run_count > 1:
                await asyncio.sleep(args.cooldown)

            strategy = make_strat()

            metrics_csv = run_dir / "metrics.csv"
            requests_csv = run_dir / "requests.csv"
            metrics_url = f"{args.sglang_url.rstrip('/')}/metrics"
            stop_event = asyncio.Event()
            collector_task = asyncio.create_task(
                collect_loop(metrics_url, metrics_csv, interval_s=0.1,
                             stop_event=stop_event)
            )

            try:
                results = await replay_with_strategy(
                    args.sglang_url, args.model, trace, args.time_scale,
                    strategy, requests_csv,
                )
            finally:
                stop_event.set()
                await asyncio.sleep(0.2)
                collector_task.cancel()
                try:
                    await collector_task
                except asyncio.CancelledError:
                    pass

            stats = compute_stats(results, f"{strat_name}@{frac:.0%}")
            stats["strategy"] = strat_name
            stats["target_fraction"] = frac
            stats["actual_outsource_pct"] = strategy.actual_fraction
            all_stats.append(stats)

    # Write summary
    summary_path = sweep_dir / "policy_invariance_summary.csv"
    cols = [
        "strategy", "target_fraction", "actual_outsource_pct",
        "label", "total_requests", "local_requests", "outsourced_requests",
        "local_success", "local_success_rate",
        "ttft_p50", "ttft_p90", "ttft_p95", "ttft_p99", "ttft_max",
        "latency_p50", "latency_p99",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_stats)

    # Print comparison matrix
    print(f"\n{'=' * 80}")
    print("POLICY INVARIANCE RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Frac':>6} | {'Strategy':<18} | {'Outsrc%':>7} {'Succ%':>7} "
          f"{'TTFT_p50':>10} {'TTFT_p99':>10}")
    print("-" * 80)

    for frac in fractions:
        for s in all_stats:
            if s["target_fraction"] == frac:
                print(f"{frac:>5.0%} | {s['strategy']:<18} | "
                      f"{s['actual_outsource_pct']:>6.1%} "
                      f"{s['local_success_rate']:>6.1%} "
                      f"{s['ttft_p50']:>9.0f}ms {s['ttft_p99']:>9.0f}ms")
        print("-" * 80)

    # Compute per-fraction max spread (indicator of policy sensitivity)
    print(f"\n{'Frac':>6} | {'TTFT_p50 spread':>16} | {'Verdict':>10}")
    print("-" * 40)
    for frac in fractions:
        ttfts = [s["ttft_p50"] for s in all_stats if s["target_fraction"] == frac]
        if ttfts and max(ttfts) > 0:
            spread = max(ttfts) - min(ttfts)
            ratio = max(ttfts) / max(min(ttfts), 1)
            verdict = "INVARIANT" if ratio < 1.5 else "SENSITIVE"
            print(f"{frac:>5.0%} | {spread:>10.0f}ms ({ratio:.2f}x) | {verdict:>10}")

    print(f"\nSummary: {summary_path}")
    print("\nIf all strategies show INVARIANT at each fraction,")
    print("Idea 5 is validated: scheduling policy doesn't matter,")
    print("only admission rate matters.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> None:
    if args.mode == "compare":
        await run_compare(args)
    elif args.mode == "knee":
        await run_knee(args)
    elif args.mode == "hysteresis":
        await run_hysteresis(args)
    elif args.mode == "bistability":
        await run_bistability(args)
    elif args.mode == "policy_invariance":
        await run_policy_invariance(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.6: Offload strategy comparison"
    )
    parser.add_argument(
        "--sglang-url", default="http://localhost:8003",
        help="SGLang base URL",
    )
    parser.add_argument(
        "--model", default="Qwen3-Coder-30B-A3B-Instruct",
        help="Model ID",
    )
    parser.add_argument(
        "--trace-file", default=str(_DEFAULT_TRACE),
        help="JSONL trace file",
    )
    parser.add_argument("--start-hours", type=float, default=0.8)
    parser.add_argument("--duration-hours", type=float, default=1.0)
    parser.add_argument("--time-scale", type=float, default=5.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cooldown", type=float, default=10)
    parser.add_argument(
        "--output-dir",
        default=str(_PROJECT_ROOT / "logs" / "phase2"),
    )

    # Mode selection
    parser.add_argument(
        "--mode", choices=["compare", "knee", "hysteresis", "bistability", "policy_invariance"],
        default="compare",
        help="'compare': 3 strategies at matched budget. "
             "'knee': fraction sweep. "
             "'hysteresis': legacy ramp up/down smoke. "
             "'bistability': steady-state hold + trigger-removal validation. "
             "'policy_invariance': validate that scheduling policy doesn't matter.",
    )
    # For compare mode
    parser.add_argument(
        "--fraction", type=float, default=0.25,
        help="Outsource budget for strategy comparison (default 0.25)",
    )
    # For compare mode: select specific strategies
    parser.add_argument(
        "--strategies", type=str, nargs="+", default=None,
        help="Strategies to run (default: all). "
             "Choose from: all_local, all_cloud, random_request, "
             "cache_disp",
    )
    # For knee mode
    parser.add_argument(
        "--fractions", type=float, nargs="+",
        default=[0.0, 0.30, 0.35, 0.40, 0.45, 0.50],
        help="Fractions for knee sweep",
    )
    parser.add_argument(
        "--bistability-trigger-fraction",
        type=float,
        default=0.0,
        help="Trigger fraction used to induce the collapsed state before removal",
    )
    parser.add_argument(
        "--bistability-hold-requests",
        type=int,
        default=2000,
        help="Requests per hold segment in --mode bistability",
    )
    parser.add_argument(
        "--bistability-max-holds",
        type=int,
        default=4,
        help="Maximum hold segments per stage in --mode bistability",
    )
    parser.add_argument(
        "--bistability-steady-windows",
        type=int,
        default=3,
        help="Recent hold segments required for steady-state check",
    )
    parser.add_argument(
        "--bistability-steady-cv",
        type=float,
        default=0.10,
        help="Max coefficient of variation for steady-state TTFT/success checks",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    try:
        asyncio.run(run(args))
    except MissingExperimentDependency as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
