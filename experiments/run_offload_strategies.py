#!/usr/bin/env python3
"""Phase 2.6: Offload Strategy Comparison.

Compares four outsourcing strategies at matched budgets to test whether
*when* and *how* you outsource matters more than *how much*:

  A) random-request       — baseline (already shown ineffective)
  B) pressure-gated       — only outsource when KV > threshold (tests timing)
  C) session-aware        — outsource entire sessions to preserve prefix chains
  D) gated-session-aware  — pressure gate + session-level sticky decisions

Also sweeps fractions around the capacity knee (35%, 40%, 50%) to find the
phase transition point where the system escapes the saturated regime.

Key hypothesis: session-aware offload at 25% >>> random offload at 25%,
because it preserves intra-session prefix continuity.

Usage:
    # Compare 3 strategies at 25% budget
    python experiments/local_deployment/phase2/run_offload_strategies.py \
        --sglang-url http://localhost:8003

    # Knee-finding sweep (random strategy, fine fractions)
    python experiments/local_deployment/phase2/run_offload_strategies.py \
        --sglang-url http://localhost:8003 \
        --mode knee --fractions 0.0 0.30 0.35 0.40 0.45 0.50
"""

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

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_collector import collect_loop

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TRACE = (
    _PROJECT_ROOT / "data" / "sharegpt_burstgpt" / "sharegpt_prompts_burstgpt_timestamps.jsonl"
)


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
            requests.append({
                "ts_sec": ts,
                "session_id": int(row.get("session_id", 0)),
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
                    if ttft_ms is None:
                        choices = cj.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            if isinstance(delta.get("content"), str) and delta["content"]:
                                ttft_ms = (time.time() - start) * 1000
                except Exception:
                    pass
            latency_ms = (time.time() - start) * 1000
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


class FIFOStrategy(OffloadStrategy):
    """Outsource the first `fraction` of requests by arrival order.

    Oblivious baseline: no per-request features, just "send the first
    N% of arrivals to the cloud, keep the rest local".  Tests whether
    intelligent selection beats arrival-order shedding.
    """

    def __init__(self, fraction: float, seed: int, trace: list[dict]):
        super().__init__(fraction, seed)
        n = len(trace)
        cutoff = int(n * fraction)
        # Sort by arrival time and mark the first `cutoff` indices.
        sorted_indices = sorted(range(n), key=lambda i: trace[i]["arrived_at"])
        self.outsource_set: set[int] = set(sorted_indices[:cutoff])
        # Map from request key (use index in trace order if no id field)
        # to outsource decision.  Caller must pass the trace in same
        # order so we can match by sequential index.
        self._idx = 0

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        self.n_total += 1
        decide = self._idx in self.outsource_set
        self._idx += 1
        if decide:
            self.n_outsourced += 1
        return decide


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


class PressureGatedStrategy(OffloadStrategy):
    """Only outsource when KV pressure exceeds threshold.

    Uses a higher per-request probability during high-pressure periods
    to match the target overall fraction.
    """

    def __init__(self, fraction: float, seed: int, kv_threshold: float = 0.90):
        super().__init__(fraction, seed)
        self.kv_threshold = kv_threshold
        # During high-pressure windows, outsource at elevated rate.
        # Empirically ~32% of time is memory-bound (Phase 2 finding),
        # so to match overall fraction f, gate rate ≈ f / 0.32
        self.gate_rate = min(fraction / 0.32, 0.95)

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        self.n_total += 1
        if kv_pressure >= self.kv_threshold:
            if self.rng.random() < self.gate_rate:
                self.n_outsourced += 1
                return True
        return False


class SessionAwareStrategy(OffloadStrategy):
    """Outsource entire sessions to preserve prefix continuity.

    Pre-selects a fraction of sessions to outsource ALL their requests.
    Once a session is outsourced, all its turns go remote.
    """

    def __init__(self, fraction: float, seed: int, trace: list[dict]):
        super().__init__(fraction, seed)
        # Pre-compute session IDs and select which ones to outsource
        session_ids = list({req["session_id"] for req in trace})
        self.rng.shuffle(session_ids)
        n_outsource = int(len(session_ids) * fraction)
        self.outsourced_sessions: set[int] = set(session_ids[:n_outsource])

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        self.n_total += 1
        if req["session_id"] in self.outsourced_sessions:
            self.n_outsourced += 1
            return True
        return False


class SizeOutsourceLongStrategy(OffloadStrategy):
    """Outsource the LONGEST requests by prefill tokens (oracle).

    Keeps short requests locally — should be 'best case' for scheduling since
    short requests consume less KV memory and finish faster.  Uses oracle
    knowledge of the full trace to set a percentile threshold.
    """

    def __init__(self, fraction: float, seed: int, trace: list[dict]):
        super().__init__(fraction, seed)
        prefills = sorted(r["num_prefill_tokens"] for r in trace)
        if fraction <= 0:
            self.threshold = float("inf")
        else:
            idx = max(0, int(len(prefills) * (1 - fraction)))
            self.threshold = prefills[min(idx, len(prefills) - 1)]

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        self.n_total += 1
        p = req["num_prefill_tokens"]
        if p > self.threshold:
            self.n_outsourced += 1
            return True
        if p == self.threshold and self.rng.random() < self.fraction:
            self.n_outsourced += 1
            return True
        return False


class SizeOutsourceShortStrategy(OffloadStrategy):
    """Outsource the SHORTEST requests by prefill tokens (oracle).

    Keeps long requests locally — should be 'worst case' since long requests
    consume more KV memory and have longer service times.
    """

    def __init__(self, fraction: float, seed: int, trace: list[dict]):
        super().__init__(fraction, seed)
        prefills = sorted(r["num_prefill_tokens"] for r in trace)
        if fraction <= 0:
            self.threshold = -1
        else:
            idx = min(int(len(prefills) * fraction), len(prefills) - 1)
            self.threshold = prefills[idx]

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        self.n_total += 1
        p = req["num_prefill_tokens"]
        if p < self.threshold:
            self.n_outsourced += 1
            return True
        if p == self.threshold and self.rng.random() < self.fraction:
            self.n_outsourced += 1
            return True
        return False


class FlopBasedStrategy(OffloadStrategy):
    """Outsource requests with highest FLOP cost (production Nimbus v1 weight).

    Uses the production SimpleFLOPCalculator from routing/outsourcing/flop_calculator.py
    with Qwen2.5-7B architecture params. Weight =
        compute_prefill_flops(prefill) + 0.6 * compute_decode_flops(decode)
    matching Nimbus v1's default decode_weight_ratio.

    Pre-computes a percentile threshold on the full trace (oracle for threshold,
    but the scoring function is what Nimbus v1 actually uses in production).
    """

    def __init__(
        self,
        fraction: float,
        seed: int,
        trace: list[dict],
        hidden_dim: int = 3584,
        num_layers: int = 28,
        num_attention_heads: int = 28,
        decode_weight_ratio: float = 0.6,
    ):
        super().__init__(fraction, seed)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.decode_weight_ratio = decode_weight_ratio

        # Pre-compute scores and threshold from full trace
        scores = [self._score(r) for r in trace]
        scores_sorted = sorted(scores)
        if fraction <= 0:
            self.threshold = float("inf")
        else:
            idx = max(0, int(len(scores_sorted) * (1 - fraction)))
            self.threshold = scores_sorted[min(idx, len(scores_sorted) - 1)]

    def _score(self, req: dict) -> float:
        """Match production SimpleFLOPCalculator formulas exactly."""
        n = req["num_prefill_tokens"]
        m = req["num_decode_tokens"]
        d = self.hidden_dim
        L = self.num_layers
        # Prefill: 2*n^2*d*L (attention) + 4*n*d^2*L (FFN)
        prefill_flops = 2 * n * n * d * L + 4 * n * d * d * L
        # Decode: m*k*d*L (kv-attention, k=n since num_processed_tokens=0 at arrival)
        #         + 4*m*d^2*L (FFN)
        decode_flops = m * n * d * L + 4 * m * d * d * L
        return prefill_flops + self.decode_weight_ratio * decode_flops

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


class GatedSessionAwareStrategy(OffloadStrategy):
    """Pressure-gated session-aware outsourcing.

    Combines *when* to shed (pressure gate) with *who* to shed (session-level
    sticky decisions).  Only makes outsource decisions for new sessions when
    KV pressure exceeds the threshold; once a session is marked outsourced or
    local, that decision is sticky for all subsequent turns.
    """

    def __init__(self, fraction: float, seed: int, kv_threshold: float = 0.90):
        super().__init__(fraction, seed)
        self.kv_threshold = kv_threshold
        self.gate_rate = min(fraction / 0.32, 0.95)
        self.outsourced_sessions: set[int] = set()
        self.local_sessions: set[int] = set()

    def should_outsource(self, req: dict, kv_pressure: float) -> bool:
        self.n_total += 1
        sid = req["session_id"]

        # Sticky: already decided
        if sid in self.outsourced_sessions:
            self.n_outsourced += 1
            return True
        if sid in self.local_sessions:
            return False

        # First request of a new session: admission decision
        if kv_pressure >= self.kv_threshold and self.rng.random() < self.gate_rate:
            self.outsourced_sessions.add(sid)
            self.n_outsourced += 1
            return True
        else:
            self.local_sessions.add(sid)
            return False


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


# ---------------------------------------------------------------------------
# Main: hysteresis verification mode
# ---------------------------------------------------------------------------
async def run_hysteresis(args: argparse.Namespace) -> None:
    """Test for hysteresis by ramping fraction UP then DOWN without cache flush."""
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


# ---------------------------------------------------------------------------
# Main: strategy comparison mode
# ---------------------------------------------------------------------------
async def run_compare(args: argparse.Namespace) -> None:
    """Compare offload strategies at matched budget."""
    frac = args.fraction
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"strategies_{ts}_f{int(frac*100):02d}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2.6: Offload Strategy Comparison ===")
    print(f"SGLang:       {args.sglang_url}")
    print(f"Window:       {args.start_hours}h–{args.start_hours + args.duration_hours}h "
          f"at {args.time_scale}x")
    print(f"Budget:       {frac:.0%} outsource")
    strat_names = args.strategies or ["random_request", "pressure_gated", "session_aware", "gated_session_aware"]
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

    all_strategies = [
        # Intuitive baselines (oblivious / extremes).
        ("all_local", lambda: AllLocalStrategy(frac, args.seed)),
        ("all_cloud", lambda: AllCloudStrategy(frac, args.seed)),
        ("fifo", lambda: FIFOStrategy(frac, args.seed, trace)),
        ("random_request", lambda: RandomRequestStrategy(frac, args.seed)),
        # System-state baselines.
        ("pressure_gated", lambda: PressureGatedStrategy(frac, args.seed)),
        ("session_aware", lambda: SessionAwareStrategy(frac, args.seed, trace)),
        ("gated_session_aware", lambda: GatedSessionAwareStrategy(frac, args.seed)),
        # Feature-aware baselines.
        ("size_long", lambda: SizeOutsourceLongStrategy(frac, args.seed, trace)),
        ("size_short", lambda: SizeOutsourceShortStrategy(frac, args.seed, trace)),
        ("flop_based", lambda: FlopBasedStrategy(frac, args.seed, trace)),
        # Ours.
        ("cache_disp", lambda: CacheDispStrategy(frac, args.seed, trace)),
    ]
    if args.strategies:
        selected = set(args.strategies)
        strategies = [(n, fn()) for n, fn in all_strategies if n in selected]
    else:
        strategies = [(n, fn()) for n, fn in all_strategies]

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
    """Sweep fractions to find the capacity knee point."""
    fractions = args.fractions
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir) / f"knee_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2.6: Capacity Knee Finding ===")
    print(f"SGLang:       {args.sglang_url}")
    print(f"Window:       {args.start_hours}h–{args.start_hours + args.duration_hours}h "
          f"at {args.time_scale}x")
    print(f"Fractions:    {[f'{f:.0%}' for f in fractions]}")
    print(f"Strategy:     session-aware (preserves prefix chains)")
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
        "strategy": "session_aware",
        "seed": args.seed,
        "timestamp": ts,
    }
    with open(sweep_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_stats: list[dict] = []

    for fi, frac in enumerate(fractions):
        tag = f"f{int(frac * 100):03d}"
        run_dir = sweep_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"Run {fi + 1}/{len(fractions)}: session-aware {frac:.0%}")
        print(f"{'#' * 60}")

        if fi > 0:
            await wait_for_cooldown(args.sglang_url)
            await asyncio.sleep(args.cooldown)

        strategy = SessionAwareStrategy(frac, args.seed, trace)

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

        stats = compute_stats(results, f"session_{frac:.0%}")
        stats["target_fraction"] = frac
        stats["actual_outsource_pct"] = strategy.actual_fraction
        all_stats.append(stats)

    # Write summary
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
    print("KNEE-FINDING SUMMARY (session-aware offload)")
    print(f"{'=' * 70}")
    print(f"{'Frac':>6} {'Outsourced%':>11} {'Local':>6} {'Success%':>9} "
          f"{'TTFT_p50':>10} {'TTFT_p99':>10}")
    print("-" * 60)
    for s in all_stats:
        print(f"{s['target_fraction']:>5.0%} {s['actual_outsource_pct']:>10.1%} "
              f"{s['local_requests']:>6} {s['local_success_rate']:>8.1%} "
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
            ("outsource_long", lambda f=frac: SizeOutsourceLongStrategy(f, args.seed, trace)),
            ("outsource_short", lambda f=frac: SizeOutsourceShortStrategy(f, args.seed, trace)),
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
        "--mode", choices=["compare", "knee", "hysteresis", "policy_invariance"],
        default="compare",
        help="'compare': 3 strategies at matched budget. "
             "'knee': fraction sweep. "
             "'hysteresis': ramp up/down to test for path-dependent behavior. "
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
             "Choose from: random_request, pressure_gated, session_aware, gated_session_aware",
    )
    # For knee mode
    parser.add_argument(
        "--fractions", type=float, nargs="+",
        default=[0.0, 0.30, 0.35, 0.40, 0.45, 0.50],
        help="Fractions for knee sweep",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
