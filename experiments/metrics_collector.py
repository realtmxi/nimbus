#!/usr/bin/env python3
"""Serving metrics collector — polls /metrics and writes time-series CSV.

Can be used as a standalone CLI or imported by run_experiment.py.

Usage (standalone):
    python metrics_collector.py --url http://localhost:8000/metrics \
        --output metrics.csv --interval 0.1
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import re
import sys
import time
from pathlib import Path

# Metrics to extract from Prometheus text format.
# Keys are Prometheus metric names; values are stable CSV column names.
METRICS_TO_COLLECT = {
    # SGLang
    "sglang:token_usage": "token_usage",
    "sglang:max_total_num_tokens": "max_total_tokens",
    "sglang:num_running_reqs": "num_running",
    "sglang:num_queue_reqs": "num_queued",
    "sglang:num_used_tokens": "num_used_tokens",
    "sglang:cache_hit_rate": "cache_hit_rate",
    "sglang:gen_throughput": "gen_throughput",
    "sglang:new_token_ratio": "new_token_ratio",
    "sglang:num_paused_reqs": "num_paused",
    # vLLM
    "vllm:num_requests_running": "num_running",
    "vllm:num_requests_waiting": "num_queued",
    "vllm:kv_cache_usage_perc": "kv_cache_usage_pct",
    "vllm:gpu_cache_usage_perc": "kv_cache_usage_pct",
    "vllm:cpu_cache_usage_perc": "cpu_cache_usage_pct",
    "vllm:avg_prompt_throughput_toks_per_s": "prompt_throughput_tps",
    "vllm:avg_generation_throughput_toks_per_s": "gen_throughput",
}

CSV_COLUMNS = ["timestamp", "elapsed_s"] + list(dict.fromkeys(METRICS_TO_COLLECT.values()))

# Matches: engine:metric_name{optional_labels} value
_METRIC_RE = re.compile(r"^((?:sglang|vllm):\w+)(?:\{[^}]*\})?\s+(\S+)$")


class MissingMetricsDependency(RuntimeError):
    """Raised when a live metrics dependency is not installed."""


def _require_aiohttp():
    try:
        import aiohttp
    except ModuleNotFoundError as exc:
        raise MissingMetricsDependency(
            "aiohttp is required to collect serving metrics. Install project deps with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    return aiohttp


def parse_metrics_text(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition and return configured metrics."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if name in METRICS_TO_COLLECT and METRICS_TO_COLLECT[name] not in values:
            try:
                values[METRICS_TO_COLLECT[name]] = float(m.group(2))
            except ValueError:
                pass
    return values


async def collect_loop(
    metrics_url: str,
    output_path: Path,
    interval_s: float = 0.1,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Poll *metrics_url* every *interval_s* seconds and append to CSV.

    Runs until *stop_event* is set or the task is cancelled.
    """
    aiohttp = _require_aiohttp()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        t0 = time.time()
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                if stop_event and stop_event.is_set():
                    break
                try:
                    async with session.get(metrics_url) as resp:
                        text = await resp.text()
                    values = parse_metrics_text(text)
                    now = time.time()
                    row: dict = {"timestamp": f"{now:.3f}", "elapsed_s": f"{now - t0:.3f}"}
                    for col in METRICS_TO_COLLECT.values():
                        row[col] = values.get(col, "")
                    writer.writerow(row)
                    f.flush()
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass

                try:
                    await asyncio.sleep(interval_s)
                except asyncio.CancelledError:
                    break


async def _main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-serving Prometheus metrics collector")
    parser.add_argument("--url", default="http://localhost:8000/metrics")
    parser.add_argument("--output", default="metrics.csv")
    parser.add_argument("--interval", type=float, default=0.1, help="Poll interval in seconds")
    args = parser.parse_args()

    print(f"Collecting metrics from {args.url} every {args.interval}s → {args.output}")
    print("Press Ctrl+C to stop.")
    sys.stdout.flush()

    stop = asyncio.Event()
    try:
        await collect_loop(args.url, Path(args.output), args.interval, stop)
    except MissingMetricsDependency as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        pass
    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(_main())
