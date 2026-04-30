"""Adapter implementations for specific serving engines.

This module contains the SGLang adapter used by the outsourcing engine. In
SGLang, accessing internal scheduler structures for metrics is discouraged.
Instead, this adapter maintains its own internal waiting queue and reads
operational metrics from SGLang's Prometheus endpoint (e.g.
http://localhost:30000/metrics) for observability.
"""

import re
import threading
import time
from collections import deque
from collections.abc import Callable

import requests

import logging
def get_logger(name): return logging.getLogger(name)

logger = get_logger(__name__)

from nimbus.queue import WaitingQueueInterface
from nimbus.request import OutsourcingRequestInfo


class SGLangWaitingQueueAdapter(WaitingQueueInterface):
    """Adapter for SGLang integration with internal queue management.

    This adapter maintains its own internal FIFO waiting queue and fetches
    performance metrics from SGLang's Prometheus endpoint for monitoring.

    Example usage:
        queue_adapter = SGLangWaitingQueueAdapter(metrics_url="http://localhost:30000/metrics")

        # Add requests to the queue
        queue_adapter.add_request(request_info)

        # Use with outsourcing engine
        engine = OutsourcingEngine(
            waiting_queue=queue_adapter,
            ...
        )
    """

    def __init__(
        self,
        metrics_url: str | None = None,
        http_timeout_s: float = 1.0,
        metrics_cache_ttl_s: float = 0.25,
    ):
        """Initialize the adapter.

        Args:
            metrics_url: Optional Prometheus metrics URL (defaults to
                "http://localhost:30000/metrics" if not provided).
            http_timeout_s: Timeout for HTTP requests to the metrics endpoint.
            metrics_cache_ttl_s: How long to treat cached metrics as fresh.
                Safe callers receive cached values immediately and trigger a
                background refresh when the cache is stale.
        """
        self.metrics_url = metrics_url or "http://localhost:30000/metrics"
        self.http_timeout_s = http_timeout_s
        self.metrics_cache_ttl_s = max(0.0, metrics_cache_ttl_s)

        # Internal waiting queue (FIFO)
        self._waiting_queue: deque[OutsourcingRequestInfo] = deque()

        # Index for fast lookup by request ID
        self._request_index: dict[str, OutsourcingRequestInfo] = {}

        # Optional hook to refresh request snapshots before returning them
        self._request_update_hook: Callable[[OutsourcingRequestInfo], None] | None = None

        # Metrics cache: request-path callers use safe=True, so never block the
        # event loop on a synchronous HTTP scrape when stale data is acceptable.
        self._metrics_cache: dict = {}
        self._metrics_cache_time: float = 0.0
        self._metrics_refresh_in_flight = False
        self._metrics_lock = threading.Lock()

    def set_request_update_hook(
        self, hook: Callable[[OutsourcingRequestInfo], None] | None
    ) -> None:
        """Install a callback that can refresh request metrics in-place."""
        self._request_update_hook = hook

    def add_request(self, request: OutsourcingRequestInfo) -> None:
        """Add a new request to the waiting queue.

        Args:
            request: Request information to add to the queue
        """
        # Update queue time to current
        current_time = time.time()
        request.queue_time = current_time - request.arrival_time

        # Add to queue and index
        self._waiting_queue.append(request)
        self._request_index[request.request_id] = request

    def get_all_waiting(self) -> list[OutsourcingRequestInfo]:
        """Get snapshot of all waiting requests in queue order (FCFS).

        Returns:
            List of OutsourcingRequestInfo for all waiting requests
        """
        # Update queue times for all requests
        current_time = time.time()
        result = []

        for req in self._waiting_queue:
            # Update queue time (in-place is fine, we return the objects)
            req.queue_time = current_time - req.arrival_time
            if self._request_update_hook is not None:
                try:
                    self._request_update_hook(req)
                except Exception as exc:  # keep queue resilient to hook failures
                    logger.warning(
                        "Request update hook failed for %s: %s",
                        req.request_id,
                        exc,
                    )
            result.append(req)

        return result

    def remove_requests(self, request_ids: set[str]) -> list[OutsourcingRequestInfo]:
        """Remove specified requests from the waiting queue.

        Args:
            request_ids: Set of request IDs to remove

        Returns:
            List of removed OutsourcingRequestInfo objects
        """
        removed = []
        ids_to_remove = set(request_ids)

        # Filter the queue, keeping only requests NOT in the removal set
        new_queue = deque()
        for req in self._waiting_queue:
            if req.request_id in ids_to_remove:
                # Remove from index and add to removed list
                self._request_index.pop(req.request_id, None)
                removed.append(req)
            else:
                # Keep in queue
                new_queue.append(req)

        self._waiting_queue = new_queue
        return removed

    def get_length(self) -> int:
        """Current number of waiting requests.

        Returns:
            Number of requests in the waiting queue
        """
        return len(self._waiting_queue)

    def peek(self) -> OutsourcingRequestInfo | None:
        """Look at the head of the queue without removing.

        Returns:
            OutsourcingRequestInfo for the first request, or None if empty
        """
        return self._waiting_queue[0] if self._waiting_queue else None

    # --------------------
    # Metrics integration
    # --------------------
    def get_metrics(self, safe: bool = False) -> dict:
        """Fetch key performance metrics from SGLang's Prometheus endpoint.

        Metrics include (when available):
          - throughput_tps: tokens per second
          - ttft_seconds: average time-to-first-token
          - inter_token_latency_seconds: average inter-token latency
          - hpu_memory_utilization: ratio (0-1) or percent (0-100)
          - request_queue: number of pending requests

        Args:
            safe: When True, suppress exceptions and return an empty dict on error.

        Returns:
            A dictionary of parsed metrics.
        """
        cached_metrics = self._get_cached_metrics()
        if cached_metrics is not None:
            return cached_metrics

        if safe:
            self._schedule_background_refresh()
            return self._get_cached_snapshot()

        return self._refresh_metrics_blocking()

    def _get_cached_snapshot(self) -> dict:
        with self._metrics_lock:
            return dict(self._metrics_cache)

    def _get_cached_metrics(self) -> dict | None:
        with self._metrics_lock:
            if not self._metrics_cache:
                return None
            age = time.monotonic() - self._metrics_cache_time
            if age <= self.metrics_cache_ttl_s:
                return dict(self._metrics_cache)
        return None

    def _store_metrics_cache(self, metrics: dict) -> dict:
        with self._metrics_lock:
            self._metrics_cache = dict(metrics)
            self._metrics_cache_time = time.monotonic()
            self._metrics_refresh_in_flight = False
            return dict(self._metrics_cache)

    def _clear_refresh_flag(self) -> None:
        with self._metrics_lock:
            self._metrics_refresh_in_flight = False

    def _refresh_metrics_blocking(self) -> dict:
        text = self._fetch_metrics_text()
        metrics = self._parse_prometheus_metrics(text)
        return self._store_metrics_cache(metrics)

    def _refresh_metrics_background(self) -> None:
        try:
            self._refresh_metrics_blocking()
        except Exception as exc:
            logger.debug("Async SGLang metrics refresh failed: %s", exc)
            self._clear_refresh_flag()

    def _schedule_background_refresh(self) -> None:
        with self._metrics_lock:
            if self._metrics_refresh_in_flight:
                return
            self._metrics_refresh_in_flight = True

        thread = threading.Thread(
            target=self._refresh_metrics_background,
            name="sglang-metrics-refresh",
            daemon=True,
        )
        thread.start()

    def _fetch_metrics_text(self) -> str:
        resp = requests.get(self.metrics_url, timeout=self.http_timeout_s)
        resp.raise_for_status()
        return resp.text

    def _parse_prometheus_metrics(self, text: str) -> dict:
        # Simple Prometheus text parser sufficient for scalar metrics and
        # histogram/counter _sum/_count pairs.
        lines = [ln.strip() for ln in text.splitlines() if ln and not ln.startswith("#")]

        # Collect raw samples by metric name
        samples: dict[str, list[tuple[float, dict[str, str]]]] = {}
        metric_re = re.compile(
            r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
        )

        def parse_labels(lbl: str) -> dict[str, str]:
            res: dict[str, str] = {}
            if not lbl:
                return res
            for part in lbl.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    res[k.strip()] = v.strip().strip('"')
            return res

        for ln in lines:
            m = metric_re.match(ln)
            if not m:
                continue
            name = m.group("name")
            labels = parse_labels(m.group("labels") or "")
            try:
                value = float(m.group("value"))
            except ValueError:
                continue
            samples.setdefault(name, []).append((value, labels))

        out: dict[str, float] = {}

        # Helper to compute average from _sum/_count
        def avg_from_sum_count(prefix: str) -> float | None:
            s_list = samples.get(f"{prefix}_sum")
            c_list = samples.get(f"{prefix}_count")
            if not s_list or not c_list:
                return None
            s_val = sum(v for v, _ in s_list)
            c_val = sum(v for v, _ in c_list)
            if c_val > 0:
                return s_val / c_val
            return None

        # Throughput (tokens/sec)
        # Prefer explicit tokens_per_second gauges if present
        for key in [
            "sglang:gen_throughput",
            "tokens_per_second",
            "throughput_tokens_per_second",
            "sglang_tokens_per_second",
        ]:
            if key in samples:
                out["throughput_tps"] = sum(v for v, _ in samples[key])
                break

        # TTFT average (seconds)
        for prefix in [
            "sglang:time_to_first_token_seconds",
            "time_to_first_token_seconds",
            "ttft_seconds",
            "sglang_ttft_seconds",
        ]:
            val = avg_from_sum_count(prefix)
            if val is not None:
                out["ttft_seconds"] = val
                break

        # Inter-token latency average (seconds)
        for prefix in [
            "inter_token_latency_seconds",
            "token_latency_seconds",
            "sglang_inter_token_latency_seconds",
        ]:
            val = avg_from_sum_count(prefix)
            if val is not None:
                out["inter_token_latency_seconds"] = val
                break

        # HPU memory utilization
        # Either a direct utilization metric, or ratio of used/total bytes
        for key in [
            "hpu_memory_utilization",
            "sglang_hpu_memory_utilization",
        ]:
            if key in samples:
                out["hpu_memory_utilization"] = sum(v for v, _ in samples[key])
                break

        if "hpu_memory_utilization" not in out:
            used = 0.0
            total = 0.0
            for key in ["hpu_memory_used_bytes", "sglang_hpu_memory_used_bytes"]:
                if key in samples:
                    used = sum(v for v, _ in samples[key])
                    break
            for key in ["hpu_memory_total_bytes", "sglang_hpu_memory_total_bytes"]:
                if key in samples:
                    total = sum(v for v, _ in samples[key])
                    break
            if total > 0:
                out["hpu_memory_utilization"] = used / total

        # Request queue length / pending requests
        for key in [
            "sglang:num_queue_reqs",
            "pending_requests",
            "request_queue_length",
            "waiting_requests",
            "sglang_pending_requests",
        ]:
            if key in samples:
                out["request_queue"] = sum(v for v, _ in samples[key])
                break

        return out
