"""Profiled FLOP calculator using Vidur profiling data.

Uses the "virtual FLOPs" technique: the calculator returns
``profiled_time_seconds * CONSTANT`` as the FLOP count and ``CONSTANT``
as the effective FLOPS/s.  When the violation detector divides the two,
the constant cancels and the result is the real profiled latency.
"""

import bisect
import csv
from pathlib import Path

from .flop_calculator import FLOPCalculatorInterface
from .request import OutsourcingRequestInfo

# Arbitrary constant that cancels in every division.  Chosen large enough
# to keep the virtual FLOP values in a comfortable numerical range.
_VIRTUAL_FLOPS_CONSTANT = 1e12


class ProfiledFLOPCalculator(FLOPCalculatorInterface):
    """FLOP calculator backed by Vidur profiling CSV data.

    Prefill: looks up ``prompt_length -> prefill_time_ms`` via sorted-array
    bisect + linear interpolation, then returns ``time_s * CONSTANT``.

    Decode: loads only ``batch_size=1`` rows, builds a
    ``kv_cache_size -> decode_time_ms`` lookup with the same interpolation.
    """

    def __init__(self, prefill_csv_path: str | Path, decode_csv_path: str | Path) -> None:
        self._prefill_lengths: list[int] = []
        self._prefill_times_ms: list[float] = []

        self._decode_kv_sizes: list[int] = []
        self._decode_times_ms: list[float] = []

        self._load_prefill_csv(Path(prefill_csv_path))
        self._load_decode_csv(Path(decode_csv_path))

        if not self._prefill_lengths:
            raise ValueError(f"No prefill data loaded from {prefill_csv_path}")
        if not self._decode_kv_sizes:
            raise ValueError(f"No batch_size=1 decode data loaded from {decode_csv_path}")

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _load_prefill_csv(self, path: Path) -> None:
        """Load ``nimbus_prefill_curve.csv``.

        Expected columns: prompt_length, prefill_time_ms, throughput_tokens_per_sec
        """
        rows: list[tuple[int, float]] = []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append((int(row["prompt_length"]), float(row["prefill_time_ms"])))
        rows.sort(key=lambda t: t[0])
        self._prefill_lengths = [r[0] for r in rows]
        self._prefill_times_ms = [r[1] for r in rows]

    def _load_decode_csv(self, path: Path) -> None:
        """Load ``nimbus_decode_table.csv`` (batch_size=1 rows only).

        Duplicate ``kv_cache_size`` entries are averaged.

        Expected columns: batch_size, kv_cache_size, decode_time_ms, throughput_tokens_per_sec
        """
        # Accumulate (sum, count) per kv_cache_size to average duplicates
        accum: dict[int, list[float]] = {}
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if int(row["batch_size"]) != 1:
                    continue
                kv = int(row["kv_cache_size"])
                accum.setdefault(kv, []).append(float(row["decode_time_ms"]))
        rows = sorted((kv, sum(vs) / len(vs)) for kv, vs in accum.items())
        self._decode_kv_sizes = [r[0] for r in rows]
        self._decode_times_ms = [r[1] for r in rows]

    # ------------------------------------------------------------------
    # Interpolation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate(xs: list[int], ys: list[float], x: int) -> float:
        """Lookup *x* in the sorted array *xs* and linearly interpolate *ys*.

        Values outside the data range are clamped to the nearest endpoint.
        """
        if not xs:
            raise ValueError("Empty profiling data")

        # Clamp to endpoints
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]

        # bisect_right gives the insertion point; the interval is [i-1, i]
        i = bisect.bisect_right(xs, x)
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = ys[i - 1], ys[i]
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def prefill_time_seconds(self, num_tokens: int) -> float:
        """Return the profiled prefill latency in seconds for *num_tokens*."""
        ms = self._interpolate(self._prefill_lengths, self._prefill_times_ms, num_tokens)
        return ms / 1000.0

    def decode_time_seconds(self, kv_cache_size: int) -> float:
        """Return the profiled per-token decode latency in seconds."""
        ms = self._interpolate(self._decode_kv_sizes, self._decode_times_ms, kv_cache_size)
        return ms / 1000.0

    # ------------------------------------------------------------------
    # FLOPCalculatorInterface
    # ------------------------------------------------------------------

    def compute_prefill_flops(self, request: OutsourcingRequestInfo, num_tokens: int) -> float:
        """Virtual FLOPs = profiled_time_seconds * CONSTANT.

        Returns 0 for ``num_tokens <= 0`` (e.g. fully cached prompts) to
        match :class:`SimpleFLOPCalculator` semantics.
        """
        if num_tokens <= 0:
            return 0.0
        return self.prefill_time_seconds(num_tokens) * _VIRTUAL_FLOPS_CONSTANT

    def get_device_flops_per_second(self) -> float:
        """Return CONSTANT (not a real hardware spec)."""
        return _VIRTUAL_FLOPS_CONSTANT

    def get_effective_flops_per_second(self, utilization: float = 0.8) -> float:
        """Return CONSTANT — utilization is already baked into the profiled data."""
        return _VIRTUAL_FLOPS_CONSTANT

    def compute_decode_flops(self, request: OutsourcingRequestInfo, num_tokens: int) -> float:
        """Virtual decode FLOPs for *num_tokens* decode steps.

        Uses the KV cache size from the request to look up per-token decode
        latency, then multiplies by *num_tokens*.
        """
        kv_cache_size = request.num_prompt_tokens + request.num_processed_tokens
        per_token_s = self.decode_time_seconds(kv_cache_size)
        return per_token_s * num_tokens * _VIRTUAL_FLOPS_CONSTANT
