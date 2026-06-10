"""Decode TPOT profiling artifact support.

The Notion design allows V2 weights to use a local TPOT that depends on the
predicted decode batch size. This module keeps the artifact format small and
testable so experiment scripts do not need ad hoc JSON parsing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TPOTPoint:
    batch_size: int
    tpot_seconds: float


@dataclass(frozen=True)
class TPOTProfile:
    deployment: dict[str, Any]
    points: tuple[TPOTPoint, ...]
    prefill_throughput_tokens_per_s: float | None = None
    b_sweet: int | None = None
    source: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "TPOTProfile":
        source = str(path)
        with open(path) as f:
            raw = json.load(f)
        return cls.from_dict(raw, source=source)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str | None = None) -> "TPOTProfile":
        raw_points = raw.get("tpots")
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError("TPOT profile must contain a non-empty `tpots` list")

        points: list[TPOTPoint] = []
        for item in raw_points:
            try:
                batch_size = int(item["batch_size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Each TPOT point must include integer `batch_size`") from exc
            if batch_size <= 0:
                raise ValueError("TPOT `batch_size` must be positive")

            if "tpot_s" in item:
                tpot_seconds = float(item["tpot_s"])
            elif "tpot_ms" in item:
                tpot_seconds = float(item["tpot_ms"]) / 1000.0
            else:
                raise ValueError("Each TPOT point must include `tpot_s` or `tpot_ms`")
            if tpot_seconds <= 0:
                raise ValueError("TPOT must be positive")

            points.append(TPOTPoint(batch_size=batch_size, tpot_seconds=tpot_seconds))

        points.sort(key=lambda point: point.batch_size)
        for prev, cur in zip(points, points[1:]):
            if prev.batch_size == cur.batch_size:
                raise ValueError(f"Duplicate TPOT batch_size: {cur.batch_size}")

        prefill_tput = raw.get("prefill_throughput_tokens_per_s")
        b_sweet = raw.get("b_sweet")
        return cls(
            deployment=dict(raw.get("deployment") or {}),
            points=tuple(points),
            prefill_throughput_tokens_per_s=(
                None if prefill_tput is None else float(prefill_tput)
            ),
            b_sweet=None if b_sweet is None else int(b_sweet),
            source=source,
        )

    def tpot_seconds_for_batch(self, batch_size: int | float) -> float:
        """Return a conservative stepwise TPOT for the predicted batch size.

        The first profiled point whose batch size is >= the requested batch is
        used. Above the largest profiled batch, use the largest point rather
        than extrapolating.
        """
        batch = max(1, int(batch_size))
        for point in self.points:
            if batch <= point.batch_size:
                return point.tpot_seconds
        return self.points[-1].tpot_seconds

    def label(self) -> str:
        if not self.deployment:
            return self.source or "profile"
        keys = ["gpu", "model", "engine", "mtp_mode"]
        parts = [str(self.deployment[key]) for key in keys if self.deployment.get(key)]
        return "/".join(parts) if parts else (self.source or "profile")
