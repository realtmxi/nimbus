"""TTFT (Time-to-First-Token) violation detection for outsourcing decisions.

Ported from vidur's outsourcing module.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from nimbus.flop_calculator import FLOPCalculatorInterface
from nimbus.request import OutsourcingRequestInfo
import logging
def get_logger(name): return logging.getLogger(name)

logger = get_logger(__name__)


@dataclass
class ViolationCheckResult:
    """Structured result from TTFT violation detection.

    Implements ``__bool__`` so existing ``if check_violations(...)`` code
    continues to work without changes.
    """

    has_violation: bool
    trigger: str = "none"  # "none" | "flop_model"
    per_request_estimates: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.has_violation


class TTFTViolationDetector:
    """Detect imminent TTFT SLO violations."""

    def __init__(
        self,
        flop_calculator: FLOPCalculatorInterface,
        mode: str = "all",
        prefill_throughput: float = 1000.0,
        max_micro_batch_size: int = 256,
        utilization_target: float = 0.8,
        default_slo_seconds: float | None = None,
    ):
        """Initialize the violation detector.

        Args:
            flop_calculator: FLOP calculator for throughput estimation
            mode: Detection mode ("all" or "head")
            prefill_throughput: Estimated prefill throughput (tokens/sec).
                               Can be updated later via set_prefill_throughput().
            max_micro_batch_size: Maximum micro-batch size
            utilization_target: Target utilization for effective FLOPS calculation
            default_slo_seconds: Default TTFT SLO used when a request does not
                carry an explicit per-request prefill SLO.
        """
        self._flop_calculator = flop_calculator
        self._mode = mode
        self._prefill_throughput = prefill_throughput
        self._max_micro_batch_size = max_micro_batch_size
        self._utilization_target = utilization_target
        self._default_slo_seconds = default_slo_seconds
        self._detector_func = self._get_detector_function(mode)

    def set_prefill_throughput(self, throughput: float) -> None:
        """Update the prefill throughput estimate.

        Args:
            throughput: New prefill throughput estimate (tokens/sec)
        """
        self._prefill_throughput = throughput

    def _get_detector_function(self, mode: str) -> Callable:
        """Get the appropriate detector function based on mode."""
        detectors = {
            "all": self._check_all_violations,
            "head": self._check_head_violation,
        }
        if mode not in detectors:
            raise ValueError(
                f"Unknown TTFT violation mode: {mode}. " f"Choose from: {list(detectors.keys())}"
            )
        return detectors[mode]

    def check_violations(
        self,
        waiting_requests: list[OutsourcingRequestInfo],
        current_time: float,
    ) -> ViolationCheckResult:
        """Check if there are any TTFT violations.

        Args:
            waiting_requests: List of waiting requests
            current_time: Current wall-clock time

        Returns:
            ViolationCheckResult with violation status, trigger, and per-request estimates.
            Supports ``bool()`` for backward compatibility.
        """
        return self._detector_func(waiting_requests, current_time)

    def _check_all_violations(
        self,
        waiting_requests: list[OutsourcingRequestInfo],
        current_time: float,
    ) -> ViolationCheckResult:
        """Check EVERY waiting request for imminent TTFT violation under FCFS.

        Returns ViolationCheckResult with per-request TTFT estimates.
        """
        _no_violation = ViolationCheckResult(
            has_violation=False,
            trigger="none",
        )
        if not waiting_requests:
            return _no_violation

        # Get effective FLOPS per second
        effective_flops = self._flop_calculator.get_effective_flops_per_second(
            self._utilization_target
        )
        if effective_flops <= 0:
            return _no_violation

        # Compute remaining prefill FLOPs for each request
        rem_flops = []
        for r in waiting_requests:
            rem_tokens = r.remaining_prompt_tokens
            flops = self._flop_calculator.compute_prefill_flops(r, rem_tokens)
            rem_flops.append(flops)

        # Prefix sum: FLOP work ahead of each request in FCFS order
        ahead = [0.0] * len(waiting_requests)
        acc = 0.0
        for i in range(len(waiting_requests)):
            ahead[i] = acc
            acc += rem_flops[i]

        # Evaluate every request and collect per-request estimates
        at_risk: set[str] = set()
        per_request_estimates: list[dict] = []

        for i, r in enumerate(waiting_requests):
            est_ttft = (ahead[i] + rem_flops[i]) / effective_flops
            deadline = self._get_prefill_deadline(r)
            time_left = deadline - current_time if deadline is not None else float("inf")
            is_at_risk = False

            if deadline is not None and est_ttft > time_left:
                at_risk.add(r.request_id)
                is_at_risk = True
                logger.debug(
                    f"Request {r.request_id}: est_ttft={est_ttft:.2f}s > "
                    f"time_left={time_left:.2f}s (VIOLATION)"
                )

            per_request_estimates.append(
                {
                    "request_id": r.request_id,
                    "est_ttft": est_ttft,
                    "time_left": time_left,
                    "at_risk": is_at_risk,
                }
            )

        has_violation = len(at_risk) > 0
        trigger = "none"
        if has_violation:
            trigger = "flop_model"

        return ViolationCheckResult(
            has_violation=has_violation,
            trigger=trigger,
            per_request_estimates=per_request_estimates,
        )

    def _check_head_violation(
        self,
        waiting_requests: list[OutsourcingRequestInfo],
        current_time: float,
    ) -> ViolationCheckResult:
        """Check if the head request has imminent TTFT violation.

        Returns ViolationCheckResult with head request estimate.
        """
        _no_violation = ViolationCheckResult(
            has_violation=False,
            trigger="none",
        )
        if not waiting_requests:
            return _no_violation

        head = waiting_requests[0]
        deadline = self._get_prefill_deadline(head)
        if deadline is None:
            return _no_violation

        est_ttft = self._estimate_fcfs_ttft(head, waiting_requests)
        time_left = deadline - current_time
        has_violation = est_ttft > time_left

        return ViolationCheckResult(
            has_violation=has_violation,
            trigger="flop_model" if has_violation else "none",
            per_request_estimates=[
                {
                    "request_id": head.request_id,
                    "est_ttft": est_ttft,
                    "time_left": time_left,
                    "at_risk": has_violation,
                }
            ],
        )

    def _get_prefill_deadline(self, request: OutsourcingRequestInfo) -> float | None:
        """Return the request deadline using explicit or model-level default SLO."""
        if request.prefill_deadline is not None:
            return request.prefill_deadline
        if self._default_slo_seconds is None:
            return None
        return request.arrival_time + self._default_slo_seconds

    def _estimate_fcfs_ttft(
        self,
        req: OutsourcingRequestInfo,
        waiting_requests: list[OutsourcingRequestInfo],
    ) -> float:
        """Estimate Time-to-First-Token under FCFS assumption.

        Returns: queueing delay + own prefill time (in seconds).
        """
        effective_flops = self._flop_calculator.get_effective_flops_per_second(
            self._utilization_target
        )
        if effective_flops <= 0:
            return float("inf")

        # Sum remaining prefill FLOPs of all waiting requests ahead of this one
        ahead_flops = 0.0
        for r in waiting_requests:
            if r.request_id == req.request_id:
                break
            rem_tokens = r.remaining_prompt_tokens
            ahead_flops += self._flop_calculator.compute_prefill_flops(r, rem_tokens)

        # Own prefill work
        rem_self = req.remaining_prompt_tokens
        self_flops = self._flop_calculator.compute_prefill_flops(req, rem_self)

        # Convert to seconds
        est = (ahead_flops + self_flops) / effective_flops
        return est
