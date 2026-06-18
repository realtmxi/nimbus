"""Outsourcing decision engine and decision dataclass.

Uses vidur-style outsourcing components:
- APICostCalculator for pricing
- CandidateSelector for request filtering
- KnapsackSolver for optimization
- RequestTracker for metrics

The live decision entry is :meth:`OutsourcingEngine.decide_by_kv_time_budget`
— a no-prediction loop that outsources from a token-second KV budget using
cache-displacement weights. (The legacy FLOP/TTFT-predictor decision path and
its TTFTViolationDetector were removed; `--weight v0` still selects the FLOP
weight via :meth:`_calculate_weight`.)
"""

from dataclasses import dataclass, field

from .candidate_selection import CandidateSelector
from .cost_calculator import APICostCalculator
from .flop_calculator import FLOPCalculatorInterface
from .knapsack import KnapsackSolver
from .queue import WaitingQueueInterface
from .request import OutsourcingRequestInfo
from .request_tracker import RequestTracker


@dataclass
class OutsourcingDecision:
    """Result of an outsourcing decision cycle."""

    should_outsource: bool
    requests_to_outsource: list[str]  # Request IDs
    requests_to_keep: list[str]  # Request IDs
    reason: str  # Human-readable explanation
    metrics: dict = field(default_factory=dict)


class OutsourcingEngine:
    """Main engine that makes outsourcing decisions using vidur-style components.

    Engine-agnostic; decides what to outsource based on:
    - a KV-time pressure budget,
    - cache-displacement knapsack optimization,
    - cost-aware prioritization.
    """

    def __init__(
        self,
        waiting_queue: WaitingQueueInterface,
        flop_calculator: FLOPCalculatorInterface,
        model_id: str = "default",
        utilization_target: float = 0.8,
        input_price_per_million: float = 1.25,
        output_price_per_million: float = 10.00,
        cached_input_price_per_million: float | None = None,
        knapsack_strategy: str = "dp_scaled",
        decode_weight_ratio: float = 0.6,
        max_outsourcing_iterations: int = 100,
        weight_mode: str = "v2_token_seconds",
        prefill_throughput_tokens_per_s: float = 50_000.0,
        tpot_seconds: float = 0.03,
    ):
        """Initialize the outsourcing engine.

        Args:
            waiting_queue: Queue interface for accessing waiting requests.
            flop_calculator: FLOP calculator (used by the v0 FLOP weight).
            model_id: Model identifier for tracking.
            utilization_target: Target utilization for effective FLOPS (v0 path).
            input_price_per_million: API input price per million tokens.
            output_price_per_million: API output price per million tokens.
            cached_input_price_per_million: API cached-input price per million
                tokens. Defaults to uncached input price.
            knapsack_strategy: Strategy for knapsack solver ("dp_scaled", "fractional", ...).
            decode_weight_ratio: v0 weight ratio for decode vs prefill FLOPs.
            max_outsourcing_iterations: Safety cap on the per-call outsource loop.
            weight_mode: Knapsack weight mode:
                "v2_token_seconds" (default), "v1_cache_displacement", or "v0_flops".
            prefill_throughput_tokens_per_s: Local prefill throughput for v2 weights.
            tpot_seconds: Local time per output token for v2 weights.
        """
        self.waiting_queue = waiting_queue
        self.flop_calculator = flop_calculator
        self.model_id = model_id
        self.utilization_target = utilization_target
        self.decode_weight_ratio = decode_weight_ratio
        self.max_outsourcing_iterations = max_outsourcing_iterations
        self.weight_mode = weight_mode
        self.prefill_throughput_tokens_per_s = prefill_throughput_tokens_per_s
        self.tpot_seconds = tpot_seconds

        self.outsourced_request_ids: set[str] = set()

        # vidur-style components
        self._cost_calculator = APICostCalculator(
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            cached_input_price_per_million=cached_input_price_per_million,
        )

        self._request_tracker = RequestTracker(
            model_id=model_id,
            cost_calculator=self._cost_calculator.calculate_cost_with_cache,
        )

        self._candidate_selector = CandidateSelector()

        self._knapsack_solver = KnapsackSolver(strategy=knapsack_strategy)

    def get_outsourcing_statistics(self) -> dict:
        """Calculate and return outsourcing statistics."""
        return self._request_tracker.get_outsourcing_statistics()

    def _build_knapsack_item(self, req: OutsourcingRequestInfo) -> dict:
        """Build a knapsack item for a request.

        Weight: cache displacement by default.
        Value: cost savings from keeping local (API cost avoided)
        """
        weight = self._calculate_weight(req)

        # Value = cost savings from NOT outsourcing (API cost avoided)
        remote_cached_tokens = int(req.metadata.get("remote_cached_tokens") or 0)
        api_cost = self._cost_calculator.calculate_cost_with_cache(
            input_tokens=req.num_prompt_tokens,
            cached_input_tokens=remote_cached_tokens,
            output_tokens=req.num_output_tokens,
        )
        value = int(api_cost * 1_000_000)  # Scale USD to micro-dollars for DP.

        return {"id": req.request_id, "weight": max(1, int(weight)), "value": max(1, value)}

    def _calculate_weight(self, req: OutsourcingRequestInfo) -> float:
        """Return the configured resource-pressure weight for a request."""
        if self.weight_mode == "v1_cache_displacement":
            return float(req.num_prompt_tokens * req.remaining_output_tokens)

        if self.weight_mode == "v2_token_seconds":
            prefill_t = (
                req.remaining_prompt_tokens
                / max(self.prefill_throughput_tokens_per_s, 1e-9)
            )
            decode_t = req.remaining_output_tokens * self.tpot_seconds
            return float(req.num_prompt_tokens * (prefill_t + decode_t))

        if self.weight_mode != "v0_flops":
            raise ValueError(
                "Unknown weight_mode: "
                f"{self.weight_mode}. Choose from v2_token_seconds, "
                "v1_cache_displacement, v0_flops."
            )

        # Legacy v0 baseline: remaining FLOPs (prefill + weighted decode).
        prefill_flops = 0.0
        if req.remaining_prompt_tokens > 0:
            prefill_flops = self.flop_calculator.compute_prefill_flops(
                req, req.remaining_prompt_tokens
            )

        decode_flops = 0.0
        if req.remaining_output_tokens > 0:
            decode_flops = self.flop_calculator.compute_decode_flops(
                req, req.remaining_output_tokens
            )

        return float(prefill_flops + self.decode_weight_ratio * decode_flops)

    def _estimated_prefill_seconds(self, req: OutsourcingRequestInfo) -> float:
        """Estimate the local prefill component used for TTFT feasibility checks."""
        if self.weight_mode == "v0_flops":
            prefill_flops = self.flop_calculator.compute_prefill_flops(
                req,
                req.remaining_prompt_tokens,
            )
            eff_flops = self.flop_calculator.get_effective_flops_per_second(
                self.utilization_target,
            )
            return prefill_flops / max(eff_flops, 1e-9)

        return (
            req.remaining_prompt_tokens
            / max(self.prefill_throughput_tokens_per_s, 1e-9)
        )

    def decide_by_kv_time_budget(
        self,
        kv_avail_tokens: float,
        horizon_seconds: float,
        max_iterations: int | None = None,
        current_time: float | None = None,
        wait_time_scale: float = 1.0,
        deadline_guard_seconds: float = 0.0,
    ) -> OutsourcingDecision:
        """Run the no-FLOP Nimbus loop using a token-second KV budget.

        The budget is ``available_kv_tokens * horizon_seconds``. We repeatedly
        solve the keep-local knapsack and outsource one request from the
        solver's overflow set until the waiting queue fits the budget.
        """
        budget = max(0, int(kv_avail_tokens * max(horizon_seconds, 0.0)))
        iteration_limit = self.max_outsourcing_iterations
        if max_iterations is not None:
            iteration_limit = max(0, min(iteration_limit, max_iterations))
        all_outsourced: list[str] = []
        iteration = 0

        while iteration < iteration_limit:
            waiting_requests = self.waiting_queue.get_all_waiting()
            candidates = self._candidate_selector.collect_candidates(
                waiting_requests=waiting_requests,
                outsourced_req_ids=self.outsourced_request_ids,
            )
            if not candidates:
                break

            def would_miss_deadline(req: OutsourcingRequestInfo) -> bool:
                wait_s = (
                    max(0.0, current_time - req.arrival_time)
                    * max(0.0, wait_time_scale)
                    if current_time is not None
                    else 0.0
                )
                if wait_s + self._estimated_prefill_seconds(req) > horizon_seconds:
                    return True
                return (
                    deadline_guard_seconds > 0
                    and wait_s + deadline_guard_seconds > horizon_seconds
                )

            infeasible = [r for r in candidates if would_miss_deadline(r)]
            if infeasible:
                req_to_outsource = max(
                    infeasible,
                    key=lambda r: (
                        self._estimated_prefill_seconds(r),
                        self._calculate_weight(r),
                    ),
                )
                outsourced = self.waiting_queue.remove_requests({req_to_outsource.request_id})
                for req in outsourced:
                    self.outsourced_request_ids.add(req.request_id)
                    self._request_tracker.track_outsourced_request(req, req.arrival_time)
                    all_outsourced.append(req.request_id)
                iteration += 1
                continue

            items = [self._build_knapsack_item(r) for r in candidates]
            total_weight = sum(item["weight"] for item in items)
            if total_weight <= budget:
                break

            _keep_ids, outsource_ids = self._knapsack_solver.solve(items, budget)
            if not outsource_ids:
                # degenerate fallback: lowest value-density request
                outsource_ids = [
                    min(items, key=lambda item: item["value"] / item["weight"])["id"]
                ]

            by_id = {item["id"]: item for item in items}
            # Kick the lowest value/weight (value-density): the request that frees the most
            # local KV-time per dollar it costs to outsource -- the principled knapsack drop
            # order. (NOT max-weight, which ignores cost; NOT min-value, which ignores capacity.)
            single_outsource = min(
                (by_id[request_id] for request_id in outsource_ids),
                key=lambda item: item["value"] / item["weight"],
            )["id"]
            outsourced = self.waiting_queue.remove_requests({single_outsource})
            for req in outsourced:
                self.outsourced_request_ids.add(req.request_id)
                self._request_tracker.track_outsourced_request(req, req.arrival_time)
                all_outsourced.append(req.request_id)
            iteration += 1

        final_waiting = self.waiting_queue.get_all_waiting()
        keep_ids = [
            r.request_id for r in final_waiting if r.request_id not in self.outsourced_request_ids
        ]

        return OutsourcingDecision(
            should_outsource=bool(all_outsourced),
            requests_to_outsource=all_outsourced,
            requests_to_keep=keep_ids,
            reason=(
                f"KV-time budget outsourcing: {len(all_outsourced)} request(s) "
                f"in {iteration} iteration(s)"
                if all_outsourced
                else "Waiting queue fits KV-time budget"
            ),
            metrics={
                "iterations": iteration,
                "budget_token_seconds": budget,
                "weight_mode": self.weight_mode,
                "outsource_count": len(all_outsourced),
                "keep_count": len(keep_ids),
            },
        )
