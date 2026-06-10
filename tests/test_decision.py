import unittest
from collections import deque

from nimbus.decision import OutsourcingEngine
from nimbus.flop_calculator import SimpleFLOPCalculator
from nimbus.queue import WaitingQueueInterface
from nimbus.request import OutsourcingRequestInfo


class InMemoryWaitingQueue(WaitingQueueInterface):
    def __init__(self, requests=None):
        self._queue = deque(requests or [])

    def add_request(self, request_info: OutsourcingRequestInfo) -> None:
        self._queue.append(request_info)

    def get_all_waiting(self) -> list[OutsourcingRequestInfo]:
        return list(self._queue)

    def remove_requests(self, request_ids: set[str]) -> list[OutsourcingRequestInfo]:
        removed = []
        kept = deque()
        for req in self._queue:
            if req.request_id in request_ids:
                removed.append(req)
            else:
                kept.append(req)
        self._queue = kept
        return removed

    def get_length(self) -> int:
        return len(self._queue)

    def peek(self) -> OutsourcingRequestInfo | None:
        return self._queue[0] if self._queue else None


def make_request(
    request_id: str,
    prompt_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    processed_tokens: int = 0,
) -> OutsourcingRequestInfo:
    return OutsourcingRequestInfo(
        request_id=request_id,
        arrival_time=0.0,
        num_prompt_tokens=prompt_tokens,
        num_output_tokens=output_tokens,
        num_cached_tokens=cached_tokens,
        num_processed_tokens=processed_tokens,
    )


def make_engine(queue, **kwargs) -> OutsourcingEngine:
    return OutsourcingEngine(
        waiting_queue=queue,
        flop_calculator=SimpleFLOPCalculator(device_tflops=1.0),
        **kwargs,
    )


class OutsourcingEngineWeightTests(unittest.TestCase):
    def test_v1_cache_displacement_weight_matches_notion_definition(self):
        req = make_request("r1", prompt_tokens=100, output_tokens=7)
        engine = make_engine(
            InMemoryWaitingQueue([req]),
            weight_mode="v1_cache_displacement",
        )

        self.assertEqual(engine._calculate_weight(req), 700.0)

    def test_v2_token_seconds_uses_remaining_prefill_and_prompt_footprint(self):
        req = make_request(
            "r1",
            prompt_tokens=100,
            output_tokens=4,
            cached_tokens=40,
            processed_tokens=10,
        )
        engine = make_engine(
            InMemoryWaitingQueue([req]),
            weight_mode="v2_token_seconds",
            prefill_throughput_tokens_per_s=100.0,
            tpot_seconds=0.5,
        )

        # remaining_prefill = 100 - 40 - 10 = 50
        # weight = prompt_tokens * (remaining_prefill / prefill_tput + decode * TPOT)
        self.assertEqual(engine._calculate_weight(req), 100 * (50 / 100 + 4 * 0.5))

    def test_unknown_weight_mode_fails_fast(self):
        req = make_request("r1", prompt_tokens=10, output_tokens=10)
        engine = make_engine(InMemoryWaitingQueue([req]), weight_mode="unknown")

        with self.assertRaises(ValueError):
            engine._calculate_weight(req)

    def test_knapsack_value_uses_remote_cached_api_cost_not_local_cache(self):
        req = make_request(
            "r1",
            prompt_tokens=1_000_000,
            output_tokens=0,
            cached_tokens=900_000,
        )
        req.metadata["remote_cached_tokens"] = 400_000
        engine = make_engine(
            InMemoryWaitingQueue([req]),
            weight_mode="v1_cache_displacement",
            input_price_per_million=1.0,
            cached_input_price_per_million=0.25,
            output_price_per_million=0.0,
        )

        item = engine._build_knapsack_item(req)

        # API cost if outsourced:
        # 600K uncached input at $1/M + 400K cached input at $0.25/M = $0.70.
        # Value is micro-dollars, so $0.70 -> 700000.
        self.assertEqual(item["value"], 700_000)


class OutsourcingEngineKvBudgetTests(unittest.TestCase):
    def test_kv_time_budget_outsources_until_remaining_queue_fits(self):
        requests = [
            make_request("big", prompt_tokens=100, output_tokens=10),
            make_request("medium", prompt_tokens=80, output_tokens=10),
            make_request("small", prompt_tokens=20, output_tokens=1),
        ]
        queue = InMemoryWaitingQueue(requests)
        engine = make_engine(
            queue,
            weight_mode="v2_token_seconds",
            prefill_throughput_tokens_per_s=1e30,
            tpot_seconds=1.0,
            input_price_per_million=0.0,
            output_price_per_million=0.0,
        )

        decision = engine.decide_by_kv_time_budget(
            kv_avail_tokens=900,
            horizon_seconds=1.0,
        )

        self.assertTrue(decision.should_outsource)
        self.assertEqual(decision.requests_to_outsource, ["big"])
        self.assertEqual(set(decision.requests_to_keep), {"medium", "small"})
        self.assertEqual(queue.get_length(), 2)
        self.assertEqual(decision.metrics["budget_token_seconds"], 900)
        self.assertEqual(decision.metrics["weight_mode"], "v2_token_seconds")

    def test_kv_time_budget_keeps_all_when_queue_fits(self):
        requests = [
            make_request("r1", prompt_tokens=10, output_tokens=2),
            make_request("r2", prompt_tokens=20, output_tokens=2),
        ]
        queue = InMemoryWaitingQueue(requests)
        engine = make_engine(
            queue,
            weight_mode="v2_token_seconds",
            prefill_throughput_tokens_per_s=1e30,
            tpot_seconds=1.0,
        )

        decision = engine.decide_by_kv_time_budget(
            kv_avail_tokens=100,
            horizon_seconds=1.0,
        )

        self.assertFalse(decision.should_outsource)
        self.assertEqual(decision.requests_to_outsource, [])
        self.assertEqual(set(decision.requests_to_keep), {"r1", "r2"})
        self.assertEqual(queue.get_length(), 2)

    def test_kv_time_budget_outsources_individually_infeasible_prefill(self):
        requests = [
            make_request("slow", prompt_tokens=200, output_tokens=1),
            make_request("fast", prompt_tokens=10, output_tokens=1),
        ]
        queue = InMemoryWaitingQueue(requests)
        engine = make_engine(
            queue,
            weight_mode="v2_token_seconds",
            prefill_throughput_tokens_per_s=100.0,
            tpot_seconds=0.0,
            input_price_per_million=0.0,
            output_price_per_million=0.0,
        )

        decision = engine.decide_by_kv_time_budget(
            kv_avail_tokens=1_000_000,
            horizon_seconds=1.0,
        )

        self.assertTrue(decision.should_outsource)
        self.assertEqual(decision.requests_to_outsource, ["slow"])
        self.assertEqual(decision.requests_to_keep, ["fast"])
        self.assertEqual(queue.get_length(), 1)

    def test_kv_time_budget_outsources_request_whose_wait_exhausts_deadline(self):
        waiting = make_request("waiting", prompt_tokens=10, output_tokens=1)
        waiting.arrival_time = 10.0
        fresh = make_request("fresh", prompt_tokens=10, output_tokens=1)
        fresh.arrival_time = 19.9
        queue = InMemoryWaitingQueue([waiting, fresh])
        engine = make_engine(
            queue,
            weight_mode="v2_token_seconds",
            prefill_throughput_tokens_per_s=100.0,
            tpot_seconds=0.0,
        )

        decision = engine.decide_by_kv_time_budget(
            kv_avail_tokens=1_000_000,
            horizon_seconds=1.0,
            current_time=20.0,
        )

        self.assertEqual(decision.requests_to_outsource, ["waiting"])
        self.assertEqual(decision.requests_to_keep, ["fresh"])
        self.assertEqual(queue.get_length(), 1)

    def test_kv_time_budget_uses_deadline_guard_for_cloud_handoff(self):
        waiting = make_request("waiting", prompt_tokens=10, output_tokens=1)
        waiting.arrival_time = 0.0
        queue = InMemoryWaitingQueue([waiting])
        engine = make_engine(
            queue,
            weight_mode="v2_token_seconds",
            prefill_throughput_tokens_per_s=1000.0,
            tpot_seconds=0.0,
        )

        decision = engine.decide_by_kv_time_budget(
            kv_avail_tokens=1_000_000,
            horizon_seconds=1.0,
            current_time=0.8,
            deadline_guard_seconds=0.3,
        )

        self.assertEqual(decision.requests_to_outsource, ["waiting"])
        self.assertEqual(queue.get_length(), 0)

    def test_zero_kv_time_budget_outsources_every_waiting_request(self):
        requests = [
            make_request("r1", prompt_tokens=1, output_tokens=1),
            make_request("r2", prompt_tokens=1, output_tokens=1),
        ]
        queue = InMemoryWaitingQueue(requests)
        engine = make_engine(
            queue,
            weight_mode="v2_token_seconds",
            prefill_throughput_tokens_per_s=1e30,
            tpot_seconds=1.0,
        )

        decision = engine.decide_by_kv_time_budget(
            kv_avail_tokens=0,
            horizon_seconds=1.0,
        )

        self.assertTrue(decision.should_outsource)
        self.assertEqual(set(decision.requests_to_outsource), {"r1", "r2"})
        self.assertEqual(decision.requests_to_keep, [])
        self.assertEqual(queue.get_length(), 0)
        self.assertEqual(decision.metrics["budget_token_seconds"], 0)

    def test_kv_time_budget_can_limit_iterations_per_call(self):
        requests = [
            make_request("r1", prompt_tokens=1, output_tokens=1),
            make_request("r2", prompt_tokens=1, output_tokens=1),
        ]
        queue = InMemoryWaitingQueue(requests)
        engine = make_engine(
            queue,
            weight_mode="v2_token_seconds",
            prefill_throughput_tokens_per_s=1e30,
            tpot_seconds=1.0,
        )

        decision = engine.decide_by_kv_time_budget(
            kv_avail_tokens=0,
            horizon_seconds=1.0,
            max_iterations=1,
        )

        self.assertEqual(len(decision.requests_to_outsource), 1)
        self.assertEqual(queue.get_length(), 1)
        self.assertEqual(decision.metrics["iterations"], 1)

    def test_outsourcing_statistics_use_remote_cached_api_cost(self):
        req = make_request("r1", prompt_tokens=1_000_000, output_tokens=0)
        req.metadata["remote_cached_tokens"] = 400_000
        queue = InMemoryWaitingQueue([req])
        engine = make_engine(
            queue,
            weight_mode="v1_cache_displacement",
            input_price_per_million=1.0,
            cached_input_price_per_million=0.25,
            output_price_per_million=0.0,
        )

        engine.decide_by_kv_time_budget(kv_avail_tokens=0, horizon_seconds=1.0)
        stats = engine.get_outsourcing_statistics()

        self.assertEqual(stats["total_remote_cached_tokens"], 400_000)
        self.assertAlmostEqual(stats["total_api_cost_usd"], 0.7)


if __name__ == "__main__":
    unittest.main()
