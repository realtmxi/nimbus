import unittest
from argparse import Namespace

from experiments.run_engine import (
    KVMetricsState,
    LocalAdmissionState,
    RealCloud,
    SimCloud,
    _effective_tpot_seconds,
    _estimate_decode_batch_size,
    _loop_sleep_seconds,
    _mock_service_times,
    _parse_kv_from_metrics,
    _queue_delay_ms,
    synthetic_burst_trace,
)
from nimbus.flop_calculator import SimpleFLOPCalculator
from nimbus.tpot_profile import TPOTProfile
from nimbus.request import OutsourcingRequestInfo


def make_request(request_id: str, session_id: int, prompt_tokens: int) -> OutsourcingRequestInfo:
    req = OutsourcingRequestInfo(
        request_id=request_id,
        arrival_time=0.0,
        num_prompt_tokens=prompt_tokens,
        num_output_tokens=100,
    )
    req.metadata["session_id"] = session_id
    return req


class RealCloudAccountingTests(unittest.TestCase):
    """Lock the cost/cache semantics of the --cloud real sink (no network).

    The streaming TTFT path is exercised as an integration test on a host with
    aiohttp + a live endpoint; here we pin the billing math and the remote
    prefix-cache (TTL) model that make sim/real cost comparable.
    """

    def _cloud(self, ttl: float = 0.0) -> RealCloud:
        return RealCloud(
            url="http://unused",
            model="m",
            api_key_env=None,
            in_price=1.0,
            cached_in_price=0.5,
            out_price=2.0,
            remote_cache_ttl_s=ttl,
        )

    def test_cost_dict_bills_uncached_input_and_output(self):
        out = self._cloud().cost_dict(
            prompt_tokens=1_000_000, cached_input_tokens=0,
            output_tokens=1_000_000, ttft_ms=120.0, success=True,
        )
        self.assertTrue(out["success"])
        self.assertEqual(out["ttft_ms"], 120.0)
        self.assertAlmostEqual(out["cost_usd"], 3.0, places=6)  # 1M*$1 + 1M*$2
        self.assertEqual(out["uncached_input_tokens"], 1_000_000)
        self.assertEqual(out["output_tokens"], 1_000_000)

    def test_cost_dict_discounts_cached_input(self):
        out = self._cloud().cost_dict(
            prompt_tokens=1_000_000, cached_input_tokens=1_000_000,
            output_tokens=0, ttft_ms=50.0, success=True,
        )
        self.assertAlmostEqual(out["cost_usd"], 0.5, places=6)  # 1M cached @ $0.5
        self.assertEqual(out["cached_input_tokens"], 1_000_000)
        self.assertEqual(out["remote_cached_tokens"], 1_000_000)

    def test_success_requires_first_token(self):
        out = self._cloud().cost_dict(
            prompt_tokens=10, cached_input_tokens=0,
            output_tokens=0, ttft_ms=None, success=True,
        )
        self.assertFalse(out["success"])  # no measured TTFT -> SLO miss

    def test_remote_cache_ttl_window(self):
        cloud = self._cloud(ttl=100.0)
        first = make_request("r1", session_id=7, prompt_tokens=500)
        self.assertEqual(cloud.estimate_remote_cached_tokens(first, now=0.0), 0)
        cloud._remember_remote_prompt(first, now=0.0)
        repeat = make_request("r2", session_id=7, prompt_tokens=500)
        self.assertEqual(cloud.estimate_remote_cached_tokens(repeat, now=50.0), 500)
        self.assertEqual(cloud.estimate_remote_cached_tokens(repeat, now=200.0), 0)

    def test_no_ttl_disables_cache(self):
        cloud = self._cloud(ttl=0.0)
        req = make_request("r1", session_id=7, prompt_tokens=500)
        cloud._remember_remote_prompt(req, now=0.0)
        self.assertEqual(cloud.estimate_remote_cached_tokens(req, now=1.0), 0)


class SimCloudRemoteCacheTests(unittest.TestCase):
    def test_remote_cache_ttl_discounts_repeated_session_prefix(self):
        cloud = SimCloud(
            in_price=1.0,
            cached_in_price=0.25,
            out_price=10.0,
            ttft_mean_ms=0.0,
            remote_cache_ttl_s=60.0,
            jitter=0.0,
            seed=0,
        )

        first = cloud.serve(make_request("r1", session_id=7, prompt_tokens=1_000), now=0.0)
        second = cloud.serve(make_request("r2", session_id=7, prompt_tokens=1_200), now=10.0)

        self.assertEqual(first["remote_cached_tokens"], 0)
        self.assertEqual(second["remote_cached_tokens"], 1_000)
        self.assertEqual(second["uncached_input_tokens"], 200)
        self.assertEqual(second["cached_input_tokens"], 1_000)

    def test_remote_cache_ttl_expiry_falls_back_to_uncached_input(self):
        cloud = SimCloud(
            in_price=1.0,
            cached_in_price=0.25,
            out_price=10.0,
            ttft_mean_ms=0.0,
            remote_cache_ttl_s=5.0,
            jitter=0.0,
            seed=0,
        )

        cloud.serve(make_request("r1", session_id=7, prompt_tokens=1_000), now=0.0)
        second = cloud.serve(make_request("r2", session_id=7, prompt_tokens=1_200), now=10.0)

        self.assertEqual(second["remote_cached_tokens"], 0)
        self.assertEqual(second["uncached_input_tokens"], 1_200)

    def test_explicit_remote_cached_tokens_override_ttl_estimate(self):
        cloud = SimCloud(
            in_price=1.0,
            cached_in_price=0.25,
            out_price=10.0,
            ttft_mean_ms=0.0,
            remote_cache_ttl_s=60.0,
            jitter=0.0,
            seed=0,
        )

        cloud.serve(make_request("r1", session_id=7, prompt_tokens=1_000), now=0.0)
        second = make_request("r2", session_id=7, prompt_tokens=1_200)
        second.metadata["remote_cached_tokens"] = 100

        out = cloud.serve(second, now=10.0)

        self.assertEqual(out["remote_cached_tokens"], 100)
        self.assertEqual(out["uncached_input_tokens"], 1_100)


class KVMetricsStateTests(unittest.TestCase):
    def test_parse_kv_from_metrics_reads_used_and_capacity(self):
        used, max_tokens = _parse_kv_from_metrics(
            "# HELP sglang:num_used_tokens current used tokens\n"
            "sglang:num_used_tokens 123\n"
            "sglang:max_total_num_tokens 1000\n"
        )

        self.assertEqual(used, 123.0)
        self.assertEqual(max_tokens, 1000.0)

    def test_parse_vllm_kv_from_metrics_uses_configured_capacity(self):
        used, max_tokens = _parse_kv_from_metrics(
            'vllm:kv_cache_usage_perc{model_name="qwen3-32b"} 0.25\n',
            serving_engine="vllm",
            fallback_max_tokens=2000,
        )

        self.assertEqual(used, 500.0)
        self.assertEqual(max_tokens, 2000.0)

    def test_missing_used_metric_is_invalid(self):
        with self.assertRaises(ValueError):
            _parse_kv_from_metrics("sglang:max_total_num_tokens 1000\n")

    def test_missing_vllm_kv_usage_is_invalid(self):
        with self.assertRaises(ValueError):
            _parse_kv_from_metrics(
                "vllm:num_requests_running 1\n",
                serving_engine="vllm",
                fallback_max_tokens=1000,
            )

    def test_metrics_fallback_uses_last_good_sample(self):
        state = KVMetricsState(fallback_max_tokens=1000)
        self.assertEqual(state.record(used_tokens=250, max_tokens=900), (250.0, 900.0))

        self.assertEqual(state.fallback_on_error(), (250.0, 900.0))
        self.assertEqual(state.failures, 1)

    def test_metrics_fallback_fails_closed_before_first_sample(self):
        state = KVMetricsState(fallback_max_tokens=1000)

        self.assertEqual(state.fallback_on_error(), (1000.0, 1000.0))
        self.assertEqual(state.failures, 1)

    def test_metrics_record_falls_back_to_configured_capacity_when_max_missing(self):
        state = KVMetricsState(fallback_max_tokens=1000)

        self.assertEqual(state.record_text("sglang:num_used_tokens 250\n"), (250.0, 1000.0))

    def test_metrics_state_parses_vllm_usage(self):
        state = KVMetricsState(fallback_max_tokens=1000, serving_engine="vllm")

        self.assertEqual(
            state.record_text('vllm:kv_cache_usage_perc{model_name="qwen3-32b"} 0.75\n'),
            (750.0, 1000.0),
        )


class ReplayAccountingTests(unittest.TestCase):
    def test_synthetic_burst_stub_prompt_keeps_tiny_text(self):
        trace = synthetic_burst_trace(5, seed=0, prompt_mode="stub", prompt_token_cap=128)

        self.assertTrue(all(row["prompt_text"] == "ping" for row in trace))
        self.assertTrue(any(row["num_prefill_tokens"] > 128 for row in trace))

    def test_synthetic_burst_sized_prompt_caps_metadata_and_text(self):
        trace = synthetic_burst_trace(20, seed=0, prompt_mode="sized", prompt_token_cap=128)

        self.assertTrue(all(row["num_prefill_tokens"] <= 128 for row in trace))
        self.assertTrue(all(len(row["prompt_text"].split()) == row["num_prefill_tokens"] for row in trace))

    def test_local_admission_state_reserves_and_releases_mock_kv_synchronously(self):
        req = make_request("r1", session_id=1, prompt_tokens=800)
        state = LocalAdmissionState()

        state.reserve(req, local_mode="mock")

        self.assertEqual(state.inflight, 1)
        self.assertEqual(state.mock_kv_used, 800)

        state.release(req, local_mode="mock")

        self.assertEqual(state.inflight, 0)
        self.assertEqual(state.mock_kv_used, 0)

    def test_local_admission_state_real_mode_only_tracks_inflight(self):
        req = make_request("r1", session_id=1, prompt_tokens=800)
        state = LocalAdmissionState()

        state.reserve(req, local_mode="real")

        self.assertEqual(state.inflight, 1)
        self.assertEqual(state.mock_kv_used, 0)

    def test_queue_delay_scales_only_when_requested(self):
        req = make_request("r1", session_id=1, prompt_tokens=100)
        req.arrival_time = 10.0

        self.assertEqual(
            _queue_delay_ms(req, now=10.25, time_scale=1.0, scale_wait=False),
            250.0,
        )
        self.assertEqual(
            _queue_delay_ms(req, now=10.25, time_scale=4.0, scale_wait=True),
            1000.0,
        )

    def test_loop_sleep_scales_down_for_mock_replay(self):
        self.assertEqual(_loop_sleep_seconds(0.1, time_scale=10.0, scale_time=True), 0.01)
        self.assertEqual(_loop_sleep_seconds(0.1, time_scale=10.0, scale_time=False), 0.1)

    def test_effective_tpot_uses_profile_when_present(self):
        profile = TPOTProfile.from_dict(
            {
                "tpots": [
                    {"batch_size": 1, "tpot_ms": 10},
                    {"batch_size": 8, "tpot_ms": 25},
                ]
            }
        )

        self.assertEqual(
            _effective_tpot_seconds(profile, fallback_tpot_s=0.03, predicted_decode_batch_size=4),
            0.025,
        )

    def test_effective_tpot_falls_back_without_profile(self):
        self.assertEqual(
            _effective_tpot_seconds(None, fallback_tpot_s=0.03, predicted_decode_batch_size=4),
            0.03,
        )

    def test_estimate_decode_batch_counts_only_immediately_admissible_waiting(self):
        state = LocalAdmissionState()
        state.inflight = 2
        waiting = [
            make_request("w1", session_id=1, prompt_tokens=100),
            make_request("w2", session_id=1, prompt_tokens=100),
            make_request("w3", session_id=1, prompt_tokens=100),
        ]

        batch = _estimate_decode_batch_size(
            state,
            waiting,
            used_tokens=0,
            max_tokens=1000,
            admit_kv=1.0,
            max_inflight=4,
        )

        self.assertEqual(batch, 4)

    def test_estimate_decode_batch_respects_kv_headroom(self):
        state = LocalAdmissionState()
        waiting = [
            make_request("w1", session_id=1, prompt_tokens=200),
            make_request("w2", session_id=1, prompt_tokens=200),
        ]

        batch = _estimate_decode_batch_size(
            state,
            waiting,
            used_tokens=750,
            max_tokens=1000,
            admit_kv=0.9,
            max_inflight=8,
        )

        self.assertEqual(batch, 1)

    def test_mock_service_v2_separates_ttft_from_decode_occupancy(self):
        req = make_request("r1", session_id=1, prompt_tokens=100)
        req.num_output_tokens = 10
        args = Namespace(mock_service_model="v2", prefill_tput=50.0, tpot_s=0.25)

        ttft_s, occupancy_s = _mock_service_times(
            req,
            args,
            SimpleFLOPCalculator(device_tflops=1.0),
            eff_flops=1.0,
            tpot_profile=None,
            decode_batch_size=1,
        )

        self.assertEqual(ttft_s, 2.0)
        self.assertEqual(occupancy_s, 4.5)

    def test_mock_service_v2_uses_effective_prefill_tput_when_present(self):
        req = make_request("r1", session_id=1, prompt_tokens=100)
        req.num_output_tokens = 1
        args = Namespace(
            mock_service_model="v2",
            prefill_tput=50.0,
            effective_prefill_tput=200.0,
            tpot_s=0.0,
        )

        ttft_s, occupancy_s = _mock_service_times(
            req,
            args,
            SimpleFLOPCalculator(device_tflops=1.0),
            eff_flops=1.0,
            tpot_profile=None,
            decode_batch_size=1,
        )

        self.assertEqual(ttft_s, 0.5)
        self.assertEqual(occupancy_s, 0.5)


if __name__ == "__main__":
    unittest.main()
