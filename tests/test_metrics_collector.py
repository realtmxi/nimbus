import unittest

from experiments.metrics_collector import CSV_COLUMNS, parse_metrics_text


class MetricsCollectorTests(unittest.TestCase):
    def test_parse_vllm_metrics(self):
        values = parse_metrics_text(
            'vllm:kv_cache_usage_perc{model_name="qwen2.5"} 0.375\n'
            "vllm:num_requests_running 3\n"
            "vllm:num_requests_waiting 2\n"
            "vllm:avg_generation_throughput_toks_per_s 412.5\n"
        )

        self.assertEqual(values["kv_cache_usage_pct"], 0.375)
        self.assertEqual(values["num_running"], 3.0)
        self.assertEqual(values["num_queued"], 2.0)
        self.assertEqual(values["gen_throughput"], 412.5)

    def test_parse_sglang_metrics_still_supported(self):
        values = parse_metrics_text(
            "sglang:num_used_tokens 1234\n"
            "sglang:max_total_num_tokens 10000\n"
            "sglang:num_queue_reqs 4\n"
        )

        self.assertEqual(values["num_used_tokens"], 1234.0)
        self.assertEqual(values["max_total_tokens"], 10000.0)
        self.assertEqual(values["num_queued"], 4.0)

    def test_csv_columns_are_unique(self):
        self.assertEqual(len(CSV_COLUMNS), len(set(CSV_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
