import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.plot_engine_sweep import (
    best_slo_rows,
    label_for_row,
    load_engine_summary,
    write_normalized_csv,
)


def write_summary(path: Path, rows: list[dict]) -> None:
    cols = [
        "policy",
        "weight",
        "fraction",
        "slo_s",
        "total",
        "outsourced",
        "outsource_pct",
        "total_cost_usd",
        "remote_cached_input_tokens",
        "remote_cache_hit_pct",
        "metrics_read_failures",
        "serving_engine",
        "tpot_profile",
        "tpot_profile_points",
        "tpot_ms_min",
        "tpot_ms_mean",
        "tpot_ms_max",
        "slo_violation_pct",
        "ttft_p50_ms",
        "ttft_p99_ms",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


class PlotEngineSweepTests(unittest.TestCase):
    def test_load_engine_summary_normalizes_rows_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(
                root / "engine_summary.csv",
                [
                    {
                        "policy": "nimbus",
                        "weight": "v2",
                        "fraction": "self",
                        "slo_s": "5.0",
                        "total": "10",
                        "outsourced": "3",
                        "outsource_pct": "0.3",
                        "total_cost_usd": "0.12",
                        "remote_cached_input_tokens": "100",
                        "remote_cache_hit_pct": "0.2",
                        "metrics_read_failures": "2",
                        "serving_engine": "vllm",
                        "tpot_profile": "RTX6000/Qwen2.5-7B/SGLang/MTP=4",
                        "tpot_profile_points": "3",
                        "tpot_ms_min": "19.2",
                        "tpot_ms_mean": "22.0",
                        "tpot_ms_max": "25.5",
                        "slo_violation_pct": "0.0",
                        "ttft_p50_ms": "100",
                        "ttft_p99_ms": "1000",
                    }
                ],
            )

            rows = load_engine_summary(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fraction"], "self")
        self.assertEqual(rows[0]["label"], "Nimbus (v2)")
        self.assertEqual(rows[0]["remote_cached_input_tokens"], 100)
        self.assertEqual(rows[0]["metrics_read_failures"], 2)
        self.assertEqual(rows[0]["serving_engine"], "vllm")
        self.assertEqual(rows[0]["tpot_profile_points"], 3)
        self.assertEqual(rows[0]["tpot_ms_max"], 25.5)

    def test_best_slo_rows_picks_lowest_cost_satisfying_row(self):
        rows = [
            {
                "label": "FLOP fixed",
                "fraction": 0.1,
                "total_cost_usd": 0.10,
                "outsource_pct": 0.1,
                "ttft_p99_ms": 7000,
                "slo_violation_pct": 0.0,
            },
            {
                "label": "FLOP fixed",
                "fraction": 0.2,
                "total_cost_usd": 0.20,
                "outsource_pct": 0.2,
                "ttft_p99_ms": 4000,
                "slo_violation_pct": 0.0,
            },
            {
                "label": "FLOP fixed",
                "fraction": 0.3,
                "total_cost_usd": 0.30,
                "outsource_pct": 0.3,
                "ttft_p99_ms": 3000,
                "slo_violation_pct": 0.0,
            },
        ]

        best = best_slo_rows(rows, ttft_slo_ms=5000, max_violation_pct=0.0)

        self.assertEqual(best["FLOP fixed"]["fraction"], 0.2)

    def test_label_for_row_includes_nimbus_weight_only(self):
        self.assertEqual(label_for_row({"policy": "nimbus", "weight": "v2"}), "Nimbus (v2)")
        self.assertEqual(label_for_row({"policy": "random", "weight": "v2"}), "Random")

    def test_write_normalized_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "normalized.csv"
            write_normalized_csv(
                [
                    {
                        "label": "Nimbus (v2)",
                        "policy": "nimbus",
                        "weight": "v2",
                        "fraction": "self",
                        "outsource_pct": 0.2,
                        "total_cost_usd": 0.1,
                        "slo_violation_pct": 0.0,
                        "ttft_p50_ms": 100.0,
                        "ttft_p99_ms": 900.0,
                        "remote_cached_input_tokens": 0,
                        "remote_cache_hit_pct": 0.0,
                        "metrics_read_failures": 0,
                        "serving_engine": "sglang",
                        "tpot_profile": "-",
                        "tpot_profile_points": 0,
                        "tpot_ms_min": 30.0,
                        "tpot_ms_mean": 30.0,
                        "tpot_ms_max": 30.0,
                    }
                ],
                path,
            )

            text = path.read_text()

        self.assertIn("Nimbus (v2)", text)
        self.assertIn("total_cost_usd", text)


if __name__ == "__main__":
    unittest.main()
