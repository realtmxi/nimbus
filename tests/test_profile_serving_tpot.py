import unittest

from experiments.profile_serving_tpot import (
    make_payload,
    percentile,
    sized_prompt,
    summarize_batch,
)
from nimbus.tpot_profile import TPOTProfile


class ProfileServingTPOTTests(unittest.TestCase):
    def test_sized_prompt_uses_requested_word_count(self):
        self.assertEqual(len(sized_prompt(7).split()), 7)
        self.assertEqual(len(sized_prompt(0).split()), 1)

    def test_payload_can_include_ignore_eos_extension(self):
        payload = make_payload(
            model="qwen",
            prompt_tokens=4,
            decode_tokens=8,
            ignore_eos=True,
        )

        self.assertEqual(payload["model"], "qwen")
        self.assertEqual(payload["max_tokens"], 8)
        self.assertTrue(payload["stream"])
        self.assertTrue(payload["ignore_eos"])
        self.assertEqual(len(payload["messages"][0]["content"].split()), 13)

    def test_percentile_uses_sorted_floor_index(self):
        self.assertEqual(percentile([5, 1, 10, 7], 0.5), 7)
        self.assertEqual(percentile([], 0.9), 0.0)

    def test_summarize_batch_outputs_tpot_profile_point(self):
        row = summarize_batch(
            4,
            [
                {
                    "success": True,
                    "tpot_ms": 10.0,
                    "ttft_ms": 100.0,
                    "latency_ms": 500.0,
                },
                {
                    "success": True,
                    "tpot_ms": 12.0,
                    "ttft_ms": 110.0,
                    "latency_ms": 520.0,
                },
                {"success": False, "tpot_ms": 100.0, "latency_ms": 1.0},
            ],
        )

        self.assertEqual(row["batch_size"], 4)
        self.assertEqual(row["tpot_ms"], 11.0)
        self.assertEqual(row["success_count"], 2)
        profile = TPOTProfile.from_dict({"tpots": [row]})
        self.assertEqual(profile.tpot_seconds_for_batch(4), 0.011)

    def test_summarize_batch_rejects_no_successes(self):
        with self.assertRaises(RuntimeError):
            summarize_batch(1, [{"success": False, "tpot_ms": 1.0}])


if __name__ == "__main__":
    unittest.main()
