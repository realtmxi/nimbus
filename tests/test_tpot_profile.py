import tempfile
import unittest
from pathlib import Path

from nimbus.tpot_profile import TPOTProfile


class TPOTProfileTests(unittest.TestCase):
    def test_loads_notion_profile_shape_and_uses_conservative_step_function(self):
        profile = TPOTProfile.from_dict(
            {
                "deployment": {
                    "gpu": "RTX6000",
                    "model": "Qwen2.5-7B",
                    "engine": "SGLang",
                    "mtp_mode": "MTP=4",
                },
                "tpots": [
                    {"batch_size": 8, "tpot_ms": 25.5},
                    {"batch_size": 1, "tpot_ms": 19.2},
                    {"batch_size": 4, "tpot_ms": 22.0},
                ],
                "prefill_throughput_tokens_per_s": 50000,
                "b_sweet": 16,
            }
        )

        self.assertEqual(profile.tpot_seconds_for_batch(1), 0.0192)
        self.assertEqual(profile.tpot_seconds_for_batch(2), 0.022)
        self.assertEqual(profile.tpot_seconds_for_batch(5), 0.0255)
        self.assertEqual(profile.tpot_seconds_for_batch(999), 0.0255)
        self.assertEqual(profile.prefill_throughput_tokens_per_s, 50000)
        self.assertEqual(profile.b_sweet, 16)
        self.assertEqual(profile.label(), "RTX6000/Qwen2.5-7B/SGLang/MTP=4")

    def test_loads_from_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                '{"tpots": [{"batch_size": 1, "tpot_s": 0.01}], '
                '"deployment": {"gpu": "A100"}}'
            )

            profile = TPOTProfile.from_json(path)

        self.assertEqual(profile.tpot_seconds_for_batch(1), 0.01)
        self.assertEqual(profile.label(), "A100")

    def test_rejects_duplicate_batch_sizes(self):
        with self.assertRaises(ValueError):
            TPOTProfile.from_dict(
                {
                    "tpots": [
                        {"batch_size": 1, "tpot_ms": 10},
                        {"batch_size": 1, "tpot_ms": 11},
                    ]
                }
            )

    def test_rejects_missing_tpot_list(self):
        with self.assertRaises(ValueError):
            TPOTProfile.from_dict({"tpots": []})


if __name__ == "__main__":
    unittest.main()
