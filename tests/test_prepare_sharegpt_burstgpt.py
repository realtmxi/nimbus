import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_sharegpt_burstgpt import compose_workload


def _write_burstgpt(path: Path) -> None:
    rows = [
        {
            "Timestamp": "100",
            "Session ID": "burst-a",
            "Elapsed time": "1",
            "Model": "GPT-4",
            "Request tokens": "10",
            "Response tokens": "2",
            "Total tokens": "12",
            "Log Type": "Conversation log",
        },
        {
            "Timestamp": "101",
            "Session ID": "",
            "Elapsed time": "1",
            "Model": "ChatGPT",
            "Request tokens": "11",
            "Response tokens": "3",
            "Total tokens": "14",
            "Log Type": "API log",
        },
        {
            "Timestamp": "102",
            "Session ID": "burst-a",
            "Elapsed time": "1",
            "Model": "GPT-4",
            "Request tokens": "12",
            "Response tokens": "4",
            "Total tokens": "16",
            "Log Type": "Conversation log",
        },
        {
            "Timestamp": "103",
            "Session ID": "burst-a",
            "Elapsed time": "1",
            "Model": "GPT-4",
            "Request tokens": "13",
            "Response tokens": "5",
            "Total tokens": "18",
            "Log Type": "Conversation log",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_sharegpt(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "sg-a",
                    "conversations": [
                        {"from": "human", "value": "A first"},
                        {"from": "gpt", "value": "A answer"},
                        {"from": "human", "value": "A second"},
                        {"from": "gpt", "value": "A second answer"},
                    ],
                },
                {
                    "id": "sg-api",
                    "conversations": [
                        {"from": "human", "value": "API prompt"},
                        {"from": "gpt", "value": "API answer"},
                    ],
                },
            ]
        )
    )


class PrepareSharegptBurstgptTests(unittest.TestCase):
    def test_compose_keeps_sharegpt_conversation_within_burst_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            burstgpt = tmp_path / "BurstGPT_3.csv"
            sharegpt = tmp_path / "sharegpt.json"
            output = tmp_path / "workload.jsonl"
            _write_burstgpt(burstgpt)
            _write_sharegpt(sharegpt)

            stats = compose_workload(burstgpt, sharegpt, output, days=1)
            records = [json.loads(line) for line in output.read_text().splitlines()]

        burst_a = [rec for rec in records if rec["session_id"] == "burst-a"]
        self.assertEqual(stats["total_requests"], 4)
        self.assertEqual([rec["arrived_at"] for rec in records], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(records[1]["session_id"], "api:1")
        self.assertEqual(records[0]["num_prefill_tokens"], 10)
        self.assertEqual(records[0]["num_decode_tokens"], 2)
        self.assertEqual(
            [rec["sharegpt_conversation_id"] for rec in burst_a],
            ["sg-a", "sg-a", "sg-a"],
        )
        self.assertEqual([rec["sharegpt_turn_index"] for rec in burst_a], [0, 1, 1])
        self.assertTrue(burst_a[1]["prompt_text"].startswith(burst_a[0]["prompt_text"]))
        self.assertEqual(burst_a[2]["prompt_text"], burst_a[1]["prompt_text"])


if __name__ == "__main__":
    unittest.main()
