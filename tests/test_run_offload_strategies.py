import unittest
import json
import tempfile
from pathlib import Path

from experiments.run_offload_strategies import (
    load_trace,
    _steady_state_reached,
    _trace_chunk,
    stream_chunk_has_content,
)


class StreamChunkTests(unittest.TestCase):
    def test_stream_chunk_has_content_ignores_role_only_delta(self):
        self.assertFalse(
            stream_chunk_has_content(
                {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
            )
        )

    def test_stream_chunk_has_content_accepts_non_empty_content(self):
        self.assertTrue(
            stream_chunk_has_content({"choices": [{"delta": {"content": "ok"}}]})
        )


class TraceLoaderTests(unittest.TestCase):
    def test_load_trace_accepts_routewise_style_string_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {
                            "arrived_at": 10,
                            "session_id": "burst-a",
                            "prompt_text": "Human: hi",
                            "num_prefill_tokens": 4,
                            "num_decode_tokens": 2,
                        },
                        {
                            "arrived_at": 11,
                            "session_id": "api:1",
                            "prompt_text": "Human: bye",
                            "num_prefill_tokens": 5,
                            "num_decode_tokens": 3,
                        },
                    ]
                )
                + "\n"
            )

            trace = load_trace(str(path), max_requests=None, duration_hours=None)

        self.assertEqual([row["session_id"] for row in trace], ["burst-a", "api:1"])


class BistabilityProtocolTests(unittest.TestCase):
    def test_steady_state_reached_uses_recent_segment_cv(self):
        steady, metrics = _steady_state_reached(
            [
                {"ttft_p50": 100.0, "local_success_rate": 0.99},
                {"ttft_p50": 104.0, "local_success_rate": 0.98},
                {"ttft_p50": 101.0, "local_success_rate": 0.99},
            ],
            min_windows=3,
            cv_threshold=0.05,
        )

        self.assertTrue(steady)
        self.assertLessEqual(metrics["ttft_p50_cv"], 0.05)

    def test_steady_state_rejects_unstable_recent_segments(self):
        steady, _metrics = _steady_state_reached(
            [
                {"ttft_p50": 100.0, "local_success_rate": 0.99},
                {"ttft_p50": 300.0, "local_success_rate": 0.98},
                {"ttft_p50": 1000.0, "local_success_rate": 0.50},
            ],
            min_windows=3,
            cv_threshold=0.05,
        )

        self.assertFalse(steady)

    def test_trace_chunk_wraps_for_long_hold_protocols(self):
        trace = [{"request_id": str(i)} for i in range(3)]

        chunk = _trace_chunk(trace, start=2, count=5)

        self.assertEqual([row["request_id"] for row in chunk], ["2", "0", "1", "2", "0"])


if __name__ == "__main__":
    unittest.main()
