"""Tests for the verbatim current-turn ShareGPT trace materializer."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from router.common import SCENARIOS
from tools.materialize_sharegpt_current_turn_trace import (
    CACHE_MODE,
    PAYLOAD_MODE,
    TOOL_PATH,
    materialize_current_turn_trace,
    parse_args,
)


class FakeTokenizer:
    name_or_path = "/models/fake"
    model_max_length = 40960
    chat_template = "fake-template-v1"
    init_kwargs = {"_commit_hash": "fake-revision"}

    def get_vocab(self):
        return {"a": 0, "b": 1, "c": 2}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        # Five chat-template tokens plus one token per whitespace-delimited word.
        return [0] * (5 + len(messages[0]["content"].split()))


class LeakyFailingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        raise RuntimeError(f"cannot tokenize {messages[0]['content']}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestCurrentTurnMaterializer(unittest.TestCase):
    def materialize(
        self,
        root: Path,
        rows: list[dict],
        **kwargs,
    ) -> tuple[dict, list[dict], Path, Path]:
        input_path = root / "source.jsonl"
        output_path = root / "retokenized.jsonl"
        write_jsonl(input_path, rows)
        manifest = materialize_current_turn_trace(
            input_path=input_path,
            output_path=output_path,
            scenario="normal",
            tokenizer=FakeTokenizer(),
            tokenizer_path="/models/fake",
            transformers_version="test",
            command_argv=["tool", "--no-prompts-here"],
            **kwargs,
        )
        output_rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ]
        return (
            manifest,
            output_rows,
            output_path,
            output_path.with_suffix(".jsonl.manifest.json"),
        )

    def test_preserves_verbatim_payload_and_provenance_but_retokenizes(self):
        start, end = SCENARIOS["normal"]
        prompt_late = "你好  world\nsecond-line"
        prompt_early = "exact  spacing is preserved"
        rows = [
            {
                "arrived_at": start + 7,
                "num_prefill_tokens": 900,
                "num_decode_tokens": 17,
                "session_id": 42,
                "prompt_text": prompt_late,
                "response_text": "must never be copied or logged",
                "block_hash_ids": "[1, 2]",
                "block_size": 16,
            },
            {
                "arrived_at": start + 2,
                "num_prefill_tokens": 800,
                "num_decode_tokens": 9,
                "session_id": "session-a",
                "prompt_text": prompt_early,
            },
            {
                "arrived_at": start + 3,
                "num_prefill_tokens": 1,
                "num_decode_tokens": 1,
                "session_id": "empty",
                "prompt_text": "",
            },
            {
                "arrived_at": end + 1,
                "num_prefill_tokens": 1,
                "num_decode_tokens": 1,
                "session_id": "outside",
                "prompt_text": "outside window",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest, output_rows, output_path, manifest_path = self.materialize(
                Path(directory), rows
            )

            self.assertEqual([row["prompt_text"] for row in output_rows], [
                prompt_early,
                prompt_late,
            ])
            self.assertEqual([row["arrived_at"] for row in output_rows], [
                start + 2,
                start + 7,
            ])
            self.assertEqual([row["num_decode_tokens"] for row in output_rows], [9, 17])
            self.assertEqual([row["session_id"] for row in output_rows], [
                "session-a",
                42,
            ])
            self.assertEqual([row["source_request_index"] for row in output_rows], [1, 0])
            self.assertEqual([row["trace_num_prefill_tokens"] for row in output_rows], [
                800,
                900,
            ])
            for row in output_rows:
                expected = 5 + len(row["prompt_text"].split())
                self.assertEqual(row["num_prefill_tokens"], expected)
                self.assertEqual(row["uncached_prompt_tokens"], expected)
                self.assertEqual(row["num_cached_tokens"], 0)
                self.assertEqual(row["trace_num_decode_tokens"], row["num_decode_tokens"])
                self.assertEqual(row["payload_mode"], PAYLOAD_MODE)
                self.assertEqual(row["cache_mode"], CACHE_MODE)
                self.assertNotIn("response_text", row)
                self.assertNotIn("block_hash_ids", row)

            output_bytes = output_path.read_bytes()
            self.assertEqual(manifest["output_sha256"], hashlib.sha256(output_bytes).hexdigest())
            self.assertEqual(manifest["scenario_rows_n"], 3)
            self.assertEqual(manifest["empty_prompt_rows_n"], 1)
            self.assertEqual(manifest["selected_n"], 2)
            self.assertEqual(manifest["n"], 2)
            self.assertEqual(manifest["unique_session_n"], 2)
            self.assertEqual(manifest["tool_path"], TOOL_PATH)
            dependency_path = (
                Path(__file__).with_name("materialize_token_aligned_trace.py")
            )
            self.assertEqual(
                manifest["dependency_sha256"][
                    "tools/materialize_token_aligned_trace.py"
                ],
                hashlib.sha256(dependency_path.read_bytes()).hexdigest(),
            )
            common_path = Path(__file__).resolve().parents[1] / "router" / "common.py"
            self.assertEqual(
                manifest["dependency_sha256"]["router/common.py"],
                hashlib.sha256(common_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["preservation_checks"], {
                "prompt_text_exact_n": 2,
                "arrived_at_exact_n": 2,
                "decode_tokens_exact_n": 2,
                "decode_source_provenance_n": 2,
                "decode_cap_respected_n": 2,
                "session_id_exact_n": 2,
                "token_metadata_aligned_n": 2,
            })
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)

            serialized_manifest = json.dumps(manifest, ensure_ascii=False)
            self.assertNotIn(prompt_early, serialized_manifest)
            self.assertNotIn(prompt_late, serialized_manifest)
            self.assertNotIn("must never be copied or logged", serialized_manifest)

    def test_decode_cap_changes_output_but_preserves_source_provenance(self):
        start, _ = SCENARIOS["normal"]
        rows = [
            {
                "arrived_at": start,
                "num_prefill_tokens": 100,
                "num_decode_tokens": 17,
                "session_id": 1,
                "prompt_text": "long decode",
            },
            {
                "arrived_at": start + 1,
                "num_prefill_tokens": 100,
                "num_decode_tokens": 9,
                "session_id": 2,
                "prompt_text": "short decode",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest, output_rows, _, _ = self.materialize(
                Path(directory), rows, max_decode_tokens=10
            )
            self.assertEqual(
                [row["num_decode_tokens"] for row in output_rows], [10, 9]
            )
            self.assertEqual(
                [row["trace_num_decode_tokens"] for row in output_rows], [17, 9]
            )
            self.assertEqual(manifest["max_decode_tokens"], 10)
            self.assertEqual(manifest["decode_cap_affected_n"], 1)
            self.assertEqual(manifest["source_decode_tokens"]["sum"], 26)
            self.assertEqual(manifest["source_decode_tokens"]["max"], 17)
            self.assertEqual(manifest["output_decode_tokens"]["sum"], 19)
            self.assertEqual(manifest["output_decode_tokens"]["max"], 10)
            self.assertEqual(
                manifest["preservation_checks"]["decode_tokens_exact_n"], 1
            )
            self.assertEqual(
                manifest["preservation_checks"]["decode_source_provenance_n"], 2
            )
            self.assertEqual(
                manifest["preservation_checks"]["decode_cap_respected_n"], 2
            )

    def test_cli_accepts_decode_cap(self):
        args = parse_args([
            "--input", "/tmp/source.jsonl",
            "--output", "/tmp/output.jsonl",
            "--scenario", "full",
            "--tokenizer", "/models/qwen3",
            "--max-decode-tokens", "1024",
        ])
        self.assertEqual(args.max_decode_tokens, 1024)

    def test_overflow_error_is_text_free_and_does_not_write_output(self):
        start, _ = SCENARIOS["normal"]
        secret_prompt = "private semantic payload"
        rows = [{
            "arrived_at": start,
            "num_prefill_tokens": 100,
            "num_decode_tokens": 20,
            "session_id": 1,
            "prompt_text": secret_prompt,
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "source.jsonl"
            output_path = root / "out.jsonl"
            write_jsonl(input_path, rows)
            with self.assertRaisesRegex(ValueError, "exceed --max-context-tokens") as raised:
                materialize_current_turn_trace(
                    input_path=input_path,
                    output_path=output_path,
                    scenario="normal",
                    tokenizer=FakeTokenizer(),
                    tokenizer_path="/models/fake",
                    transformers_version="test",
                    max_context_tokens=10,
                    overflow_policy="error",
                )
            self.assertNotIn(secret_prompt, str(raised.exception))
            self.assertFalse(output_path.exists())
            self.assertFalse(output_path.with_suffix(".jsonl.manifest.json").exists())

    def test_overflow_drop_is_auditable_without_prompt_in_manifest(self):
        start, _ = SCENARIOS["normal"]
        dropped_prompt = "drop this private prompt"
        kept_prompt = "keep"
        rows = [
            {
                "arrived_at": start,
                "num_prefill_tokens": 100,
                "num_decode_tokens": 20,
                "session_id": 1,
                "prompt_text": dropped_prompt,
            },
            {
                "arrived_at": start + 1,
                "num_prefill_tokens": 10,
                "num_decode_tokens": 1,
                "session_id": 2,
                "prompt_text": kept_prompt,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest, output_rows, _, _ = self.materialize(
                Path(directory),
                rows,
                max_context_tokens=10,
                overflow_policy="drop",
            )
            self.assertEqual([row["prompt_text"] for row in output_rows], [kept_prompt])
            self.assertEqual(manifest["context_overflow_affected_n"], 1)
            self.assertEqual(manifest["n"], 1)
            self.assertNotIn(dropped_prompt, json.dumps(manifest))

    def test_dropped_capped_row_is_not_mixed_into_emitted_decode_stats(self):
        start, _ = SCENARIOS["normal"]
        rows = [
            {
                "arrived_at": start,
                "num_prefill_tokens": 100,
                "num_decode_tokens": 50,
                "session_id": 1,
                "prompt_text": "overflow after decode cap",
            },
            {
                "arrived_at": start + 1,
                "num_prefill_tokens": 10,
                "num_decode_tokens": 5,
                "session_id": 2,
                "prompt_text": "keep",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest, output_rows, _, _ = self.materialize(
                Path(directory),
                rows,
                max_decode_tokens=10,
                max_context_tokens=15,
                overflow_policy="drop",
            )
            self.assertEqual(len(output_rows), 1)
            self.assertEqual(manifest["context_overflow_affected_n"], 1)
            self.assertEqual(manifest["decode_cap_affected_n"], 0)
            self.assertEqual(manifest["source_decode_tokens"]["sum"], 5)
            self.assertEqual(manifest["output_decode_tokens"]["sum"], 5)

    def test_tokenizer_exception_cannot_leak_prompt(self):
        start, _ = SCENARIOS["normal"]
        secret_prompt = "do not echo this tokenizer input"
        rows = [{
            "arrived_at": start,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 1,
            "session_id": 1,
            "prompt_text": secret_prompt,
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "source.jsonl"
            output_path = root / "out.jsonl"
            write_jsonl(input_path, rows)
            with self.assertRaisesRegex(ValueError, "source row 0") as raised:
                materialize_current_turn_trace(
                    input_path=input_path,
                    output_path=output_path,
                    scenario="normal",
                    tokenizer=LeakyFailingTokenizer(),
                    tokenizer_path="/models/fake",
                    transformers_version="test",
                )
            self.assertNotIn(secret_prompt, str(raised.exception))
            self.assertIsNone(raised.exception.__cause__)
            self.assertFalse(output_path.exists())

    def test_manifest_write_failure_preserves_existing_published_pair(self):
        start, _ = SCENARIOS["normal"]
        rows = [{
            "arrived_at": start,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 1,
            "session_id": 1,
            "prompt_text": "replacement payload",
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "source.jsonl"
            output_path = root / "out.jsonl"
            manifest_path = output_path.with_suffix(".jsonl.manifest.json")
            write_jsonl(input_path, rows)
            output_path.write_bytes(b"old-output\n")
            manifest_path.write_bytes(b"old-manifest\n")

            with mock.patch(
                "tools.materialize_sharegpt_current_turn_trace._write_manifest_temp",
                side_effect=OSError("injected manifest failure"),
            ), self.assertRaisesRegex(OSError, "injected manifest failure"):
                materialize_current_turn_trace(
                    input_path=input_path,
                    output_path=output_path,
                    scenario="normal",
                    tokenizer=FakeTokenizer(),
                    tokenizer_path="/models/fake",
                    transformers_version="test",
                )

            self.assertEqual(output_path.read_bytes(), b"old-output\n")
            self.assertEqual(manifest_path.read_bytes(), b"old-manifest\n")
            self.assertEqual(
                [path for path in root.iterdir() if path.suffix == ".tmp"],
                [],
            )


if __name__ == "__main__":
    unittest.main()
