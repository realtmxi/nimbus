"""Lightweight tests for the pinned Qwen3-32B snapshot verifier."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import verify_qwen3_32b_snapshot as verifier


def header_bytes(*, dtype: str = "BF16") -> tuple[bytes, int]:
    payload = {
        "tensor.a": {
            "dtype": dtype,
            "shape": [2],
            "data_offsets": [0, 4],
        },
        "tensor.b": {
            "dtype": "BF16",
            "shape": [3],
            "data_offsets": [4, 10],
        },
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return raw, 8 + len(raw) + 10


class VerifyQwen3SnapshotTests(unittest.TestCase):
    def test_embedded_official_contract_is_complete(self) -> None:
        verifier._validate_embedded_contract()
        self.assertEqual(len(verifier.OFFICIAL_FILES), 27)
        self.assertEqual(
            verifier._official_manifest_sha256(),
            "6597f6b6ebb926354d721f44fa3cc5ea97f61c4d645858440a08f46cbfd9020e",
        )

    def test_bf16_headers_must_match_index_exactly(self) -> None:
        raw, shard_size = header_bytes()
        shard = "model-00001-of-00001.safetensors"
        official = (verifier.OfficialFile(shard, shard_size, "sha256", "0" * 64),)
        weight_map = {"tensor.a": shard, "tensor.b": shard}
        with (
            patch.object(verifier, "OFFICIAL_FILES", official),
            patch.object(verifier, "EXPECTED_TENSOR_COUNT", 2),
            patch.object(verifier, "EXPECTED_INDEX_TOTAL_SIZE", 10),
        ):
            stats = verifier._validate_headers({shard: raw}, weight_map)
            self.assertEqual(stats, {"tensor_count": 2, "tensor_bytes": 10})
            with self.assertRaisesRegex(
                verifier.SnapshotVerificationError, "headers and index disagree"
            ):
                verifier._validate_headers(
                    {shard: raw}, {"tensor.a": shard, "tensor.b": "wrong"}
                )

    def test_file_hashing_binds_lfs_and_git_blob_digests(self) -> None:
        lfs_body = b"expanded-lfs-object"
        git_body = b'{"small":"metadata"}\n'
        git_digest = hashlib.sha1(
            f"blob {len(git_body)}\0".encode("ascii") + git_body
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lfs").write_bytes(lfs_body)
            (root / "git").write_bytes(git_body)
            verifier._hash_and_capture_file(
                root,
                verifier.OfficialFile(
                    "lfs", len(lfs_body), "sha256", hashlib.sha256(lfs_body).hexdigest()
                ),
            )
            captured = verifier._hash_and_capture_file(
                root,
                verifier.OfficialFile("git", len(git_body), "git-sha1", git_digest),
                capture_body=True,
            )
            self.assertEqual(captured, git_body)
            (root / "lfs").write_bytes(b"X" * len(lfs_body))
            with self.assertRaisesRegex(
                verifier.SnapshotVerificationError, "pinned revision"
            ):
                verifier._hash_and_capture_file(
                    root,
                    verifier.OfficialFile(
                        "lfs",
                        len(lfs_body),
                        "sha256",
                        hashlib.sha256(lfs_body).hexdigest(),
                    ),
                )

    def test_non_bf16_header_fails_closed(self) -> None:
        raw, shard_size = header_bytes(dtype="F32")
        with self.assertRaisesRegex(
            verifier.SnapshotVerificationError, "dtype is not BF16"
        ):
            verifier._parse_safetensors_header(raw, shard_size)

    def test_tree_rejects_extra_file_and_symlink(self) -> None:
        official = (verifier.OfficialFile("only", 1, "sha256", "0" * 64),)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "only").write_bytes(b"x")
            with (
                patch.object(verifier, "OFFICIAL_FILES", official),
                patch.object(verifier, "EXPECTED_FILE_COUNT", 1),
            ):
                verifier._validate_exact_tree(root)
                (root / "extra").write_bytes(b"x")
                with self.assertRaisesRegex(
                    verifier.SnapshotVerificationError, "file set"
                ):
                    verifier._validate_exact_tree(root)
                (root / "extra").unlink()
                (root / "only").unlink()
                os.symlink("missing", root / "only")
                with self.assertRaisesRegex(
                    verifier.SnapshotVerificationError, "regular non-symlink"
                ):
                    verifier._validate_exact_tree(root)

    def test_evidence_write_is_atomic_private_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            verifier.write_json_atomic(output, {"status": "pass"})
            self.assertEqual(json.loads(output.read_text()), {"status": "pass"})
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
