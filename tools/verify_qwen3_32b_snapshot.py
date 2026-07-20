#!/usr/bin/env python3
"""Verify the exact dense-BF16 Qwen3-32B snapshot used by E12.

The contract is intentionally embedded and pinned.  Verification never trusts
model-directory metadata, never follows symlinks, and never loads tensor data
into memory.  It checks every repository file byte-for-byte, then independently
checks the config, index, and safetensors headers.  The emitted evidence contains
only public model facts and verification results; filesystem paths and model
contents are deliberately omitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable


MODEL_ID = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
EXPECTED_FILE_COUNT = 27
EXPECTED_SHARD_COUNT = 17
EXPECTED_LFS_FILE_COUNT = 18
EXPECTED_TENSOR_COUNT = 707
EXPECTED_INDEX_TOTAL_SIZE = 65_524_246_528
EXPECTED_CONFIG_SHA256 = (
    "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"
)
EXPECTED_INDEX_SHA256 = (
    "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0"
)
MAX_HEADER_BYTES = 16 * 1024 * 1024
READ_CHUNK_BYTES = 8 * 1024 * 1024


class SnapshotVerificationError(ValueError):
    """A sanitized, content-free snapshot-verification failure."""


@dataclass(frozen=True)
class OfficialFile:
    name: str
    size: int
    digest_algorithm: str
    digest: str


# Captured from the public Hugging Face model API with ``blobs=true`` at the
# exact revision above.  LFS objects are bound by their official SHA-256 and
# expanded size.  Ordinary Git blobs are bound by their Git SHA-1 and size.
OFFICIAL_FILES: tuple[OfficialFile, ...] = (
    OfficialFile(".gitattributes", 1570, "git-sha1", "52373fe24473b1aa44333d318f578ae6bf04b49b"),
    OfficialFile("LICENSE", 11343, "git-sha1", "6634c8cc3133b3848ec74b9f275acaaa1ea618ab"),
    OfficialFile("README.md", 16636, "git-sha1", "d8d96d1482cb71e6b8fa6b8109fe3b449e7cc5d5"),
    OfficialFile("config.json", 728, "git-sha1", "d66d65fbc7960c2b3c254293f74df73b47fef6d3"),
    OfficialFile("generation_config.json", 239, "git-sha1", "20a8a9156fc8c3f25295ca067f61fdf120d517c5"),
    OfficialFile("merges.txt", 1671853, "git-sha1", "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
    OfficialFile("model-00001-of-00017.safetensors", 3957109648, "sha256", "52562b2ff97b61764260273e71bf5b4cf8a66f569399398f26dec0300fcf1316"),
    OfficialFile("model-00002-of-00017.safetensors", 3900791760, "sha256", "e26764b2c6878e3fb7198895fa833ec62838d84a19665e6abfbae43c6daf02b3"),
    OfficialFile("model-00003-of-00017.safetensors", 3900791760, "sha256", "6c5ba7bed9c52bc121e75cbe8a7be46936d0006cc80f42a6d5886ed40b4c2a62"),
    OfficialFile("model-00004-of-00017.safetensors", 3900791800, "sha256", "f736f6ac4d8c30866107fb1185a05b3c3cfce9717720082f466fa44e691bcec8"),
    OfficialFile("model-00005-of-00017.safetensors", 3900791800, "sha256", "a52ed375c083209c54d42ac510afeb1fbb5af4f193be2dc7d103f665a0f212d3"),
    OfficialFile("model-00006-of-00017.safetensors", 3900791800, "sha256", "37fae28990b0e4a70228549d040c0393e87bee3820d59e58e47844974d8dff5b"),
    OfficialFile("model-00007-of-00017.safetensors", 3900791800, "sha256", "37776006aeaba29eca8bc73b2b963fe3477e1c2e3f6a27cb9527be75b905e1bf"),
    OfficialFile("model-00008-of-00017.safetensors", 3900791800, "sha256", "73e74e9129674fe330948005075d70ab4fa0b92b68fb220c5a693f9cea553730"),
    OfficialFile("model-00009-of-00017.safetensors", 3900791800, "sha256", "a044b3602a01bd8ea62ff51badf9cc038ab1d73d97399480e6a55b4c86fa7fa6"),
    OfficialFile("model-00010-of-00017.safetensors", 3900791800, "sha256", "9966612ba7ecfc2cd2e592fb95224b86d743271ff88e172cc272a4b26382aa75"),
    OfficialFile("model-00011-of-00017.safetensors", 3900791800, "sha256", "e2a058a0ac7d4b992b731c29221ecfb4b76b8a48d9004d0e8a62ba44f699845c"),
    OfficialFile("model-00012-of-00017.safetensors", 3900791800, "sha256", "58a1aa89093fea07325f787072a468e3482a470ff4b7fe5ead5f749683907c40"),
    OfficialFile("model-00013-of-00017.safetensors", 3900791800, "sha256", "35f3381bab31a23370c37d922290aeecdf603418336058fb86fe42d8f51ac40c"),
    OfficialFile("model-00014-of-00017.safetensors", 3900791800, "sha256", "8713b062ddc178acf5917610b7f4b64eede833b2ea4aa37bd562dcf2f3a3339d"),
    OfficialFile("model-00015-of-00017.safetensors", 3900791800, "sha256", "bec439d23931821a236d8f62fa79deecf5551bd25602278aa2ae0ce432b378cf"),
    OfficialFile("model-00016-of-00017.safetensors", 3900791800, "sha256", "e569139fadd61fe7c8f9eb1c976d9a627cae48c57ddf228cfbd0593c59c64ff7"),
    OfficialFile("model-00017-of-00017.safetensors", 3055341992, "sha256", "1f47c318fcd7797c0f85b4233cb754438b10e795b8bc874889090c416a94bd38"),
    OfficialFile("model.safetensors.index.json", 58330, "git-sha1", "2bc093f8069057e0b28f628398a05a273efd1297"),
    OfficialFile("tokenizer.json", 11422654, "sha256", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    OfficialFile("tokenizer_config.json", 9732, "git-sha1", "417d038a63fa3de29cfde265caedae14d1a58d92"),
    OfficialFile("vocab.json", 2776833, "git-sha1", "4783fe10ac3adce15ac8f358ef5462739852c569"),
)

EXPECTED_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 25600,
    "max_position_embeddings": 40960,
    "max_window_layers": 64,
    "model_type": "qwen3",
    "num_attention_heads": 64,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1_000_000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _official_manifest_sha256(files: tuple[OfficialFile, ...] = OFFICIAL_FILES) -> str:
    return _canonical_sha256([
        {
            "name": item.name,
            "size": item.size,
            "digest_algorithm": item.digest_algorithm,
            "digest": item.digest,
        }
        for item in files
    ])


def _validate_embedded_contract() -> None:
    if len(OFFICIAL_FILES) != EXPECTED_FILE_COUNT:
        raise SnapshotVerificationError("embedded file count is invalid")
    names = [item.name for item in OFFICIAL_FILES]
    if len(set(names)) != len(names) or names != sorted(names):
        raise SnapshotVerificationError("embedded filenames are invalid")
    for item in OFFICIAL_FILES:
        if (
            not item.name
            or "/" in item.name
            or item.name in {".", ".."}
            or isinstance(item.size, bool)
            or item.size < 0
        ):
            raise SnapshotVerificationError("embedded file metadata is invalid")
        expected_length = 64 if item.digest_algorithm == "sha256" else 40
        if item.digest_algorithm not in {"sha256", "git-sha1"}:
            raise SnapshotVerificationError("embedded digest algorithm is invalid")
        if len(item.digest) != expected_length or any(
            character not in "0123456789abcdef" for character in item.digest
        ):
            raise SnapshotVerificationError("embedded digest is invalid")
    shards = [name for name in names if name.endswith(".safetensors")]
    expected_shards = [
        f"model-{index:05d}-of-{EXPECTED_SHARD_COUNT:05d}.safetensors"
        for index in range(1, EXPECTED_SHARD_COUNT + 1)
    ]
    if shards != expected_shards:
        raise SnapshotVerificationError("embedded shard set is invalid")
    if sum(item.digest_algorithm == "sha256" for item in OFFICIAL_FILES) != EXPECTED_LFS_FILE_COUNT:
        raise SnapshotVerificationError("embedded LFS file count is invalid")


def _read_exact(source: BinaryIO, size: int, label: str) -> bytes:
    data = source.read(size)
    if len(data) != size:
        raise SnapshotVerificationError(f"{label} is truncated")
    return data


def _parse_json_bytes(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SnapshotVerificationError(f"{label} is not valid JSON") from None


def _hash_and_capture_file(
    root: Path,
    item: OfficialFile,
    *,
    capture_body: bool = False,
    capture_safetensors_header: bool = False,
) -> bytes | None:
    path = root / item.name
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise SnapshotVerificationError("unable to open an expected model file") from None
    captured: bytes | None = None
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != item.size:
                raise SnapshotVerificationError("model file type or size does not match")
            if item.digest_algorithm == "sha256":
                digest = hashlib.sha256()
            else:
                digest = hashlib.sha1()
                digest.update(f"blob {item.size}\0".encode("ascii"))

            if capture_safetensors_header:
                prefix = _read_exact(source, 8, "safetensors header prefix")
                digest.update(prefix)
                header_size = struct.unpack("<Q", prefix)[0]
                if header_size <= 1 or header_size > MAX_HEADER_BYTES:
                    raise SnapshotVerificationError("safetensors header size is invalid")
                if 8 + header_size > item.size:
                    raise SnapshotVerificationError("safetensors header exceeds its file")
                captured = _read_exact(source, header_size, "safetensors header")
                digest.update(captured)
            elif capture_body:
                if item.size > MAX_HEADER_BYTES:
                    raise SnapshotVerificationError("captured metadata file is too large")
                captured = _read_exact(source, item.size, "model metadata file")
                digest.update(captured)

            bytes_read = source.tell()
            while True:
                chunk = source.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                bytes_read += len(chunk)
                digest.update(chunk)
            if bytes_read != item.size or digest.hexdigest() != item.digest:
                raise SnapshotVerificationError("model file content does not match the pinned revision")
            after = os.fstat(source.fileno())
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if identity_before != identity_after:
                raise SnapshotVerificationError("model file changed during verification")
    except OSError:
        raise SnapshotVerificationError("unable to read an expected model file") from None
    return captured


def _validate_exact_tree(root: Path) -> None:
    try:
        metadata = root.lstat()
    except OSError:
        raise SnapshotVerificationError("snapshot directory is unavailable") from None
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise SnapshotVerificationError("snapshot root must be a real directory")
    try:
        children = list(root.iterdir())
    except OSError:
        raise SnapshotVerificationError("snapshot directory cannot be listed") from None
    expected = {item.name for item in OFFICIAL_FILES}
    actual = {child.name for child in children}
    if actual != expected or len(children) != EXPECTED_FILE_COUNT:
        raise SnapshotVerificationError("snapshot file set does not exactly match the pinned revision")
    for child in children:
        try:
            child_metadata = child.lstat()
        except OSError:
            raise SnapshotVerificationError("snapshot entry cannot be inspected") from None
        if not stat.S_ISREG(child_metadata.st_mode) or stat.S_ISLNK(child_metadata.st_mode):
            raise SnapshotVerificationError("snapshot entries must be regular non-symlink files")


def _validate_config(raw: bytes) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise SnapshotVerificationError("config SHA-256 does not match")
    payload = _parse_json_bytes(raw, "config")
    if payload != EXPECTED_CONFIG:
        raise SnapshotVerificationError("config semantics do not match dense BF16 Qwen3-32B")
    return payload


def _validate_index(raw: bytes) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != EXPECTED_INDEX_SHA256:
        raise SnapshotVerificationError("index SHA-256 does not match")
    payload = _parse_json_bytes(raw, "safetensors index")
    if not isinstance(payload, dict) or set(payload) != {"metadata", "weight_map"}:
        raise SnapshotVerificationError("safetensors index schema is invalid")
    if payload["metadata"] != {"total_size": EXPECTED_INDEX_TOTAL_SIZE}:
        raise SnapshotVerificationError("safetensors index total size does not match")
    weight_map = payload["weight_map"]
    if not isinstance(weight_map, dict) or len(weight_map) != EXPECTED_TENSOR_COUNT:
        raise SnapshotVerificationError("safetensors index tensor count does not match")
    shards = {
        item.name for item in OFFICIAL_FILES if item.name.endswith(".safetensors")
    }
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(shard, str)
        or shard not in shards
        for name, shard in weight_map.items()
    ):
        raise SnapshotVerificationError("safetensors index references are invalid")
    if set(weight_map.values()) != shards:
        raise SnapshotVerificationError("safetensors index does not reference every shard")
    return weight_map


def _parse_safetensors_header(raw: bytes, shard_size: int) -> dict[str, dict[str, Any]]:
    payload = _parse_json_bytes(raw, "safetensors header")
    if not isinstance(payload, dict):
        raise SnapshotVerificationError("safetensors header schema is invalid")
    metadata = payload.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise SnapshotVerificationError("safetensors metadata is invalid")
    if not payload:
        raise SnapshotVerificationError("safetensors shard has no tensors")
    tensors: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int]] = []
    for name, descriptor in payload.items():
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            raise SnapshotVerificationError("safetensors tensor descriptor is invalid")
        if set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise SnapshotVerificationError("safetensors tensor descriptor fields are invalid")
        if descriptor["dtype"] != "BF16":
            raise SnapshotVerificationError("safetensors tensor dtype is not BF16")
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if (
            not isinstance(shape, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in offsets)
            or offsets[0] > offsets[1]
        ):
            raise SnapshotVerificationError("safetensors tensor shape or offsets are invalid")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if offsets[1] - offsets[0] != elements * 2:
            raise SnapshotVerificationError("safetensors BF16 byte size is inconsistent")
        intervals.append((offsets[0], offsets[1]))
        tensors[name] = descriptor
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise SnapshotVerificationError("safetensors data offsets are not contiguous")
        cursor = end
    if 8 + len(raw) + cursor != shard_size:
        raise SnapshotVerificationError("safetensors data extent does not match shard size")
    return tensors


def _validate_headers(
    headers: dict[str, bytes],
    weight_map: dict[str, str],
) -> dict[str, int]:
    official_by_name = {item.name: item for item in OFFICIAL_FILES}
    actual_tensor_shard: dict[str, str] = {}
    tensor_bytes = 0
    for shard in sorted(headers):
        tensors = _parse_safetensors_header(
            headers[shard], official_by_name[shard].size
        )
        for name, descriptor in tensors.items():
            if name in actual_tensor_shard:
                raise SnapshotVerificationError("tensor appears in more than one shard")
            actual_tensor_shard[name] = shard
            tensor_bytes += descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
    if actual_tensor_shard != weight_map:
        raise SnapshotVerificationError("safetensors headers and index disagree")
    if len(actual_tensor_shard) != EXPECTED_TENSOR_COUNT:
        raise SnapshotVerificationError("safetensors header tensor count does not match")
    if tensor_bytes != EXPECTED_INDEX_TOTAL_SIZE:
        raise SnapshotVerificationError("safetensors tensor bytes do not match index total size")
    return {"tensor_count": len(actual_tensor_shard), "tensor_bytes": tensor_bytes}


def verify_snapshot(
    root: Path,
    *,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    _validate_embedded_contract()
    _validate_exact_tree(root)

    config_raw: bytes | None = None
    index_raw: bytes | None = None
    headers: dict[str, bytes] = {}
    for item in OFFICIAL_FILES:
        is_shard = item.name.endswith(".safetensors")
        captured = _hash_and_capture_file(
            root,
            item,
            capture_body=item.name in {"config.json", "model.safetensors.index.json"},
            capture_safetensors_header=is_shard,
        )
        if item.name == "config.json":
            config_raw = captured
        elif item.name == "model.safetensors.index.json":
            index_raw = captured
        elif is_shard:
            if captured is None:
                raise SnapshotVerificationError("safetensors header was not captured")
            headers[item.name] = captured
    if config_raw is None or index_raw is None or len(headers) != EXPECTED_SHARD_COUNT:
        raise SnapshotVerificationError("required model metadata was not captured")

    config = _validate_config(config_raw)
    weight_map = _validate_index(index_raw)
    header_stats = _validate_headers(headers, weight_map)
    captured_at = now() if now is not None else datetime.now(timezone.utc)
    if captured_at.tzinfo is None:
        raise SnapshotVerificationError("evidence time must be timezone-aware")

    # Explicit whitelist: do not emit root/output paths, arbitrary source JSON,
    # tensor names, file contents, process state, environment, or credentials.
    return {
        "schema_version": 1,
        "status": "pass",
        "captured_at_utc": captured_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "official_manifest_sha256": _official_manifest_sha256(),
        "files": {
            "count": EXPECTED_FILE_COUNT,
            "bytes": sum(item.size for item in OFFICIAL_FILES),
            "lfs_sha256_count": EXPECTED_LFS_FILE_COUNT,
            "git_blob_sha1_count": EXPECTED_FILE_COUNT - EXPECTED_LFS_FILE_COUNT,
            "exact_top_level_set": True,
            "regular_non_symlink": True,
            "size_and_digest_exact": True,
        },
        "config": {
            "sha256": EXPECTED_CONFIG_SHA256,
            "architecture": config["architectures"][0],
            "model_type": config["model_type"],
            "torch_dtype": config["torch_dtype"],
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config["num_key_value_heads"],
            "semantic_exact": True,
        },
        "index": {
            "sha256": EXPECTED_INDEX_SHA256,
            "total_size": EXPECTED_INDEX_TOTAL_SIZE,
            "tensor_count": len(weight_map),
            "referenced_shard_count": len(set(weight_map.values())),
            "references_complete": True,
        },
        "safetensors": {
            "shard_count": len(headers),
            "tensor_count": header_stats["tensor_count"],
            "tensor_bytes": header_stats["tensor_bytes"],
            "dtype_counts": {"BF16": header_stats["tensor_count"]},
            "headers_match_index": True,
        },
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            os.fchmod(target.fileno(), 0o600)
            json.dump(payload, target, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        raise SnapshotVerificationError("unable to publish verification evidence") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # Keep the final path component unresolved so _validate_exact_tree can
        # reject a snapshot-root symlink instead of silently following it.
        snapshot = args.snapshot.expanduser().absolute()
        output = args.output.resolve()
        try:
            output.relative_to(snapshot)
        except ValueError:
            pass
        else:
            raise SnapshotVerificationError("evidence output must be outside the snapshot")
        evidence = verify_snapshot(snapshot)
        write_json_atomic(output, evidence)
    except (OSError, SnapshotVerificationError) as exc:
        if isinstance(exc, OSError):
            message = "snapshot path cannot be resolved"
        else:
            message = str(exc)
        print(f"Qwen3-32B snapshot verification failed: {message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
