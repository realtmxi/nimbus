#!/usr/bin/env python3
"""Compose the Nimbus ShareGPT+BurstGPT workload.

BurstGPT supplies arrival times, session structure, and token counts. ShareGPT
supplies reusable prompt/response text. This mirrors the RouteWise workload
preparation path while writing Nimbus's canonical primary trace location.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BURSTGPT = _REPO_ROOT / "data" / ".cache" / "BurstGPT_3.csv"
_DEFAULT_SHAREGPT = (
    _REPO_ROOT / "data" / ".cache" / "ShareGPT_V3_unfiltered_cleaned_split.json"
)
_DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "data"
    / "sharegpt_burstgpt"
    / "sharegpt_prompts_burstgpt_timestamps.jsonl"
)

_SECONDS_PER_DAY = 86_400
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_BURSTGPT_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv"
_SHAREGPT_URL = "https://huggingface.co/datasets/learnanything/sharegpt_v3_unfiltered_cleaned_split/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
_BURSTGPT_SHA256 = "2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f"
_SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str, *, verify: bool = True) -> None:
    if not verify:
        return
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {path}\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual_sha256}"
        )


def download_file(
    name: str,
    url: str,
    path: Path,
    expected_sha256: str,
    *,
    force: bool = False,
    verify: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0 and not force:
        verify_file(path, expected_sha256, verify=verify)
        logger.info("[skip] %s already exists: %s", name, path)
        return

    tmp_path = path.with_name(f"{path.name}.part")
    if tmp_path.exists():
        tmp_path.unlink()

    logger.info("[download] %s", name)
    logger.info("           %s", url)
    logger.info("        -> %s", path)

    request = Request(url, headers={"User-Agent": "Nimbus-workload-prep/1.0"})
    with urlopen(request) as response, tmp_path.open("wb") as handle:
        while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
            handle.write(chunk)

    verify_file(tmp_path, expected_sha256, verify=verify)
    tmp_path.replace(path)
    logger.info("[ok] %s saved: %s", name, path)


def ensure_raw_workloads(
    *,
    burstgpt_path: Path,
    sharegpt_path: Path,
    force_download: bool = False,
    verify: bool = True,
) -> None:
    download_file(
        "BurstGPT_3",
        os.environ.get("BURSTGPT_URL", _BURSTGPT_URL),
        burstgpt_path,
        _BURSTGPT_SHA256,
        force=force_download,
        verify=verify,
    )
    download_file(
        "ShareGPT_V3",
        os.environ.get("SHAREGPT_URL", _SHAREGPT_URL),
        sharegpt_path,
        _SHAREGPT_SHA256,
        force=force_download,
        verify=verify,
    )


def _text_value(message: dict[str, Any]) -> str:
    value = message.get("value", "")
    return value if isinstance(value, str) else str(value)


def _append_transcript_part(parts: list[str], role: str, text: str) -> None:
    if text:
        parts.append(f"{role}: {text}")


def load_sharegpt_conversations(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"ShareGPT trace not found: {path}\n"
            "Run: bash scripts/download_data.sh"
        )

    logger.info("Loading ShareGPT text pool from %s ...", path)
    with path.open() as handle:
        conversations = json.load(handle)

    reusable_conversations: list[dict[str, object]] = []
    human_roles = {"human", "user"}
    assistant_roles = {"gpt", "chatgpt", "assistant"}

    for conv_idx, conv in enumerate(conversations):
        if not isinstance(conv, dict):
            continue
        conv_id = str(conv.get("id") or conv.get("conversation_id") or conv_idx)
        messages = conv.get("conversations") or conv.get("messages") or []
        if not isinstance(messages, list):
            continue

        transcript_parts: list[str] = []
        turns: list[dict[str, object]] = []
        human_turn_index = 0
        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("from") or message.get("role") or "").lower()
            if role in assistant_roles:
                _append_transcript_part(transcript_parts, "Assistant", _text_value(message))
                continue
            if role not in human_roles:
                continue

            prompt_text = _text_value(message)
            response_text = ""
            if idx + 1 < len(messages):
                next_message = messages[idx + 1]
                if isinstance(next_message, dict):
                    next_role = str(
                        next_message.get("from") or next_message.get("role") or ""
                    ).lower()
                    if next_role in assistant_roles:
                        response_text = _text_value(next_message)

            if prompt_text:
                _append_transcript_part(transcript_parts, "Human", prompt_text)
                turns.append(
                    {
                        "turn_index": human_turn_index,
                        "prompt_text": "\n\n".join(transcript_parts),
                        "response_text": response_text,
                    }
                )
                human_turn_index += 1

        if turns:
            reusable_conversations.append(
                {
                    "conversation_id": conv_id,
                    "turns": turns,
                }
            )

    if not reusable_conversations:
        raise ValueError(f"No ShareGPT human turns found in {path}")

    turn_count = sum(len(conv["turns"]) for conv in reusable_conversations)
    logger.info(
        "Loaded %d reusable ShareGPT conversations (%d human turns)",
        len(reusable_conversations),
        turn_count,
    )
    return reusable_conversations


def _int_field(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def compose_workload(
    burstgpt_path: Path,
    sharegpt_path: Path,
    output_path: Path,
    *,
    days: int = 30,
    start_day: int = 0,
    max_requests: int | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    if start_day < 0:
        raise ValueError(f"start_day must be non-negative, got {start_day}")
    if max_requests is not None and max_requests <= 0:
        raise ValueError(f"max_requests must be positive when set, got {max_requests}")
    if not burstgpt_path.exists():
        raise FileNotFoundError(
            f"BurstGPT trace not found: {burstgpt_path}\n"
            "Run: bash scripts/download_data.sh"
        )

    sharegpt_conversations = load_sharegpt_conversations(sharegpt_path)
    sharegpt_turn_count = sum(len(conv["turns"]) for conv in sharegpt_conversations)

    with burstgpt_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        try:
            first = next(reader)
        except StopIteration as exc:
            raise ValueError(f"BurstGPT trace is empty: {burstgpt_path}") from exc
        t0 = float(first["Timestamp"])

    window_start = t0 + start_day * _SECONDS_PER_DAY
    window_end = window_start + days * _SECONDS_PER_DAY
    logger.info(
        "Extraction window: day %d -> day %d (%.1f -> %.1f raw seconds)",
        start_day,
        start_day + days,
        window_start,
        window_end,
    )

    output_handle = None
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w")

    total_requests = 0
    skipped_rows = 0
    api_log_rows = 0
    sessions_seen: set[str] = set()
    session_to_conversation: dict[str, int] = {}
    session_turn_index: dict[str, int] = {}
    token_totals: list[int] = []
    last_arrived_at = 0.0

    try:
        with burstgpt_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader):
                ts = float(row["Timestamp"])
                if ts < window_start:
                    continue
                if ts >= window_end:
                    break
                if max_requests is not None and total_requests >= max_requests:
                    break

                try:
                    request_tokens = _int_field(row, "Request tokens")
                    response_tokens = _int_field(row, "Response tokens")
                    total_tokens = _int_field(row, "Total tokens")
                except (KeyError, ValueError):
                    skipped_rows += 1
                    continue
                if total_tokens <= 0:
                    skipped_rows += 1
                    continue

                session_id = str(row.get("Session ID", "")).strip()
                if not session_id:
                    session_id = f"api:{row_idx}"
                    api_log_rows += 1
                sessions_seen.add(session_id)

                if session_id not in session_to_conversation:
                    session_to_conversation[session_id] = (
                        len(session_to_conversation) % len(sharegpt_conversations)
                    )
                    session_turn_index[session_id] = 0

                conversation = sharegpt_conversations[
                    session_to_conversation[session_id]
                ]
                turns = conversation["turns"]
                if not isinstance(turns, list):
                    raise TypeError("ShareGPT conversation turns must be a list")
                turn_index = min(session_turn_index[session_id], len(turns) - 1)
                turn = turns[turn_index]
                session_turn_index[session_id] += 1

                arrived_at = ts - window_start
                record = {
                    "request_id": total_requests,
                    "arrived_at": arrived_at,
                    "session_id": session_id,
                    "num_prefill_tokens": request_tokens,
                    "num_decode_tokens": response_tokens,
                    "model": row.get("Model", ""),
                    "log_type": row.get("Log Type", ""),
                    "elapsed_time_sec": float(row.get("Elapsed time") or 0.0),
                    "prompt_text": turn["prompt_text"],
                    "response_text": turn["response_text"],
                    "sharegpt_conversation_id": conversation["conversation_id"],
                    "sharegpt_turn_index": turn["turn_index"],
                }

                if output_handle is not None:
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_requests += 1
                token_totals.append(total_tokens)
                last_arrived_at = arrived_at
    finally:
        if output_handle is not None:
            output_handle.close()

    stats: dict[str, object] = {
        "burstgpt_file": str(burstgpt_path),
        "sharegpt_file": str(sharegpt_path),
        "output_file": str(output_path),
        "days_requested": days,
        "start_day": start_day,
        "max_requests": max_requests,
        "total_requests": total_requests,
        "unique_sessions": len(sessions_seen),
        "api_log_rows": api_log_rows,
        "skipped_rows": skipped_rows,
        "sharegpt_conversation_pool": len(sharegpt_conversations),
        "sharegpt_turn_pool": sharegpt_turn_count,
        "duration_seconds": last_arrived_at,
        "requests_per_day": total_requests / days,
    }
    if sharegpt_conversations:
        stats["sharegpt_session_reuse_factor"] = (
            len(sessions_seen) / len(sharegpt_conversations)
        )
    if sharegpt_turn_count:
        stats["sharegpt_turn_reuse_factor"] = total_requests / sharegpt_turn_count
    if token_totals:
        sorted_tokens = sorted(token_totals)
        stats["median_total_tokens"] = int(sorted_tokens[len(sorted_tokens) // 2])
        stats["mean_total_tokens"] = sum(token_totals) / len(token_totals)
    if output_path.exists() and not dry_run:
        stats["output_size_mb"] = round(output_path.stat().st_size / (1024 * 1024), 1)

    logger.info(
        "Composition complete: %d requests, %d sessions",
        total_requests,
        len(sessions_seen),
    )
    if dry_run:
        logger.info("Dry run -- no file written.")
    else:
        logger.info("Written %s", output_path)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw traces and compose Nimbus's ShareGPT+BurstGPT workload."
    )
    parser.add_argument("--burstgpt", type=Path, default=_DEFAULT_BURSTGPT)
    parser.add_argument("--sharegpt", type=Path, default=_DEFAULT_SHAREGPT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite output if it exists")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true", help="skip SHA-256 checks")
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=None,
        help="optional JSON file for composition stats",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if (
        args.output.exists()
        and args.output.stat().st_size > 0
        and not args.force
        and not args.dry_run
    ):
        logger.info("[skip] workload already exists: %s", args.output)
        return

    if not args.skip_download:
        ensure_raw_workloads(
            burstgpt_path=args.burstgpt,
            sharegpt_path=args.sharegpt,
            force_download=args.force_download,
            verify=not args.no_verify,
        )

    stats = compose_workload(
        args.burstgpt,
        args.sharegpt,
        args.output,
        days=args.days,
        start_day=args.start_day,
        max_requests=args.max_requests,
        dry_run=args.dry_run,
    )

    print(json.dumps(stats, indent=2, sort_keys=True))
    if args.stats_file is not None and not args.dry_run:
        args.stats_file.parent.mkdir(parents=True, exist_ok=True)
        args.stats_file.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
