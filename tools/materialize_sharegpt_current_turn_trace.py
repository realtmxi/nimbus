#!/usr/bin/env python3
"""Retokenize verbatim ShareGPT current-turn prompts for a no-cache replay.

The primary ShareGPT/BurstGPT trace records cumulative conversation length in
``num_prefill_tokens`` while ``prompt_text`` contains only the current user
turn.  This tool keeps that current-turn text verbatim and makes the scheduling
metadata describe the payload that the OpenAI-compatible endpoint will receive:

``num_prefill_tokens = uncached_prompt_tokens = chat_template(prompt_text)``
``num_cached_tokens = 0``

Arrival time, decode cap, session id, original trace length, and source-line
index are retained as provenance.  Response text and source cache-block fields
are deliberately not copied because they are neither sent nor needed for this
no-cache experiment.

The JSONL and its ``.manifest.json`` sidecar are written atomically.  The
manifest and stdout contain hashes and aggregate token statistics, never prompt
or response text.

Example (on the serving host):

    python tools/materialize_sharegpt_current_turn_trace.py \
      --input /path/sharegpt_prompts_burstgpt_timestamps.jsonl \
      --output /scratch/me/sharegpt_current_turn_qwen3.jsonl \
      --scenario extreme_burst_1200 \
      --tokenizer /path/Qwen3-32B \
      --max-decode-tokens 1024 \
      --max-context-tokens 40960
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

# Keep ``python tools/...py`` runnable from a checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.common import SCENARIOS
from tools.materialize_token_aligned_trace import (
    _chat_token_count,
    _percentile,
    _tokenizer_fingerprint,
)


PAYLOAD_MODE = "sharegpt_current_turn_retokenized"
CACHE_MODE = "none"
TOOL_PATH = "tools/materialize_sharegpt_current_turn_trace.py"


def _distribution(values: list[float | int]) -> dict[str, float | int | None]:
    """Return compact aggregate evidence without exposing source text."""
    if not values:
        return {
            "n": 0,
            "sum": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "n": len(values),
        "sum": sum(values),
        "min": min(values),
        "p50": _percentile([float(value) for value in values], 0.50),
        "p95": _percentile([float(value) for value in values], 0.95),
        "p99": _percentile([float(value) for value in values], 0.99),
        "max": max(values),
    }


def _scenario_bounds(scenario: str) -> tuple[float, float]:
    if scenario == "full":
        return -math.inf, math.inf
    return SCENARIOS[scenario]


def _selected_index_hash(indices: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for source_index in indices:
        digest.update(str(source_index).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_selected(
    input_path: Path,
    *,
    scenario: str,
    limit: int,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    start, end = _scenario_bounds(scenario)
    input_hash = hashlib.sha256()
    input_rows_n = 0
    scenario_rows_n = 0
    empty_prompt_rows_n = 0
    selected: list[tuple[int, dict[str, Any]]] = []

    with input_path.open("rb") as source:
        for source_index, raw in enumerate(source):
            input_hash.update(raw)
            input_rows_n += 1
            try:
                obj = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Do not include the source line (which can contain prompt text)
                # in an error destined for a terminal or experiment log.
                raise ValueError(
                    f"invalid JSON in source row {source_index}"
                ) from None

            try:
                arrived_at = int(obj["arrived_at"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid arrived_at in source row {source_index}"
                ) from exc
            if not start <= arrived_at <= end:
                continue
            scenario_rows_n += 1

            prompt = obj.get("prompt_text")
            if prompt is None or prompt == "":
                empty_prompt_rows_n += 1
                continue
            if not isinstance(prompt, str):
                raise ValueError(
                    f"prompt_text is not a string in source row {source_index}"
                )
            selected.append((source_index, obj))

    selected.sort(key=lambda item: (int(item[1]["arrived_at"]), item[0]))
    selected_before_limit_n = len(selected)
    if limit:
        selected = selected[:limit]

    load_stats = {
        "input_sha256": input_hash.hexdigest(),
        "input_rows_n": input_rows_n,
        "scenario_rows_n": scenario_rows_n,
        "empty_prompt_rows_n": empty_prompt_rows_n,
        "selected_before_limit_n": selected_before_limit_n,
        "selected_n": len(selected),
        "selected_source_indices_sha256": _selected_index_hash(
            source_index for source_index, _ in selected
        ),
    }
    return selected, load_stats


def _retokenize_rows(
    tokenizer: Any,
    selected: list[tuple[int, dict[str, Any]]],
    *,
    max_decode_tokens: int,
    max_context_tokens: int,
    overflow_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overflows: list[dict[str, int | str]] = []
    actual_prompt_tokens: list[int] = []
    trace_prompt_tokens: list[int] = []
    actual_minus_trace: list[int] = []
    source_decode_tokens_all: list[int] = []
    output_decode_tokens_all: list[int] = []
    decode_cap_affected_n = 0
    arrivals: list[int] = []
    sessions: set[str] = set()
    session_id_missing_n = 0

    for source_index, obj in selected:
        prompt = obj["prompt_text"]
        try:
            trace_prompt_tokens_value = int(obj["num_prefill_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid num_prefill_tokens in source row {source_index}"
            ) from exc
        try:
            source_decode_tokens = int(obj["num_decode_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid num_decode_tokens in source row {source_index}"
            ) from exc
        if trace_prompt_tokens_value < 0 or source_decode_tokens < 0:
            raise ValueError(
                f"negative token metadata in source row {source_index}"
            )
        decode_tokens = source_decode_tokens
        decode_was_capped = bool(
            max_decode_tokens and source_decode_tokens > max_decode_tokens
        )
        if decode_was_capped:
            decode_tokens = max_decode_tokens

        try:
            actual = int(_chat_token_count(tokenizer, prompt))
        except Exception:
            # A third-party tokenizer exception is allowed to contain its input.
            # Replace it with a source-index-only diagnostic before it reaches
            # stdout/stderr or a captured experiment log.
            raise ValueError(
                f"tokenization failed for source row {source_index}"
            ) from None
        if actual <= 0:
            raise ValueError(
                f"tokenizer returned no prompt tokens for source row {source_index}"
            )

        if max_context_tokens and actual + decode_tokens > max_context_tokens:
            overflow = {
                "source_request_index": source_index,
                "actual_prompt_tokens": actual,
                "decode_tokens": decode_tokens,
                "total_tokens": actual + decode_tokens,
                "action": overflow_policy,
            }
            overflows.append(overflow)
            if overflow_policy == "drop":
                continue
            # Delay failure until all selected rows have been inspected so the
            # count is deterministic, while never including prompt text.
            continue

        arrived_at = int(obj["arrived_at"])
        session_id = obj.get("session_id")
        if "session_id" not in obj:
            session_id_missing_n += 1
        sessions.add(json.dumps(session_id, sort_keys=True, separators=(",", ":")))

        row = {
            "arrived_at": arrived_at,
            "num_prefill_tokens": actual,
            "uncached_prompt_tokens": actual,
            "num_cached_tokens": 0,
            "num_decode_tokens": decode_tokens,
            "trace_num_decode_tokens": source_decode_tokens,
            "session_id": session_id,
            "prompt_text": prompt,
            "trace_num_prefill_tokens": trace_prompt_tokens_value,
            "source_request_index": source_index,
            "payload_mode": PAYLOAD_MODE,
            "cache_mode": CACHE_MODE,
        }
        rows.append(row)
        if decode_was_capped:
            # Keep this count on the same emitted-row population as the source
            # and output decode distributions below.  Dropped overflow rows are
            # already accounted for by context_overflow_affected_n.
            decode_cap_affected_n += 1
        actual_prompt_tokens.append(actual)
        trace_prompt_tokens.append(trace_prompt_tokens_value)
        actual_minus_trace.append(actual - trace_prompt_tokens_value)
        source_decode_tokens_all.append(source_decode_tokens)
        output_decode_tokens_all.append(decode_tokens)
        arrivals.append(arrived_at)

    if overflows and overflow_policy == "error":
        sample = [
            {
                "source_request_index": event["source_request_index"],
                "actual_prompt_tokens": event["actual_prompt_tokens"],
                "decode_tokens": event["decode_tokens"],
                "total_tokens": event["total_tokens"],
            }
            for event in overflows[:5]
        ]
        raise ValueError(
            f"{len(overflows)} request(s) exceed --max-context-tokens; "
            f"choose --overflow-policy drop. sample={json.dumps(sample)}"
        )

    stats = {
        "n": len(rows),
        "context_overflow_affected_n": len(overflows),
        "context_overflow_events": overflows[:100],
        "unique_session_n": len(sessions),
        "source_session_id_missing_n": session_id_missing_n,
        "arrival": {
            "min": min(arrivals) if arrivals else None,
            "max": max(arrivals) if arrivals else None,
            "span_s": max(arrivals) - min(arrivals) if arrivals else None,
        },
        "actual_prompt_tokens": _distribution(actual_prompt_tokens),
        "original_trace_prompt_tokens": _distribution(trace_prompt_tokens),
        "actual_minus_original_trace_tokens": _distribution(actual_minus_trace),
        "decode_cap_affected_n": decode_cap_affected_n,
        "source_decode_tokens": _distribution(source_decode_tokens_all),
        "output_decode_tokens": _distribution(output_decode_tokens_all),
        "preservation_checks": {
            "prompt_text_exact_n": len(rows),
            "arrived_at_exact_n": len(rows),
            "decode_tokens_exact_n": sum(
                row["num_decode_tokens"] == row["trace_num_decode_tokens"]
                for row in rows
            ),
            "decode_source_provenance_n": len(rows),
            "decode_cap_respected_n": sum(
                not max_decode_tokens
                or row["num_decode_tokens"] <= max_decode_tokens
                for row in rows
            ),
            "session_id_exact_n": len(rows),
            "token_metadata_aligned_n": sum(
                row["num_prefill_tokens"] == row["uncached_prompt_tokens"]
                and row["num_cached_tokens"] == 0
                for row in rows
            ),
        },
    }
    return rows, stats


def _write_jsonl_temp(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, str]:
    """Write and fsync a sibling temp file without publishing it."""
    output_hash = hashlib.sha256()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            tmp_path = Path(target.name)
            for row in rows:
                encoded = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                target.write(encoded)
                output_hash.update(encoded)
            target.flush()
            os.fsync(target.fileno())
        assert tmp_path is not None
        result = tmp_path
        tmp_path = None
        return result, output_hash.hexdigest()
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _write_manifest_temp(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> Path:
    """Write and fsync a sibling manifest temp file without publishing it."""
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            tmp_path = Path(target.name)
            json.dump(manifest, target, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        assert tmp_path is not None
        result = tmp_path
        tmp_path = None
        return result
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def materialize_current_turn_trace(
    *,
    input_path: Path,
    output_path: Path,
    scenario: str,
    tokenizer: Any,
    tokenizer_path: str,
    transformers_version: str,
    limit: int = 0,
    max_decode_tokens: int = 0,
    max_context_tokens: int = 0,
    overflow_policy: str = "error",
    chat_template_source: dict[str, Any] | None = None,
    command_argv: list[str] | None = None,
) -> dict[str, Any]:
    """Build a trace and return its text-free manifest (used by tests/CLI)."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("output must differ from input")
    if scenario not in {*SCENARIOS, "full"}:
        raise ValueError(f"unknown scenario: {scenario}")
    if limit < 0 or max_decode_tokens < 0 or max_context_tokens < 0:
        raise ValueError(
            "limit, max_decode_tokens, and max_context_tokens must be >= 0"
        )
    if overflow_policy not in {"error", "drop"}:
        raise ValueError("overflow_policy must be error or drop")

    selected, load_stats = _load_selected(
        input_path,
        scenario=scenario,
        limit=limit,
    )
    rows, row_stats = _retokenize_rows(
        tokenizer,
        selected,
        max_decode_tokens=max_decode_tokens,
        max_context_tokens=max_context_tokens,
        overflow_policy=overflow_policy,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    tmp_output_path: Path | None = None
    tmp_manifest_path: Path | None = None
    try:
        tmp_output_path, output_sha256 = _write_jsonl_temp(output_path, rows)

        dependency_path = Path(__file__).with_name(
            "materialize_token_aligned_trace.py"
        )
        common_path = Path(__file__).resolve().parents[1] / "router" / "common.py"
        manifest = {
            "schema_version": 1,
            "tool_path": TOOL_PATH,
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {
                "tools/materialize_token_aligned_trace.py": hashlib.sha256(
                    dependency_path.read_bytes()
                ).hexdigest(),
                "router/common.py": hashlib.sha256(
                    common_path.read_bytes()
                ).hexdigest(),
            },
            "command_argv": list(command_argv or []),
            "python_version": sys.version,
            "transformers_version": transformers_version,
            "input": str(input_path),
            "input_sha256": load_stats["input_sha256"],
            "output": str(output_path),
            "output_sha256": output_sha256,
            "scenario": scenario,
            "tokenizer": tokenizer_path,
            "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
            "chat_template_source": chat_template_source or {
                "kind": "tokenizer_default"
            },
            "payload_mode": PAYLOAD_MODE,
            "cache_mode": CACHE_MODE,
            "semantic_scope": "verbatim_current_user_turn_only",
            "token_count_method": {
                "api": "tokenizer.apply_chat_template",
                "messages": [{"role": "user", "content": "<prompt_text>"}],
                "tokenize": True,
                "add_generation_prompt": True,
            },
            "source_fields_deliberately_not_copied": [
                "response_text",
                "block_hash_ids",
                "block_size",
            ],
            "limit": limit or None,
            "max_decode_tokens": max_decode_tokens or None,
            "max_context_tokens": max_context_tokens or None,
            "overflow_policy": overflow_policy,
            **load_stats,
            **row_stats,
        }
        tmp_manifest_path = _write_manifest_temp(manifest_path, manifest)

        # A crash between these replaces yields a hash-mismatched pair, which
        # the matrix driver rejects; no partially written file is published.
        os.replace(tmp_output_path, output_path)
        tmp_output_path = None
        os.replace(tmp_manifest_path, manifest_path)
        tmp_manifest_path = None
    finally:
        for partial in (tmp_output_path, tmp_manifest_path):
            if partial is not None:
                partial.unlink(missing_ok=True)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "full"], required=True)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Hugging Face tokenizer path used by the serving deployment",
    )
    parser.add_argument(
        "--chat-template",
        type=Path,
        help="optional exact chat-template override also configured on the server",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional number of selected requests (0 = all)",
    )
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=0,
        help="optional output-token cap (0 = preserve source decode cap)",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=0,
        help="deployment max model length (0 = unchecked)",
    )
    parser.add_argument(
        "--overflow-policy",
        choices=("error", "drop"),
        default="error",
        help="fail atomically or drop rows whose prompt + decode exceeds context",
    )
    args = parser.parse_args(argv)
    if args.limit < 0 or args.max_decode_tokens < 0 or args.max_context_tokens < 0:
        parser.error(
            "--limit, --max-decode-tokens, and --max-context-tokens must be >= 0"
        )
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")
    return args


def main() -> None:
    args = parse_args()
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise SystemExit(
            "transformers is required in the materialization environment"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    chat_template_source: dict[str, Any] = {"kind": "tokenizer_default"}
    if args.chat_template is not None:
        template = args.chat_template.read_text(encoding="utf-8")
        tokenizer.chat_template = template
        chat_template_source = {
            "kind": "file_override",
            "path": str(args.chat_template.resolve()),
            "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        }

    manifest = materialize_current_turn_trace(
        input_path=args.input,
        output_path=args.output,
        scenario=args.scenario,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        transformers_version=transformers.__version__,
        limit=args.limit,
        max_context_tokens=args.max_context_tokens,
        overflow_policy=args.overflow_policy,
        chat_template_source=chat_template_source,
        command_argv=sys.argv,
        max_decode_tokens=args.max_decode_tokens,
    )
    # Manifest is intentionally text-free; this is safe to tee into run logs.
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
