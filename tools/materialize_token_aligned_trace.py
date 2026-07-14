#!/usr/bin/env python3
"""Materialize a no-cache trace whose payload tokens match scheduler metadata.

The primary ShareGPT/BurstGPT trace stores cumulative conversation-token counts
but its ``prompt_text`` field is only the current user turn.  Sending that field
directly can turn a scheduler-visible 1,000-token request into a 20-token HTTP
payload.  That is fatal to resource/selector experiments.

This tool preserves arrival times and decode lengths while replacing each prompt
with deterministic, request-unique text.  It uses the deployment tokenizer and
chat template to make ``num_prefill_tokens`` equal the actual payload length.
The output deliberately represents a *no-cache* experiment; run the serving
engine with prefix caching disabled and treat cache-aware evaluation as a
separate leg.

Example (on the GPU host):

    python tools/materialize_token_aligned_trace.py \
      --input /path/sharegpt_prompts_burstgpt_timestamps.jsonl \
      --output /scratch/me/extreme_token_aligned.jsonl \
      --scenario extreme_burst_1200 \
      --tokenizer /path/Qwen3-32B \
      --salt dense32b-nocache-v1
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
from typing import Any

# Keep the documented ``python tools/...py`` form runnable from a checkout.
# When Python executes a file by path it otherwise puts only ``tools/`` on
# sys.path, so sibling package ``router`` cannot be imported.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.common import SCENARIOS


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _chat_token_count(tokenizer: Any, text: str) -> int:
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(tokens, "shape"):
        return int(tokens.shape[-1])
    return len(tokens)


def _tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    """Stable tokenizer evidence beyond a mutable model-directory path."""
    vocab_hash = hashlib.sha256()
    for token, index in sorted(
        tokenizer.get_vocab().items(), key=lambda item: (item[1], item[0])
    ):
        vocab_hash.update(str(index).encode())
        vocab_hash.update(b"\0")
        vocab_hash.update(token.encode("utf-8", errors="surrogatepass"))
        vocab_hash.update(b"\n")
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": len(tokenizer.get_vocab()),
        "vocab_sha256": vocab_hash.hexdigest(),
        "chat_template_sha256": hashlib.sha256(
            (getattr(tokenizer, "chat_template", None) or "").encode()
        ).hexdigest(),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "revision": init_kwargs.get("_commit_hash"),
    }


def _unique_prefix(salt: str, source_index: int) -> str:
    # Put request-specific material immediately after the common chat-template
    # tokens.  It therefore differs inside the first KV block instead of sharing
    # a long repeated filler prefix across requests.
    digest = hashlib.sha256(f"{salt}\0{source_index}".encode()).hexdigest()[:12]
    return f"{digest} "


def sized_unique_prompt(
    tokenizer: Any,
    target_tokens: int,
    *,
    salt: str,
    source_index: int,
) -> tuple[str, int]:
    """Return deterministic text with chat-template count nearest ``target``."""
    prefix = _unique_prefix(salt, source_index)
    filler = " calibration"

    cache: dict[int, tuple[str, int]] = {}

    def candidate(repetitions: int) -> tuple[str, int]:
        repetitions = max(0, repetitions)
        if repetitions not in cache:
            text = prefix + filler * repetitions
            cache[repetitions] = (text, _chat_token_count(tokenizer, text))
        return cache[repetitions]

    target = max(1, int(target_tokens))
    base_text, base_count = candidate(0)
    if base_count >= target:
        # Unique payloads have a minimum chat-template + nonce size.  Record the
        # actual count in output metadata instead of pretending the target fit.
        return base_text, base_count

    lo, hi = 0, max(1, target - base_count)
    while candidate(hi)[1] < target:
        lo = hi
        hi *= 2

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if candidate(mid)[1] < target:
            lo = mid
        else:
            hi = mid

    choices = [candidate(lo), candidate(hi)]
    return min(choices, key=lambda item: (abs(item[1] - target), item[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "full"], required=True)
    parser.add_argument("--tokenizer", required=True,
                        help="Hugging Face model/tokenizer path used by the server")
    parser.add_argument("--salt", required=True,
                        help="stable workload id included in each unique prompt")
    parser.add_argument("--limit", type=int, default=0,
                        help="optional number of selected requests (0 = all)")
    parser.add_argument("--max-prompt-tokens", type=int, default=0,
                        help="optional target-token cap (0 = preserve trace target)")
    parser.add_argument("--max-decode-tokens", type=int, default=0,
                        help="optional decode cap applied uniformly (0 = preserve)")
    parser.add_argument("--max-context-tokens", type=int, default=0,
                        help="deployment max model length (0 = unchecked)")
    parser.add_argument(
        "--overflow-policy", choices=("error", "cap", "drop"), default="error",
        help="what to do when prompt target + decode exceeds max context",
    )
    args = parser.parse_args()
    if min(args.limit, args.max_prompt_tokens, args.max_decode_tokens,
           args.max_context_tokens) < 0:
        parser.error("--limit/token limits must be >= 0")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")
    return args


def main() -> None:
    args = parse_args()
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise SystemExit("transformers is required in the materialization environment") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer_fingerprint = _tokenizer_fingerprint(tokenizer)
    if args.scenario == "full":
        start, end = -math.inf, math.inf
    else:
        start, end = SCENARIOS[args.scenario]

    selected: list[tuple[int, dict[str, Any]]] = []
    input_hash = hashlib.sha256()
    with args.input.open("rb") as source:
        for source_index, raw in enumerate(source):
            input_hash.update(raw)
            obj = json.loads(raw)
            arrived_at = int(obj["arrived_at"])
            if start <= arrived_at <= end and obj.get("prompt_text"):
                selected.append((source_index, obj))

    selected.sort(key=lambda item: (int(item[1]["arrived_at"]), item[0]))
    if args.limit:
        selected = selected[:args.limit]

    selected_n = len(selected)
    prompt_cap_affected = 0
    decode_cap_affected = 0
    overflow_events: list[dict[str, Any]] = []
    prepared: list[tuple[int, dict[str, Any], int, int]] = []
    for source_index, obj in selected:
        trace_tokens = int(obj.get("num_prefill_tokens") or 0)
        trace_decode_tokens = int(obj["num_decode_tokens"])
        decode_tokens = trace_decode_tokens
        if args.max_decode_tokens and decode_tokens > args.max_decode_tokens:
            decode_tokens = args.max_decode_tokens
            decode_cap_affected += 1
        desired = trace_tokens
        if args.max_prompt_tokens and desired > args.max_prompt_tokens:
            desired = args.max_prompt_tokens
            prompt_cap_affected += 1

        if (args.max_context_tokens
                and desired + decode_tokens > args.max_context_tokens):
            event = {
                "source_request_index": source_index,
                "trace_prompt_tokens": trace_tokens,
                "target_prompt_tokens_before_context_policy": desired,
                "decode_tokens": decode_tokens,
            }
            if args.overflow_policy == "error":
                overflow_events.append({**event, "action": "error"})
                continue
            if args.overflow_policy == "drop" or decode_tokens >= args.max_context_tokens:
                overflow_events.append({**event, "action": "drop"})
                continue
            desired = args.max_context_tokens - decode_tokens
            overflow_events.append({
                **event, "action": "cap", "adjusted_prompt_tokens": desired,
            })
        prepared.append((source_index, obj, desired, decode_tokens))

    if any(event["action"] == "error" for event in overflow_events):
        sample = overflow_events[:5]
        raise SystemExit(
            "context overflow; choose --overflow-policy cap|drop. sample="
            + json.dumps(sample, ensure_ascii=False)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    output_hash = hashlib.sha256()
    deltas: list[float] = []
    trace_deltas: list[float] = []
    tmp_output_path: Path | None = None
    tmp_manifest_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            tmp_output_path = Path(target.name)
            for source_index, obj, desired, decode_tokens in prepared:
                trace_tokens = int(obj.get("num_prefill_tokens") or 0)
                prompt, actual = sized_unique_prompt(
                    tokenizer,
                    desired,
                    salt=args.salt,
                    source_index=source_index,
                )
                if (args.max_context_tokens
                        and actual + decode_tokens > args.max_context_tokens):
                    # A very short target can be smaller than the unique nonce
                    # plus chat template. Never emit an invalid request silently.
                    raise SystemExit(
                        f"materialized request {source_index} still exceeds context: "
                        f"{actual}+{decode_tokens}>{args.max_context_tokens}"
                    )
                deltas.append(float(actual - desired))
                trace_deltas.append(float(actual - trace_tokens))
                row = {
                    "arrived_at": int(obj["arrived_at"]),
                    "num_prefill_tokens": actual,
                    "uncached_prompt_tokens": actual,
                    "num_cached_tokens": 0,
                    "num_decode_tokens": decode_tokens,
                    "trace_num_decode_tokens": int(obj["num_decode_tokens"]),
                    "session_id": obj.get("session_id"),
                    "prompt_text": prompt,
                    "trace_num_prefill_tokens": trace_tokens,
                    "materialized_target_tokens": desired,
                    "source_request_index": source_index,
                    "payload_mode": "token_aligned_unique",
                    "cache_mode": "none",
                }
                encoded = (
                    json.dumps(
                        row, ensure_ascii=False, separators=(",", ":")
                    ) + "\n"
                ).encode()
                target.write(encoded)
                output_hash.update(encoded)
            target.flush()
            os.fsync(target.fileno())

        manifest = {
            "schema_version": 1,
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "command_argv": sys.argv,
            "python_version": sys.version,
            "transformers_version": transformers.__version__,
            "input": str(args.input),
            "input_sha256": input_hash.hexdigest(),
            "output": str(args.output),
            "output_sha256": output_hash.hexdigest(),
            "scenario": args.scenario,
            "tokenizer": args.tokenizer,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "salt": args.salt,
            "cache_mode": "none",
            "limit": args.limit or None,
            "selected_n": selected_n,
            "n": len(prepared),
            "max_prompt_tokens": args.max_prompt_tokens or None,
            "prompt_cap_affected_n": prompt_cap_affected,
            "max_decode_tokens": args.max_decode_tokens or None,
            "decode_cap_affected_n": decode_cap_affected,
            "max_context_tokens": args.max_context_tokens or None,
            "overflow_policy": args.overflow_policy,
            "context_overflow_affected_n": len(overflow_events),
            "context_overflow_events": overflow_events[:100],
            "actual_minus_target_tokens": {
                "exact_n": sum(delta == 0 for delta in deltas),
                "changed_n": sum(delta != 0 for delta in deltas),
                "p50": _percentile(deltas, 0.50),
                "p95": _percentile(deltas, 0.95),
                "min": min(deltas) if deltas else None,
                "max": max(deltas) if deltas else None,
            },
            "actual_minus_original_trace_tokens": {
                "exact_n": sum(delta == 0 for delta in trace_deltas),
                "changed_n": sum(delta != 0 for delta in trace_deltas),
                "p50": _percentile(trace_deltas, 0.50),
                "p95": _percentile(trace_deltas, 0.95),
                "min": min(trace_deltas) if trace_deltas else None,
                "max": max(trace_deltas) if trace_deltas else None,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as manifest_file:
            tmp_manifest_path = Path(manifest_file.name)
            json.dump(manifest, manifest_file, indent=2)
            manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())

        # A crash between these replaces yields a hash-mismatched pair, which
        # the matrix driver rejects; it never yields a silently partial trace.
        os.replace(tmp_output_path, args.output)
        tmp_output_path = None
        os.replace(tmp_manifest_path, manifest_path)
        tmp_manifest_path = None
    finally:
        for partial in (tmp_output_path, tmp_manifest_path):
            if partial is not None:
                partial.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
