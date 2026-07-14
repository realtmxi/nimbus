#!/usr/bin/env python3
"""Profile warm TTFT/TPOT for prompt length x local concurrency.

The output is a deployment-specific calibration artifact for ``ttft_pred``.
Prompts are tokenizer-sized and request-unique.  Use a server with prefix
caching disabled for the no-cache calibration leg.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Keep the documented ``python tools/...py`` form runnable from a checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.common import Endpoint, one_request
from tools.materialize_token_aligned_trace import (
    _tokenizer_fingerprint,
    sized_unique_prompt,
)


def _cells(value: str) -> list[tuple[int, int]]:
    parsed = []
    try:
        for item in value.split(","):
            offered, prompt = item.strip().lower().split("x", 1)
            parsed.append((int(offered), int(prompt)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated offered_concurrency x prompt_tokens cells"
        ) from exc
    if not parsed or any(offered <= 0 or prompt <= 0 for offered, prompt in parsed):
        raise argparse.ArgumentTypeError("profile cells must be positive")
    return parsed


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--server-log", type=Path, required=True,
                        help="active vLLM log proving enable_prefix_caching=False")
    parser.add_argument("--server-pid", type=int, required=True,
                        help="recorded parent PID for the active vLLM lifecycle")
    parser.add_argument(
        "--cells", type=_cells,
        default=_cells(
            "1x32,1x512,1x4096,1x32768,8x32,8x512,8x4096,"
            "16x512,16x4096,32x2048,64x1024,128x512"
        ),
        help="comma-separated offered_concurrency x prompt_tokens cells",
    )
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5,
                        help="independent measured batches per cell")
    parser.add_argument("--seed", type=int, default=0,
                        help="randomized measured-block order")
    parser.add_argument("--kv-capacity-tokens", type=int, required=True)
    parser.add_argument("--allow-overload-cells", action="store_true")
    parser.add_argument("--warmup-batch", type=int, default=8)
    parser.add_argument("--warmup-prompt-tokens", type=int, default=256)
    parser.add_argument("--warmup-decode-tokens", type=int, default=16)
    parser.add_argument("--steady-warmup-max-ttft-ms", type=float, default=5000.0)
    parser.add_argument("--nimbus-tick-ms", type=float, default=250.0,
                        help="periodic trigger delay included in guard recommendation")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--salt", default="nimbus-ttft-profile-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.decode_tokens <= 1 or args.repeats < 2:
        parser.error("--decode-tokens must be > 1 and --repeats must be >= 2")
    if min(args.warmup_batch, args.warmup_prompt_tokens,
           args.warmup_decode_tokens) <= 0:
        parser.error("warmup parameters must be positive")
    if args.timeout_s <= 0 or args.kv_capacity_tokens <= 0:
        parser.error("--timeout-s and --kv-capacity-tokens must be positive")
    if args.server_pid <= 0:
        parser.error("--server-pid must be positive")
    if args.nimbus_tick_ms <= 0 or args.steady_warmup_max_ttft_ms <= 0:
        parser.error("--nimbus-tick-ms and --steady-warmup-max-ttft-ms must be positive")
    return args


async def _server_json(session: Any, url: str) -> Any:
    try:
        async with session.get(url, timeout=10) as response:
            return {"status": response.status, "body": await response.json()}
    except Exception as exc:  # fingerprint failure is evidence, not fatal
        return {"error": f"{type(exc).__name__}: {exc}"}


async def run() -> None:
    args = parse_args()
    try:
        os.kill(args.server_pid, 0)
    except OSError as exc:
        raise SystemExit(
            f"recorded server PID is not alive: {args.server_pid} ({exc})"
        ) from exc
    proc_cmdline_path = Path(f"/proc/{args.server_pid}/cmdline")
    if not proc_cmdline_path.is_file():
        raise SystemExit(
            "profile requires Linux /proc evidence for the recorded server PID"
        )
    server_proc_cmdline = proc_cmdline_path.read_bytes()
    if not server_proc_cmdline:
        raise SystemExit("recorded server PID has an empty /proc cmdline")
    server_log = args.server_log.read_bytes()
    if b"enable_prefix_caching=False" not in server_log:
        raise SystemExit(
            "no-cache profile requires an active server log containing "
            "enable_prefix_caching=False"
        )
    try:
        import aiohttp
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - deployment dependencies
        raise SystemExit("aiohttp and transformers are required") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer_fingerprint = _tokenizer_fingerprint(tokenizer)
    base = args.base_url.rstrip("/")
    endpoint = Endpoint(
        name="local", url=base + "/v1/chat/completions", model=args.model
    )
    raw_samples: list[dict[str, Any]] = []
    nonce = 0

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        fingerprint = {
            "version": await _server_json(session, base + "/version"),
            "models": await _server_json(session, base + "/v1/models"),
        }

        async def one_batch(
            offered_concurrency: int,
            prompt_tokens: int,
            decode_tokens: int,
            phase: str,
            repeat: int,
        ) -> list[dict[str, Any]]:
            nonlocal nonce
            requests = []
            for _ in range(offered_concurrency):
                prompt, scheduled_prompt_tokens = sized_unique_prompt(
                    tokenizer,
                    prompt_tokens,
                    salt=args.salt,
                    source_index=nonce,
                )
                req = {
                    "request_id": nonce,
                    "arrived_at": 0,
                    "relative_arrival_s": 0.0,
                    "prompt": prompt,
                    "prompt_tokens": scheduled_prompt_tokens,
                    "uncached_prompt_tokens": scheduled_prompt_tokens,
                    "max_tokens": decode_tokens,
                }
                nonce += 1
                requests.append(req)

            actual_peak_tokens = sum(
                req["prompt_tokens"] + decode_tokens for req in requests
            )
            if (actual_peak_tokens > args.kv_capacity_tokens
                    and not args.allow_overload_cells):
                raise RuntimeError(
                    "materialized profile batch exceeds declared KV capacity: "
                    f"offered={offered_concurrency} target_prompt={prompt_tokens} "
                    f"actual_peak={actual_peak_tokens} "
                    f"capacity={args.kv_capacity_tokens}"
                )

            due = time.perf_counter()
            results = await asyncio.gather(*[
                one_request(
                    session, endpoint, req, due,
                    timeout_s=args.timeout_s,
                    on_output_progress=lambda _tokens: None,
                    temperature=0.0,
                    ignore_eos=True,
                )
                for req in requests
            ])
            rows = []
            for req, result in zip(requests, results):
                row = {
                    "phase": phase,
                    "repeat": repeat,
                    "offered_concurrency": offered_concurrency,
                    "target_prompt_tokens": prompt_tokens,
                    "scheduler_prompt_tokens": req["prompt_tokens"],
                    "decode_tokens_requested": decode_tokens,
                    "actual_batch_estimated_peak_tokens": actual_peak_tokens,
                    "request_id": req["request_id"],
                    **result,
                }
                rows.append(row)
            invalid = [
                row for row in rows
                if not row.get("success")
                or row.get("prompt_tokens") != row["scheduler_prompt_tokens"]
                or row.get("completion_tokens") != decode_tokens
                or row.get("ttft_ms") is None
                or row.get("tpot_ms") is None
            ]
            if invalid:
                sample = [
                    {key: row.get(key) for key in (
                        "request_id", "success", "error_type", "error",
                        "scheduler_prompt_tokens", "prompt_tokens",
                        "decode_tokens_requested", "completion_tokens",
                        "ttft_ms", "tpot_ms",
                    )}
                    for row in invalid[:5]
                ]
                raise RuntimeError(f"invalid profile batch: {sample}")
            return rows

        warmup = await one_batch(
            args.warmup_batch,
            args.warmup_prompt_tokens,
            args.warmup_decode_tokens,
            "warmup_discard",
            0,
        )
        if not all(row.get("success") for row in warmup):
            raise RuntimeError("warmup failed; refusing to profile a bad server")

        warmup_validation = await one_batch(
            args.warmup_batch,
            args.warmup_prompt_tokens,
            args.warmup_decode_tokens,
            "warmup_validate",
            1,
        )
        validation_ttfts = [
            float(row["ttft_ms"]) for row in warmup_validation
            if row.get("success") and row.get("ttft_ms") is not None
        ]
        if (len(validation_ttfts) != len(warmup_validation)
                or max(validation_ttfts) > args.steady_warmup_max_ttft_ms):
            raise RuntimeError(
                "second independent warmup batch is not steady: "
                f"ttft_ms={validation_ttfts}"
            )

        # Warm every requested shape once with independent prompts, then
        # randomize measured blocks so cell identity is not confounded with
        # thermal drift or fixed execution order.
        cell_warmups = []
        for offered_concurrency, prompt_tokens in args.cells:
            cell_warmups.extend(await one_batch(
                offered_concurrency,
                prompt_tokens,
                args.decode_tokens,
                "cell_warmup_discard",
                -1,
            ))

        blocks = [
            (offered_concurrency, prompt_tokens, repeat)
            for offered_concurrency, prompt_tokens in args.cells
            for repeat in range(args.repeats)
        ]
        random.Random(args.seed).shuffle(blocks)
        for offered_concurrency, prompt_tokens, repeat in blocks:
            rows = await one_batch(
                offered_concurrency,
                prompt_tokens,
                args.decode_tokens,
                "measure",
                repeat,
            )
            raw_samples.extend(rows)
            failures = [row for row in rows if not row.get("success")]
            print(
                f"P={prompt_tokens} offered={offered_concurrency} "
                f"repeat={repeat} ok={len(rows)-len(failures)}/{len(rows)}",
                flush=True,
            )

    cells = []
    for offered_concurrency, prompt_tokens in args.cells:
        rows = [
            row for row in raw_samples
            if row["target_prompt_tokens"] == prompt_tokens
            and row["offered_concurrency"] == offered_concurrency
        ]
        repeat_summaries = []
        for repeat in range(args.repeats):
            repeat_rows = [row for row in rows if row["repeat"] == repeat]
            repeat_summaries.append({
                "repeat": repeat,
                "n": len(repeat_rows),
                "scheduler_prompt_median_tokens": statistics.median(
                    int(row["scheduler_prompt_tokens"]) for row in repeat_rows
                ),
                "ttft_median_ms": statistics.median(
                    float(row["ttft_ms"]) for row in repeat_rows
                ),
                "tpot_median_ms": statistics.median(
                    float(row["tpot_ms"]) for row in repeat_rows
                ),
                "e2e_median_ms": statistics.median(
                    float(row["e2e_ms"]) for row in repeat_rows
                ),
            })
        cells.append({
            "prompt_tokens": prompt_tokens,
            "offered_concurrency": offered_concurrency,
            "actual_estimated_peak_tokens_max": max(
                int(row["actual_batch_estimated_peak_tokens"]) for row in rows
            ),
            "fits_declared_kv_capacity": max(
                int(row["actual_batch_estimated_peak_tokens"]) for row in rows
            ) <= args.kv_capacity_tokens,
            "independent_batch_repeats": len(repeat_summaries),
            "repeat_summaries": repeat_summaries,
            "batch_median_ttft_ms": _summary([
                row["ttft_median_ms"] for row in repeat_summaries
            ]),
            "batch_median_tpot_ms": _summary([
                row["tpot_median_ms"] for row in repeat_summaries
            ]),
            "request_pooled_ttft_ms_diagnostic_only": _summary([
                float(row["ttft_ms"]) for row in rows
            ]),
            "request_pooled_tpot_ms_diagnostic_only": _summary([
                float(row["tpot_ms"]) for row in rows
            ]),
        })

    # Reserve one independently scheduled repeat as held-out evidence.  This
    # calibration is deliberately conservative and deployment-specific: the
    # slowest effective prefill rate and slowest repeat-median TPOT from the
    # calibration blocks feed the online predictor.  It is not a claim that
    # the engine literally executes one request at that scalar rate.
    heldout_repeat = args.repeats - 1
    calibration_repeat_summaries = [
        summary
        for cell in cells
        for summary in cell["repeat_summaries"]
        if summary["repeat"] != heldout_repeat
    ]
    recommended_tpot_ms = max(
        float(summary["tpot_median_ms"])
        for summary in calibration_repeat_summaries
    )
    rate_candidates = []
    for summary in calibration_repeat_summaries:
        prompt_tokens = float(summary["scheduler_prompt_median_tokens"])
        prefill_component_s = max(
            0.001,
            (
                float(summary["ttft_median_ms"])
                - recommended_tpot_ms
            ) / 1000.0,
        )
        rate_candidates.append(prompt_tokens / prefill_component_s)
    recommended_prefill_tput = min(rate_candidates)
    if (not math.isfinite(recommended_prefill_tput)
            or recommended_prefill_tput <= 0):
        raise RuntimeError("could not derive a positive prefill calibration")

    heldout_rows = [
        row for row in raw_samples if row["repeat"] == heldout_repeat
    ]
    heldout_underprediction_ms = []
    heldout_predictions_ms = []
    for row in heldout_rows:
        predicted_ms = (
            float(row["scheduler_prompt_tokens"])
            / recommended_prefill_tput * 1000.0
            + recommended_tpot_ms
        )
        heldout_predictions_ms.append(predicted_ms)
        heldout_underprediction_ms.append(
            max(0.0, float(row["ttft_ms"]) - predicted_ms)
        )
    error_guard_ms = math.ceil(
        _percentile(heldout_underprediction_ms, 0.99) or 0.0
    )
    recommended_guard_ms = math.ceil(error_guard_ms + args.nimbus_tick_ms)
    uncovered_n = sum(
        float(row["ttft_ms"]) > predicted_ms + error_guard_ms
        for row, predicted_ms in zip(heldout_rows, heldout_predictions_ms)
    )

    artifact = {
        "schema_version": 1,
        "created_at_unix_s": time.time(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command_argv": sys.argv,
        "python_version": sys.version,
        "transformers_version": transformers.__version__,
        "aiohttp_version": getattr(aiohttp, "__version__", None),
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "salt": args.salt,
        "valid": True,
        "sampling": {"temperature": 0.0, "ignore_eos": True,
                     "continuous_usage_stats": True},
        "kv_capacity_tokens": args.kv_capacity_tokens,
        "profile_config": {
            "cells": [
                {"offered_concurrency": offered, "prompt_tokens": prompt}
                for offered, prompt in args.cells
            ],
            "decode_tokens": args.decode_tokens,
            "repeats": args.repeats,
            "seed": args.seed,
            "allow_overload_cells": args.allow_overload_cells,
            "steady_warmup_max_ttft_ms": args.steady_warmup_max_ttft_ms,
            "nimbus_tick_ms": args.nimbus_tick_ms,
            "timeout_s": args.timeout_s,
        },
        "temporary_directory": tempfile.gettempdir(),
        "chat_template_sha256": hashlib.sha256(
            (tokenizer.chat_template or "").encode()
        ).hexdigest(),
        "cache_mode_required": "none",
        "server_log": str(args.server_log),
        "server_pid": args.server_pid,
        "server_proc_cmdline": server_proc_cmdline.replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip(),
        "server_proc_cmdline_sha256_at_start": hashlib.sha256(
            server_proc_cmdline
        ).hexdigest(),
        "server_log_sha256_at_start": hashlib.sha256(server_log).hexdigest(),
        "server_log_bytes_at_start": len(server_log),
        "warmup": {
            "offered_concurrency": args.warmup_batch,
            "prompt_tokens": args.warmup_prompt_tokens,
            "decode_tokens": args.warmup_decode_tokens,
            "discarded": True,
            "success_n": sum(bool(row.get("success")) for row in warmup),
            "discard_samples": warmup,
            "validation_samples": warmup_validation,
            "cell_warmup_samples": cell_warmups,
        },
        "fingerprint": fingerprint,
        "predictor_calibration": {
            "method": (
                "slowest calibration-block effective prefill rate plus "
                "slowest calibration-block median TPOT; final repeat held out"
            ),
            "scope": (
                "independent no-queue service TTFT under offered concurrency; "
                "queue-level validation is still required before a full matrix"
            ),
            "calibration_repeats": list(range(heldout_repeat)),
            "heldout_repeat": heldout_repeat,
            "recommended_prefill_tput_tokens_per_s": recommended_prefill_tput,
            "recommended_tpot_ms": recommended_tpot_ms,
            "heldout_underprediction_ms": _summary(
                heldout_underprediction_ms
            ),
            "prediction_error_guard_ms_p99": error_guard_ms,
            "nimbus_tick_guard_ms": args.nimbus_tick_ms,
            "recommended_ttft_guard_ms": recommended_guard_ms,
            "heldout_n": len(heldout_rows),
            "heldout_uncovered_after_error_guard_n": uncovered_n,
        },
        "cells": cells,
        "samples": raw_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".tmp",
            delete=False,
        ) as artifact_file:
            tmp_path = Path(artifact_file.name)
            json.dump(artifact, artifact_file, indent=2)
            artifact_file.write("\n")
            artifact_file.flush()
            os.fsync(artifact_file.fileno())
        os.replace(tmp_path, args.output)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    print(f"artifact: {args.output}")


if __name__ == "__main__":
    asyncio.run(run())
