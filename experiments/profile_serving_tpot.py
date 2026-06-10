#!/usr/bin/env python3
"""Profile decode TPOT for an OpenAI-compatible serving endpoint.

The output JSON matches :mod:`nimbus.tpot_profile` and can be passed to
``experiments/run_engine.py --tpot-profile``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def sized_prompt(prompt_tokens: int) -> str:
    return " ".join(["hello"] * max(1, int(prompt_tokens)))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * q), len(ordered) - 1)
    return ordered[idx]


def make_payload(
    model: str,
    prompt_tokens: int,
    decode_tokens: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    sized_prompt(prompt_tokens)
                    + "\nContinue with the token `x` until the output limit."
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": int(decode_tokens),
        "stream": True,
    }
    if ignore_eos:
        # vLLM accepts this OpenAI-compatible extension; unsupported servers will
        # reject the request, so keep it opt-in.
        payload["ignore_eos"] = True
    return payload


async def stream_once(
    session: Any,
    url: str,
    model: str,
    prompt_tokens: int,
    decode_tokens: int,
    request_id: str,
    ignore_eos: bool,
) -> dict[str, Any]:
    payload = make_payload(model, prompt_tokens, decode_tokens, ignore_eos)
    headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
    start = time.perf_counter()
    first_content_s: float | None = None
    first_choice_s: float | None = None
    chunks = 0
    status = 0
    error = ""
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            status = resp.status
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                chunks += 1
                if first_choice_s is None:
                    first_choice_s = time.perf_counter() - start
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if first_content_s is None and isinstance(content, str) and content:
                    first_content_s = time.perf_counter() - start
    except Exception as exc:  # pragma: no cover - exercised in integration runs
        error = str(exc)[:300]
    latency_s = time.perf_counter() - start
    ttft_s = first_content_s if first_content_s is not None else first_choice_s
    success = status == 200 and ttft_s is not None
    decode_tail_s = max(0.0, latency_s - (ttft_s or latency_s))
    tpot_s = decode_tail_s / max(1, int(decode_tokens) - 1)
    return {
        "request_id": request_id,
        "success": success,
        "status": status,
        "ttft_ms": None if ttft_s is None else ttft_s * 1000.0,
        "latency_ms": latency_s * 1000.0,
        "tpot_ms": tpot_s * 1000.0,
        "chunks": chunks,
        "error": error,
    }


async def run_batch(
    session: Any,
    serving_url: str,
    model: str,
    batch_size: int,
    prompt_tokens: int,
    decode_tokens: int,
    repeat_idx: int,
    ignore_eos: bool,
) -> list[dict[str, Any]]:
    url = f"{serving_url.rstrip('/')}/v1/chat/completions"
    tasks = [
        stream_once(
            session,
            url,
            model,
            prompt_tokens,
            decode_tokens,
            request_id=f"tpot-b{batch_size}-r{repeat_idx}-{i}",
            ignore_eos=ignore_eos,
        )
        for i in range(batch_size)
    ]
    return await asyncio.gather(*tasks)


def summarize_batch(batch_size: int, measurements: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [m for m in measurements if m.get("success")]
    if not ok:
        raise RuntimeError(f"No successful measurements for batch_size={batch_size}")
    tpot = [float(m["tpot_ms"]) for m in ok]
    ttft = [float(m["ttft_ms"]) for m in ok if m.get("ttft_ms") is not None]
    latency = [float(m["latency_ms"]) for m in ok]
    return {
        "batch_size": batch_size,
        "tpot_ms": round(statistics.median(tpot), 4),
        "tpot_p90_ms": round(percentile(tpot, 0.90), 4),
        "ttft_ms": round(statistics.median(ttft), 4) if ttft else None,
        "latency_ms": round(statistics.median(latency), 4),
        "success_count": len(ok),
        "sample_count": len(measurements),
    }


async def profile(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("aiohttp is required for profile_serving_tpot.py") from exc

    all_measurements: dict[int, list[dict[str, Any]]] = {
        int(batch): [] for batch in args.batch_sizes
    }
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for batch in args.batch_sizes:
            batch = int(batch)
            for repeat_idx in range(args.warmup + args.repeats):
                measurements = await run_batch(
                    session,
                    args.serving_url,
                    args.model,
                    batch,
                    args.prompt_tokens,
                    args.decode_tokens,
                    repeat_idx,
                    args.ignore_eos,
                )
                if repeat_idx >= args.warmup:
                    all_measurements[batch].extend(measurements)
                ok = sum(1 for item in measurements if item.get("success"))
                phase = "warmup" if repeat_idx < args.warmup else "profile"
                print(
                    f"[{phase}] batch={batch} repeat={repeat_idx + 1} "
                    f"success={ok}/{len(measurements)}",
                    flush=True,
                )

    tpots = [
        summarize_batch(batch, all_measurements[batch])
        for batch in sorted(all_measurements)
    ]
    raw = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "deployment": {
            "engine": args.engine,
            "model": args.model,
            "gpu": args.gpu,
            "serving_url": args.serving_url,
        },
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "ignore_eos": args.ignore_eos,
        "tpots": tpots,
        "raw_measurements": {
            str(batch): all_measurements[batch]
            for batch in sorted(all_measurements)
        },
    }
    if args.prefill_throughput_tokens_per_s is not None:
        raw["prefill_throughput_tokens_per_s"] = args.prefill_throughput_tokens_per_s
    if args.b_sweet is not None:
        raw["b_sweet"] = args.b_sweet
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile decode TPOT for serving engine")
    parser.add_argument("--serving-url", default="http://127.0.0.1:18200")
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--prefill-throughput-tokens-per-s", type=float, default=None)
    parser.add_argument("--b-sweet", type=int, default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(batch <= 0 for batch in args.batch_sizes):
        raise SystemExit("--batch-sizes must be positive")
    if args.prompt_tokens <= 0 or args.decode_tokens <= 1:
        raise SystemExit("--prompt-tokens must be positive and --decode-tokens must be > 1")
    raw = asyncio.run(profile(args))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
