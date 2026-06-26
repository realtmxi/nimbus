#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import aiohttp


SCENARIOS = {
    "normal": (1477007, 1478206),
    "burst": (1698942, 1702541),
    "burst_1200": (1700032, 1701231),
    "burst_300": (1700593, 1700892),
    "extreme_burst_1200": (1260532, 1261731),
    "extreme_burst": (1255679, 1266478),
}

DEFAULT_TIMEOUT_S: float = float(os.environ.get("TIMEOUT_S", "600"))


async def get_spec_metrics(session: aiohttp.ClientSession, api_url: str) -> tuple[str, dict[str, float]]:
    spec_url = api_url.split("/v1/", 1)[0].rstrip("/") + "/metrics"

    async with session.get(spec_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        text = await resp.text()

    values: dict[str, float] = {}

    for line in text.splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "spec_decode" not in line:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        name = parts[0].split("{", 1)[0]

        if name.endswith("_total"):
            name = name[:-6]

        value = float(parts[1])

        if name.startswith("vllm:spec_decode_"):
            values[name] = values.get(name, 0.0) + value

    return spec_url, values


def load_trace(path: Path, scenario: str) -> list[dict[str, Any]]:
    start, end = SCENARIOS[scenario]
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            arrived_at = int(obj["arrived_at"])
            prompt = obj.get("prompt_text", "")

            if start <= arrived_at <= end and prompt:
                rows.append({
                    "request_id": len(rows),
                    "arrived_at": arrived_at,
                    "relative_arrival_s": arrived_at - start,
                    "prompt": prompt,
                    "max_tokens": int(obj["num_decode_tokens"]),
                })

    rows.sort(key=lambda x: (x["arrived_at"], x["request_id"]))

    for i, row in enumerate(rows):
        row["request_id"] = i

    return rows


def make_payload(args: argparse.Namespace, req: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": args.model,
        "stream": True,
        "max_tokens": args.max_tokens if args.max_tokens is not None else req["max_tokens"],
        "messages": [{"role": "user", "content": req["prompt"]}],
        "stream_options": {"include_usage": True},
    }


async def one_request(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    req: dict[str, Any],
    due_time: float,
    *,
    chunk_file=None,
    chunk_lock: asyncio.Lock | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    first_token_time: float | None = None
    end_time: float | None = None
    chunks = 0
    usage: dict[str, Any] = {}
    output_parts: list[str] = []
    # Per-chunk metrics
    last_chunk_time: float | None = None
    chunk_intervals_ms: list[float] = []
    chunk_sizes_chars: list[int] = []
    cumulative_chars = 0

    result = {
        "request_id": req["request_id"],
        "arrived_at": req["arrived_at"],
        "relative_arrival_s": req["relative_arrival_s"],
        "scheduled_lag_ms": max(0, (start - due_time) * 1000),
        "model": args.model,
        "success": False,
        "error": None,
        "error_type": None,
        "http_status": None,
        "ttft_ms": None,
        "e2e_ms": None,
        "tpot_ms": None,
        "chunks": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "output_text": "",
    }

    headers = {"Content-Type": "application/json"}

    if args.api_key_env and os.environ.get(args.api_key_env):
        headers["Authorization"] = f"Bearer {os.environ[args.api_key_env]}"

    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT_S)

    def record_error(error_type: str, message: str) -> None:
        result["error_type"] = error_type
        result["error"] = message
        result["e2e_ms"] = (time.perf_counter() - start) * 1000
        result["chunks"] = chunks

        if first_token_time is not None:
            result["ttft_ms"] = (first_token_time - start) * 1000

    try:
        async with session.post(
            args.url,
            headers=headers,
            json=make_payload(args, req),
            timeout=timeout,
        ) as resp:
            result["http_status"] = resp.status

            if resp.status >= 400:
                error_text = (await resp.text())[:500]
                record_error(f"HTTP {resp.status}", f"HTTP {resp.status}: {error_text}")
                return result

            buffer = ""
            done = False

            async for raw in resp.content.iter_chunked(8192):
                buffer += raw.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    if data == "[DONE]":
                        end_time = time.perf_counter()
                        done = True
                        break

                    obj = json.loads(data)

                    if obj.get("error"):
                        error = obj["error"]
                        message = error.get("message") if isinstance(error, dict) else str(error)
                        record_error("StreamError", message)
                        return result

                    if obj.get("usage"):
                        usage = obj["usage"]

                    choices = obj.get("choices") or []
                    delta = choices[0].get("delta") if choices else {}
                    token = delta.get("content") if isinstance(delta, dict) else ""

                    if token:
                        now = time.perf_counter()
                        first_token_time = first_token_time or now
                        chunks += 1
                        output_parts.append(token)
                        # Update per-chunk metrics
                        interval_ms: float
                        if last_chunk_time is None:
                            interval_ms = (now - start) * 1000.0  # equals TTFT for first chunk
                        else:
                            interval_ms = (now - last_chunk_time) * 1000.0
                        last_chunk_time = now
                        size_chars = len(token)
                        cumulative_chars += size_chars
                        chunk_intervals_ms.append(interval_ms)
                        chunk_sizes_chars.append(size_chars)
                        # Persist a chunk event immediately if requested
                        if chunk_file is not None:
                            rec = {
                                "type": "chunk",
                                "request_id": req["request_id"],
                                "chunk_index": chunks,
                                "relative_arrival_s": req["relative_arrival_s"],
                                "scheduled_lag_ms": max(0, (start - due_time) * 1000),
                                "rel_time_ms": (now - start) * 1000.0,
                                "delta_ms": interval_ms,
                                "size_chars": size_chars,
                                "cumulative_chars": cumulative_chars,
                            }
                            if chunk_lock is not None:
                                async with chunk_lock:
                                    chunk_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                    chunk_file.flush()
                            else:
                                chunk_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                chunk_file.flush()

                if done:
                    break

        end_time = end_time or time.perf_counter()

        result["success"] = True
        result["e2e_ms"] = (end_time - start) * 1000
        result["chunks"] = chunks

        if first_token_time is not None:
            result["ttft_ms"] = (first_token_time - start) * 1000

        result["prompt_tokens"] = usage.get("prompt_tokens")
        result["completion_tokens"] = usage.get("completion_tokens")
        result["output_text"] = "".join(output_parts)
        result["output_tokens"] = output_parts

        gen_count = result["completion_tokens"] or chunks
        if gen_count and gen_count > 1 and result["ttft_ms"] is not None:
            result["tpot_ms"] = (result["e2e_ms"] - result["ttft_ms"]) / (gen_count - 1)

        # Removed per-request chunk derived stats

    except asyncio.TimeoutError:
        record_error("TimeoutError", f"timeout after {DEFAULT_TIMEOUT_S}s")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        record_error(type(exc).__name__, str(exc))
    except aiohttp.ClientError as exc:
        record_error(type(exc).__name__, str(exc) or repr(exc))
    except Exception as exc:
        record_error(type(exc).__name__, str(exc) or repr(exc))

    return result


async def replay(args: argparse.Namespace, trace: list[dict[str, Any]]) -> Path:
    model_slug = args.model.replace(":", "_").replace("/", "_")
    out = args.output or Path(f"{args.scenario}_{model_slug}.jsonl")

    if not out.is_absolute():
        out = args.out_dir / out

    out.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        spec_url, spec_before = await get_spec_metrics(session, args.url)

        run_start = time.perf_counter()
        pending = set()

        # Prepare a streaming chunk log living next to the raw results
        chunk_path = Path(str(out) + ".chunks.jsonl")
        chunk_lock = asyncio.Lock()
        # Open once for the whole run; close after all tasks complete
        chunk_file = chunk_path.open("a", encoding="utf-8")

        with out.open("w", encoding="utf-8") as f:
            for req in trace:
                due = run_start + req["relative_arrival_s"]
                wait = due - time.perf_counter()

                if wait > 0:
                    await asyncio.sleep(wait)

                pending.add(asyncio.create_task(
                    one_request(session, args, req, due, chunk_file=chunk_file, chunk_lock=chunk_lock)
                ))

                finished = {task for task in pending if task.done()}

                for task in finished:
                    pending.remove(task)
                    f.write(json.dumps(task.result(), ensure_ascii=False) + "\n")

                if finished:
                    f.flush()

            for task in asyncio.as_completed(pending):
                f.write(json.dumps(await task, ensure_ascii=False) + "\n")
                f.flush()

        # Close chunk file after all tasks have drained
        chunk_file.close()

        _, spec_after = await get_spec_metrics(session, args.url)

    spec_delta = {
        k: spec_after.get(k, 0.0) - spec_before.get(k, 0.0)
        for k in sorted(set(spec_before) | set(spec_after))
    }

    drafts = spec_delta.get("vllm:spec_decode_num_drafts", 0.0)
    draft_tokens = spec_delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    accepted_tokens = spec_delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)

    spec_result = {
        "metrics_url": spec_url,
        "spec_delta": spec_delta,
        "derived": {
            "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
            "avg_draft_tokens_per_draft": draft_tokens / drafts if drafts else None,
            "avg_accepted_tokens_per_draft": accepted_tokens / drafts if drafts else None,
            "approx_avg_committed_tokens_per_decode_step": (
                1 + accepted_tokens / drafts if drafts else None
            ),
        },
    }

    spec_path = Path(str(out) + ".spec_metrics.json")
    spec_path.write_text(
        json.dumps(spec_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"spec metrics: {spec_path}")

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--url",
        required=True,
        help="Full endpoint, e.g. http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)

    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    trace = load_trace(args.data, args.scenario)

    print(f"loaded {len(trace)} requests")
    raw_path = await replay(args, trace)
    print(f"raw: {raw_path}")


if __name__ == "__main__":
    asyncio.run(main())
