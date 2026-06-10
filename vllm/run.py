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
    payload = {
        "model": args.model,
        "stream": True,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens if args.max_tokens is not None else req["max_tokens"],
        "messages": [{"role": "user", "content": req["prompt"]}],
    }

    if args.top_p is not None:
        payload["top_p"] = args.top_p

    return payload


async def one_request(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    req: dict[str, Any],
    due_time: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    first_token_time = None
    end_time = None
    chunks = 0
    usage: dict[str, Any] = {}
    raw_usage_events: list[dict[str, Any]] = []
    output_parts: list[str] = []

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
        "total_tokens": None,
        "raw_usage_events": [],
        "output_text": "",
    }

    headers = {"Content-Type": "application/json"}
    if args.api_key_env and os.environ.get(args.api_key_env):
        headers["Authorization"] = f"Bearer {os.environ[args.api_key_env]}"

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)

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
                        raw_usage_events.append(usage)

                    choices = obj.get("choices") or []
                    delta = choices[0].get("delta") if choices else {}
                    token = delta.get("content") if isinstance(delta, dict) else ""

                    if token:
                        now = time.perf_counter()
                        first_token_time = first_token_time or now
                        chunks += 1
                        output_parts.append(token)

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
        result["total_tokens"] = usage.get("total_tokens")
        result["raw_usage_events"] = raw_usage_events
        result["output_text"] = "".join(output_parts)

        gen_count = result["completion_tokens"] or chunks
        if gen_count and gen_count > 1 and result["ttft_ms"] is not None:
            result["tpot_ms"] = (result["e2e_ms"] - result["ttft_ms"]) / (gen_count - 1)

    except asyncio.TimeoutError:
        record_error("TimeoutError", f"timeout after {args.timeout_s}s")
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
        run_start = time.perf_counter()
        pending = set()

        with out.open("w", encoding="utf-8") as f:
            for req in trace:
                due = run_start + req["relative_arrival_s"]
                wait = due - time.perf_counter()

                if wait > 0:
                    await asyncio.sleep(wait)

                pending.add(asyncio.create_task(one_request(session, args, req, due)))

                finished = {task for task in pending if task.done()}
                for task in finished:
                    pending.remove(task)
                    f.write(json.dumps(task.result(), ensure_ascii=False) + "\n")

                if finished:
                    f.flush()

            for task in asyncio.as_completed(pending):
                f.write(json.dumps(await task, ensure_ascii=False) + "\n")
                f.flush()

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--url", required=True, help="Full endpoint, e.g. http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
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
