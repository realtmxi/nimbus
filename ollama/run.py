#!/usr/bin/env python3


import argparse
import asyncio
import csv
import json
import math
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


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - i) + xs[hi] * (i - lo)


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
                    "trace_prefill_tokens": obj.get("num_prefill_tokens"),
                    "trace_decode_tokens": obj.get("num_decode_tokens"),
                })

    rows.sort(key=lambda x: (x["arrived_at"], x["request_id"]))
    for i, row in enumerate(rows):
        row["request_id"] = i
    return rows


def make_payload(args: argparse.Namespace, req: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": args.temperature}

    if args.num_ctx is not None:
        options["num_ctx"] = args.num_ctx

    if args.num_predict is not None:
        options["num_predict"] = args.num_predict
    elif args.use_trace_decode and req.get("trace_decode_tokens"):
        options["num_predict"] = max(1, int(req["trace_decode_tokens"]))

    return {
        "model": args.model,
        "prompt": req["prompt"],
        "stream": True,
        "options": options,
    }


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
    chars = 0
    final = {}

    result = {
        "request_id": req["request_id"],
        "arrived_at": req["arrived_at"],
        "relative_arrival_s": req["relative_arrival_s"],
        "scheduled_lag_ms": max(0, (start - due_time) * 1000),
        "model": args.model,
        "success": False,
        "error": None,
        "http_status": None,
        "ttft_ms": None,
        "e2e_ms": None,
        "tpot_ms": None,
        "chunks": 0,
        "chars": 0,
        "trace_prefill_tokens": req.get("trace_prefill_tokens"),
        "trace_decode_tokens": req.get("trace_decode_tokens"),
        "prompt_eval_count": None,
        "eval_count": None,
        "total_duration_ms": None,
        "load_duration_ms": None,
        "prompt_eval_duration_ms": None,
        "eval_duration_ms": None,
        "eval_tps": None,
    }

    headers = {"Content-Type": "application/json"}
    if args.api_key_env and os.environ.get(args.api_key_env):
        headers["Authorization"] = f"Bearer {os.environ[args.api_key_env]}"

    timeout = aiohttp.ClientTimeout(total=args.timeout_s, connect=args.connect_timeout_s)

    try:
        async with session.post(args.url, headers=headers, json=make_payload(args, req), timeout=timeout) as resp:
            result["http_status"] = resp.status

            if resp.status >= 400:
                result["error"] = f"HTTP {resp.status}: {(await resp.text())[:500]}"
                result["e2e_ms"] = (time.perf_counter() - start) * 1000
                return result

            async for raw in resp.content:
                now = time.perf_counter()
                obj = json.loads(raw.decode("utf-8"))

                token = obj.get("response", "")
                if token:
                    first_token_time = first_token_time or now
                    chunks += 1
                    chars += len(token)

                if obj.get("done"):
                    final = obj
                    end_time = now
                    break

        end_time = end_time or time.perf_counter()
        result["success"] = True
        result["e2e_ms"] = (end_time - start) * 1000
        result["chunks"] = chunks
        result["chars"] = chars

        if first_token_time is not None:
            result["ttft_ms"] = (first_token_time - start) * 1000

        result["prompt_eval_count"] = final.get("prompt_eval_count")
        result["eval_count"] = final.get("eval_count")

        for k in ["total_duration", "load_duration", "prompt_eval_duration", "eval_duration"]:
            if final.get(k) is not None:
                result[f"{k}_ms"] = final[k] / 1e6

        if result["eval_count"] and result["eval_duration_ms"]:
            result["eval_tps"] = result["eval_count"] / (result["eval_duration_ms"] / 1000)

        gen_count = result["eval_count"] or chunks
        if gen_count and gen_count > 1 and result["ttft_ms"] is not None:
            result["tpot_ms"] = (result["e2e_ms"] - result["ttft_ms"]) / (gen_count - 1)

    except asyncio.TimeoutError:
        result["error"] = f"timeout after {args.timeout_s}s"
        result["e2e_ms"] = (time.perf_counter() - start) * 1000

    return result


async def replay(args: argparse.Namespace, trace: list[dict[str, Any]]) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.scenario}_{args.model.replace(':', '_')}.jsonl"

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        run_start = time.perf_counter()
        pending = set()
        done_count = 0
        error_count = 0

        async def run(req: dict[str, Any], due: float) -> dict[str, Any]:
            nonlocal done_count, error_count
            res = await one_request(session, args, req, due)
            done_count += 1
            error_count += int(not res["success"])
            if done_count % 100 == 0 or done_count == len(trace):
                print(f"{done_count}/{len(trace)} done, errors={error_count}", flush=True)
            return res

        with out.open("w", encoding="utf-8") as f:
            for req in trace:
                due = run_start + req["relative_arrival_s"] / args.replay_speedup
                wait = due - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)

                pending.add(asyncio.create_task(run(req, due)))

                finished = {t for t in pending if t.done()}
                for t in finished:
                    pending.remove(t)
                    f.write(json.dumps(t.result(), ensure_ascii=False) + "\n")
                if finished:
                    f.flush()

            for t in asyncio.as_completed(pending):
                f.write(json.dumps(await t, ensure_ascii=False) + "\n")
                f.flush()

    return out


def summarize(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ok = [r for r in rows if r["success"]]

    ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    e2e = [r["e2e_ms"] for r in ok if r["e2e_ms"] is not None]
    tpot = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
    eval_tps = [r["eval_tps"] for r in ok if r["eval_tps"] is not None]

    return {
        "scenario": args.scenario,
        "model": args.model,
        "total": len(rows),
        "success": len(ok),
        "errors": len(rows) - len(ok),
        "success_rate": len(ok) / len(rows) if rows else None,
        "ttft_p50_s": None if pct(ttft, 50) is None else pct(ttft, 50) / 1000,
        "ttft_p95_s": None if pct(ttft, 95) is None else pct(ttft, 95) / 1000,
        "e2e_p50_s": None if pct(e2e, 50) is None else pct(e2e, 50) / 1000,
        "e2e_p95_s": None if pct(e2e, 95) is None else pct(e2e, 95) / 1000,
        "tpot_p50_ms": pct(tpot, 50),
        "tpot_p95_ms": pct(tpot, 95),
        "eval_tps_p50": pct(eval_tps, 50),
        "e2e_slo_30s_success": sum(x <= 30000 for x in e2e),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--url", required=True, help="Full endpoint, e.g. http://127.0.0.1:11435/api/generate")
    p.add_argument("--model", required=True)
    p.add_argument("--scenario", choices=SCENARIOS, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--replay-speedup", type=float, default=1.0)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--connect-timeout-s", type=float, default=30.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-ctx", type=int, default=None)
    p.add_argument("--num-predict", type=int, default=None)
    p.add_argument("--use-trace-decode", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    trace = load_trace(args.data, args.scenario)
    print(f"loaded {len(trace)} requests")

    raw_path = await replay(args, trace)
    summary = summarize(raw_path, args)

    summary_path = args.out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    print(f"raw: {raw_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
