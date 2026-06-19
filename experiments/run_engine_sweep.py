#!/usr/bin/env python3
"""Sweep online Nimbus and baselines under the same run_engine accounting loop.

This is the driver to use for iso-SLO cost curves:
- Nimbus self-selects its outsource fraction using the KV-time budget.
- Baselines sweep explicit fixed fractions.
- all_local and all_cloud run once because their strategies ignore fraction.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from run_engine import BASELINE_POLICIES, run_engine


CONSTANT_FRACTION_POLICIES = {
    "all_local": 0.0,
    "all_cloud": 1.0,
}


def _build_run_args(
    args: argparse.Namespace,
    policy: str,
    fraction: float,
    weight: str,
) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.policy = policy
    run_args.fraction = fraction
    run_args.weight = weight
    return run_args


def _run_tag(policy: str, fraction: float, weight: str) -> str:
    if policy == "nimbus":
        return f"{policy}_{weight}"
    return f"{policy}_f{int(fraction * 100):03d}"


def _planned_runs(args: argparse.Namespace) -> list[tuple[str, float, str]]:
    runs: list[tuple[str, float, str]] = []
    for policy in args.policies:
        if policy == "nimbus":
            for weight in args.nimbus_weights:
                runs.append((policy, args.fractions[0], weight))
        elif policy in CONSTANT_FRACTION_POLICIES:
            runs.append((policy, CONSTANT_FRACTION_POLICIES[policy], args.nimbus_weights[0]))
        else:
            for fraction in args.fractions:
                runs.append((policy, fraction, args.nimbus_weights[0]))
    return runs


def _prepare_output_dir(out_dir: Path, runs: list[tuple[str, float, str]], append: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if append:
        return

    for path in [out_dir / "engine_summary.csv", out_dir / "engine_sweep_config.json"]:
        if path.exists():
            path.unlink()
    for policy, fraction, weight in runs:
        request_csv = out_dir / f"requests_{_run_tag(policy, fraction, weight)}.csv"
        if request_csv.exists():
            request_csv.unlink()


async def run_sweep(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    runs = _planned_runs(args)
    _prepare_output_dir(out_dir, runs, args.append)
    with open(out_dir / "engine_sweep_config.json", "w") as f:
        json.dump(
            {
                "policies": args.policies,
                "fractions": args.fractions,
                "nimbus_weights": args.nimbus_weights,
                "local": args.local,
                "cloud": args.cloud,
                "serving_engine": args.serving_engine,
                "trace_file": args.trace_file,
                "synthetic_burst": args.synthetic_burst,
                "synthetic_n": args.synthetic_n,
                "synthetic_prompt_mode": args.synthetic_prompt_mode,
                "synthetic_prompt_token_cap": args.synthetic_prompt_token_cap,
                "synthetic_prompt_salt": args.synthetic_prompt_salt,
                "scenario": args.scenario,
                "time_scale": args.time_scale,
                "slo_s": args.slo_s,
                "remote_cache_ttl_s": args.remote_cache_ttl_s,
                "tpot_profile": args.tpot_profile,
                "mock_service_model": args.mock_service_model,
                "cloud_ttft_guard_multiplier": args.cloud_ttft_guard_multiplier,
                "in_price": args.in_price,
                "cached_in_price": args.cached_in_price,
                "out_price": args.out_price,
            },
            f,
            indent=2,
        )

    total_runs = len(runs)
    for run_idx, (policy, fraction, weight) in enumerate(runs, start=1):
        label = f"{policy} weight={weight}" if policy == "nimbus" else f"{policy} fraction={fraction:.0%}"
        print(f"\n### Sweep run {run_idx}/{total_runs}: {label}")
        await run_engine(_build_run_args(args, policy, fraction, weight))

    print(f"\nSweep complete. Summary: {out_dir / 'engine_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep run_engine policies")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["nimbus", "cachedisp_oracle", "random", "all_local", "all_cloud"],
        choices=["nimbus", *sorted(BASELINE_POLICIES)],
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--nimbus-weights", nargs="+", choices=["v0", "v1", "v2"], default=["v2"])
    parser.add_argument(
        "--sglang-url",
        "--serving-url",
        dest="sglang_url",
        default="http://localhost:8000",
        help="OpenAI-compatible serving base URL; --sglang-url is a legacy alias",
    )
    parser.add_argument("--serving-engine", choices=["sglang", "vllm"], default="vllm")
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--synthetic-burst", action="store_true", help="in-memory bursty trace")
    parser.add_argument("--synthetic-n", type=int, default=400)
    parser.add_argument(
        "--synthetic-prompt-mode",
        choices=["stub", "sized"],
        default="stub",
        help="stub uses tiny prompt text; sized creates capped word-count prompts",
    )
    parser.add_argument("--synthetic-prompt-token-cap", type=int, default=2048)
    parser.add_argument("--synthetic-prompt-salt", default="")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--output-dir", default="logs/engine_sweep")
    parser.add_argument("--local", choices=["real", "mock"], default="mock")
    parser.add_argument("--cloud", choices=["sim"], default="sim")
    parser.add_argument("--start-hours", type=float, default=0.0)
    parser.add_argument("--duration-hours", type=float, default=1.0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slo-s", type=float, default=5.0)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--admit-kv", type=float, default=0.90)
    parser.add_argument("--local-kv-tokens", type=float, default=200000.0)
    parser.add_argument("--tick-s", type=float, default=0.25)
    parser.add_argument("--mock-cap-s", type=float, default=2.0)
    parser.add_argument("--mock-service-model", choices=["v2", "flops"], default="v2")
    parser.add_argument("--hidden-dim", type=int, default=3584)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--num-heads", type=int, default=28)
    parser.add_argument("--device-tflops", type=float, default=200.0)
    parser.add_argument("--util", type=float, default=0.8)
    parser.add_argument("--prefill-tput", type=float, default=50000.0)
    parser.add_argument("--tpot-s", type=float, default=0.03)
    parser.add_argument("--tpot-profile", default=None)
    parser.add_argument("--in-price", type=float, default=0.15)
    parser.add_argument("--cached-in-price", type=float, default=None)
    parser.add_argument("--out-price", type=float, default=1.20)
    parser.add_argument("--remote-cache-ttl-s", type=float, default=0.0)
    parser.add_argument("--cloud-ttft-ms", type=float, default=900.0)
    parser.add_argument("--cloud-ttft-guard-multiplier", type=float, default=1.5)
    parser.add_argument("--append", action="store_true", help="append to existing engine_summary.csv")
    args = parser.parse_args()

    asyncio.run(run_sweep(args))


if __name__ == "__main__":
    main()
