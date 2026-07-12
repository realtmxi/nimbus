#!/usr/bin/env python3
"""Reconstruct a router-run timeline from its result JSONL and audit nimbus kicks.

This is the audit that exposed the hybrid-GDN gauge finding (docs/
v3_experiments_2026-07.md Sections 3.3-4): it answers "was there real token-KV
pressure at the moments the policy kicked?" from the per-request records alone.

Usage:
    python3 tools/analyze_eb1200.py RESULTS.jsonl [KV_CAPACITY_TOKENS]
default capacity: 216512.
"""
import json
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit(__doc__)
PATH = sys.argv[1]
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 216_512

rows = [json.loads(l) for l in open(PATH)]
t0 = min(r["relative_arrival_s"] for r in rows)
for r in rows:
    r["t"] = r["relative_arrival_s"] - t0

local = [r for r in rows if r["endpoint"] == "local"]
cloud = [r for r in rows if r["endpoint"] == "cloud"]
print(f"n={len(rows)} local={len(local)} cloud(kicked)={len(cloud)}")
dur = max(r["t"] for r in rows) or 1
print(f"trace span: {dur:.0f}s   overall rate={len(rows)/dur:.1f} req/s")

# --- arrival & kick rate per 10s bucket ---
buck_arr = defaultdict(int)
buck_kick = defaultdict(int)
buck_ptok = defaultdict(int)
for r in rows:
    b = int(r["t"] // 10) * 10
    buck_arr[b] += 1
    buck_ptok[b] += r["prompt_tokens"]
    if r["endpoint"] == "cloud":
        buck_kick[b] += 1

print("\n== arrival & kick rate per 10s bucket (buckets with kicks, first 40) ==")
print(f"{'t':>6} {'arr':>5} {'kick':>5} {'kick%':>6} {'avg_ptok':>8}")
shown = 0
for b, n in sorted(buck_arr.items()):
    k = buck_kick.get(b, 0)
    if k > 0:
        print(f"{b:>6} {n:>5} {k:>5} {100*k/max(n,1):>5.0f}% {buck_ptok[b]/max(n,1):>8.0f}")
        shown += 1
        if shown >= 40:
            print("... (truncated)")
            break


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


kp = [r["prompt_tokens"] for r in cloud]
lp = [r["prompt_tokens"] for r in local]
kd = [r["completion_tokens"] for r in cloud]   # cloud rows: trace max_tokens (REPLACE semantics)
ld = [r["completion_tokens"] for r in local]   # local rows: actual generated
print("\n== kicked vs kept characteristics ==")
if cloud:
    print(f"kicked prompt p50={q(kp,.5)} p95={q(kp,.95)}  decode(trace) p50={q(kd,.5)} p95={q(kd,.95)}")
    print(f"kicked footprint p50={q([a+b for a, b in zip(kp, kd)],.5)}")
print(f"local  prompt p50={q(lp,.5)} p95={q(lp,.95)}  decode(actual) p50={q(ld,.5)} p95={q(ld,.95)}")
print(f"local  footprint p50={q([a+b for a, b in zip(lp, ld)],.5)}")

# --- state at each kick moment (footprint UPPER bound: full footprint while inflight) ---
events = []
for r in local:
    d = r["t"] + r["queue_delay_ms"] / 1000.0
    c = r["t"] + r["e2e_ms"] / 1000.0
    events.append((d, +1, r["prompt_tokens"] + r["completion_tokens"]))
    events.append((c, -1, r["prompt_tokens"] + r["completion_tokens"]))
events.sort()
kick_times = sorted(r["t"] for r in cloud)

inflight = 0
kv_committed = 0
ei = 0
samples = []
for kt in kick_times:
    while ei < len(events) and events[ei][0] <= kt:
        _, s, fp = events[ei]
        inflight += s
        kv_committed += s * fp
        ei += 1
    samples.append((inflight, kv_committed))

if samples:
    infs = [s[0] for s in samples]
    kvs = [s[1] for s in samples]
    print(f"\n== state AT the {len(kick_times)} kick moments ==")
    print(f"inflight: p50={q(infs,.5)} max={max(infs)}")
    print(f"KV committed UPPER bound: p50={q(kvs,.5):,} p95={q(kvs,.95):,} "
          f"max={max(kvs):,}  (capacity={CAP:,})")
    for thresh in (0.5, 0.9):
        frac = sum(1 for v in kvs if v > CAP * thresh) / len(kvs)
        print(f"kicks with committed bound > {int(100*thresh)}% capacity: {frac:.1%}")
    print("If the upper bound never approaches capacity, token-KV pressure did not")
    print("exist at kick time: the trigger fired on something else (see docs).")
    print(f"\nkicks span t=[{kick_times[0]:.0f}s .. {kick_times[-1]:.0f}s] of [0..{dur:.0f}s]")
