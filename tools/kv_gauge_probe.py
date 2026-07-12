#!/usr/bin/env python3
"""Live probe: does vllm:kv_cache_usage_perc track TOKENS or CONCURRENT SEQUENCES?

Established (2026-07-12, Qwen3.6-35B-A3B, vLLM v0.19): on hybrid-GDN models the
gauge is dominated by the per-sequence state pool (~2.4%/seq, saturates ~41
running), NOT by token volume. See docs/v3_experiments_2026-07.md Section 4.

stdlib only. Usage:
    python3 tools/kv_gauge_probe.py [BASE_URL] [MODEL]
defaults: http://127.0.0.1:8010  Qwen3.6-35B-A3B

Phases:
  A) fixed tiny prompt, max_tokens=200, concurrency 8 -> 32 -> 64.
     Slot-dominated gauge <=> usage ~ linear in #running despite tiny token totals.
  B) 2 concurrent ~8k-token prompts (per-token contribution check).
"""
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.6-35B-A3B"


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=3).read().decode()
    out = {}
    for ln in txt.splitlines():
        if ln.startswith("vllm:kv_cache_usage_perc{"):
            out["usage"] = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:num_requests_running{"):
            out["running"] = float(ln.rsplit(" ", 1)[1])
        elif ln.startswith("vllm:num_requests_waiting{"):
            out["waiting"] = float(ln.rsplit(" ", 1)[1])
    return out


def one(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=300).read()
    except Exception as e:  # noqa: BLE001 - probe keeps going, error is data
        print("req err:", e)


def phase(name, n, prompt, max_tokens, settle=4, verbose=False):
    print(f"\n=== {name}: {n} concurrent, max_tokens={max_tokens} ===", flush=True)
    ths = [threading.Thread(target=one, args=(f"{prompt} (variant {i})", max_tokens))
           for i in range(n)]
    t0 = time.time()
    for t in ths:
        t.start()
    peak = 0.0
    peak_run = 0
    while any(t.is_alive() for t in ths):
        m = metrics()
        peak = max(peak, m.get("usage", 0))
        peak_run = max(peak_run, int(m.get("running", 0)))
        if verbose:
            print(f"  t={time.time()-t0:5.1f}s running={m.get('running',0):>4.0f} "
                  f"waiting={m.get('waiting',0):>3.0f} usage={100*m.get('usage',0):6.2f}%",
                  flush=True)
        time.sleep(0.5)
    for t in ths:
        t.join()
    time.sleep(settle)
    m = metrics()
    print(f"  PHASE PEAK: usage={100*peak:.2f}% at running<= {peak_run} | "
          f"after settle: usage={100*m.get('usage',0):.2f}%", flush=True)
    return peak, peak_run


if __name__ == "__main__":
    print(f"target {BASE} model {MODEL}")
    print(f"idle baseline: {metrics()}")
    p8, _ = phase("A1", 8, "Count slowly from one to two hundred in English words.", 200)
    p32, _ = phase("A2", 32, "Count slowly from one to two hundred in English words.", 200)
    p64, r64 = phase("A3", 64, "Count slowly from one to two hundred in English words.", 200)
    longp = "The quick brown fox jumps over the lazy dog. " * 900   # ~8-9k tokens
    pB, _ = phase("B", 2, longp, 32)

    print("\n===== SUMMARY =====")
    print(f"A1  8-way small: peak usage {100*p8:.2f}%")
    print(f"A2 32-way small: peak usage {100*p32:.2f}%")
    print(f"A3 64-way small: peak usage {100*p64:.2f}% (engine ran <= {r64})")
    print(f"B  2 x ~8k-token prompt: peak usage {100*pB:.2f}%")
    print("Slot-dominated gauge <=> A-phases scale ~linearly with concurrency while")
    print("far exceeding their token fraction, and B stays near its token fraction.")
