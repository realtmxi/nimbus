#!/usr/bin/env bash
# Cache Displacement vs FLOP knee sweep
# Runs 4 strategies (flop_based, cache_disp, session_aware, oracle_size)
# across a fraction grid to find the knee point.
#
# Expected runtime: ~1h trace × 5x speedup × 4 strategies × N fractions
# At 7 fractions: ~7h wall time
#
# Usage: bash run_cachedisp_knee.sh [output_tag]

set -euo pipefail

TAG="${1:-knee1}"
SGLANG_URL="http://localhost:8200"
MODEL="Qwen2.5-7B-Instruct"
TRACE="data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl"
OUT="logs/cachedisp_${TAG}"
PY="/home/murphy/local-deployment/.venv/bin/python"
SCRIPT="experiments/local_deployment/phase2/run_offload_strategies.py"

STRATEGIES="flop_based cache_disp session_aware oracle_size"
FRACTIONS="0.0 0.15 0.20 0.25 0.30 0.35 0.50"
SEED=42
START_HOURS=0.8
DURATION_HOURS=1.0
TIME_SCALE=5.0

mkdir -p "$OUT"
echo "=== Cache Displacement Knee Sweep ===" | tee "$OUT/driver.log"
echo "Started: $(date)" | tee -a "$OUT/driver.log"
echo "Output:  $OUT" | tee -a "$OUT/driver.log"
echo "Fractions: $FRACTIONS" | tee -a "$OUT/driver.log"
echo "Strategies: $STRATEGIES" | tee -a "$OUT/driver.log"
echo "" | tee -a "$OUT/driver.log"

for frac in $FRACTIONS; do
    echo "========================================" | tee -a "$OUT/driver.log"
    echo "FRACTION $frac @ $(date)" | tee -a "$OUT/driver.log"
    echo "========================================" | tee -a "$OUT/driver.log"

    $PY $SCRIPT \
        --sglang-url "$SGLANG_URL" \
        --model "$MODEL" \
        --trace-file "$TRACE" \
        --start-hours $START_HOURS \
        --duration-hours $DURATION_HOURS \
        --time-scale $TIME_SCALE \
        --mode compare \
        --fraction $frac \
        --strategies $STRATEGIES \
        --output-dir "$OUT" \
        --seed $SEED 2>&1 | tee -a "$OUT/driver.log"

    # Give the server 30s cooldown between fractions
    sleep 30
done

echo "" | tee -a "$OUT/driver.log"
echo "Completed: $(date)" | tee -a "$OUT/driver.log"
