#!/usr/bin/env bash
# Near-knee repeat runs for error bars
# 3 seeds x 3 fractions x 2 strategies (cache_disp, flop_based) = 18 runs
# Plus session_aware for comparison = 27 runs
# ~20 min per run = ~9h

set -euo pipefail

SGLANG_URL="http://localhost:8200"
MODEL="Qwen2.5-7B-Instruct"
TRACE="data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl"
OUT="logs/cachedisp_repeats"
PY="/home/murphy/local-deployment/.venv/bin/python"
SCRIPT="experiments/local_deployment/phase2/run_offload_strategies.py"

STRATEGIES="flop_based cache_disp session_aware"
FRACTIONS="0.15 0.20 0.25"
SEEDS="42 123 456"
START_HOURS=0.8
DURATION_HOURS=1.0
TIME_SCALE=5.0

mkdir -p "$OUT"
echo "=== Near-Knee Repeat Runs ===" | tee "$OUT/driver.log"
echo "Started: $(date)" | tee -a "$OUT/driver.log"
echo "Fractions: $FRACTIONS" | tee -a "$OUT/driver.log"
echo "Seeds: $SEEDS" | tee -a "$OUT/driver.log"
echo "" | tee -a "$OUT/driver.log"

for seed in $SEEDS; do
    for frac in $FRACTIONS; do
        echo "========================================" | tee -a "$OUT/driver.log"
        echo "SEED=$seed FRACTION=$frac @ $(date)" | tee -a "$OUT/driver.log"
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
            --output-dir "$OUT/seed_${seed}" \
            --seed $seed 2>&1 | tee -a "$OUT/driver.log"

        sleep 15
    done
done

echo "" | tee -a "$OUT/driver.log"
echo "Completed: $(date)" | tee -a "$OUT/driver.log"
