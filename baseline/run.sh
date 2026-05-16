export OLLAMA_API_KEY='3ca5442b0e8342c68694ed9fefb15236.4_Z5j4ypJJcyAZBvoEsp-lMg'
DATA=/scratch/jialu/workloads/Burst_ShareGPT/sharegpt_prompts_burstgpt_timestamps.jsonl
SCRIPT=/scratch/jialu/workloads/Burst_ShareGPT/v2/run.py


python "$SCRIPT" \
  --data "$DATA" \
  --url http://127.0.0.1:11437/api/generate \
  --model ministral-3:3b \
  --scenario extreme_burst_1200 \
  --num-predict 2048 \
  --timeout-s 100 \
  --out-dir results_ministral_3b_extreme_burst1200_64