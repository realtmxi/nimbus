# Data

Trace files are not committed (multi-GB). Download from gpu1 via the
provided script.

## Quick download

Primary trace (ShareGPT+BurstGPT, ~491 MB):
```bash
bash scripts/download_data.sh
```

All traces (includes RouteWise rednote/freeinference logs, ~20 GB):
```bash
bash scripts/download_data.sh --all
```

The script uses rsync over SSH to `/scratch/murphy/workloads/` on gpu1.

If you SSH to gpu1 differently than `ssh -p 10021 localhost`, set:
```bash
export NIMBUS_GPU1_SSH="-p 10022 murphy@your.host"
bash scripts/download_data.sh
```

## Available datasets

### ShareGPT + BurstGPT (primary)

`data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl`

200,957 requests, 58,845 sessions. ShareGPT prompts replayed under BurstGPT
inter-arrival timestamps. Includes per-request `block_hash_ids` for prefix
overlap reconstruction (cross-session overlap is zero by construction).

Schema (one JSON object per line):
```json
{
  "arrived_at": 1255458,
  "num_prefill_tokens": 34,
  "num_decode_tokens": 244,
  "block_hash_ids": "[0, 1, 2, ...]",
  "block_size": 16,
  "session_id": 0,
  "prompt_text": "...",
  "response_text": "..."
}
```

### RouteWise traces (optional, large)

`data/routewise/sharegpt_prompts_7d.jsonl` (~506 MB)
`data/routewise/freeinference_logs.csv` (~3.2 GB)
`data/routewise/rednote_logs.csv` (~15 GB)

Used for cross-trace generalization and long-context stress tests.
Rednote median prefill is ~5400 tokens (10x ShareGPT) — useful for
evaluating cache-displacement-heavy workloads.
