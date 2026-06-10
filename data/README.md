# Data

Trace files are not committed (multi-GB). The primary workload is composed
locally from public BurstGPT and ShareGPT traces, using the same approach as
RouteWise. Optional RouteWise mirror traces can still be copied from a remote
trace host.

## Quick download

Primary trace (ShareGPT+BurstGPT, ~491 MB):
```bash
bash scripts/download_data.sh
```

Re-compose the primary trace:
```bash
bash scripts/download_data.sh --force
```

All traces (includes RouteWise rednote/freeinference logs, ~20 GB; requires a
remote trace mirror):
```bash
bash scripts/download_data.sh --all
```

The default primary path downloads raw traces into `data/.cache/`:
- `BurstGPT_3.csv` from the BurstGPT GitHub release
- `ShareGPT_V3_unfiltered_cleaned_split.json` from Hugging Face

To copy from a remote mirror instead:
```bash
export NIMBUS_TRACE_SSH=user@host
export NIMBUS_REMOTE_DATA_ROOT=/remote/path/to/workloads
bash scripts/download_data.sh --remote
```

## Available datasets

### ShareGPT + BurstGPT (primary)

`data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl`

ShareGPT prompts replayed under BurstGPT inter-arrival timestamps. The default
local composition uses the first 30 BurstGPT days and preserves BurstGPT
session IDs, model labels, elapsed time, and token counts. A remote mirror may
include additional optional prefix fields such as `block_hash_ids`.

Schema (one JSON object per line):
```json
{
  "arrived_at": 1255458,
  "session_id": "12345",
  "num_prefill_tokens": 34,
  "num_decode_tokens": 244,
  "model": "GPT-4",
  "log_type": "Conversation log",
  "elapsed_time_sec": 1.23,
  "prompt_text": "...",
  "response_text": "...",
  "sharegpt_conversation_id": "abc",
  "sharegpt_turn_index": 0
}
```

### RouteWise traces (optional, large)

`data/routewise/sharegpt_prompts_7d.jsonl` (~506 MB)
`data/routewise/freeinference_logs.csv` (~3.2 GB)
`data/routewise/rednote_logs.csv` (~15 GB)

Used for cross-trace generalization and long-context stress tests.
Rednote median prefill is ~5400 tokens (10x ShareGPT) — useful for
evaluating cache-displacement-heavy workloads.
