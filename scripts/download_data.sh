#!/usr/bin/env bash
#
# Download or compose trace data.
#
# Primary data is composed locally from public BurstGPT and ShareGPT traces,
# following the RouteWise workload preparation path. Optional RouteWise mirror
# traces can still be copied from a remote trace host.
#
# Usage:
#   bash scripts/download_data.sh                # Download primary trace only
#   bash scripts/download_data.sh --force        # Re-compose primary trace
#   bash scripts/download_data.sh --remote       # Copy primary trace from remote
#   bash scripts/download_data.sh --all          # Primary + RouteWise traces
#   bash scripts/download_data.sh --routewise    # RouteWise traces (large)
#
# Output goes to data/ at the repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"

PYTHON="${PYTHON:-python3}"
TRACE_SSH="${NIMBUS_TRACE_SSH:-}"
REMOTE_DATA_ROOT="${NIMBUS_REMOTE_DATA_ROOT:-}"
RSYNC_RSH="${NIMBUS_RSYNC_RSH:-ssh}"

mkdir -p "$DATA_DIR/sharegpt_burstgpt"

download_primary() {
    echo "[1/1] Preparing ShareGPT+BurstGPT trace from public sources..."
    "$PYTHON" "$REPO_ROOT/scripts/prepare_sharegpt_burstgpt.py" "$@"
}

require_remote_config() {
    if [[ -z "$TRACE_SSH" || -z "$REMOTE_DATA_ROOT" ]]; then
        cat >&2 <<'EOF'
Set NIMBUS_TRACE_SSH and NIMBUS_REMOTE_DATA_ROOT before copying remote traces.

Example:
  NIMBUS_TRACE_SSH=user@host \
  NIMBUS_REMOTE_DATA_ROOT=/remote/path/to/workloads \
      bash scripts/download_data.sh --routewise
EOF
        exit 2
    fi
}

download_primary_remote() {
    require_remote_config
    echo "[1/1] Copying ShareGPT+BurstGPT trace from remote mirror..."
    rsync -avh --progress -e "$RSYNC_RSH" \
        "${TRACE_SSH}:${REMOTE_DATA_ROOT%/}/Burst_ShareGPT/sharegpt_prompts_burstgpt_timestamps.jsonl" \
        "$DATA_DIR/sharegpt_burstgpt/"
    rsync -avh -e "$RSYNC_RSH" \
        "${TRACE_SSH}:${REMOTE_DATA_ROOT%/}/Burst_ShareGPT/README.md" \
        "$DATA_DIR/sharegpt_burstgpt/" 2>/dev/null || true
}

download_routewise() {
    require_remote_config
    echo "[+] Downloading RouteWise traces (large: ~19 GB total)..."
    mkdir -p "$DATA_DIR/routewise"
    rsync -avh --progress -e "$RSYNC_RSH" \
        "${TRACE_SSH}:${REMOTE_DATA_ROOT%/}/RouteWise/sharegpt_prompts_7d.jsonl" \
        "$DATA_DIR/routewise/"
    rsync -avh --progress -e "$RSYNC_RSH" \
        "${TRACE_SSH}:${REMOTE_DATA_ROOT%/}/RouteWise/freeinference_logs.csv" \
        "$DATA_DIR/routewise/"
    rsync -avh --progress -e "$RSYNC_RSH" \
        "${TRACE_SSH}:${REMOTE_DATA_ROOT%/}/RouteWise/rednote_logs.csv" \
        "$DATA_DIR/routewise/"
}

case "${1:-}" in
    --all)
        shift
        download_primary "$@"
        download_routewise
        ;;
    --force)
        shift
        download_primary --force "$@"
        ;;
    --remote)
        shift
        download_primary_remote
        ;;
    --routewise)
        shift
        download_routewise
        ;;
    --help|-h)
        "$PYTHON" "$REPO_ROOT/scripts/prepare_sharegpt_burstgpt.py" --help
        exit 0
        ;;
    *)
        download_primary "$@"
        ;;
esac

echo
echo "Done. Downloaded files:"
find "$DATA_DIR" -type f -not -name "README.md" | xargs -I {} du -h {} 2>/dev/null
