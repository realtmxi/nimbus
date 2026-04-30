#!/usr/bin/env bash
#
# Download trace data from gpu1 (/scratch/murphy/workloads).
#
# Requires SSH access to gpu1 (port 10021 on localhost via tunnel,
# or direct hostname inside Berkeley network).
#
# Usage:
#   bash scripts/download_data.sh                # Download primary trace only
#   bash scripts/download_data.sh --all          # Download all available traces
#   bash scripts/download_data.sh --routewise    # RouteWise traces (large)
#
# Output goes to data/ at the repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"

# Default SSH target. Override with NIMBUS_GPU1_SSH env var if needed.
GPU1_SSH="${NIMBUS_GPU1_SSH:--p 10021 localhost}"

mkdir -p "$DATA_DIR/sharegpt_burstgpt"

download_primary() {
    echo "[1/1] Downloading ShareGPT+BurstGPT trace (~491 MB)..."
    rsync -avh --progress -e "ssh ${GPU1_SSH%% *}" \
        "${GPU1_SSH##* }":/scratch/murphy/workloads/Burst_ShareGPT/sharegpt_prompts_burstgpt_timestamps.jsonl \
        "$DATA_DIR/sharegpt_burstgpt/"
    rsync -avh -e "ssh ${GPU1_SSH%% *}" \
        "${GPU1_SSH##* }":/scratch/murphy/workloads/Burst_ShareGPT/README.md \
        "$DATA_DIR/sharegpt_burstgpt/" 2>/dev/null || true
}

download_routewise() {
    echo "[+] Downloading RouteWise traces (large: ~19 GB total)..."
    mkdir -p "$DATA_DIR/routewise"
    rsync -avh --progress -e "ssh ${GPU1_SSH%% *}" \
        "${GPU1_SSH##* }":/scratch/murphy/workloads/RouteWise/sharegpt_prompts_7d.jsonl \
        "$DATA_DIR/routewise/"
    rsync -avh --progress -e "ssh ${GPU1_SSH%% *}" \
        "${GPU1_SSH##* }":/scratch/murphy/workloads/RouteWise/freeinference_logs.csv \
        "$DATA_DIR/routewise/"
    rsync -avh --progress -e "ssh ${GPU1_SSH%% *}" \
        "${GPU1_SSH##* }":/scratch/murphy/workloads/RouteWise/rednote_logs.csv \
        "$DATA_DIR/routewise/"
}

case "${1:-}" in
    --all)
        download_primary
        download_routewise
        ;;
    --routewise)
        download_routewise
        ;;
    *)
        download_primary
        ;;
esac

echo
echo "Done. Downloaded files:"
find "$DATA_DIR" -type f -not -name "README.md" | xargs -I {} du -h {} 2>/dev/null
