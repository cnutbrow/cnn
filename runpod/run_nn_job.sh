#!/usr/bin/env bash
# Kick off the NN training job on a RunPod GPU pod and walk away.
# Assumes: pod already provisioned, repo + data/processed already synced to
# the pod (e.g. via `runpodctl send` or a mounted network volume) so this is
# a clean scriptable job, not a notebook session.
#
# Usage (run on the pod, or via `runpodctl exec` / ssh from your machine):
#   bash runpod/run_nn_job.sh [config_path]
set -euo pipefail

CONFIG="${1:-configs/config.yaml}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

cd "$(dirname "$0")/.."
pip install -q -r requirements.txt

STAMP=$(date +%Y%m%d_%H%M%S)
nohup python3 -m models.nn.train_nn --config "$CONFIG" --device cuda \
    > "$LOG_DIR/nn_train_${STAMP}.log" 2>&1 &

echo "Training launched in background, PID $!"
echo "Logs: $LOG_DIR/nn_train_${STAMP}.log"
echo "Results land in results/nn_fold_results.csv and results/nn_summary.csv when done."
