#!/usr/bin/env bash
# Relaunch ZeroG-GPS training on Tillicum from latest checkpoint of job 81263.
# Usage: bash scripts/relaunch_gps.sh [--now]
#   Without --now: waits 6 hours then relaunches.
#   With --now: relaunches immediately.

set -e

CHECKPOINT_DIR="/gpfs/scrubbed/patyin/uwlab/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/81263"
TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-GPS-v0"

if [[ "$1" != "--now" ]]; then
    echo "[$(date)] Waiting 6 hours before relaunching..."
    sleep 21600
fi

# Find latest checkpoint
LATEST_CKPT=$(ssh tillicum "ls -1 ${CHECKPOINT_DIR}/model_*.pt | sort -t_ -k2 -n | tail -1" 2>/dev/null | grep -o '/gpfs/.*')

if [[ -z "$LATEST_CKPT" ]]; then
    echo "[ERROR] No checkpoint found in ${CHECKPOINT_DIR}"
    exit 1
fi

echo "[$(date)] Found latest checkpoint: ${LATEST_CKPT}"
echo "[$(date)] Launching new Tillicum job..."

cd "$(dirname "$0")/.."

./docker/cluster/cluster_interface.sh --cluster tillicum job base \
    --gpus 8 \
    --task "$TASK" \
    --num_envs 8125 --logger wandb --headless --distributed \
    --resume_path "$LATEST_CKPT" \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole

echo "[$(date)] Done. New job submitted."
