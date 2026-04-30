#!/usr/bin/env bash
# Coordinate-descent sweep over FastSAC's batch_size, UTD, and num_envs.
# Submits one Hyak SLURM job per grid point. Each run is tagged with --run_name
# so the runs are easy to filter/group in wandb.
#
# Usage:
#   bash scripts/reinforcement_learning/holosoma/sweep_fastsac.sh [SWEEP_TAG]
#
# Defaults:
#   - 2 GPUs / 1 node, ckpt partition (weirdlab-ckpt)
#   - Easy off-policy peg task with expert checkpoint BC
#
# Override per-cell defaults by editing the GRID arrays below.

set -e

SWEEP_TAG=${1:-"sweep_$(date +%Y%m%d_%H%M)"}
echo "[INFO] Sweep tag: $SWEEP_TAG"

# Baseline (center of the cross). Single-axis sweeps perturb one coord at a time.
BASE_BATCH=16384
BASE_NUM_UPDATES=8
BASE_POLICY_FREQ=4   # UTD = num_updates / policy_frequency = 2
BASE_NUM_ENVS=4096
BASE_TARGET_ENTROPY_RATIO=0.0

# Sweep grids (axis values). Baseline value is included so re-running shows the
# center cell; `submit_one` dedupes on (batch, nu, pf, ne, ter).
BATCH_GRID=(4096 16384 65536)
# UTD axis varied by num_updates with policy_frequency held fixed (UTD = 1, 2, 4)
NUM_UPDATES_GRID=(4 8 16)
NUM_ENVS_GRID=(1024 4096 16384)
TARGET_ENTROPY_RATIO_GRID=(0.0 0.5 1.0)

submitted=()

submit_one() {
    local batch=$1
    local nu=$2
    local pf=$3
    local ne=$4
    local ter=$5

    local key="b${batch}_nu${nu}_pf${pf}_ne${ne}_ter${ter}"
    for prev in "${submitted[@]}"; do
        if [[ "$prev" == "$key" ]]; then
            echo "[SKIP] $key (already submitted)"
            return
        fi
    done
    submitted+=("$key")

    local run_name="${SWEEP_TAG}_${key}"
    echo "[SUBMIT] $run_name"

    NODES=1 GPUS_PER_NODE=2 PARTITION=ckpt ACCOUNT=weirdlab-ckpt \
    CLUSTER_PYTHON_EXECUTABLE=scripts/reinforcement_learning/holosoma/train.py \
    bash docker/cluster/cluster_interface.sh --cluster hyak job base \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-OffPolicy-v0 \
        --num_envs "$ne" \
        --logger wandb \
        --headless \
        --distributed \
        --run_name "$run_name" \
        env.scene.insertive_object=peg \
        env.scene.receptive_object=peghole \
        agent.batch_size="$batch" \
        agent.num_updates="$nu" \
        agent.policy_frequency="$pf" \
        agent.target_entropy_ratio="$ter" \
        2>&1 | tail -3
}

echo "=== batch_size axis ==="
for b in "${BATCH_GRID[@]}"; do
    submit_one "$b" "$BASE_NUM_UPDATES" "$BASE_POLICY_FREQ" "$BASE_NUM_ENVS" "$BASE_TARGET_ENTROPY_RATIO"
done

echo "=== num_updates (UTD) axis ==="
for nu in "${NUM_UPDATES_GRID[@]}"; do
    submit_one "$BASE_BATCH" "$nu" "$BASE_POLICY_FREQ" "$BASE_NUM_ENVS" "$BASE_TARGET_ENTROPY_RATIO"
done

echo "=== num_envs axis ==="
for ne in "${NUM_ENVS_GRID[@]}"; do
    submit_one "$BASE_BATCH" "$BASE_NUM_UPDATES" "$BASE_POLICY_FREQ" "$ne" "$BASE_TARGET_ENTROPY_RATIO"
done

echo "=== target_entropy_ratio axis ==="
for ter in "${TARGET_ENTROPY_RATIO_GRID[@]}"; do
    submit_one "$BASE_BATCH" "$BASE_NUM_UPDATES" "$BASE_POLICY_FREQ" "$BASE_NUM_ENVS" "$ter"
done

echo "[DONE] Submitted ${#submitted[@]} unique runs (tag: $SWEEP_TAG)"
