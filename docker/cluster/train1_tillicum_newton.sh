#!/usr/bin/env bash
#SBATCH --job-name=newton-train1
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/scrubbed/profjat/uwlab3/logs/slurm-%j.out
#SBATCH --error=/gpfs/scrubbed/profjat/uwlab3/logs/slurm-%j.err
set -uo pipefail
UWLAB_DIR=/gpfs/scrubbed/profjat/uwlab3
SIF=/gpfs/scrubbed/profjat/sif/uwlab_isaaclab3_cu130.sif
SCRATCH="${SLURM_TMPDIR:-/tmp}"; CACHE_DIR="$SCRATCH/isaaclab_cache_$SLURM_JOB_ID"; FAKE_HOME="$SCRATCH/omni_home_$SLURM_JOB_ID"; OVERLAY_DIR="$SCRATCH/overlay_$SLURM_JOB_ID"
mkdir -p "$CACHE_DIR" "$FAKE_HOME" "$OVERLAY_DIR/upper" "$OVERLAY_DIR/work"
PP=/workspace/uwlab/rsl_rl_patched:/workspace/uwlab/source/uwlab:/workspace/uwlab/source/uwlab_assets:/workspace/uwlab/source/uwlab_rl:/workspace/uwlab/source/uwlab_tasks
echo "[INFO] node $(hostname)"
for v in "16384 single-rank:16384"; do
  echo "##### ${v%%:*}"
  apptainer exec --nv --containall --overlay "$OVERLAY_DIR" --bind "$UWLAB_DIR":/workspace/uwlab:rw --bind "$CACHE_DIR":/tmp/isaaclab_cache:rw --home "$FAKE_HOME":/root --pwd /workspace/uwlab \
    --env TMPDIR=/tmp/isaaclab_cache --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONPATH="$PP" --env UWLAB_ROBOT_ASSETS_DIR=/workspace/uwlab/usd/patched_newton_mass --env WANDB_MODE=disabled \
    "$SIF" /opt/venv/bin/python -u /workspace/uwlab/scripts/reinforcement_learning/rsl_rl/train_expert_init.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Newton-v0 --num_envs "${v#*:}" --max_iterations 2 --headless --resume_path /workspace/uwlab/expert_seed0_rslrl52.pt --run_name prof1 env.scene.insertive_object=peg env.scene.receptive_object=peghole env.events.reset_from_reset_states.params.dataset_dir=/workspace/uwlab/Datasets/OmniReset_patched agent.algorithm.schedule=fixed 2>&1 | grep "step-time\|Learning iteration\|Collection time\|Traceback\|Error:\|Aborted" | sed "s/\x1b\[[0-9;]*m//g"
done
echo "##### done"
