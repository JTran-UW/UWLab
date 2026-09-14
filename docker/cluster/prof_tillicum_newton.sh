#!/usr/bin/env bash
#SBATCH --job-name=newton-prof
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scrubbed/profjat/uwlab3/logs/slurm-%j.out
#SBATCH --error=/gpfs/scrubbed/profjat/uwlab3/logs/slurm-%j.err
set -uo pipefail
UWLAB_DIR=/gpfs/scrubbed/profjat/uwlab3
SIF=/gpfs/scrubbed/profjat/sif/uwlab_isaaclab3_cu130.sif
SCRATCH="${SLURM_TMPDIR:-/tmp}"; CACHE_DIR="$SCRATCH/isaaclab_cache_$SLURM_JOB_ID"; FAKE_HOME="$SCRATCH/omni_home_$SLURM_JOB_ID"; OVERLAY_DIR="$SCRATCH/overlay_$SLURM_JOB_ID"
mkdir -p "$CACHE_DIR" "$FAKE_HOME" "$OVERLAY_DIR/upper" "$OVERLAY_DIR/work"
PP=/workspace/uwlab/rsl_rl_patched:/workspace/uwlab/source/uwlab:/workspace/uwlab/source/uwlab_assets:/workspace/uwlab/source/uwlab_rl:/workspace/uwlab/source/uwlab_tasks
echo "[INFO] node $(hostname) cpus=$SLURM_CPUS_PER_TASK"; nproc; nvidia-smi --query-gpu=name --format=csv,noheader
for v in "16384 pa baseline:UWLAB_X=1:16384:ObjectPartiallyAssembledEEGrasped:2000" "16384 pa ncon256 nj512:UWLAB_NEWTON_NCONMAX=256:16384:ObjectPartiallyAssembledEEGrasped:512" "16384 pa substeps2:UWLAB_NEWTON_SUBSTEPS=2:16384:ObjectPartiallyAssembledEEGrasped:2000"; do
  echo "##### ${v%%:*}"
  apptainer exec --nv --containall --overlay "$OVERLAY_DIR" --bind "$UWLAB_DIR":/workspace/uwlab:rw --bind "$CACHE_DIR":/tmp/isaaclab_cache:rw --home "$FAKE_HOME":/root --pwd /workspace/uwlab \
    --env TMPDIR=/tmp/isaaclab_cache --env OMNI_KIT_ACCEPT_EULA=YES --env PYTHONPATH="$PP" --env UWLAB_ROBOT_ASSETS_DIR=/workspace/uwlab/usd/patched_newton_mass --env "$(echo "$v" | cut -d: -f2)" --env UWLAB_NEWTON_NJMAX="$(echo "$v" | cut -d: -f5)" \
    "$SIF" /opt/venv/bin/python -u /workspace/uwlab/tools/newton_probes/prof.py --headless --num_envs "$(echo "$v" | cut -d: -f3)" --steps 40 --reset_type "$(echo "$v" | cut -d: -f4)" env.scene.insertive_object=peg env.scene.receptive_object=peghole 2>&1 | grep "^PROFILE\|^SOLVER\|^NEWTON\|Traceback\|Error:\|Aborted\|CUDA graph"
done
echo "##### done"
