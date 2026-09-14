#!/usr/bin/env bash
#SBATCH --job-name=omnireset-train
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/profjat/uwlab3/logs/slurm-%j.out
#SBATCH --error=/gpfs/scrubbed/profjat/uwlab3/logs/slurm-%j.err

# OmniReset 4-GPU distributed training on Tillicum (H200), IsaacLab 3.0 image.
# Adapted from docker/cluster/submit_hyak_omnireset.sh (klone); see that file
# for the rationale behind every apptainer flag. Tillicum differences:
# account=weirdlab / partition=gpu-h200, no --constraint (nodes have no
# features; any constraint is unschedulable), filesystem root /gpfs, node-local
# /tmp is a 3.5 TB NVMe, driver 610 (CUDA 13) -> cu130 image.
#
# Usage: sbatch docker/cluster/submit_tillicum_omnireset.sh
#   NUM_ENVS (per process, default 14336 = measured PhysX material-cap ceiling)
#   RESUME_PATH (optional checkpoint)

set -euo pipefail

UWLAB_DIR=/gpfs/scrubbed/profjat/uwlab3
SIF=/gpfs/scrubbed/profjat/sif/uwlab_isaaclab3_cu130.sif
NUM_ENVS="${NUM_ENVS:-14336}"

SCRATCH="${SLURM_TMPDIR:-/tmp}"
CACHE_DIR="$SCRATCH/isaaclab_cache_${SLURM_JOB_ID:-manual}"
OVERLAY_DIR="$SCRATCH/overlay_${SLURM_JOB_ID:-manual}"
FAKE_HOME="$SCRATCH/omni_home_${SLURM_JOB_ID:-manual}"
mkdir -p "$CACHE_DIR" "$FAKE_HOME" "$UWLAB_DIR/logs" "$OVERLAY_DIR/upper" "$OVERLAY_DIR/work"

if [ -f "$HOME/.netrc" ]; then
    cp "$HOME/.netrc" "$FAKE_HOME/.netrc" && chmod 600 "$FAKE_HOME/.netrc"
    echo "[INFO] staged ~/.netrc for wandb auth"
else
    echo "[WARN] no ~/.netrc found -- relying on WANDB_API_KEY"
fi

echo "[INFO] node        : $(hostname)"
echo "[INFO] gpus        : ${SLURM_GPUS_ON_NODE:-unknown}"
echo "[INFO] sif         : $SIF"
echo "[INFO] uwlab       : $UWLAB_DIR"
echo "[INFO] num_envs    : $NUM_ENVS per process x 4"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# --- rsl_rl gSDE fix (docker/cluster/patches/README.md) -------------------------
# The image's rsl_rl trains a wrong network under gSDE. Overlay a patched copy via
# PYTHONPATH; build it from the image on first use; abort if it is not active.
ensure_rsl_rl_patched() {
    local overlay="$UWLAB_DIR/rsl_rl_patched"
    if ! python3 "$UWLAB_DIR/docker/cluster/patches/apply_rsl_rl_gsde_fix.py" --check "$overlay/rsl_rl" >/dev/null 2>&1; then
        echo "[INFO] building patched rsl_rl overlay at $overlay"
        rm -rf "$overlay" && mkdir -p "$overlay"
        apptainer exec --bind "$UWLAB_DIR":/workspace/uwlab "$SIF" \
            cp -r /opt/venv/lib/python3.12/site-packages/rsl_rl /workspace/uwlab/rsl_rl_patched/rsl_rl
        python3 "$UWLAB_DIR/docker/cluster/patches/apply_rsl_rl_gsde_fix.py" "$overlay/rsl_rl" || { echo "[FATAL] could not patch rsl_rl"; exit 1; }
    fi
    local ok
    ok=$(apptainer exec --bind "$UWLAB_DIR":/workspace/uwlab --env PYTHONPATH="$RSL_RL_PYTHONPATH" "$SIF" \
        /opt/venv/bin/python -c "import rsl_rl,os; p=os.path.dirname(rsl_rl.__file__); print('[INFO] rsl_rl   :', p); print('[INFO] gsde fix :', 'children = list(self.mlp)' in open(os.path.join(p,'models/mlp_model.py')).read())" 2>/dev/null)
    echo "$ok"
    echo "$ok" | grep -q "gsde fix : True" || { echo "[FATAL] rsl_rl gSDE fix is not active inside the container -- refusing to train"; exit 1; }
}
RSL_RL_PYTHONPATH=/workspace/uwlab/rsl_rl_patched:/workspace/uwlab/source/uwlab:/workspace/uwlab/source/uwlab_assets:/workspace/uwlab/source/uwlab_rl:/workspace/uwlab/source/uwlab_tasks
ensure_rsl_rl_patched
# ---------------------------------------------------------------------------------

# Pre-warm the shared node-local asset cache before the ranks start. With 4
# ranks downloading the same metadata.yaml concurrently, a rank can read a
# half-written file ("metadata.yaml is empty or failed to load", job 287617).
HF=https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main
for rel in Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/metadata.yaml \
           Props/Custom/Peg/metadata.yaml Props/Custom/PegHole/metadata.yaml; do
    dst="$CACHE_DIR/datasets/UW-Lab/uwlab-assets/resolve/main/$rel"
    mkdir -p "$(dirname "$dst")"
    curl -fsSL "$HF/$rel" -o "$dst" && echo "[INFO] pre-warmed $rel ($(wc -c < "$dst") bytes)" || echo "[WARN] pre-warm failed for $rel"
done

apptainer exec \
    --nv \
    --containall \
    --overlay "$OVERLAY_DIR" \
    --bind "$UWLAB_DIR":/workspace/uwlab:rw \
    --bind "$CACHE_DIR":/tmp/isaaclab_cache:rw \
    --bind /dev/shm:/dev/shm \
    --home "$FAKE_HOME":/root \
    --pwd /workspace/uwlab \
    --env NCCL_CUMEM_ENABLE=0 \
    --env NCCL_DEBUG=WARN \
    --env TMPDIR=/tmp/isaaclab_cache \
    --env UWLAB_PATH=/workspace/uwlab \
    --env WANDB_MODE="${WANDB_MODE:-online}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env PYTHONPATH="$RSL_RL_PYTHONPATH" \
    "$SIF" \
    /opt/venv/bin/python -m torch.distributed.run \
        --nnodes 1 \
        --nproc_per_node 4 \
        /workspace/uwlab/scripts/reinforcement_learning/rsl_rl/train.py \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
        --num_envs "$NUM_ENVS" \
        --logger wandb \
        ${RUN_NAME:+--run_name "$RUN_NAME"} \
        --headless \
        --distributed \
        env.scene.insertive_object=peg \
        env.scene.receptive_object=peghole \
        env.events.reset_from_reset_states.params.dataset_dir=/workspace/uwlab/Datasets/OmniReset \
        ${RESUME_PATH:+--resume_path "$RESUME_PATH"}

echo "[INFO] done"
