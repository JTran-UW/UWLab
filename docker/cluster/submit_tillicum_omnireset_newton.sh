#!/usr/bin/env bash
#SBATCH --job-name=omnireset-newton
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=960G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scrubbed/profjat/uwlab3/logs/slurm-%j.out
#SBATCH --error=/gpfs/scrubbed/profjat/uwlab3/logs/slurm-%j.err

# OmniReset 4-GPU distributed training on Tillicum (H200) on the NEWTON (MuJoCo-Warp) backend,
# resuming actor+critic from a PhysX-trained expert (train_expert_init.py). Requires the patched
# Robotiq USDs (usd/patched_newton_mass) and converted datasets (Datasets/OmniReset_patched);
# see ISAACLAB_3_GRASP_HANDOFF.md §13-§15.
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
NUM_ENVS="${NUM_ENVS:-16384}"
EXPERT="${EXPERT:-/workspace/uwlab/expert_seed0_rslrl52.pt}"

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
echo "[INFO] osc clamp   : ${UWLAB_OSC_VEL_CLAMP:-1}"
echo "[INFO] backend     : Newton/MuJoCo-Warp, resume from ${RESUME_PATH:-$EXPERT}"
echo "[INFO] algorithm   : stock omnireset PPO cfg"
test -d "$UWLAB_DIR/usd/patched_newton_mass" || { echo "[FATAL] usd/patched_newton_mass missing"; exit 1; }
test -d "$UWLAB_DIR/Datasets/OmniReset_patched" || { echo "[FATAL] Datasets/OmniReset_patched missing"; exit 1; }
for f in "_notify_newton_fixed_base_moved:source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/events.py" "TrainNewtonCfg:source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/__init__.py" "isfinite:source/uwlab/uwlab/envs/mdp/terminations.py" "_NanSafeRslRlVecEnvWrapper:scripts/reinforcement_learning/rsl_rl/train_expert_init.py" "ProgressContextNanSafe:source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/rewards.py" "_apply_local_object_assets_if_configured:source/uwlab_tasks/uwlab_tasks/utils/hydra.py"; do
    grep -q "${f%%:*}" "$UWLAB_DIR/${f#*:}" || { echo "[FATAL] fix marker ${f%%:*} missing in ${f#*:}"; exit 1; }
done
echo "[INFO] all Newton fix markers present"
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

# FROM_SCRATCH=1 -> plain train.py with the stock recipe (the Newton control); otherwise resume/fine-tune
# from RESUME_PATH (default: the PhysX expert) with train_expert_init.py, iteration counter continued.
if [ "${FROM_SCRATCH:-0}" = "1" ]; then
    TRAIN_SCRIPT=/workspace/uwlab/scripts/reinforcement_learning/rsl_rl/train.py
    RESUME_ARGS=""
    echo "[INFO] mode        : from scratch (train.py)"
else
    TRAIN_SCRIPT=/workspace/uwlab/scripts/reinforcement_learning/rsl_rl/train_expert_init.py
    RESUME_ARGS="--resume_path ${RESUME_PATH:-$EXPERT}"
    echo "[INFO] mode        : resume from ${RESUME_PATH:-$EXPERT} (iteration counter continues)"
fi
ATTEMPT="${ATTEMPT:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
echo "[INFO] attempt     : $ATTEMPT / $MAX_ATTEMPTS"
set +e
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
    --env UWLAB_ROBOT_ASSETS_DIR=/workspace/uwlab/usd/patched_newton_mass \
    --env UWLAB_OSC_VEL_CLAMP="${UWLAB_OSC_VEL_CLAMP:-1}" \
    --env UWLAB_NEWTON_SUBSTEPS="${UWLAB_NEWTON_SUBSTEPS:-4}" \
    --env UWLAB_NEWTON_EQ_TIMECONST="${UWLAB_NEWTON_EQ_TIMECONST:-0.005}" \
    --env UWLAB_NEWTON_CONTACT_KD="${UWLAB_NEWTON_CONTACT_KD:-400}" \
    --env UWLAB_NEWTON_CONTACT_KE="${UWLAB_NEWTON_CONTACT_KE:-4e4}" \
    "$SIF" \
    /opt/venv/bin/python -m torch.distributed.run \
        --nnodes 1 \
        --nproc_per_node 4 \
        "$TRAIN_SCRIPT" \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Newton-v0 \
        --num_envs "$NUM_ENVS" \
        --logger wandb \
        ${RUN_NAME:+--run_name "$RUN_NAME"} \
        --headless \
        --distributed \
        env.scene.insertive_object=peg \
        env.scene.receptive_object=peghole \
        env.events.reset_from_reset_states.params.dataset_dir=/workspace/uwlab/Datasets/OmniReset_patched \
        $RESUME_ARGS

rc=$?
set -e
LOG="$UWLAB_DIR/logs/slurm-${SLURM_JOB_ID}.out"
ERRLOG="$UWLAB_DIR/logs/slurm-${SLURM_JOB_ID}.err"
if [ "$rc" -ne 0 ] && ! grep -q "Learning iteration" "$LOG" 2>/dev/null && grep -q "tcache\|SIGABRT\|Aborted\|double free\|exitcode  : -6\|exitcode  : -9" "$LOG" "$ERRLOG" 2>/dev/null; then
    # Newton's USD import aborts with a libc heap-corruption (malloc(): unaligned tcache chunk) on
    # ~10 % of launches (hand-off §15g); with 4 ranks that is ~35 % per job. Retry before training starts.
    if [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; then
        echo "[WARN] died before the first learning iteration (rc=$rc); resubmitting attempt $((ATTEMPT+1))"
        cd "$UWLAB_DIR" && ATTEMPT=$((ATTEMPT+1)) MAX_ATTEMPTS="$MAX_ATTEMPTS" NUM_ENVS="$NUM_ENVS" FROM_SCRATCH="${FROM_SCRATCH:-0}" UWLAB_OSC_VEL_CLAMP="${UWLAB_OSC_VEL_CLAMP:-1}" RUN_NAME="${RUN_NAME:-}" RESUME_PATH="${RESUME_PATH:-}" \
            sbatch docker/cluster/submit_tillicum_omnireset_newton.sh
    else
        echo "[FATAL] gave up after $ATTEMPT attempts"
    fi
    exit "$rc"
elif [ "$rc" -ne 0 ]; then
    echo "[FATAL] job failed (rc=$rc) with a non-flake error; not resubmitting"
    exit "$rc"
fi
grep -q "local object assets redirected" "$LOG" || echo "[WARN] no asset-redirect line in the log: Newton ran with cloud (SDF) object USDs?"
echo "[INFO] done (rc=$rc)"
