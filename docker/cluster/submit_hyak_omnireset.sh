#!/usr/bin/env bash
#SBATCH --job-name=omnireset-train
# On klone the *account* carries the lab+GPU-type name and the *partition* is
# just the GPU type: account=gpu-l40s-weirdlab pairs with partition=gpu-l40s.
# (`sacctmgr show assoc user=profjat` lists the accounts; `sinfo` the partitions.)
#SBATCH --account=gpu-l40s-weirdlab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/weirdlab/profjat/uwlab/logs/slurm-%j.out
#SBATCH --error=/gscratch/weirdlab/profjat/uwlab/logs/slurm-%j.err

# Submit the OmniReset distributed training run on Hyak (klone).
#
# Does NOT use docker/cluster/run_singularity.sh: that script targets the Isaac
# Sim *binary* image layout (it calls /isaac-sim/python.sh) produced from the NGC
# Docker image. The image this uses is a pip install of isaacsim inside a venv at
# /opt/venv, which is the configuration actually validated in
# ISAACLAB_3_MIGRATION.md, so the entrypoint differs.
#
# Usage (from a klone login node):
#   sbatch docker/cluster/submit_hyak_omnireset.sh
#
# Adjust the account/partition pair to whatever has 4 free GPUs. Valid pairs for
# profjat (all these node types carry 8 GPUs each, so 4 fits on one node):
#
#   --account=gpu-l40s-weirdlab  --partition=gpu-l40s   (25 nodes)
#   --account=gpu-l40-weirdlab   --partition=gpu-l40    (15 nodes)
#   --account=gpu-a40-weirdlab   --partition=gpu-a40    (27 nodes)
#   --account=ckpt-weirdlab      --partition=ckpt       (preemptible/scavenger)

set -euo pipefail

UWLAB_DIR=/gscratch/weirdlab/profjat/uwlab
SIF=/gscratch/weirdlab/profjat/sif/uwlab_isaaclab3_cu130.sif

# Node-local scratch for the Isaac Lab asset cache. Deliberately NOT on
# /gscratch: that share is at ~97% of its inode quota, and the asset cache is
# many small files. It is also why the environment ships as a single .sif.
#
# Note SLURM_TMPDIR is NOT set on klone, so this resolves to /tmp -- which on a
# compute node is a 2.8 TB local NVMe (/dev/nvme0n1p3), exactly what we want.
SCRATCH="${SLURM_TMPDIR:-/tmp}"
CACHE_DIR="$SCRATCH/isaaclab_cache_${SLURM_JOB_ID:-manual}"
# Disk-backed writable overlay for the read-only image (see --overlay below).
OVERLAY_DIR="$SCRATCH/overlay_${SLURM_JOB_ID:-manual}"
# Isaac Sim/Kit wants a writable HOME for its config, caches and EULA marker.
# --containall hides the real one, so give it a node-local scratch home.
FAKE_HOME="${SLURM_TMPDIR:-/tmp}/omni_home_${SLURM_JOB_ID:-manual}"
# An apptainer directory overlay needs the upper/work layout underneath it.
mkdir -p "$CACHE_DIR" "$FAKE_HOME" "$UWLAB_DIR/logs" "$OVERLAY_DIR/upper" "$OVERLAY_DIR/work"

# wandb auth: --containall hides ~/.netrc, and FAKE_HOME is mounted at /root, so
# copy the credentials in. Without this WANDB_MODE=online fails to authenticate
# and the run silently falls back to offline (nothing appears on wandb.ai).
if [ -f "$HOME/.netrc" ]; then
    cp "$HOME/.netrc" "$FAKE_HOME/.netrc" && chmod 600 "$FAKE_HOME/.netrc"
    echo "[INFO] staged ~/.netrc for wandb auth"
else
    echo "[WARN] no ~/.netrc found -- wandb will not authenticate; set WANDB_API_KEY or use WANDB_MODE=offline"
fi

echo "[INFO] node        : $(hostname)"
echo "[INFO] gpus        : ${SLURM_GPUS_ON_NODE:-unknown}"
echo "[INFO] sif         : $SIF"
echo "[INFO] uwlab       : $UWLAB_DIR"
echo "[INFO] cache (local): $CACHE_DIR"
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

# --nv exposes the GPUs; --containall keeps the host env out; the code is bound
# in rather than baked, so a code change needs only an rsync, not a rebuild.
apptainer exec \
    --nv \
    --containall \
    `# Kit insists on writing inside the image (user.config.json, kit/data/...)` \
    `# and Isaac Lab's logging handler writes a lot during asset init. The SIF is` \
    `# read-only, so it needs an overlay -- but NOT --writable-tmpfs, whose` \
    `# default is a ~64 MiB RAM overlay that fills within a minute and throws` \
    `# "OSError: [Errno 28] No space left on device" out of logging.flush().` \
    `# A directory overlay on the node-local NVMe (2.8 TB) has no such limit.` \
    --overlay "$OVERLAY_DIR" \
    --bind "$UWLAB_DIR":/workspace/uwlab:rw \
    --bind "$CACHE_DIR":/tmp/isaaclab_cache:rw \
    `# NCCL needs real shared memory for its collectives. --containall gives the` \
    `# container a small private /dev/shm, which makes the multi-GPU parameter` \
    `# sync die with SIGSEGV (signal 11) right after "Synchronizing parameters` \
    `# for rank N..." -- a hard crash with no Python traceback. Bind the host's.` \
    --bind /dev/shm:/dev/shm \
    `# HOME must be set with --home, not --env: apptainer refuses the latter` \
    `# ("Overriding HOME environment variable with APPTAINERENV_HOME is not` \
    `# permitted") and silently leaves HOME pointing inside the container.` \
    --home "$FAKE_HOME":/root \
    `# Without an explicit --pwd, cwd falls back to a path that does not exist` \
    `# in the container, so the run's relative log_dir lands in the small` \
    `# private tmpfs and dump_yaml() dies with Errno 28 while writing env.yaml.` \
    --pwd /workspace/uwlab \
    `# The 4-rank broadcast used to segfault here. Cause was NOT /dev/shm or` \
    `# P2P (both tried, neither helped) -- it was the cu128 torch in the old` \
    `# image shipping NCCL cuda12.9 against Hyak's CUDA 13.0 driver. The cu130` \
    `# image fixes it; verified with a 4-rank all_reduce via torch.distributed.run` \
    `# (note: mp.spawn gives a false negative here, so test with the real launcher).` \
    `# Isaac Sim's warp/PhysX layer allocates through the CUDA virtual-memory` \
    `# APIs, which collide with NCCL's cuMem path: every collective is enqueued` \
    `# and then the process dies with SIGSEGV and no Python traceback, right` \
    `# after "Synchronizing parameters for rank N...". Disabling cuMem fixes it.` \
    `# This is why each piece passed in isolation -- single-GPU Isaac Sim, and a` \
    `# 4-rank all_reduce with no Isaac Sim, are both fine; only the combination` \
    `# in one process fails. Reproduced on 2 GPUs locally, outside any container` \
    `# or Slurm, which is what finally made it debuggable.` \
    --env NCCL_CUMEM_ENABLE=0 \
    --env NCCL_DEBUG=WARN \
    --env TMPDIR=/tmp/isaaclab_cache \
    --env UWLAB_PATH=/workspace/uwlab \
    --env WANDB_MODE="${WANDB_MODE:-online}" \
    `# Isaac Sim prompts "Do you accept the EULA? (Yes/No):" on first run. A` \
    `# batch job has no stdin, so without this every rank exits 1 within` \
    `# seconds. It is normally cached in HOME after one interactive accept,` \
    `# which --containall hides.` \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env PYTHONPATH="$RSL_RL_PYTHONPATH" \
    "$SIF" \
    /opt/venv/bin/python -m torch.distributed.run \
        --nnodes 1 \
        --nproc_per_node 4 \
        /workspace/uwlab/scripts/reinforcement_learning/rsl_rl/train.py \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
        `# --num_envs is PER PROCESS under --distributed: 14336 x 4 = 57344 total.` \
        `#` \
        `# 14336 is the measured ceiling for this scene. PhysX caps materials at` \
        `# 64K per scene, and each env clones 4 material prims authored inside the` \
        `# asset USDs (InsertiveObject/PhysicsMaterial, ReceptiveObject/` \
        `# PhysicsMaterial, Robot/PhysicsMaterial, Robot/fingertip_material) --` \
        `# measured at exactly 4.00/env at 64, 8192, 12288 and 14336 envs.` \
        `#` \
        `# Verified ladder: 8192 OK, 12288 OK, 14336 OK, 16383 FAIL, 16384 FAIL.` \
        `#` \
        `# Not fixable from config: removing the material-randomization events does` \
        `# NOT help (the materials come from the USDs, not the events), and` \
        `# clone_in_fabric=True does not help either -- both tested. Reaching 16384` \
        `# needs one fewer material per env, i.e. an edit to the source USD.` \
        `#` \
        `# Root cause of the regression vs 2.3.2: interactive_scene.py gates the` \
        `# physics-replication path to Newton, so PhysX takes clone_usd=True and` \
        `# copies materials per env instead of sharing them.` \
        --num_envs 14336 \
        --logger wandb \
        --headless \
        --distributed \
        env.scene.insertive_object=peg \
        env.scene.receptive_object=peghole \
        env.events.reset_from_reset_states.params.dataset_dir=/workspace/uwlab/Datasets/OmniReset \
        ${RESUME_PATH:+--resume_path "$RESUME_PATH"}

echo "[INFO] done"
