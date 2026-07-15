#!/usr/bin/env bash
# Self-dispatching GPU job queue for PointNet/FlattenMLP BC trainings.
#
# Watches all GPUs and launches the next queued job the instant a GPU goes idle, so
# queued runs flow onto GPUs as in-flight ones finish -- no manual polling. Used to
# queue the seed-1/seed-2 runs of the arch x dataset 2x2 behind the already-running
# seed-0 runs (which occupy 4 GPUs); the 4 idle GPUs take the first 4 queued jobs
# immediately, the rest dispatch as GPUs free.
#
# "Idle" = the GPU has NO running compute process AND no live job THIS scheduler put
# there. We detect via nvidia-smi compute-apps (not memory) because FlattenMLP uses
# <1 GB VRAM and would look idle by a memory threshold.
#
# Run detached on the HOST (the trainings are host processes):
#   nohup bash scripts/imitation_learning/point_cloud/gpu_job_queue.sh > /tmp/pcbc_queue.log 2>&1 &

set -u
REPO=/mnt/storage/lti/UWLab-patrick-private
PY=/home/ubuntu/miniforge3/envs/patlab/bin/python
TRAIN="$REPO/scripts/imitation_learning/point_cloud/train_point_net.py"
CFGDIR="$REPO/scripts/imitation_learning/point_cloud/configs"
GPUS=(0 1 2 3 4 5 6 7)
POLL=30          # seconds between dispatch sweeps
STAGGER=25       # seconds after a dispatch before reusing the sweep (lets the job claim the GPU)

# Job queue: "run_name|config|dataset|seed". Seed-0 of each config is already running
# (unsuffixed run names); here we queue seeds 1 and 2 for all four configs = 8 jobs.
QUEUE=(
  "pointnet_contactseg_s1|point_net|contact_seg_100k|1"
  "flatmlp_contactseg_s1|flatten_mlp|contact_seg_100k|1"
  "pointnet_cleanscenepc_s1|point_net|clean_scenepc_100k|1"
  "flatmlp_cleanscenepc_s1|flatten_mlp|clean_scenepc_100k|1"
  "pointnet_contactseg_s2|point_net|contact_seg_100k|2"
  "flatmlp_contactseg_s2|flatten_mlp|contact_seg_100k|2"
  "pointnet_cleanscenepc_s2|point_net|clean_scenepc_100k|2"
  "flatmlp_cleanscenepc_s2|flatten_mlp|clean_scenepc_100k|2"
)

declare -A GPU_PID   # physical GPU index -> pid of the job this scheduler dispatched there

log() { echo "[$(date '+%F %T')] $*"; }

gpu_idle() {
  local g=$1
  # Our own freshly-dispatched job may still be loading data into RAM (no CUDA ctx yet);
  # treat the GPU as busy while that pid is alive so we don't double-book it.
  local pid=${GPU_PID[$g]:-}
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && return 1
  # Otherwise idle iff no compute process is on the device.
  local n
  n=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')
  [ "$n" -eq 0 ]
}

dispatch() {
  local g=$1 job=$2 name cfg ds seed
  IFS='|' read -r name cfg ds seed <<< "$job"
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" "$TRAIN" \
    --dataset "$REPO/demos/${ds}.hdf5" \
    --config "$CFGDIR/${cfg}.yaml" \
    --wandb_project pc_bc --run_name "$name" --out_dir "$REPO/logs/pc_bc" \
    --seed "$seed" --ckpt_every_n_epochs 5 \
    > "/tmp/pcbc_${name}.log" 2>&1 &
  GPU_PID[$g]=$!
  log "dispatched $name (seed $seed, $cfg, $ds) -> GPU $g (pid $!)"
}

log "queue start: ${#QUEUE[@]} jobs"
idx=0
while [ "$idx" -lt "${#QUEUE[@]}" ]; do
  for g in "${GPUS[@]}"; do
    [ "$idx" -lt "${#QUEUE[@]}" ] || break
    if gpu_idle "$g"; then
      dispatch "$g" "${QUEUE[$idx]}"
      idx=$((idx + 1))
      sleep "$STAGGER"
    fi
  done
  [ "$idx" -lt "${#QUEUE[@]}" ] && sleep "$POLL"
done
log "all ${#QUEUE[@]} jobs dispatched; scheduler exiting (jobs keep running)"
