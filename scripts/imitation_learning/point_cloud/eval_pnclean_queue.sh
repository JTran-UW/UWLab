#!/usr/bin/env bash
# Self-dispatching GPU eval queue for the four pnclean BC PointNet runs.
#
# Evals each run's epoch-20/30/40 + final checkpoint on the NotEECentric clean-PC eval
# env (512 envs / 1024 episodes), one job per GPU, dispatching the next queued eval the
# instant a GPU goes idle. 4 runs x 4 ckpts = 16 evals total.
#
# Checkpoints live on the HOST under logs/pc_bc/<run>/ but the eval runs inside the
# uw-lab-base container (logs/ there is a separate docker volume), so each ckpt is
# `docker cp`-ed into the container /tmp before its play.py rollout -- same pattern as
# eval_combo_ladder.sh.
#
# "20/30/40" map to the 0-indexed filenames point-net-019/029/039 (every-5 cadence,
# 1-indexed epoch count). "last" = last.ckpt.
#
# Run detached on the HOST:
#   nohup bash scripts/imitation_learning/point_cloud/eval_pnclean_queue.sh > /tmp/pnclean_eval_queue.log 2>&1 &
# Live results land in /tmp/pnclean_eval_results.txt

set -u
REPO=/mnt/storage/lti/UWLab-patrick-private
CONTAINER=uw-lab-base
TASK=${TASK:-OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetCleanNotEECentricEval-v0}
NUM_ENVS=512
NUM_EPISODES=1024
GPUS=(0 1 2 3 4 5 6 7)
POLL=30          # seconds between dispatch sweeps
STAGGER=40       # seconds after a dispatch before reusing the sweep (lets Isaac claim the GPU)
RESULTS=${RESULTS:-/tmp/pnclean_eval_results.txt}

# Runs to eval: defaults to the original four, override by passing run names as args, e.g.
#   bash eval_pnclean_queue.sh pnclean_widepool512_aux
if [ "$#" -gt 0 ]; then
  RUNS=("$@")
else
  RUNS=(pnclean_widepool1024 pnclean_bighead_widepool pnclean_widepool512 pnclean_xl)
fi
EPOCHS=(019 029 039 last)

# Build the job queue: "tag|host_ckpt_path"
QUEUE=()
for run in "${RUNS[@]}"; do
  for ep in "${EPOCHS[@]}"; do
    if [ "$ep" = "last" ]; then
      ckpt="$REPO/logs/pc_bc/${run}/last.ckpt"
    else
      # epochs/point-net-<ep>-<val>.ckpt (val varies); glob for the single match
      ckpt=$(ls "$REPO/logs/pc_bc/${run}/epochs/point-net-${ep}-"*.ckpt 2>/dev/null | head -1)
    fi
    if [ -z "${ckpt:-}" ] || [ ! -f "$ckpt" ]; then
      echo "WARN: missing ckpt for ${run} epoch ${ep} -- skipping" >&2
      continue
    fi
    QUEUE+=("${run}__ep${ep}|${ckpt}")
  done
done

declare -A GPU_PID   # physical GPU index -> pid of the eval this scheduler dispatched there

log() { echo "[$(date '+%F %T')] $*"; }

gpu_idle() {
  local g=$1
  local pid=${GPU_PID[$g]:-}
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && return 1
  local n
  n=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')
  [ "$n" -eq 0 ]
}

run_eval() {
  # cp ckpt into container, roll out, parse success rate, append to RESULTS. Backgrounded.
  local g=$1 tag=$2 ckpt=$3
  local clog="/tmp/pnclean_eval_${tag}.log"
  docker cp "$ckpt" "${CONTAINER}:/tmp/${tag}.ckpt" >/dev/null 2>&1
  docker exec -e CUDA_VISIBLE_DEVICES="$g" "$CONTAINER" bash -lc "
    cd /workspace/uwlab && \
    /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
      --task ${TASK} \
      --num_envs ${NUM_ENVS} --num_episodes ${NUM_EPISODES} --headless \
      --bc_checkpoint /tmp/${tag}.ckpt" > "$clog" 2>&1
  local sr eps
  sr=$(grep -a "Success rate:" "$clog" | tail -1 | sed -E 's/.*Success rate: *//')
  eps=$(grep -a "Number of episodes:" "$clog" | tail -1 | sed -E 's/.*Number of episodes: *//')
  echo "$(date '+%F %T')  ${tag}  success=${sr:-NA}  episodes=${eps:-NA}  (GPU $g)" >> "$RESULTS"
  log "DONE ${tag}: success=${sr:-NA} episodes=${eps:-NA}"
}

: > "$RESULTS"
log "eval queue start: ${#QUEUE[@]} jobs on task ${TASK}"
idx=0
while [ "$idx" -lt "${#QUEUE[@]}" ]; do
  for g in "${GPUS[@]}"; do
    [ "$idx" -lt "${#QUEUE[@]}" ] || break
    if gpu_idle "$g"; then
      IFS='|' read -r tag ckpt <<< "${QUEUE[$idx]}"
      log "dispatch ${tag} -> GPU $g"
      run_eval "$g" "$tag" "$ckpt" &
      GPU_PID[$g]=$!
      idx=$((idx + 1))
      sleep "$STAGGER"
    fi
  done
  [ "$idx" -lt "${#QUEUE[@]}" ] && sleep "$POLL"
done
log "all ${#QUEUE[@]} evals dispatched; waiting for in-flight evals to finish"
wait
log "ALL EVALS DONE -- results in $RESULTS"
echo "===== FINAL =====" >> "$RESULTS"
sort "$RESULTS" >> "$RESULTS" 2>/dev/null || true
