#!/usr/bin/env bash
# Screen the DAgger retrain's per-epoch ladder (+ the BC baseline) on BCPointNetCleanEval to find
# the EARLIEST epoch reaching ~80% success (and the peak-before-overtrain). Parallel across free
# GPUs; docker cp each host ckpt into uw-lab-base, run play.py, parse "Success rate:".
# Screening size (fast, relative curve); re-eval the winner at 1024/10000 for the headline number.
set -u
REPO=/mnt/storage/lti/UWLab-patrick-private
CONTAINER=uw-lab-base
TASK=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetCleanEval-v0
NUM_ENVS=1024
NUM_EPISODES=${NUM_EPISODES:-2048}     # screening; override for the precise re-eval
GPUS=(3 4 5 6 7)                        # 0 wedged; 1/2 busy with the two training runs
RESULTS=${RESULTS:-/tmp/dagger_ladder_eval.txt}
DAG=$REPO/logs/pc_bc/pndagger_xl_residual_clean/epochs

# job queue: "tag|host_ckpt"
QUEUE=()
QUEUE+=("baseline_last|$REPO/logs/pc_bc/pnclean_xl_residual_big_ee/last.ckpt")
for ep in 02 04 06 08 10 12 14 16 18 20; do
  ck=$(ls "$DAG/residual-point-net-0${ep}-"*.ckpt 2>/dev/null | grep -v -- '-v1' | head -1)
  [ -n "${ck:-}" ] && [ -f "$ck" ] && QUEUE+=("dagger_ep${ep}|${ck}")
done

log() { echo "[$(date '+%T')] $*"; }
: > "$RESULTS"
log "queue: ${#QUEUE[@]} evals on $TASK (${NUM_ENVS}env/${NUM_EPISODES}ep)"

run_eval() {
  local g=$1 tag=$2 ckpt=$3
  local clog="/tmp/dagger_eval_${tag}.log"
  docker cp "$ckpt" "${CONTAINER}:/tmp/${tag}.ckpt" >/dev/null 2>&1
  docker exec -e CUDA_VISIBLE_DEVICES="$g" "$CONTAINER" bash -lc "
    cd /workspace/uwlab && \
    /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
      --task ${TASK} --num_envs ${NUM_ENVS} --num_episodes ${NUM_EPISODES} --headless \
      --bc_checkpoint /tmp/${tag}.ckpt" > "$clog" 2>&1
  local sr eps
  sr=$(grep -a "Success rate:" "$clog" | tail -1 | sed -E 's/.*Success rate: *//')
  eps=$(grep -a "Number of episodes:" "$clog" | tail -1 | sed -E 's/.*Number of episodes: *//')
  echo "${tag}  success=${sr:-NA}  episodes=${eps:-NA}  (GPU $g)" >> "$RESULTS"
  log "DONE ${tag}: success=${sr:-NA} eps=${eps:-NA}"
}

# Static round-robin: assign queue[i] to GPUS[i % nGPU], run nGPU at a time.
i=0; n=${#QUEUE[@]}; ng=${#GPUS[@]}
while [ $i -lt $n ]; do
  pids=()
  for g in "${GPUS[@]}"; do
    [ $i -lt $n ] || break
    IFS='|' read -r tag ckpt <<< "${QUEUE[$i]}"
    log "dispatch ${tag} -> GPU $g"
    run_eval "$g" "$tag" "$ckpt" & pids+=($!)
    i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done   # barrier per wave
done
log "ALL DONE -> $RESULTS"
sort "$RESULTS"
