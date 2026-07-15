#!/usr/bin/env bash
# Wait for the combo ladder training to finish, then eval a spread of its epoch-ladder
# checkpoints in sim (512 envs / 1024 episodes each), in parallel across free GPUs.
# Result: where on the training curve sim success peaks vs val/loss.
set -u
REPO=/mnt/storage/lti/UWLab-patrick-private
LADDER_DIR="$REPO/logs/pc_bc/combo/epochs"
TRAIN_LOG=/tmp/pcbc_combo_ladder.log
RESULTS=/tmp/combo_ladder_results.txt
GPUS=(3 4 5 6 7 1)            # GPU 1 frees up once training ends
: > "$RESULTS"

# 1) wait for training to finish
until grep -qa "\[train\] DONE" "$TRAIN_LOG" 2>/dev/null || ! pgrep -f "run_name combo" >/dev/null 2>&1; do
  sleep 60
done
echo "[ladder] training done; ladder ckpts:" >> "$RESULTS"
ls -1 "$LADDER_DIR"/*.ckpt >> "$RESULTS" 2>/dev/null

# 2) pick a spread of rungs (~every 10 epochs) + always the last one
mapfile -t ALL < <(ls -1 "$LADDER_DIR"/pointnet-*.ckpt 2>/dev/null | sort)
PICK=()
i=0
for f in "${ALL[@]}"; do
  ep=$(basename "$f" | sed -E 's/pointnet-0*([0-9]+)-.*/\1/')
  # keep every rung whose epoch is a multiple of 10, plus the final rung
  if (( ep % 10 == 0 )) || [[ "$f" == "${ALL[-1]}" ]]; then PICK+=("$f"); fi
done
echo "[ladder] evaluating ${#PICK[@]} rungs: ${PICK[*]}" >> "$RESULTS"

# 3) eval each rung; throttle to one job per GPU at a time
run_eval () {
  local ckpt="$1" gpu="$2"
  local tag; tag=$(basename "$ckpt" .ckpt)
  docker cp "$ckpt" "uw-lab-base:/tmp/${tag}.ckpt" >/dev/null 2>&1
  local log="/tmp/bc_eval_${tag}.log"
  docker exec -e CUDA_VISIBLE_DEVICES="$gpu" uw-lab-base bash -lc "
    cd /workspace/uwlab && \
    /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetEval-v0 \
      --num_envs 512 --num_episodes 1024 --headless \
      --bc_checkpoint /tmp/${tag}.ckpt" > "$log" 2>&1
  local res; res=$(grep -A3 "^==================================================" "$log" | tail -3 | tr '\n' ' ')
  echo "RUNG $tag :: $res" >> "$RESULTS"
}

n=${#PICK[@]}; g=${#GPUS[@]}
idx=0
while (( idx < n )); do
  pids=()
  for (( j=0; j<g && idx<n; j++, idx++ )); do
    run_eval "${PICK[$idx]}" "${GPUS[$j]}" &
    pids+=($!)
  done
  wait "${pids[@]}"
done
echo "[ladder] ALL DONE" >> "$RESULTS"
