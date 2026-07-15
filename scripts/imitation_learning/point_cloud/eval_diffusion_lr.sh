#!/usr/bin/env bash
# Eval the diffusion LR sweep (v-pred / zero-SNR head, 100ep) last.ckpt per LR at 1024env/10000ep
# on BCPointNetCleanEval, one run per GPU (1-4). Compare to the 78.5% MSE baseline.
set -u
REPO=/mnt/storage/lti/UWLab-patrick-private
C=uw-lab-base
TASK=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetCleanEval-v0
NUM_ENVS=1024
NUM_EPISODES=10000
RESULTS=/tmp/diffusion_lr_eval.txt

TAGS=(lr1e4 lr3e4 lr5e4 lr1e3)
GPUS=(1 2 3 4)

: > "$RESULTS"
echo "[lr-eval] $TASK  ${NUM_ENVS}env/${NUM_EPISODES}ep"
pids=()
for i in "${!TAGS[@]}"; do
  tag=${TAGS[$i]}; g=${GPUS[$i]}
  ck="$REPO/logs/pc_bc/pndiff2_${tag}/last.ckpt"
  clog="/tmp/diff_lr_${tag}.log"
  echo "[lr-eval] dispatch $tag -> GPU $g"
  (
    docker cp "$ck" "${C}:/tmp/diff_lr_${tag}.ckpt" >/dev/null 2>&1
    docker exec -e CUDA_VISIBLE_DEVICES="$g" "$C" bash -lc "
      cd /workspace/uwlab && \
      /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
        --task ${TASK} --num_envs ${NUM_ENVS} --num_episodes ${NUM_EPISODES} --headless \
        --bc_checkpoint /tmp/diff_lr_${tag}.ckpt" > "$clog" 2>&1
    sr=$(grep -a "Success rate:" "$clog" | tail -1 | sed -E 's/.*Success rate: *//')
    ep=$(grep -a "Number of episodes:" "$clog" | tail -1 | sed -E 's/.*Number of episodes: *//')
    echo "${tag}  success=${sr:-NA}  episodes=${ep:-NA}  (GPU $g)" >> "$RESULTS"
    echo "[lr-eval] DONE ${tag}: ${sr:-NA}"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "LR_EVAL_DONE"
sort "$RESULTS"
