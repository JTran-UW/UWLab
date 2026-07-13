#!/usr/bin/env bash
# Eval the 2 continuing diffusion runs (new-dataset, 100ep) last.ckpt at 10k on the FIXED
# BCPointNetCleanEval (hard reset dist). GPUs 1 & 3 (2 & 4 busy with MSE runs).
set -u
REPO=/mnt/storage/lti/UWLab-patrick-private
C=uw-lab-base
TASK=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetCleanEval-v0
NUM_ENVS=1024; NUM_EPISODES=10000
RESULTS=/tmp/pndiff3_eval.txt
TAGS=(pndiff3_lr1e4 pndiff3_lr5e4); GPUS=(1 3)
: > "$RESULTS"; echo "[pndiff3-eval] $TASK ${NUM_ENVS}env/${NUM_EPISODES}ep"
pids=()
for i in "${!TAGS[@]}"; do
  tag=${TAGS[$i]}; g=${GPUS[$i]}; ck="$REPO/logs/pc_bc/${tag}/last.ckpt"; clog="/tmp/eval_${tag}.log"
  echo "[pndiff3-eval] dispatch $tag -> GPU $g"
  (
    docker cp "$ck" "${C}:/tmp/${tag}.ckpt" >/dev/null 2>&1
    docker exec -e CUDA_VISIBLE_DEVICES="$g" "$C" bash -lc "
      cd /workspace/uwlab && /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
        --task ${TASK} --num_envs ${NUM_ENVS} --num_episodes ${NUM_EPISODES} --headless \
        --bc_checkpoint /tmp/${tag}.ckpt" > "$clog" 2>&1
    sr=$(grep -a "Success rate:" "$clog" | tail -1 | sed -E 's/.*Success rate: *//')
    ep=$(grep -a "Number of episodes:" "$clog" | tail -1 | sed -E 's/.*Number of episodes: *//')
    echo "${tag}  success=${sr:-NA}  episodes=${ep:-NA}  (GPU $g)" >> "$RESULTS"
    echo "[pndiff3-eval] DONE ${tag}: ${sr:-NA}"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "PNDIFF3_EVAL_DONE"; sort "$RESULTS"
