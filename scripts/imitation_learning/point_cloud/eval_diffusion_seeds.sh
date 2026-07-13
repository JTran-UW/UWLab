#!/usr/bin/env bash
# Eval the 4 diffusion-head seeds (last.ckpt each) at 1024env/10000ep on BCPointNetCleanEval,
# one seed per GPU in parallel. Diffusion sampling (10 DDIM steps) is a bit slower than MSE.
set -u
REPO=/mnt/storage/lti/UWLab-patrick-private
C=uw-lab-base
TASK=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetCleanEval-v0
NUM_ENVS=1024
NUM_EPISODES=10000
RESULTS=/tmp/diffusion_seeds_eval.txt

TAGS=(diff_s0 diff_s1 diff_s2 diff_s3)
GPUS=(1 2 3 4)
CKPTS=(
  "$REPO/logs/pc_bc/pndiff_xl_residual_clean/last.ckpt"
  "$REPO/logs/pc_bc/pndiff_xl_residual_clean_s1/last.ckpt"
  "$REPO/logs/pc_bc/pndiff_xl_residual_clean_s2/last.ckpt"
  "$REPO/logs/pc_bc/pndiff_xl_residual_clean_s3/last.ckpt"
)

: > "$RESULTS"
echo "[diff-eval] $TASK  ${NUM_ENVS}env/${NUM_EPISODES}ep  (4 seeds)"
pids=()
for i in "${!TAGS[@]}"; do
  tag=${TAGS[$i]}; g=${GPUS[$i]}; ck=${CKPTS[$i]}; clog="/tmp/${tag}.log"
  echo "[diff-eval] dispatch $tag -> GPU $g"
  (
    docker cp "$ck" "${C}:/tmp/${tag}.ckpt" >/dev/null 2>&1
    docker exec -e CUDA_VISIBLE_DEVICES="$g" "$C" bash -lc "
      cd /workspace/uwlab && \
      /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
        --task ${TASK} --num_envs ${NUM_ENVS} --num_episodes ${NUM_EPISODES} --headless \
        --bc_checkpoint /tmp/${tag}.ckpt" > "$clog" 2>&1
    sr=$(grep -a "Success rate:" "$clog" | tail -1 | sed -E 's/.*Success rate: *//')
    ep=$(grep -a "Number of episodes:" "$clog" | tail -1 | sed -E 's/.*Number of episodes: *//')
    echo "${tag}  success=${sr:-NA}  episodes=${ep:-NA}  (GPU $g)" >> "$RESULTS"
    echo "[diff-eval] DONE ${tag}: ${sr:-NA}"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "DIFF_EVAL_DONE"
sort "$RESULTS"
