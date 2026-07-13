#!/usr/bin/env bash
# Precise 1024-env / 10000-episode eval of the BC baseline + selected DAgger epochs on
# BCPointNetCleanEval, one job per GPU in parallel. Headline numbers (screening runs noisy).
set -u
REPO=/mnt/storage/lti/UWLab-patrick-private
C=uw-lab-base
TASK=OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetCleanEval-v0
NUM_ENVS=1024
NUM_EPISODES=10000
RESULTS=/tmp/precise_eval.txt
DAG=$REPO/logs/pc_bc/pndagger_xl_residual_clean/epochs

# Parallel arrays: tag / gpu / host-ckpt-path
TAGS=(baseline dagger_ep08 dagger_ep12 dagger_ep16)
GPUS=(3 4 5 6)
CKPTS=(
  "$REPO/logs/pc_bc/pnclean_xl_residual_big_ee/last.ckpt"
  "$(ls $DAG/residual-point-net-008-*.ckpt | grep -v -- -v1 | head -1)"
  "$(ls $DAG/residual-point-net-012-*.ckpt | grep -v -- -v1 | head -1)"
  "$(ls $DAG/residual-point-net-016-*.ckpt | grep -v -- -v1 | head -1)"
)

: > "$RESULTS"
echo "[precise] $TASK  ${NUM_ENVS}env/${NUM_EPISODES}ep"
pids=()
for i in "${!TAGS[@]}"; do
  tag=${TAGS[$i]}; g=${GPUS[$i]}; ck=${CKPTS[$i]}; clog="/tmp/pr_${tag}.log"
  echo "[precise] dispatch $tag -> GPU $g  ($ck)"
  (
    docker cp "$ck" "${C}:/tmp/pr_${tag}.ckpt" >/dev/null 2>&1
    docker exec -e CUDA_VISIBLE_DEVICES="$g" "$C" bash -lc "
      cd /workspace/uwlab && \
      /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
        --task ${TASK} --num_envs ${NUM_ENVS} --num_episodes ${NUM_EPISODES} --headless \
        --bc_checkpoint /tmp/pr_${tag}.ckpt" > "$clog" 2>&1
    sr=$(grep -a "Success rate:" "$clog" | tail -1 | sed -E 's/.*Success rate: *//')
    ep=$(grep -a "Number of episodes:" "$clog" | tail -1 | sed -E 's/.*Number of episodes: *//')
    echo "${tag}  success=${sr:-NA}  episodes=${ep:-NA}  (GPU $g)" >> "$RESULTS"
    echo "[precise] DONE ${tag}: ${sr:-NA}"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "PRECISE_DONE"
sort "$RESULTS"
