#!/usr/bin/env bash
# Overnight PointNet-BC sweep launcher.
#
# Launches the 7 variant configs (base is run separately / first) as background jobs,
# one per GPU, round-robined over the free GPUs. Each job logs to /tmp/pcbc_<name>.log
# and to wandb (project pc_bc, run name = config name).
#
# Override the dataset / project / gpu list via env vars:
#   DATASET=demos/foo.hdf5 GPUS="4 5 6 7 0 1" bash run_sweep.sh
set -uo pipefail

REPO=/mnt/storage/lti/UWLab-patrick-private
PY=/home/ubuntu/miniforge3/envs/patlab/bin/python
TRAIN=$REPO/scripts/imitation_learning/point_cloud/train_point_net.py
CFG_DIR=$REPO/scripts/imitation_learning/point_cloud/configs

DATASET=${DATASET:-$REPO/demos/bench_16k_100k.hdf5}
PROJECT=${PROJECT:-pc_bc}
OUT_DIR=${OUT_DIR:-$REPO/logs/pc_bc}
# GPU 2 is intentionally skipped (busy). Edit to taste.
read -r -a GPUS <<< "${GPUS:-4 5 6 7 0 1 4}"

# Overnight variants (base is launched separately first).
CONFIGS=(nll wide_encoder subsample1024 lr1e3 lr1e4_long strong_reg big_model)

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset not found: $DATASET" >&2
  exit 1
fi

echo "sweep: ${#CONFIGS[@]} jobs | dataset=$DATASET | project=$PROJECT"
for i in "${!CONFIGS[@]}"; do
  name=${CONFIGS[$i]}
  gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
  log=/tmp/pcbc_${name}.log
  echo "  [$((i+1))/${#CONFIGS[@]}] $name -> GPU $gpu  ($log)"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" "$TRAIN" \
    --dataset "$DATASET" --config "$CFG_DIR/${name}.yaml" \
    --wandb_project "$PROJECT" --run_name "$name" --out_dir "$OUT_DIR" \
    > "$log" 2>&1 &
  # Stagger startup so the per-process ~77GB dataset loads don't land at once.
  sleep 45
done
echo "all launched. tail logs: tail -f /tmp/pcbc_*.log"
