#!/usr/bin/env bash
#
# sweep_seeds.sh — submit the same cluster training job across multiple seeds.
#
# Thin wrapper around cluster_interface.sh: for each seed it appends `--seed <s>`
# to the forwarded job args and submits one independent SLURM job (one sbatch
# each, so they queue separately).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CLUSTER_INTERFACE="$SCRIPT_DIR/cluster_interface.sh"

usage() {
    cat <<'EOF'
Usage:
  sweep_seeds.sh --seeds "0 4 8 12"  -- <cluster_interface.sh args...>
  sweep_seeds.sh --n 5 [--start 0] [--step K] -- <cluster_interface.sh args...>

Everything after `--` is passed verbatim to cluster_interface.sh. Example:
  ./docker/cluster/sweep_seeds.sh --n 5 -- \
      --cluster hyak job base --gpus 4 --partition gpu-a40 --account weirdlab \
      --time 24:00:00 \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-Train-v0 \
      --num_envs 16384 --logger wandb --headless --distributed \
      env.scene.insertive_object=peg env.scene.receptive_object=peghole \
      env.curriculum.gravity_curriculum.params.reduction=monitor_mean \
      env.curriculum.gravity_curriculum.params.floor=0.1

Options:
  --seeds "L"   Explicit seed list (space- or comma-separated). Overrides --n.
  --n N         Generate N seeds: start, start+step, start+2*step, ...
  --start B     First seed when using --n (default: 0).
  --step K      Seed spacing for --n. Default: the --gpus value in the
                passthrough args (else 1). With --distributed each rank uses
                seed = base + local_rank, so spacing >= gpus keeps runs from
                overlapping.
  --name P      Run-name prefix. Each job gets --run_name "<P>_s<seed>" (or
                "s<seed>" if P is empty), which suffixes the log dir and the
                W&B run name so seeds are distinguishable. Default: empty.
  --no-run-name Do not inject --run_name (keep whatever is in the passthrough).
  --dry-run     Print the commands without submitting.
  -h, --help    Show this help.
EOF
}

SEEDS_LIST=""
N_SEEDS=""
START=0
STEP=""
NAME_PREFIX=""
INJECT_RUN_NAME=true
DRY_RUN=false
PASSTHROUGH=()

# Parse sweep options up to `--`; everything after `--` is for cluster_interface.sh
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)   SEEDS_LIST="$2"; shift 2 ;;
        --n)       N_SEEDS="$2"; shift 2 ;;
        --start)   START="$2"; shift 2 ;;
        --step)    STEP="$2"; shift 2 ;;
        --name)    NAME_PREFIX="$2"; shift 2 ;;
        --no-run-name) INJECT_RUN_NAME=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --)        shift; PASSTHROUGH=("$@"); break ;;
        *)
            echo "[ERROR] Unknown sweep option '$1' (forgot '--' before the cluster_interface args?)" >&2
            usage; exit 1 ;;
    esac
done

if [[ ${#PASSTHROUGH[@]} -eq 0 ]]; then
    echo "[ERROR] No passthrough args. Put cluster_interface.sh args after '--'." >&2
    usage; exit 1
fi
if [[ ! -x "$CLUSTER_INTERFACE" ]]; then
    echo "[ERROR] cluster_interface.sh not found or not executable at $CLUSTER_INTERFACE" >&2
    exit 1
fi

# Detect --gpus in the passthrough to use as the default seed step.
GPUS=1
for ((i = 0; i < ${#PASSTHROUGH[@]}; i++)); do
    if [[ "${PASSTHROUGH[$i]}" == "--gpus" ]]; then
        GPUS="${PASSTHROUGH[$((i + 1))]:-1}"
        break
    fi
done
[[ -z "$STEP" ]] && STEP="$GPUS"

# Build the seed list.
SEEDS=()
if [[ -n "$SEEDS_LIST" ]]; then
    IFS=', ' read -r -a SEEDS <<< "$SEEDS_LIST"
elif [[ -n "$N_SEEDS" ]]; then
    for ((k = 0; k < N_SEEDS; k++)); do
        SEEDS+=( $((START + k * STEP)) )
    done
else
    echo "[ERROR] Specify seeds with --seeds \"...\" or --n N." >&2
    usage; exit 1
fi

echo "[INFO] Sweeping ${#SEEDS[@]} seed(s): ${SEEDS[*]}"
echo "[INFO] Base command: $CLUSTER_INTERFACE ${PASSTHROUGH[*]} --seed <seed>"
$INJECT_RUN_NAME && echo "[INFO] Injecting per-seed --run_name \"${NAME_PREFIX:+${NAME_PREFIX}_}s<seed>\""
$DRY_RUN && echo "[INFO] --dry-run: nothing will be submitted."

for seed in "${SEEDS[@]}"; do
    # Per-seed extra args appended after the passthrough (last value wins in argparse).
    extra=( --seed "$seed" )
    if $INJECT_RUN_NAME; then
        extra+=( --run_name "${NAME_PREFIX:+${NAME_PREFIX}_}s${seed}" )
    fi

    echo "----------------------------------------------------------------"
    echo "[INFO] Submitting seed=$seed"
    if $DRY_RUN; then
        echo "       $CLUSTER_INTERFACE ${PASSTHROUGH[*]} ${extra[*]}"
    else
        "$CLUSTER_INTERFACE" "${PASSTHROUGH[@]}" "${extra[@]}"
        # cluster_interface.sh stamps each code snapshot with a second-resolution
        # datetime; pause so concurrent submits land in distinct snapshot dirs.
        sleep 1
    fi
done

echo "----------------------------------------------------------------"
echo "[INFO] Submitted ${#SEEDS[@]} job(s)."
