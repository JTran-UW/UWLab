#!/usr/bin/env bash

# Read environment variables for node and GPU counts, with defaults for single-GPU runs.
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

# Python script to run (can be overridden per-launch via CLUSTER_PYTHON_EXECUTABLE env var)
CLUSTER_PYTHON_EXECUTABLE=${CLUSTER_PYTHON_EXECUTABLE:-scripts/reinforcement_learning/holosoma/train.py}

# Defaults for Tillicum (discovered 2026-08-20)
# gpu-h200: 22 nodes x 8x H200(140GB), 64 CPU, ~2TB RAM, MaxTime UNLIMITED, QoS normal.
# NOTE: tillicum BILLS GPU-hours (srun prints a cost estimate). 8 GPUs x 24h is real money.
ACCOUNT=${ACCOUNT:-weirdlab}
PARTITION=${PARTITION:-gpu-h200}
# CPU/Mem defaults - safe for A100/A40/L40s
CPUS_PER_TASK=${CPUS_PER_TASK:-6}
MEM_PER_GPU=${MEM_PER_GPU:-60G}
# Tillicum nodes report AvailableFeatures=(null): ANY --constraint makes a job
# permanently PENDING (never an error). Leave empty unless features are added later.
CONSTRAINT=${CONSTRAINT:-""}
# Comma-separated nodes to avoid, e.g. EXCLUDE_NODES=g018. A bad GPU/driver on a node surfaces as
# `nvmlDeviceGetHandleByIndex(N) failed` in the NCCL warmup, before training starts (seen 2026-08-24).
EXCLUDE_NODES=${EXCLUDE_NODES:-""}
# Time limit (default 24 hours, use shorter for testing requeue, e.g., TIME=00:05:00)
TIME=${TIME:-24:00:00}

# SLURM log directory on shared storage
# TODO: replace with your scrubbed path
SLURM_LOG_DIR=${SLURM_LOG_DIR:-/gpfs/scrubbed/profjat/slurm_logs}

# Calculate total tasks for SLURM
NTASKS=$((NODES * GPUS_PER_NODE))
echo "Requesting ${NODES} node(s) with ${GPUS_PER_NODE} GPU(s) per node."

echo "----------------------------------------------------------------"
echo "Submitting Job to Tillicum"
echo "----------------------------------------------------------------"
echo "Account:       ${ACCOUNT}"
echo "Partition:     ${PARTITION}"
echo "Nodes:         ${NODES}"
echo "GPUs per Node: ${GPUS_PER_NODE}"
echo "CPUs per Task: ${CPUS_PER_TASK}"
echo "Mem per GPU:   ${MEM_PER_GPU}"
if [ -n "${CONSTRAINT}" ]; then
    echo "Constraint:    ${CONSTRAINT}"
fi
echo "Time Limit:    ${TIME}"
echo "----------------------------------------------------------------"

# Tillicum has no preemptible tier, so requeue is off by default. This also avoids
# klone's trap where --requeue + a SIGTERM handler makes plain `scancel` resurrect a job
# (there, cancelling needs `scontrol update JobId=<id> Requeue=0` first).
# Set REQUEUE=1 to opt back in (e.g. for auto-resume at the time limit).
if [ "${REQUEUE:-0}" = "1" ]; then REQUEUE_FLAG="#SBATCH --requeue"; else REQUEUE_FLAG=""; fi

# Tillicum: no checkpoint/preemptible partition, so no ckpt-account special casing.
if [ -n "${REQUEUE_FLAG}" ]; then echo "Requeue: ENABLED (REQUEUE=1)"; else echo "Requeue: disabled"; fi

# Ensure slurm log directory exists
mkdir -p "$SLURM_LOG_DIR"

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<'EOFSCRIPT' > job.sh
#!/bin/bash

# ------------------ Job Metadata ------------------
#SBATCH --job-name="uwlab-dist-DATETIME_PLACEHOLDER"
#SBATCH --output=SLURMLOG_PLACEHOLDER/%x-%j.out
#SBATCH --error=SLURMLOG_PLACEHOLDER/%x-%j.err
#SBATCH --open-mode=append                        # Append to log files on requeue instead of overwriting

# ------------------ Resource Requests ------------------
#SBATCH --account=ACCOUNT_PLACEHOLDER
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --nodes=NODES_PLACEHOLDER
#SBATCH --ntasks-per-node=1                       # one task per node (launch script handles distribution)
#SBATCH --gpus-per-node=GPUS_PLACEHOLDER
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER

# Signal handler: send USR1 30 seconds before time limit to trigger requeue
#SBATCH --signal=B:USR1@30

# Optional Constraint
CONSTRAINT_PLACEHOLDER
# Optional node exclusion
EXCLUDE_PLACEHOLDER

# Requeue flag
REQUEUE_PLACEHOLDER

# --- Requeue Handler for Time Limits and Preemption ---
requeue_handler() {
    echo "[$(date)] Caught signal: $1 - marking job $SLURM_JOB_ID for requeue"
    scontrol requeue $SLURM_JOB_ID
    # Don't exit - let the job continue to save checkpoints until forced kill
}

# Install signal handlers for both time limit (USR1) and preemption (TERM).
# Gated on REQUEUE: without this the handler requeues the job at every time limit even when
# `Requeue: disabled` was printed, because that path only drops the `#SBATCH --requeue` flag.
# Observed on job 252047 -- it silently re-ran three times, each restart resuming from the
# ORIGINAL --resume_path and discarding the curriculum state it had built up.
REQUEUE_ENABLED=REQUEUE_ENABLED_PLACEHOLDER
if [ "${REQUEUE_ENABLED}" = "1" ]; then
    trap 'requeue_handler USR1' USR1
    trap 'requeue_handler TERM' TERM
fi

echo "[$(date)] Job $SLURM_JOB_ID starting on $(hostname)"
if [ "${REQUEUE_ENABLED}" = "1" ]; then echo "[$(date)] Requeue handler installed - job will auto-resume if preempted or hits time limit"; else echo "[$(date)] Requeue handler NOT installed (REQUEUE=0) - job will end at the time limit"; fi

# --- PyTorch Distributed Setup ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(($SLURM_JOB_ID % 55535 + 10000))

echo "Master Node: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# --- NCCL settings for multi-GPU ---
# 4-GPU PPO dies at the first collective (ppo.broadcast_parameters -> NCCL broadcast) with
# "Cuda failure 'invalid argument'", even though all ranks bind distinct devices (cuda:0..3) and
# build their Isaac envs. run_singularity.sh forwards these with `--env`; an earlier
# APPTAINERENV_* attempt never reached the container, so nothing was actually being set.
# Plain exports here: srun --export=ALL makes them visible to run_singularity.sh, which passes
# them in explicitly.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,ENV}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}

# --- Execute the Job ---
# Script to run inside the container (set at submission time)
export CLUSTER_PYTHON_EXECUTABLE="CLUSTER_PYTHON_EXECUTABLE_PLACEHOLDER"

srun --export=ALL bash "UWLAB_DIR_PLACEHOLDER/docker/cluster/run_singularity.sh" "UWLAB_DIR_PLACEHOLDER" "PROFILE_PLACEHOLDER" \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    JOB_ARGS_PLACEHOLDER &

# Wait for srun but allow trap handlers to run
wait $!
EXIT_CODE=$?

echo "[$(date)] Job finished with exit code: $EXIT_CODE"
EOFSCRIPT

# Replace placeholders with actual values
# Using % as delimiter to avoid conflicts with | (in CONSTRAINT) and # (in SBATCH comments)
sed -i "s%DATETIME_PLACEHOLDER%$(date +"%Y-%m-%dT%H-%M")%g" job.sh
sed -i "s%SLURMLOG_PLACEHOLDER%${SLURM_LOG_DIR}%g" job.sh
sed -i "s%ACCOUNT_PLACEHOLDER%${ACCOUNT}%g" job.sh
sed -i "s%PARTITION_PLACEHOLDER%${PARTITION}%g" job.sh
sed -i "s%NODES_PLACEHOLDER%${NODES}%g" job.sh
sed -i "s%GPUS_PLACEHOLDER%${GPUS_PER_NODE}%g" job.sh
sed -i "s%CPUS_PLACEHOLDER%$((GPUS_PER_NODE * CPUS_PER_TASK))%g" job.sh
sed -i "s%MEM_PLACEHOLDER%$((GPUS_PER_NODE * $(echo ${MEM_PER_GPU} | tr -dc '0-9') ))G%g" job.sh
sed -i "s%TIME_PLACEHOLDER%${TIME}%g" job.sh
sed -i "s%UWLAB_DIR_PLACEHOLDER%$1%g" job.sh
sed -i "s%PROFILE_PLACEHOLDER%$2%g" job.sh
sed -i "s%JOB_ARGS_PLACEHOLDER%${*:3}%g" job.sh
sed -i "s%CLUSTER_PYTHON_EXECUTABLE_PLACEHOLDER%${CLUSTER_PYTHON_EXECUTABLE}%g" job.sh

# Handle optional constraint
if [ -n "${CONSTRAINT}" ]; then
    sed -i "s%CONSTRAINT_PLACEHOLDER%#SBATCH --constraint=\"${CONSTRAINT}\"%g" job.sh
else
    sed -i "s%CONSTRAINT_PLACEHOLDER%%g" job.sh
fi

# Handle optional node exclusion
if [ -n "${EXCLUDE_NODES}" ]; then
    sed -i "s%EXCLUDE_PLACEHOLDER%#SBATCH --exclude=${EXCLUDE_NODES}%g" job.sh
else
    sed -i "s%EXCLUDE_PLACEHOLDER%%g" job.sh
fi

# Handle requeue flag
sed -i "s%REQUEUE_PLACEHOLDER%${REQUEUE_FLAG}%g" job.sh
sed -i "s%REQUEUE_ENABLED_PLACEHOLDER%${REQUEUE:-0}%g" job.sh

# Submit
sbatch < job.sh
rm job.sh
