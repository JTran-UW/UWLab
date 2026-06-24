#!/usr/bin/env bash

# Read environment variables for node and GPU counts, with defaults for single-GPU runs.
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

# Defaults for Hyak (Conservative defaults that fit most partitions)
ACCOUNT=${ACCOUNT:-weirdlab}
PARTITION=${PARTITION:-gpu-a40}
# CPU/Mem defaults - safe for A100/A40/L40s
CPUS_PER_TASK=${CPUS_PER_TASK:-6}
MEM_PER_GPU=${MEM_PER_GPU:-60G}
CONSTRAINT=${CONSTRAINT:-"h200|l40|l40s|a40"}
# Time limit (default 24 hours, use shorter for testing requeue, e.g., TIME=00:05:00)
TIME=${TIME:-24:00:00}
# QOS (optional; empty = use the partition/account default QOS). Required by some
# pools, e.g. ckpt GPU jobs typically need QOS=ckpt-gpu.
QOS=${QOS:-}

# Resolve the SLURM log directory to an absolute path BEFORE it is used in the
# sbatch --output/--error directives. Fall back to <uwlab>/slurm_logs (derived
# from the persistent uwlab dir passed as $1) when not provided via the env.
SLURM_LOGS_DIR=${SLURM_LOGS_DIR:-"$(dirname "$1")/slurm_logs"}
# Guard: an empty or relative path makes SLURM silently write slurm-<jobid>.out
# into the submit dir, which on Hyak is the auto-deleted timestamped code copy
# (REMOVE_CODE_COPY_AFTER_JOB=true) -- i.e. the logs vanish. Fail loudly instead.
if [[ -z "$SLURM_LOGS_DIR" || "$SLURM_LOGS_DIR" != /* ]]; then
    echo "[ERROR] SLURM_LOGS_DIR is empty or not absolute ('$SLURM_LOGS_DIR'); refusing to submit." >&2
    exit 1
fi
mkdir -p "$SLURM_LOGS_DIR"

# Calculate total tasks for SLURM
NTASKS=$((NODES * GPUS_PER_NODE))
echo "Requesting ${NODES} node(s) with ${GPUS_PER_NODE} GPU(s) per node."

echo "----------------------------------------------------------------"
echo "Submitting Job to Hyak"
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
if [ -n "${QOS}" ]; then
    echo "QOS:           ${QOS}"
fi
echo "Time Limit:    ${TIME}"
echo "SLURM Logs:    ${SLURM_LOGS_DIR}"
echo "----------------------------------------------------------------"

# Enable requeue for all jobs
REQUEUE_FLAG="#SBATCH --requeue"

# Additional check for checkpoint partitions
if [[ "$PARTITION" == *"ckpt"* ]]; then
    echo "Detected checkpoint partition. Requeue enabled for preemption handling."
    if [[ "$ACCOUNT" != *"-ckpt" ]]; then
        echo "WARNING: Partition is '$PARTITION' but Account is '$ACCOUNT'. Checkpoint jobs usually require an account ending in '-ckpt'."
    fi
fi

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<'EOFSCRIPT' > job.sh
#!/bin/bash

# ------------------ Job Metadata ------------------
#SBATCH --job-name="uwlab-dist-DATETIME_PLACEHOLDER"
#SBATCH --output=LOGS_PLACEHOLDER/%x-%j.out
#SBATCH --error=LOGS_PLACEHOLDER/%x-%j.err
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

# Optional QOS
QOS_PLACEHOLDER

# Optional Constraint
CONSTRAINT_PLACEHOLDER

# Requeue flag
REQUEUE_PLACEHOLDER

# --- Requeue Handler for Time Limits and Preemption ---
requeue_handler() {
    echo "[$(date)] Caught signal: $1 - marking job $SLURM_JOB_ID for requeue"
    scontrol requeue $SLURM_JOB_ID
    # Don't exit - let the job continue to save checkpoints until forced kill
}

# Install signal handlers for both time limit (USR1) and preemption (TERM)
trap 'requeue_handler USR1' USR1
trap 'requeue_handler TERM' TERM

echo "[$(date)] Job $SLURM_JOB_ID starting on $(hostname)"
echo "[$(date)] Requeue handler installed - job will auto-resume if preempted or hits time limit"

# --- PyTorch Distributed Setup ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(($SLURM_JOB_ID % 55535 + 10000))

echo "Master Node: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# --- Execute the Job ---
srun --export=ALL bash "UWLAB_DIR_PLACEHOLDER/docker/cluster/run_singularity.sh" "UWLAB_DIR_PLACEHOLDER" "PROFILE_PLACEHOLDER" \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --run_id="$SLURM_JOB_ID" \
    JOB_ARGS_PLACEHOLDER &

# Wait for srun but allow trap handlers to run
wait $!
EXIT_CODE=$?

echo "[$(date)] Job finished with exit code: $EXIT_CODE"

# Propagate the real exit code so SLURM reports FAILED (not COMPLETED) when the
# training process crashes. Requeue on preemption/time-limit is handled by the
# USR1/TERM traps above, so this only affects the terminal job state.
exit $EXIT_CODE
EOFSCRIPT

# Replace placeholders with actual values
# Using % as delimiter to avoid conflicts with | (in CONSTRAINT) and # (in SBATCH comments)
sed -i "s%DATETIME_PLACEHOLDER%$(date +"%Y-%m-%dT%H-%M")%g" job.sh
sed -i "s%LOGS_PLACEHOLDER%${SLURM_LOGS_DIR}%g" job.sh
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

# Handle optional constraint
if [ -n "${CONSTRAINT}" ]; then
    sed -i "s%CONSTRAINT_PLACEHOLDER%#SBATCH --constraint=\"${CONSTRAINT}\"%g" job.sh
else
    sed -i "s%CONSTRAINT_PLACEHOLDER%%g" job.sh
fi

# Handle optional QOS
if [ -n "${QOS}" ]; then
    sed -i "s%QOS_PLACEHOLDER%#SBATCH --qos=${QOS}%g" job.sh
else
    sed -i "s%QOS_PLACEHOLDER%%g" job.sh
fi

# Handle requeue flag
sed -i "s%REQUEUE_PLACEHOLDER%${REQUEUE_FLAG}%g" job.sh

# Submit
sbatch --export=ALL < job.sh
rm job.sh
