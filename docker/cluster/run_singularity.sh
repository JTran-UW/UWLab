#!/usr/bin/env bash

echo "(run_singularity.sh): Initial call on compute node."

# --- Argument Parsing ---
UWLAB_DIR_ARG="$1"
PROFILE_ARG="$2"
shift 2

# Use job-specific paths to prevent multiple jobs on the same node from interfering
# When SLURM schedules multiple jobs to the same node, they share $TMPDIR
# Without unique paths, jobs delete each other's files causing squashfuse crashes
JOB_TMPDIR="$TMPDIR/job_$SLURM_JOB_ID"
mkdir -p "$JOB_TMPDIR"
echo "[INFO] Using job-specific temp directory: $JOB_TMPDIR"

CLI_ARGS=()
DIST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes=*|--nproc_per_node=*|--rdzv_id=*|--rdzv_endpoint=*)
            DIST_ARGS+=("$1")
            shift
            ;;
        *)
            CLI_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Parsed Distributed Args: ${DIST_ARGS[@]}"
echo "Parsed Script CLI Args: ${CLI_ARGS[@]}"

#==
# Helper functions
#==

setup_directories() {
    # Check and create directories for the persistent cache on GPFS
    # Must include ALL subdirectories Isaac Sim expects, otherwise it crashes with:
    # "filesystem error: cannot get free space: No such file or directory"
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/kit/DerivedDataCache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/kit/nv_shadercache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data/Kit/Isaac-Sim/5.1/pip3-envs/default" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data/Kit/Isaac-Sim/5.1/exts" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data/documents/Kit/apps/Isaac-Sim/scripts" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data/documents/Kit/shared" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents/Kit/shared"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}

#==
# Main
#==

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the UW Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# Make sure that all directories exist in the persistent cache directory on GPFS
setup_directories

# warn if nested docker-isaac-sim directories exist (caused by older scripts)
if find "$CLUSTER_ISAAC_SIM_CACHE_DIR" -mindepth 2 -type d -name docker-isaac-sim | grep -q .; then
    echo "[WARN] Detected nested 'docker-isaac-sim' directories under $CLUSTER_ISAAC_SIM_CACHE_DIR."
    echo "[WARN] Please remove them (e.g., 'rm -rf $CLUSTER_ISAAC_SIM_CACHE_DIR/docker-isaac-sim') to avoid cache recursion."
fi

# paths inside the cache that can become read-only (Isaac Sim managed)
RSYNC_EXCLUDES=(
    --exclude "data/documents/Kit/shared/screenshots"
)

# Clean any leftover data from previous runs (in case of requeue with same job ID)
rm -rf "$JOB_TMPDIR/docker-isaac-sim"
rm -rf "$JOB_TMPDIR/$PROFILE_ARG.sif"

# Copy all persistent cache files to the node's local temporary storage
# Use rsync with trailing slashes to ensure contents are copied, not the folder itself
mkdir -p "$JOB_TMPDIR/docker-isaac-sim"
rsync -a "${RSYNC_EXCLUDES[@]}" $CLUSTER_ISAAC_SIM_CACHE_DIR/ "$JOB_TMPDIR/docker-isaac-sim/"

# Ensure critical directories exist on the local node (Isaac Sim crashes without these)
for dir in \
    "$JOB_TMPDIR/docker-isaac-sim/kit/DerivedDataCache" \
    "$JOB_TMPDIR/docker-isaac-sim/kit/nv_shadercache" \
    "$JOB_TMPDIR/docker-isaac-sim/data/Kit/Isaac-Sim/5.1/pip3-envs/default" \
    "$JOB_TMPDIR/docker-isaac-sim/data/Kit/Isaac-Sim/5.1/exts" \
    "$JOB_TMPDIR/docker-isaac-sim/data/documents/Kit/apps/Isaac-Sim/scripts" \
    "$JOB_TMPDIR/docker-isaac-sim/data/documents/Kit/shared" \
    "$JOB_TMPDIR/docker-isaac-sim/documents/Kit/shared"; do
    mkdir -p "$dir"
done
echo "[INFO] Verified critical cache directories exist on node."

# make sure logs directory exists (in the permanent uwlab directory)
mkdir -p "$CLUSTER_UWLAB_DIR/logs"
touch "$CLUSTER_UWLAB_DIR/logs/.keep"

# copy the temporary uwlab directory with the latest changes to the compute node
cp -r "$UWLAB_DIR_ARG" "$JOB_TMPDIR"
# Get the directory name
dir_name=$(basename "$UWLAB_DIR_ARG")

# Extract container to job-specific directory
# Retry logic handles transient GPFS read failures
MAX_RETRIES=3
RETRY_DELAY=5
for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[INFO] Extracting container (attempt $attempt/$MAX_RETRIES)..."
    if tar -xf "$CLUSTER_SIF_PATH/$PROFILE_ARG.tar" -C "$JOB_TMPDIR" 2>&1; then
        # Verify the .sif file was extracted and is readable
        if [ -f "$JOB_TMPDIR/$PROFILE_ARG.sif" ] && [ -s "$JOB_TMPDIR/$PROFILE_ARG.sif" ]; then
            echo "[INFO] Container extracted successfully."
            break
        else
            echo "[WARN] Container file missing or empty after extraction."
        fi
    else
        echo "[WARN] Tar extraction failed."
    fi

    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "[INFO] Retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
        rm -f "$JOB_TMPDIR/$PROFILE_ARG.sif"  # Clean up partial extraction
    else
        echo "[ERROR] Failed to extract container after $MAX_RETRIES attempts."
        exit 1
    fi
done

# Create a persistent /tmp on GPFS (the default --containall /tmp is a tiny RAM tmpfs)
mkdir -p "$CLUSTER_UWLAB_DIR/tmp"

# Build optional cloud-assets bind mount (skipped if dir doesn't exist)
CLOUD_ASSETS_BIND=""
CLOUD_ASSETS_ENV=""
if [ -n "$CLUSTER_CLOUD_ASSETS_DIR" ] && [ -d "$CLUSTER_CLOUD_ASSETS_DIR" ]; then
    CLOUD_ASSETS_BIND="-B $CLUSTER_CLOUD_ASSETS_DIR:/workspace/cloud_assets:ro"
    CLOUD_ASSETS_ENV="--env UWLAB_CLOUD_ASSETS_DIR=/workspace/cloud_assets"
    echo "[INFO] Binding cloud assets from $CLUSTER_CLOUD_ASSETS_DIR"
else
    echo "[WARN] No local cloud assets found at ${CLUSTER_CLOUD_ASSETS_DIR:-<unset>}; assets will be downloaded at runtime"
fi

# Build optional rsl_rl bind mount (overrides container-installed rsl_rl with local fork)
RSL_RL_BIND=""
if [ -n "$CLUSTER_RSL_RL_DIR" ] && [ -d "$CLUSTER_RSL_RL_DIR" ]; then
    RSL_RL_BIND="-B $CLUSTER_RSL_RL_DIR:/workspace/rsl_rl:rw"
    echo "[INFO] Binding rsl_rl from $CLUSTER_RSL_RL_DIR"
else
    echo "[WARN] CLUSTER_RSL_RL_DIR not set or not found; using container-installed rsl_rl"
fi

# Build optional holosoma bind mounts:
#   - source dir (read-only) — picked up via editable install .pth file
#   - deps dir (read-write, persistent) — pip installs holosoma deps here on first run
HOLOSOMA_BIND=""
HOLOSOMA_ENV=""
if [ -n "$CLUSTER_HOLOSOMA_DIR" ] && [ -d "$CLUSTER_HOLOSOMA_DIR" ]; then
    mkdir -p "$CLUSTER_HOLOSOMA_DEPS_DIR"
    HOLOSOMA_BIND="-B $CLUSTER_HOLOSOMA_DIR:/workspace/holosoma:rw -B $CLUSTER_HOLOSOMA_DEPS_DIR:/workspace/holosoma-deps:rw"
    HOLOSOMA_ENV="--env PYTHONUSERBASE=/workspace/holosoma-deps"
    echo "[INFO] Binding holosoma from $CLUSTER_HOLOSOMA_DIR (deps cache: $CLUSTER_HOLOSOMA_DEPS_DIR)"
else
    echo "[WARN] CLUSTER_HOLOSOMA_DIR not set or not found; holosoma will not be available"
fi

# execute command in singularity container
singularity exec \
    $CLOUD_ASSETS_BIND \
    $CLOUD_ASSETS_ENV \
    $RSL_RL_BIND \
    $HOLOSOMA_BIND \
    $HOLOSOMA_ENV \
    -B "$CLUSTER_UWLAB_DIR/tmp":/tmp:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/kit":${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/cache":${DOCKER_USER_HOME}/.cache:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/computecache":${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/logs":${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/data":${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/data":/isaac-sim/kit/data:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/documents":${DOCKER_USER_HOME}/Documents:rw \
    -B "$JOB_TMPDIR/$dir_name":/workspace/uwlab:rw \
    -B $CLUSTER_UWLAB_DIR/logs:/workspace/uwlab/logs:rw \
    -B $CLUSTER_MOUNT_DIR:$CLUSTER_MOUNT_DIR:rw \
    -B /etc/resolv.conf:/etc/resolv.conf:ro \
    -B /etc/pki:/etc/pki:ro \
    -B /etc/ssl:/etc/ssl:ro \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
    --env SSL_CERT_DIR=/etc/pki/tls/certs \
    --env REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt \
    --env CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt \
    --nv --containall "$JOB_TMPDIR/$PROFILE_ARG.sif" \
    bash -c 'export PYTHONUSERBASE=/workspace/holosoma-deps && if [ -d /workspace/rsl_rl ]; then echo "[INFO] Using rsl_rl from local fork (PYTHONPATH override)" && export PYTHONPATH=/workspace/rsl_rl:${PYTHONPATH}; fi && echo "[INFO] Installing holosoma deps (cached after first run)..." && /isaac-sim/python.sh -m pip install --user -q -e /workspace/holosoma/src/holosoma && export PYTHONPATH=/workspace/holosoma-deps/lib/python3.11/site-packages:${PYTHONPATH} && export VK_LOADER_LAYERS_DISABLE=VK_LAYER_NV_optimus && export UWLAB_PATH=/workspace/uwlab && cd /workspace/uwlab && /isaac-sim/python.sh -m torch.distributed.run --rdzv_backend=c10d "$@"' _ ${DIST_ARGS[@]} ${CLUSTER_PYTHON_EXECUTABLE} ${CLI_ARGS[@]} --distributed

# copy resulting cache files back to the persistent storage
rsync -azPv "${RSYNC_EXCLUDES[@]}" "$JOB_TMPDIR/docker-isaac-sim/" $CLUSTER_ISAAC_SIM_CACHE_DIR

# Clean up job-specific temp directory to free node-local disk space
rm -rf "$JOB_TMPDIR"
echo "[INFO] Cleaned up job temp directory: $JOB_TMPDIR"

# if defined, remove the temporary uwlab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf "$UWLAB_DIR_ARG"
fi

echo "(run_singularity.sh): Return"
