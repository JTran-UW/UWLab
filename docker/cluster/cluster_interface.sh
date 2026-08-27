#!/usr/bin/env bash

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4 2>/dev/null || true

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#==
# Functions
#==
# Function to display warnings in red
display_warning() {
    echo -e "\033[31mWARNING: $1\033[0m"
}

# Helper function to compare version numbers
version_gte() {
    # Returns 0 if the first version is greater than or equal to the second, otherwise 1
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]
}

# Function to check docker versions
check_docker_version() {
    # check if docker is installed
    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed! Please check the 'Docker Guide' for instruction." >&2;
        exit 1
    fi
    # Retrieve Docker version
    docker_version=$(docker --version | awk '{ print $3 }')
    apptainer_version=$(apptainer --version | awk '{ print $3 }')

    # Check if Docker version is exactly 24.0.7 or Apptainer version is exactly 1.2.5
    if [ "$docker_version" = "24.0.7" ] && [ "$apptainer_version" = "1.2.5" ]; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."

    # Check if Docker version is >= 27.0.0 and Apptainer version is >= 1.3.4
    elif version_gte "$docker_version" "27.0.0" && version_gte "$apptainer_version" "1.3.4"; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."

    # Else, display a warning for non-tested versions
    else
        display_warning "Docker version ${docker_version} and Apptainer version ${apptainer_version} are non-tested versions. There could be issues, please try to update them. More info: https://uw-lab.github.io/UWLab/source/deployment/cluster.html"
    fi
}

# Checks if a docker image exists, otherwise prints warning and exists
check_image_exists() {
    image_name="$1"
    if ! docker image inspect $image_name &> /dev/null; then
        echo "[Error] The '$image_name' image does not exist!" >&2;
        echo "[Error] You might be able to build it with /UWLab/docker/container.py." >&2;
        exit 1
    fi
}

# Check if the singularity image exists on the remote host, otherwise print warning and exit
check_singularity_image_exists() {
    image_name="$1"
    if ! ssh "$CLUSTER_LOGIN" "[ -f $CLUSTER_SIF_PATH/$image_name.tar ]"; then
        echo "[Error] The '$image_name' image does not exist on the remote host $CLUSTER_LOGIN!" >&2;
        exit 1
    fi
}

submit_job() {

    echo "[INFO] Arguments passed to job script ${@}"

    case $CLUSTER_JOB_SCHEDULER in
        "SLURM")
            job_script_file=submit_job_slurm_${CLUSTER_NAME}.sh
            ;;
        "PBS")
            job_script_file=submit_job_pbs.sh
            ;;
        *)
            echo "[ERROR] Unsupported job scheduler specified: '$CLUSTER_JOB_SCHEDULER'. Supported options are: ['SLURM', 'PBS']"
            exit 1
            ;;
    esac

    # We copy the cluster-specific env file to .env.cluster on the remote side
    # so that run_singularity.sh works without modification. Then patch the
    # CLUSTER_*_DIR lines so the in-container mounts point at the suffixed
    # paths chosen for this submission (latest vs timestamped).
    ssh $CLUSTER_LOGIN "cd $CLUSTER_UWLAB_DIR && \
        cp -f docker/cluster/.env.${CLUSTER_NAME} docker/cluster/.env.cluster && \
        sed -i 's|^CLUSTER_UWLAB_DIR=.*|CLUSTER_UWLAB_DIR=${CLUSTER_UWLAB_DIR}|' docker/cluster/.env.cluster && \
        sed -i 's|^CLUSTER_RSL_RL_DIR=.*|CLUSTER_RSL_RL_DIR=${CLUSTER_RSL_RL_DIR}|' docker/cluster/.env.cluster && \
        sed -i 's|^CLUSTER_HOLOSOMA_DIR=.*|CLUSTER_HOLOSOMA_DIR=${CLUSTER_HOLOSOMA_DIR}|' docker/cluster/.env.cluster && \
        sed -i 's|^REMOVE_CODE_COPY_AFTER_JOB=.*|REMOVE_CODE_COPY_AFTER_JOB=${REMOVE_CODE_COPY_AFTER_JOB}|' docker/cluster/.env.cluster && \
        NODES=${NODES} GPUS_PER_NODE=${GPUS_PER_NODE} \
        PARTITION=${PARTITION} ACCOUNT=${ACCOUNT} TIME=${TIME} \
        CPUS_PER_TASK=${CPUS_PER_TASK} MEM_PER_GPU=${MEM_PER_GPU} CONSTRAINT='${CONSTRAINT}' \
        EXCLUDE_NODES='${EXCLUDE_NODES}' \
        CLUSTER_PYTHON_EXECUTABLE=${CLUSTER_PYTHON_EXECUTABLE} \
        RANK_ISOLATE_DISABLE=${RANK_ISOLATE_DISABLE:-} \
        NCCL_DEBUG=${NCCL_DEBUG:-} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-} \
        NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-} NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-} \
        NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} \
        bash $CLUSTER_UWLAB_DIR/docker/cluster/$job_script_file \"$CLUSTER_UWLAB_DIR\" \"uw-lab-$profile\" ${@}"
}

#==
# Main
#==

help() {
    echo -e "\nusage: $(basename "$0") [-h] [--cluster <name>] <command> [<profile>] [<job_args>...] -- Utility for interfacing between UWLab and compute clusters."
    echo -e "\noptions:"
    echo -e "  -h              Display this help message."
    echo -e "  --cluster       Specify the cluster configuration to use (e.g. hyak). Defaults to 'hyak'."
    echo -e "\ncommands:"
    echo -e "  push [<profile>]              Push the docker image to the cluster."
    echo -e "  job [<profile>] [<job_args>]  Submit a job to the cluster."
    echo -e "  sync-assets                   Sync cloud assets to the cluster for offline use."
    echo -e "\nwhere:"
    echo -e "  <profile>  is the optional container profile specification. Defaults to 'base'."
    echo -e "  <job_args> are optional arguments specific to the job command."
    echo -e "\n" >&2
}

# Parse options
CLUSTER_NAME="hyak"
CLI_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -h|--help)
            help
            exit 0
            ;;
        *)
            CLI_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${CLI_ARGS[@]}"

# Check for command
if [ $# -lt 1 ]; then
    echo "Error: Command is required." >&2
    help
    exit 1
fi

command=$1
shift
profile="base"

echo "[INFO] Using cluster configuration: $CLUSTER_NAME"

case $command in
    push)
        if [ $# -gt 1 ]; then
            echo "Error: Too many arguments for push command." >&2
            help
            exit 1
        fi
        [ $# -eq 1 ] && profile=$1
        echo "Executing push command"
        [ -n "$profile" ] && echo "Using profile: $profile"
        if ! command -v apptainer &> /dev/null; then
            echo "[INFO] Exiting because apptainer was not installed"
            echo "[INFO] You may follow the installation procedure from here: https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages"
            exit
        fi
        # Check if Docker image exists
        check_image_exists uw-lab-$profile:latest
        # Check docker and apptainer version
        check_docker_version
        # source env file to get cluster login and path information
        if [ -f "$SCRIPT_DIR/.env.$CLUSTER_NAME" ]; then
            source "$SCRIPT_DIR/.env.$CLUSTER_NAME"
        else
            echo "[ERROR] Environment file for cluster '$CLUSTER_NAME' not found at $SCRIPT_DIR/.env.$CLUSTER_NAME"
            exit 1
        fi
        # make sure exports directory exists
        mkdir -p /$SCRIPT_DIR/exports
        # clear old exports for selected profile
        rm -rf /$SCRIPT_DIR/exports/uw-lab-$profile*
        # create singularity image as a single .sif file (no --sandbox to avoid noexec issues on some clusters)
        cd /$SCRIPT_DIR/exports
        APPTAINER_NOHTTPS=1 apptainer build --fakeroot uw-lab-$profile.sif docker-daemon://uw-lab-$profile:latest
        # tar image (faster to send single file as opposed to directory with many files)
        tar -cvf /$SCRIPT_DIR/exports/uw-lab-$profile.tar uw-lab-$profile.sif
        # make sure target directory exists
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_SIF_PATH"
        # send image to cluster
        scp $SCRIPT_DIR/exports/uw-lab-$profile.tar $CLUSTER_LOGIN:$CLUSTER_SIF_PATH/uw-lab-$profile.tar
        ;;
    job)
        if [ $# -ge 1 ]; then
            passed_profile=$1
            if [ -f "$SCRIPT_DIR/../.env.$passed_profile" ]; then
                profile=$passed_profile
                shift
            fi
        fi
        job_args="$@"
        echo "[INFO] Executing job command"
        [ -n "$profile" ] && echo -e "\tUsing profile: $profile"
        [ -n "$job_args" ] && echo -e "\tJob arguments: $job_args"

        # source env file
        if [ -f "$SCRIPT_DIR/.env.$CLUSTER_NAME" ]; then
            source "$SCRIPT_DIR/.env.$CLUSTER_NAME"
        else
            echo "[ERROR] Environment file for cluster '$CLUSTER_NAME' not found at $SCRIPT_DIR/.env.$CLUSTER_NAME"
            exit 1
        fi

        # Snapshot suffix selects whether all three trees go to shared "_latest"
        # directories (default — fast, shared fate with other live jobs) or to
        # per-submission "_<timestamp>" directories (no shared fate).
        # Triggered from the caller by setting UWLAB_TIMESTAMP_SNAPSHOT=1.
        if [ "${UWLAB_TIMESTAMP_SNAPSHOT:-0}" = "1" ]; then
            SNAPSHOT_SUFFIX=$(date +"%Y%m%d_%H%M%S")
            echo "[INFO] Snapshot mode: timestamp ($SNAPSHOT_SUFFIX) — no shared fate"
        else
            SNAPSHOT_SUFFIX="latest"
            REMOVE_CODE_COPY_AFTER_JOB=false
            echo "[INFO] Snapshot mode: latest — shared fate with other live jobs"
        fi
        # Apply the suffix to all three tree paths (UWLab, rsl_rl, holosoma).
        # These modified paths are used for rsync targets here and re-injected
        # into .env.cluster inside submit_job so the in-container mounts agree.
        CLUSTER_UWLAB_DIR="${CLUSTER_UWLAB_DIR}_${SNAPSHOT_SUFFIX}"
        if [ -n "$CLUSTER_RSL_RL_DIR" ]; then
            CLUSTER_RSL_RL_DIR="${CLUSTER_RSL_RL_DIR}_${SNAPSHOT_SUFFIX}"
        fi
        if [ -n "$CLUSTER_HOLOSOMA_DIR" ]; then
            CLUSTER_HOLOSOMA_DIR="${CLUSTER_HOLOSOMA_DIR}_${SNAPSHOT_SUFFIX}"
        fi
        # Check if singularity image exists on the remote host
        check_singularity_image_exists uw-lab-$profile
        # make sure target directory exists
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_UWLAB_DIR"
        # Sync UW Lab code
        echo "[INFO] Syncing UW Lab code to $CLUSTER_UWLAB_DIR ..."
        rsync -ah --exclude="*.git*" --filter=':- .dockerignore' /$SCRIPT_DIR/../.. $CLUSTER_LOGIN:$CLUSTER_UWLAB_DIR
        # Sync rsl_rl if configured
        if [ -n "$LOCAL_RSL_RL_DIR" ] && [ -n "$CLUSTER_RSL_RL_DIR" ]; then
            echo "[INFO] Syncing rsl_rl code to $CLUSTER_RSL_RL_DIR ..."
            ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_RSL_RL_DIR"
            rsync -ah --exclude="*.git*" --exclude="__pycache__" --exclude="*.pyc" \
                --exclude="logs/" \
                "$LOCAL_RSL_RL_DIR/" "$CLUSTER_LOGIN:$CLUSTER_RSL_RL_DIR/"
        fi
        # Sync holosoma if configured
        if [ -n "$LOCAL_HOLOSOMA_DIR" ] && [ -n "$CLUSTER_HOLOSOMA_DIR" ]; then
            echo "[INFO] Syncing holosoma code to $CLUSTER_HOLOSOMA_DIR ..."
            ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_HOLOSOMA_DIR"
            rsync -ah --exclude="*.git*" --exclude="__pycache__" --exclude="*.pyc" \
                --exclude=".wandb" --exclude="logs/" \
                --exclude="src/holosoma_retargeting/" \
                --exclude="src/holosoma/holosoma/data/" \
                "$LOCAL_HOLOSOMA_DIR/" "$CLUSTER_LOGIN:$CLUSTER_HOLOSOMA_DIR/"
        fi
        # execute job script
        echo "[INFO] Executing job script..."
        submit_job $job_args
        ;;
    sync-assets)
        # source env file
        if [ -f "$SCRIPT_DIR/.env.$CLUSTER_NAME" ]; then
            source "$SCRIPT_DIR/.env.$CLUSTER_NAME"
        else
            echo "[ERROR] Environment file for cluster '$CLUSTER_NAME' not found at $SCRIPT_DIR/.env.$CLUSTER_NAME"
            exit 1
        fi
        if [ -z "$CLUSTER_CLOUD_ASSETS_DIR" ]; then
            echo "[ERROR] CLUSTER_CLOUD_ASSETS_DIR is not set in .env.$CLUSTER_NAME"
            exit 1
        fi
        LOCAL_SRC="${1:-$LOCAL_CLOUD_ASSETS_DIR}"
        if [ -z "$LOCAL_SRC" ] || [ ! -d "$LOCAL_SRC" ]; then
            echo "[ERROR] Provide a local assets directory to sync from."
            echo "Usage:  $(basename "$0") --cluster $CLUSTER_NAME sync-assets /path/to/local/cloud_assets"
            echo "   or:  set LOCAL_CLOUD_ASSETS_DIR in your environment"
            exit 1
        fi
        echo "[INFO] Syncing cloud assets to cluster..."
        echo "[INFO]   Source: $LOCAL_SRC"
        echo "[INFO]   Dest:   $CLUSTER_LOGIN:$CLUSTER_CLOUD_ASSETS_DIR"
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_CLOUD_ASSETS_DIR"
        rsync -ahP "$LOCAL_SRC/" "$CLUSTER_LOGIN:$CLUSTER_CLOUD_ASSETS_DIR/"
        echo "[INFO] Asset sync complete."
        ;;
    *)
        echo "Error: Invalid command: $command" >&2
        help
        exit 1
        ;;
esac
