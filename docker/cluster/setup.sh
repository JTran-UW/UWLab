#!/usr/bin/env bash

# UWLab Cluster Interactive Setup Script
# This script helps users configure their .env.user file.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
USER_ENV_FILE="$SCRIPT_DIR/.env.user"
TEMPLATE_FILE="$SCRIPT_DIR/.env.user.template"

read_with_default() {
    local label="$1"
    local default="$2"
    local var_name="$3"
    local val

    if [ -n "$default" ]; then
        read -p "$label [$default]: " val
    else
        read -p "$label: " val
    fi

    printf -v "$var_name" "%s" "${val:-$default}"
}

echo "----------------------------------------------------------------"
echo "        UWLab Cluster Setup Wizard"
echo "----------------------------------------------------------------"

# Load existing values if they exist
if [ -f "$USER_ENV_FILE" ]; then
    echo "[INFO] Loading existing configuration from .env.user"
    set +e
    source "$USER_ENV_FILE"
    set -e
fi

echo "This wizard will help you set up your cluster environment variables."
echo "Press ENTER to accept the [default] value."
echo ""

current_netid="${CLUSTER_USER:-}"
[[ "$current_netid" == "<your-uw-netid>" ]] && current_netid=""

current_hyak_acc="${HYAK_ACCOUNT:-}"
[[ "$current_hyak_acc" == "<your-lab-group>" ]] && current_hyak_acc=""

read_with_default "UW NetID" "$current_netid" netid
if [ -z "$netid" ]; then echo "[ERROR] NetID cannot be empty."; exit 1; fi

read_with_default "Hyak Lab Group / Slurm Account" "$current_hyak_acc" hyak_account
if [ -z "$hyak_account" ]; then echo "[ERROR] Lab Group cannot be empty."; exit 1; fi

hyak_partition="${HYAK_PARTITION:-gpu-a40}"
hyak_cpus="${HYAK_CPUS_PER_TASK:-6}"
hyak_mem="${HYAK_MEM_PER_GPU:-60G}"
hyak_constraint="${HYAK_CONSTRAINT:-h200|l40|l40s|a40}"
hyak_constraint="${hyak_constraint%\"}"
hyak_constraint="${hyak_constraint#\"}"
hyak_time="${HYAK_TIME:-24:00:00}"
tillicum_time="${TILLICUM_TIME:-24:00:00}"
tillicum_qos="${TILLICUM_QOS:-normal}"

echo ""
read -p "Do you want to customize SLURM defaults (partitions, limits)? (y/N) [n]: " customize
if [[ "$customize" =~ ^[Yy]$ ]]; then
    echo ""
    echo "--- Hyak Customization ---"
    read_with_default "  Hyak Partition" "$hyak_partition" hyak_partition
    read_with_default "  Hyak CPUs per Task" "$hyak_cpus" hyak_cpus
    read_with_default "  Hyak Memory per GPU" "$hyak_mem" hyak_mem
    read_with_default "  Hyak Constraint" "$hyak_constraint" hyak_constraint
    read_with_default "  Hyak Time Limit" "$hyak_time" hyak_time

    echo ""
    echo "--- Tillicum Customization ---"
    read_with_default "  Tillicum Time Limit" "$tillicum_time" tillicum_time
    read_with_default "  Tillicum QOS" "$tillicum_qos" tillicum_qos
fi

echo ""
echo "[INFO] Generating .env.user..."
cp "$TEMPLATE_FILE" "$USER_ENV_FILE"

sed -i "s%<your-uw-netid>%${netid}%g" "$USER_ENV_FILE"
sed -i "s%<your-lab-group>%${hyak_account}%g" "$USER_ENV_FILE"

sed -i "s%^HYAK_PARTITION=.*%HYAK_PARTITION=${hyak_partition}                # Default: gpu-a40%g" "$USER_ENV_FILE"
sed -i "s%^HYAK_CPUS_PER_TASK=.*%HYAK_CPUS_PER_TASK=${hyak_cpus}                  # Default: 6%g" "$USER_ENV_FILE"
sed -i "s%^HYAK_MEM_PER_GPU=.*%HYAK_MEM_PER_GPU=${hyak_mem}                  # Default: 60G%g" "$USER_ENV_FILE"

if [[ "$hyak_constraint" != \"*\" ]]; then
    hyak_constraint_quoted="\"$hyak_constraint\""
else
    hyak_constraint_quoted="$hyak_constraint"
fi
sed -i "s%^HYAK_CONSTRAINT=.*%HYAK_CONSTRAINT=${hyak_constraint_quoted} # Default: \"h200|l40|l40s|a40\"%g" "$USER_ENV_FILE"

sed -i "s%^HYAK_TIME=.*%HYAK_TIME=${hyak_time}                    # Default: 24:00:00%g" "$USER_ENV_FILE"
sed -i "s%^TILLICUM_TIME=.*%TILLICUM_TIME=${tillicum_time}                # Default: 24:00:00%g" "$USER_ENV_FILE"
sed -i "s%^TILLICUM_QOS=.*%TILLICUM_QOS=${tillicum_qos}                   # Default: normal%g" "$USER_ENV_FILE"

source "$USER_ENV_FILE"

echo "----------------------------------------------------------------"
echo "[SUCCESS] Configuration complete!"
echo ""
echo "Summary of your configuration in .env.user:"
echo "  NetID:            $CLUSTER_USER"
echo "  Hyak Account:     $HYAK_ACCOUNT"
echo ""
echo "  Hyak Paths:"
echo "    Main Dir:       $HYAK_DIR"
echo "    Cache Dir:      $HYAK_CACHE_DIR"
echo "    SIF Path:       $HYAK_SIF_PATH"
echo ""
echo "  Tillicum Paths:"
echo "    Main Dir:       $TILLICUM_DIR"
echo "    Cache Dir:      $TILLICUM_CACHE_DIR"
echo "    SIF Path:       $TILLICUM_SIF_PATH"
echo ""
echo "  Hyak SLURM Defaults:"
echo "    Partition:      $HYAK_PARTITION"
echo "    CPUs:           $HYAK_CPUS_PER_TASK"
echo "    Memory:         $HYAK_MEM_PER_GPU"
echo "    Constraint:     $HYAK_CONSTRAINT"
echo "    Time Limit:     $HYAK_TIME"
echo ""
echo "  Tillicum SLURM Defaults:"
echo "    Time Limit:     $TILLICUM_TIME"
echo "    QOS:            $TILLICUM_QOS"
echo ""
echo "The required directories on the cluster will be created automatically"
echo "the first time you run a 'push' or 'job' command."
echo "----------------------------------------------------------------"
