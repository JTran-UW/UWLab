#!/usr/bin/env bash
# Cancel SLURM jobs without letting them requeue.
# Usage: ./kill_jobs.sh <jobid> [jobid ...]

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <jobid> [jobid ...]" >&2
    exit 1
fi

for jobid in "$@"; do
    echo "==> Disabling requeue and cancelling job $jobid"
    scontrol update JobId="$jobid" Requeue=0 || echo "    (scontrol update failed for $jobid, continuing)"
    scancel "$jobid" || echo "    (scancel failed for $jobid, continuing)"
done

echo "Done."
