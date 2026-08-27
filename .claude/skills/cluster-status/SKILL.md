---
name: cluster-status
description: Check the status of running SLURM jobs and resource usage on Hyak (klone) and Tillicum
argument-hint: "[--storage] [--cluster klone|tillicum|all]"
allowed-tools: Bash(ssh *)
---

# Cluster Status

Check the status of the user's training jobs and cluster resource usage.

## Arguments

- `--storage`: also report storage/quota usage
- `--cluster klone|tillicum|all`: which cluster to query (default: `all`)

## Cluster Details

The SLURM username on BOTH clusters is **`profjat`** (NOT `jtran`, which is only the
local workstation user and the directory name on klone's gscratch). `squeue -u jtran`
fails with `Invalid user: jtran`. Prefer `-u $(whoami)` inside the ssh command.

### Hyak (Klone)
- SSH: `ssh klone-login`
- Account: `weirdlab`; partitions: `ckpt` (preemptible), `gpu-l40s`, etc.
- Resource availability: `hyakalloc`
- Storage: `hyakstorage`
- Logs: `/mmfs1/gscratch/scrubbed/jtran/slurm_logs`
- Runs: `/mmfs1/gscratch/weirdlab/jtran/uwlab_latest/logs/...`

### Tillicum
- SSH: `ssh tillicum`
- Account: `weirdlab`; partition: `gpu-h200` (22 nodes x 8x H200 140GB, 64 CPU, ~2TB RAM)
- No preemptible tier; `MaxTime=UNLIMITED`
- **Billed by GPU-hour** — surface the cost estimate when jobs are large
- Storage: `hyakstorage`; logs `/gpfs/scrubbed/profjat/slurm_logs`
- Runs: `/gpfs/scrubbed/profjat/uwlab_latest/logs/...`

## Instructions

1. Parse `$ARGUMENTS` for `--storage` and `--cluster`.
2. For each selected cluster run:
   `ssh <alias> 'squeue -u $(whoami) -o "%.10i %.45j %.9T %.10M %.12P %R"'`
3. Availability: `hyakalloc` on klone; on tillicum use
   `sinfo -p gpu-h200 -o "%.10P %.6a %.6D %.20G %.10T"` plus
   `squeue -p gpu-h200 -h -t R -o %b | wc -l` for GPUs in use.
4. If `--storage`, run `hyakstorage` on the selected clusters.
5. Present a concise summary: running/pending counts, available GPUs, storage if
   requested, and warnings (failed jobs, full queues, quotas near limit).

Note when a job's state is `COMPLETED` with a tiny elapsed time — on these clusters
that usually means the container's python died, not that the run succeeded. Check the
`.err` file (loguru writes to stderr) rather than trusting the SLURM state.

Keep output concise. Use a table for job listings if there are many jobs.
