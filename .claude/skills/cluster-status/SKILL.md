---
name: cluster-status
description: Check the status of running SLURM jobs and resource usage on Hyak
argument-hint: "[--storage]"
allowed-tools: Bash(ssh *)
---

# Cluster Status

Check the status of the user's training jobs and cluster resource usage on Hyak (Klone).

## Arguments

- `--storage` flag: also check storage usage via `hyakstorage`

## Cluster Details

### Hyak (Klone)
- SSH: `ssh klone-login`
- User: `jtran`
- Resource availability: `hyakalloc`

## Instructions

1. Parse `$ARGUMENTS` for the `--storage` flag.
2. Run `ssh klone-login 'squeue -u jtran'` to get running/pending jobs.
3. Run `ssh klone-login 'hyakalloc'` to show available GPU resources.
4. If `--storage` is specified, run `ssh klone-login 'hyakstorage'`.
5. Present a concise summary:
   - Number of running/pending jobs
   - Available GPUs by partition
   - Storage status if requested
   - Any warnings (failed jobs, full queues)

Keep output concise. Use a table for job listings if there are many jobs.
