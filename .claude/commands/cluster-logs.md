---
name: cluster-logs
description: Inspect SLURM stdout/stderr logs for training jobs on Hyak
argument-hint: "[job-id|latest|all] [--errors] [--tail N]"
disable-model-invocation: true
allowed-tools: Bash(ssh *)
---

# Cluster Logs

Read and summarize SLURM job logs from Hyak.

## Arguments

- Job identifier:
  - A SLURM job ID (e.g. `34263645`) -- looks for files matching `*<job-id>*` in the logs dir
  - `latest` -- reads the most recently modified log files
  - `all` -- lists all log files with timestamps so the user can pick
  - If omitted, default to `latest`
- `--errors` -- only show `.err` files, or filter for ERROR/WARN/Exception lines
- `--tail N` -- show last N lines (default: 80)

## Cluster Details

### Hyak
- SSH: `ssh klone-login`
- SLURM logs dir: `/mmfs1/gscratch/scrubbed/jtran/slurm_logs`

## Instructions

1. Parse `$ARGUMENTS` for job identifier and flags.
2. Based on the job identifier:
   - **Specific job ID**: `ssh klone-login 'ls -lt /mmfs1/gscratch/scrubbed/jtran/slurm_logs/*<job_id>*'` to find matching files, then read them.
   - **`latest`**: `ssh klone-login 'ls -t /mmfs1/gscratch/scrubbed/jtran/slurm_logs/ | head -4'` to find the most recent files, then read the `.out` and `.err` for that job.
   - **`all`**: `ssh klone-login 'ls -lt /mmfs1/gscratch/scrubbed/jtran/slurm_logs/ | head -40'` and present the listing.
3. Read log content with `ssh klone-login 'tail -n <N> <log_file>'` (default N=80).
4. If `--errors` is specified, filter with: `ssh klone-login 'grep -iE "error|exception|traceback|warn|fatal|killed|oom" <log_file> | tail -40'`
5. Summarize the log output:
   - Is the job still running or did it complete/fail?
   - Any errors, warnings, or OOM kills?
   - Last reported training iteration/step if visible
   - Any reward or loss values from the tail of the output
6. If both `.out` and `.err` exist for a job, check both. Often errors appear in `.err` while progress appears in `.out`.

Keep output concise. Show the raw log tail only if the user needs it or if there are issues to diagnose. Otherwise, summarize.
