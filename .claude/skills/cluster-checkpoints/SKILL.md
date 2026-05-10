---
name: cluster-checkpoints
description: List training checkpoints and experiment progress on Hyak
argument-hint: "[experiment-name] [--active] [--size]"
disable-model-invocation: true
allowed-tools: Bash(ssh *)
---

# Cluster Checkpoints

List checkpoints and assess training progress for experiments on Hyak.

## Arguments

- Experiment name: optional filter to look at a specific experiment directory
- `--active` -- only show experiments modified within last 24h (likely still training)
- `--size` -- show disk usage per experiment

## Cluster Details

### Hyak
- SSH: `ssh klone-login`
- Persistent logs: `/mmfs1/gscratch/scrubbed/jtran/uwlab/logs/rsl_rl`

Note: `cluster_interface.sh` timestamps each job's code directory (e.g. `uwlab_20260414_172200`),
so the actual logs base will be `/mmfs1/gscratch/scrubbed/jtran/uwlab_*/logs/rsl_rl`. When listing,
check both `/mmfs1/gscratch/scrubbed/jtran/uwlab/logs/rsl_rl` and timestamped variants.

## Instructions

1. Parse `$ARGUMENTS` for experiment filter and flags.
2. **List experiments**:
   - `ssh klone-login 'ls -lt /mmfs1/gscratch/scrubbed/jtran/uwlab*/logs/rsl_rl/ 2>/dev/null'` to list experiment directories across all code copies.
   - If an experiment name is provided, filter to just that one.
   - If `--active` is specified, filter: `ssh klone-login 'find /mmfs1/gscratch/scrubbed/jtran/uwlab*/logs/rsl_rl -maxdepth 2 -name "*.pt" -mmin -1440 -printf "%h\n" | sort -u'`
3. **For each experiment** (or the specified one), list the run directories and their checkpoints:
   - `ssh klone-login 'ls -lt <logs_base>/<experiment>/'` to show run directories
   - For each run: `ssh klone-login 'ls -lh <logs_base>/<experiment>/<run>/*.pt 2>/dev/null'` to list checkpoint files
4. **Assess progress**:
   - Checkpoint filenames typically encode the iteration number (e.g., `model_1000.pt`)
   - Report: experiment name, run ID, number of checkpoints, latest checkpoint iteration, last modified time
   - If `--size` is specified: `ssh klone-login 'du -sh <logs_base>/<experiment>/'`
5. **Summarize**:
   - Total number of experiments
   - Which are actively training (recently modified)
   - Latest checkpoint iteration for active experiments
   - Any experiments that appear stalled

If there are many experiments, present a table. For a single experiment, show more detail.
