# Get Checkpoint

Retrieve checkpoint paths for a training run on Hyak.

## Usage

`/get-checkpoint <run-name-or-partial-timestamp>`

## Instructions

1. SSH to klone-login and search for a matching run directory:
   ```
   ssh klone-login 'find /mmfs1/gscratch/scrubbed/jtran/uwlab/logs/rsl_rl -maxdepth 2 -type d -name "*<ARG>*"'
   ```
2. If multiple directories match, list them all and ask the user to clarify.
3. List the checkpoints in the matched directory:
   ```
   ssh klone-login 'ls <dir>'
   ```
4. Report:
   - Full directory path
   - All checkpoint files (model_*.pt)
   - Full path to the **latest** checkpoint (highest iteration number)
