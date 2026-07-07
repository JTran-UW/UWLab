# Get Checkpoint

Download checkpoints from a Hyak training run into `./checkpoints/<run_name>/`.

## Usage

`/get-checkpoint <run-name-or-partial-timestamp>`

## Instructions

1. **Locate matching run dir(s) on Hyak.** Runs live under both migrated (weirdlab) and legacy (scrubbed) trees:
   ```
   ssh klone-login 'find \
     /mmfs1/gscratch/weirdlab/jtran/uwlab_latest/logs/rsl_rl \
     /mmfs1/gscratch/scrubbed/jtran/uwlab_latest/logs/rsl_rl \
     -maxdepth 3 -type d -name "*<ARG>*" 2>/dev/null'
   ```
   Use `-maxdepth 3` because the layout is `.../logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`.

2. **If zero matches** → tell the user and stop.

3. **If multiple matches**, list them with checkpoint counts and ask the user which one:
   ```
   ssh klone-login 'for d in <matched dirs>; do
     count=$(ls "$d"/model_*.pt 2>/dev/null | wc -l)
     echo "$d ($count checkpoints)"
   done'
   ```
   Do not proceed until the user picks one.

4. **List the checkpoints** in the chosen dir and figure out the run_name (strip the leading `<timestamp>_` off the dir name):
   ```
   ssh klone-login 'ls <chosen dir>/model_*.pt | xargs -n1 basename | sort'
   ```

5. **Confirm scope with the user** if the run has many checkpoints (e.g., >20). Ask whether they want all of them, the latest N, or specific iterations. Otherwise default to all.

6. **Create local destination** and rsync the selected `model_*.pt` files. Local destination is `./checkpoints/<run_name>/` **without** the timestamp prefix (matches the existing convention). Use rsync with a specific pattern so only checkpoint files come across, not wandb/params/tensorboard:
   ```
   mkdir -p ./checkpoints/<run_name>
   rsync -h --info=progress2 'klone-login:<chosen dir>/model_*.pt' ./checkpoints/<run_name>/
   ```
   For a subset (e.g., latest 9), pass the specific files:
   ```
   rsync -h 'klone-login:<chosen dir>/model_0011000.pt' 'klone-login:<chosen dir>/model_0012000.pt' … ./checkpoints/<run_name>/
   ```
   (Or on the remote, glob them into a temp list — either works, single-glob rsync is simpler when it's contiguous.)

7. **Verify and report**:
   - Local dir path
   - Count of files downloaded
   - Full path of the latest checkpoint locally (highest iteration number)
   - Note if the run dir also had `model_final.pt` — that's the end-of-training save from a clean exit, distinct from periodic saves.

## Notes

- If the user asks for "the latest checkpoints" (plural, no count), default to the N highest-step checkpoints where N ≈ 9. If they ask for "the latest checkpoint" (singular), just grab the highest-step one.
- If the same `run_name` appears in multiple timestamped dirs (retries, requeues), do NOT flatten them together — checkpoint filenames collide (`model_0001000.pt` in both). If the user wants both, put each under `./checkpoints/<run_name>/<timestamp>/`. If they only want one, ask which.
- The `<experiment_name>` layer between `rsl_rl/` and the timestamped run dir is set by the RSL-RL runner config (e.g., `ur5e_robotiq_2f85_omnireset_agent`); it's part of the search path but not part of the local destination.
- The search covers both weirdlab and scrubbed paths because different runs may live in either — the migration to weirdlab happened on 2026-07-06, older runs may still be on scrubbed if they weren't cleaned.
