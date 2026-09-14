# Runtime patches for the IsaacLab 3.0 stack

These are applied at runtime instead of being committed upstream / baked into the image,
so the fix is reproducible from this tree alone.

## `apply_rsl_rl_gsde_fix.py` — rsl_rl gSDE forward drops activations

Root cause of the 3.0 OmniReset t0/t1 plateau (see `ISAACLAB_3_GRASP_HANDOFF.md` §10).

* **Local env (`env_isaaclab3`)**: run once per (re)install:
  `python docker/cluster/patches/apply_rsl_rl_gsde_fix.py $(python -c "import rsl_rl,os;print(os.path.dirname(rsl_rl.__file__))")`
* **Cluster jobs**: every `docker/cluster/submit_*omnireset*.sh` calls `ensure_rsl_rl_patched`,
  which copies rsl_rl out of the `.sif` into `$UWLAB_DIR/rsl_rl_patched/` (once), applies
  this script, and prepends it to `PYTHONPATH` inside the container. The job aborts if the
  in-container tripwire does not report `gsde fix : True`.
* Checkpoints trained without the fix are 1-hidden-layer policies; do not resume from them.
