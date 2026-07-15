"""Generate ScenePointCloud canonical-points cache file(s) for upload to HF.

Set ``UWLAB_GENERATE_SCENE_PC_CACHE=1`` in your env so ``ScenePointCloud.__init__``
samples fresh + saves to ``~/.cache/uwlab/scene_pc_cache/<key>.pt`` instead of
the default STRICT path (HF-only).

After generation, the script prints the ``huggingface-cli upload`` command to
publish — run it to make the canonical points available to every other machine.

Usage:
    UWLAB_GENERATE_SCENE_PC_CACHE=1 python scripts/generate_scene_pc_cache.py \\
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-WristSide-PCTeacher-v0 \\
        env.scene.insertive_object=peg env.scene.receptive_object=peghole
"""

from __future__ import annotations

import argparse
import os
import sys

# Set generate-mode env var BEFORE importing AppLauncher / uwlab_tasks so the
# ScenePointCloud term picks it up at observation manager init.
os.environ.setdefault("UWLAB_GENERATE_SCENE_PC_CACHE", "1")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, help="Gym task id (e.g. OmniReset-...-PCTeacher-v0)")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument(
    "overrides",
    nargs=argparse.REMAINDER,
    help="Hydra overrides (e.g. env.scene.insertive_object=peg env.scene.receptive_object=peghole)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import uwlab_tasks  # noqa: F401, E402  (registers tasks)


def _apply_overrides(env_cfg, overrides: list[str]) -> None:
    """Apply minimal Hydra-style overrides for the bits we need (peg variant)."""
    for ov in overrides:
        if "=" not in ov:
            continue
        key, value = ov.split("=", 1)
        # Only handle the variant overrides we use; everything else is ignored
        if key == "env.scene.insertive_object":
            v = getattr(env_cfg, "variants", {}).get("scene.insertive_object", {})
            if value in v:
                env_cfg.scene.insertive_object = v[value]
        elif key == "env.scene.receptive_object":
            v = getattr(env_cfg, "variants", {}).get("scene.receptive_object", {})
            if value in v:
                env_cfg.scene.receptive_object = v[value]
        else:
            print(f"[generate] ignored override: {ov} (only scene.insertive/receptive_object supported)")


def main() -> int:
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    _apply_overrides(env_cfg, args.overrides or [])

    print(f"[generate] Building env {args.task} with num_envs={args.num_envs}")
    print(f"[generate] UWLAB_GENERATE_SCENE_PC_CACHE={os.environ.get('UWLAB_GENERATE_SCENE_PC_CACHE')}")

    env = gym.make(args.task, cfg=env_cfg)
    # init alone is enough — ScenePointCloud samples + saves in __init__
    env.close()
    print("[generate] Done. Look for '[ScenePointCloud] cache SAVED to ...' line above for the upload command.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    app.close()
