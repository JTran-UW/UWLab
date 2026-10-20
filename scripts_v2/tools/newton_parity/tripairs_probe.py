# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Peak Newton triangle-pair usage vs the allocated budget, with a policy driving the task."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Triangle-pair budget probe.")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--hole_sdf", type=int, default=None, help="Newton SDF max resolution for the hole collision mesh.")
parser.add_argument(
    "--peg_sdf", type=int, default=None, help="Collide the peg as an SDF mesh with this max resolution."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if not any(a.startswith("presets=") for a in remaining_args):
    remaining_args.append("presets=newton_mjwarp")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import subprocess
import torch

import warp as wp
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_newton.physics import NewtonManager
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from uwlab_tasks.utils.hydra import hydra_task_compose

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "reinforcement_learning", "rsl_rl")
)
import cli_args  # noqa: E402  isort: skip


@hydra_task_compose(args_cli.task, args_cli.agent, hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    from isaaclab_newton.sim.schemas import NewtonSDFCollisionPropertiesCfg

    if args_cli.hole_sdf:
        env_cfg.scene.receptive_object.spawn.collision_props = NewtonSDFCollisionPropertiesCfg(
            sdf_max_resolution=args_cli.hole_sdf
        )
    if args_cli.peg_sdf:
        peg_spawn = env_cfg.scene.insertive_object.spawn
        if hasattr(peg_spawn, "collision_approximation"):
            peg_spawn.collision_approximation = "sdf"
        peg_spawn.collision_props = NewtonSDFCollisionPropertiesCfg(sdf_max_resolution=args_cli.peg_sdf)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    narrow = NewtonManager._collision_pipeline.narrow_phase
    model = NewtonManager._model
    sidx = model._shape_sdf_index.numpy() if getattr(model, "_shape_sdf_index", None) is not None else None
    n_sdf = int((sidx >= 0).sum()) if sidx is not None else 0
    print(f"[tripairs] shapes with SDF: {n_sdf} of {model.shape_count} ({n_sdf / max(1, args_cli.num_envs):.1f}/env)")
    budget = narrow.max_triangle_pairs
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    def used_bytes():
        out = subprocess.run(
            ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        return int(out.strip()) * 2**20

    peak, peak_mem = 0, 0
    obs = env.get_observations()
    with torch.inference_mode():
        for k in range(args_cli.steps):
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            peak = max(peak, int(wp.to_torch(narrow.triangle_pairs_count).max()))
            if k % 20 == 0:
                peak_mem = max(peak_mem, used_bytes())
    n = args_cli.num_envs
    print(
        f"[tripairs] envs {n}: budget {budget:,} ({budget / n:,.0f}/env), peak {peak:,} ({peak / n:,.0f}/env, "
        f"{peak / budget:.0%} of budget), overflow {peak > budget}, peak GPU mem {peak_mem / 2**30:.1f} GiB"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
