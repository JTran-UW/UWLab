# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Track the object through the grasp-sampling loop, step by step.

Builds the same env as ``record_grasps.py`` (hydra overrides such as
``env.scene.object=peg`` work), drives it with the same constant ``close``
action, and prints per-step statistics on where the object is and what the
gripper joints are doing. No recorder, no dataset output.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", type=str, default="OmniReset-Robotiq2f85-GraspSampling-v0")
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--action", type=float, default=-1.0, help="binary gripper action; sampler uses -1")
parser.add_argument("--every", type=int, default=2, help="print every N steps")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402

GRIPPER_JOINTS = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    obj = env.scene["object"]
    grip = env.scene["robot"]
    jn = list(grip.joint_names)
    jidx = {n: jn.index(n) for n in GRIPPER_JOINTS if n in jn}
    bn = list(grip.body_names)
    pads = [i for i, b in enumerate(bn) if "inner_finger" in b and "knuckle" not in b]
    print(f"\njoints: {jn}")
    print(f"bodies: {bn}")
    print(f"pad bodies: {[bn[i] for i in pads]}")
    lim = grip.data.joint_pos_limits.torch[0]
    for n, i in jidx.items():
        print(f"  {n:34s} limits=({lim[i,0].item():+.3f},{lim[i,1].item():+.3f})")
    print(f"masses: {dict(zip(bn, [round(v,4) for v in grip.data.default_mass[0].tolist()]))}")
    act_term = env.action_manager.get_term("gripper")
    print(f"action term: {type(act_term).__name__} open={act_term._open_command.tolist()} close={act_term._close_command.tolist()}")

    actions = torch.full(env.action_space.shape, args_cli.action, device=env.device)
    steps_per_ep = int(env.max_episode_length)
    print(f"max_episode_length={steps_per_ep} steps, step_dt={env.step_dt}\n")

    hdr = (f"{'ep':>2} {'t':>3} {'grav':>4} | {'peg_z mean':>10} {'min':>7} {'drop%':>6} {'dev%':>5} {'|v|':>6} | "
           f"{'fing_q':>7} {'q_tgt':>6} {'q_std':>6} {'pad_gap':>7} | {'pas_q(mean of 5)':>16} {'|jv|max':>8} {'abn%':>5}")
    print(hdr)
    for ep in range(args_cli.episodes):
        p0 = obj.data.root_pos_w.torch.clone()
        for t in range(steps_per_ep):
            _, _, term, trunc, extras = env.step(actions)
            p = obj.data.root_pos_w.torch
            dz = p[:, 2] - p0[:, 2]
            dev = (p - p0).norm(dim=1)
            q = grip.data.joint_pos.torch
            qv = grip.data.joint_vel.torch
            tgt = grip.data.joint_pos_target.torch
            fi = jidx["finger_joint"]
            pas = [jidx[n] for n in GRIPPER_JOINTS[1:] if n in jidx]
            gap = (grip.data.body_pos_w.torch[:, pads[0]] - grip.data.body_pos_w.torch[:, pads[1]]).norm(dim=1) if len(pads) == 2 else torch.zeros(1)
            abn = (qv.abs() > grip.data.joint_vel_limits.torch * 2).any(dim=1).float().mean().item() * 100
            grav = env.event_manager.get_term_cfg("global_physics_control_event").func.gravity_enabled
            if t % args_cli.every == 0 or t == steps_per_ep - 1:
                print(f"{ep:>2} {t:>3} {'on' if grav else 'off':>4} | {p[:,2].mean().item():>10.4f} {p[:,2].min().item():>7.3f} "
                      f"{(dz < -0.05).float().mean().item()*100:>6.1f} {(dev > 0.05).float().mean().item()*100:>5.1f} "
                      f"{obj.data.root_lin_vel_w.torch.norm(dim=1).mean().item():>6.3f} | "
                      f"{q[:,fi].mean().item():>7.4f} {tgt[:,fi].mean().item():>6.3f} {q[:,fi].std().item():>6.3f} {gap.mean().item():>7.4f} | "
                      f"{' '.join(f'{q[:,i].mean().item():+.2f}' for i in pas):>16} {qv.abs().max().item():>8.2f} {abn:>5.1f}")
            if (term | trunc).any():
                n_done = (term | trunc).sum().item()
                succ = env.termination_manager.get_term("success").float().mean().item() * 100
                print(f"   -> {n_done} envs done at t={t}; success term true on {succ:.1f}%")
                p0 = obj.data.root_pos_w.torch.clone()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
