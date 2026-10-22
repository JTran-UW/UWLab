# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManiSkill-parity push-T probe: TCP at reset, holds under zero action, reward/intersection vs hand-computed
ManiSkill values, success on a teleported T, joint-delta targets, and a scripted (diff-IK) push."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="PushT-PandaStick-JointDeltaPos-State-v0")
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import uwlab_tasks  # noqa: E402, F401
from uwlab_tasks.manager_based.manipulation.push_t import mdp  # noqa: E402
from uwlab_tasks.manager_based.manipulation.push_t.mdp import maniskill  # noqa: E402
from uwlab_tasks.manager_based.manipulation.push_t.mdp.t_block import t_part_centers  # noqa: E402


def fmt(t, n=4):
    return [round(v, n) for v in t.tolist()]


def ms_quat_to_z_euler(q_wxyz):
    """ManiSkill PushTEnv.quat_to_z_euler, verbatim."""
    signs = torch.ones_like(q_wxyz[:, -1])
    signs[q_wxyz[:, -1] < 0] = -1.0
    return 2 * (q_wxyz[:, 0] * signs).acos()


def ms_reward(t_pos, t_quat_xyzw, g_pos, g_yaw, tcp_pos, success):
    """ManiSkill compute_normalized_dense_reward, written from their source on raw poses."""
    q = t_quat_xyzw[:, [3, 0, 1, 2]]
    rot = (ms_quat_to_z_euler(q) - g_yaw).cos()
    r = (((rot + 1) / 2) ** 2) / 2
    r += ((1 - torch.tanh(5 * torch.linalg.norm(t_pos[:, :2] - g_pos[:, :2], axis=1))) ** 2) / 2
    r += ((1 - torch.tanh(5 * torch.linalg.norm(t_pos - tcp_pos, axis=1))).sqrt()) / 20
    r[success] = 3
    return r / 3


def ms_pseudo_render(t_xy, t_yaw, g_xy, g_yaw, bar, stem, res=64, uv_half_width=0.15):
    """ManiSkill pseudo_render_intersection ported to our T boxes (poses as planar xy + yaw)."""
    dev = t_xy.device
    oned = torch.arange(res, dtype=torch.float32).view(1, res).repeat(res, 1) - (res / 2)
    uv = (torch.cat([oned.unsqueeze(0), (-1 * oned.T).unsqueeze(0)], dim=0) + 0.5) / ((res / 2) / uv_half_width)
    homo = torch.cat([uv, torch.ones_like(uv[0]).unsqueeze(0)], dim=0).to(dev)
    bar_y, stem_y = t_part_centers(bar, stem)

    def box(cy, w, h):
        return torch.tensor([[-w / 2, cy + h / 2], [w / 2, cy + h / 2], [-w / 2, cy - h / 2], [w / 2, cy - h / 2]])

    boxes = [box(bar_y, *bar), box(stem_y, *stem)]
    render = torch.zeros(res, res)
    for b in boxes:
        b = (b * ((res / 2) / uv_half_width) + res / 2).long()
        render.T[b[0, 0] : b[1, 0], b[2, 1] : b[0, 1]] = 1
    render = render.flip(0).to(dev)

    def se2(xy, yaw):
        m = torch.zeros(len(yaw), 3, 3, device=dev)
        m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = yaw.cos(), -yaw.sin(), yaw.sin(), yaw.cos()
        m[:, :2, 2], m[:, 2, 2] = xy, 1.0
        return m

    t2g = torch.linalg.inv(se2(g_xy, g_yaw)) @ se2(t_xy, t_yaw)
    n = len(t_yaw)
    pts = (t2g @ homo.view(3, -1)).view(n, 3, res, res)
    pts = pts[:, 0:2] / pts[:, -1].unsqueeze(1)
    coords = pts[:, :, render == 1].view(n, 2, -1)
    idx = (coords * ((res / 2) / uv_half_width) + (res / 2)).long().view(n, 2, -1)
    bi = torch.arange(n, device=dev).view(-1, 1).repeat(1, idx.shape[-1])
    bad = (idx[:, 0] < 0) | (idx[:, 0] >= res) | (idx[:, 1] < 0) | (idx[:, 1] >= res)
    idx[:, 0][bad] = 0
    idx[:, 1][bad] = 0
    out = torch.zeros(n, res, res, device=dev)
    out[bi, idx[:, 0], idx[:, 1]] = 1
    out = out.permute(0, 2, 1).flip(1)
    return (out.bool() & render.bool()).sum(dim=[-1, -2]).float() / render.bool().sum().float()


def teleport_t(env, xy, yaw):
    t = env.scene["t_block"]
    n = env.num_envs
    z = env.cfg.table_top_z + 0.5 * t.cfg.spawn.height + 5e-4
    pose = torch.zeros(n, 7, device=env.device)
    pose[:, :2] = xy + env.scene.env_origins[:, :2]
    pose[:, 2] = z + env.scene.env_origins[:, 2]
    pose[:, 5], pose[:, 6] = torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)
    t.write_root_pose_to_sim_index(root_pose=pose)
    t.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device))


def main():
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    n, dev = env.num_envs, env.device
    robot, origin = env.scene["robot"], env.scene.env_origins
    act_dim = env.action_manager.total_action_dim
    zero = torch.zeros(n, act_dim, device=dev)
    print(f"[probe] step_dt {env.step_dt} max_episode_length {env.max_episode_length} action dim {act_dim}")
    print(f"[probe] obs dims { {k: tuple(v.shape) for k, v in env.observation_manager.compute().items()} }")
    print(f"[probe] reward weight {env.reward_manager.get_term_cfg('maniskill_dense').weight}")
    print(f"[probe] bodies {robot.body_names}")
    print(f"[probe] joints {robot.joint_names}")
    print(f"[probe] body masses {fmt(robot.data.body_mass.torch[0].flatten())}")
    print(
        f"[probe] stiffness {fmt(robot.data.joint_stiffness.torch[0])} damping {fmt(robot.data.joint_damping.torch[0])}"
    )
    print(
        f"[probe] effort lim {fmt(robot.data.joint_effort_limits.torch[0])} "
        f"vel lim {fmt(robot.data.joint_vel_limits.torch[0])}"
    )
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/envs/env_0/Robot") and prim.HasAttribute("physxRigidBody:disableGravity"):
            print(
                f"[probe] {path} disableGravity={prim.GetAttribute('physxRigidBody:disableGravity').Get()} "
                f"instance_proxy={prim.IsInstanceProxy()}"
            )
    root_p, root_q = mdp.robot_root_pose_w(env)
    print(f"[probe] body0 {robot.body_names[0]} pos {fmt(root_p[0] - origin[0])} quat {fmt(root_q[0])}")

    # --- TCP at the noise-free ManiSkill keyframe (FK refreshed by one physics substep at zero velocity)
    q_def = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=q_def)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_def))
    robot.set_joint_position_target_index(target=q_def)
    robot.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    tcp_w, _ = mdp.ee_tip_pose_w(env)
    print(f"[probe] TCP rel base at keyframe (no noise) {fmt((tcp_w - root_p)[0])}; ManiSkill (0.2943, 0.2835, 0.0154)")
    env.reset()
    env.step(zero)

    # --- TCP at reset (+ N(0, 0.02) joint noise), after one zero-action step
    tcp_w, tcp_q = mdp.ee_tip_pose_w(env)
    tcp_b = tcp_w - root_p
    print(
        f"[probe] TCP rel base (reset, all envs mean) {fmt(tcp_b.mean(0))} std {fmt(tcp_b.std(0))}; "
        f"ManiSkill keyframe (0.2943, 0.2835, 0.0154)"
    )
    print(f"[probe] TCP height above table: {fmt(tcp_w[:, 2] - origin[:, 2] - env.cfg.table_top_z)}")
    t_pos, t_yaw, t_quat = mdp.t_pose(env)
    spawn = env.scene["t_block"].cfg.spawn
    grid = maniskill._t_grid(spawn.bar_size, spawn.stem_size, 0.0025, dev)
    # TCP distance to the nearest T point (planar) at reset
    c, s = torch.cos(t_yaw), torch.sin(t_yaw)
    gx = c[:, None] * grid[:, 0] - s[:, None] * grid[:, 1] + t_pos[:, None, 0]
    gy = s[:, None] * grid[:, 0] + c[:, None] * grid[:, 1] + t_pos[:, None, 1]
    tcp_e = tcp_w - origin
    dmin = torch.sqrt((gx - tcp_e[:, None, 0]) ** 2 + (gy - tcp_e[:, None, 1]) ** 2).min(dim=1).values
    print(f"[probe] reset: min planar TCP->T distance {dmin.min().item():.4f} m (stick radius 0.008)")

    # --- zero-action hold
    q0 = robot.data.joint_pos.torch.clone()
    t0_pos, t0_yaw, _ = mdp.t_pose(env)
    rew = None
    for i in range(50):
        _, rew, term, trunc, _ = env.step(zero)
        if i == 0:
            # reward at reset vs hand-computed ManiSkill value (poses after the step, i.e. what the reward saw)
            t_pos, t_yaw, t_quat = mdp.t_pose(env)
            g_pos, g_yaw = mdp.goal_pose(env)
            tcp_w, _ = mdp.ee_tip_pose_w(env)
            inter = maniskill.intersection_area(env)
            ref = ms_reward(t_pos, t_quat, g_pos, g_yaw, tcp_w - origin, inter >= 0.9)
            print(f"[probe] step1 reward env {fmt(rew, 6)}")
            print(
                f"[probe] step1 reward ManiSkill-by-hand {fmt(ref, 6)}; max |diff| {(rew - ref).abs().max().item():.2e}"
            )
            print(f"[probe] step1 intersection {fmt(inter)}")
    print(f"[probe] zero-action 50 steps: signed dq env0 {fmt((robot.data.joint_pos.torch - q0)[0], 5)}")
    dq = (robot.data.joint_pos.torch - q0).abs()
    p, y, _ = mdp.t_pose(env)
    print(
        f"[probe] zero-action 50 steps: max |dq| per joint {fmt(dq.max(0).values, 5)} rad; "
        f"max |qd| {robot.data.joint_vel.torch.abs().max().item():.5f}"
    )
    print(
        f"[probe] zero-action 50 steps: T drift max {(p - t0_pos).norm(dim=-1).max().item():.6f} m, "
        f"yaw {(y - t0_yaw).abs().max().item():.6f} rad; terminated {int(term.sum())} truncated {int(trunc.sum())}"
    )

    # --- joint delta target: +1 on joint k -> target = qpos(before step) + 0.1; 2.0 clamps to +0.1
    for k, a in ((3, 1.0), (5, -1.0), (1, 2.0)):
        act = zero.clone()
        act[:, k] = a
        qb = robot.data.joint_pos.torch.clone()
        env.step(act)
        tgt = robot.data.joint_pos_target.torch
        print(
            f"[probe] action[{k}]={a}: target-qpos_before joint {k}: {fmt(tgt[:, k] - qb[:, k], 5)}; "
            f"others max |.| {(tgt - qb).abs()[:, [j for j in range(7) if j != k]].max().item():.2e}; "
            f"qpos moved {fmt(robot.data.joint_pos.torch[:2, k] - qb[:2, k], 4)}"
        )

    # --- intersection cross-check vs ManiSkill pseudo-render on random poses
    g_pos, g_yaw = mdp.goal_pose(env)
    m = 4096
    gen = torch.Generator(device=dev).manual_seed(0)
    g_xy = g_pos[:1, :2].repeat(m, 1)
    gy_ = g_yaw[:1].repeat(m)
    t_xy = g_xy + (torch.rand(m, 2, device=dev, generator=gen) - 0.5) * 0.08
    t_y = gy_ + (torch.rand(m, device=dev, generator=gen) - 0.5) * 1.0
    ours = maniskill.t_overlap(t_xy, t_y, g_xy, gy_, grid, spawn.bar_size, spawn.stem_size)
    ms = ms_pseudo_render(t_xy, t_y, g_xy, gy_, spawn.bar_size, spawn.stem_size)
    err = (ours - ms).abs()
    agree = ((ours >= 0.9) == (ms >= 0.9)).float().mean().item()
    args_id = (g_xy[:1], gy_[:1], g_xy[:1], gy_[:1])
    ours_id = maniskill.t_overlap(*args_id, grid, spawn.bar_size, spawn.stem_size).item()
    ms_id = ms_pseudo_render(*args_id, spawn.bar_size, spawn.stem_size).item()
    print(
        f"[probe] overlap vs ManiSkill pseudo-render ({m} random poses): mean |diff| {err.mean().item():.4f} "
        f"max {err.max().item():.4f}; success-label agreement {agree:.4f}; ours@identity {ours_id:.4f} "
        f"ms@identity {ms_id:.4f}"
    )
    shifted = maniskill.t_overlap(
        g_xy[:1] + torch.tensor([[0.01, 0.0]], device=dev),
        gy_[:1],
        g_xy[:1],
        gy_[:1],
        grid,
        spawn.bar_size,
        spawn.stem_size,
    )
    print(f"[probe] overlap with T shifted 1 cm along goal-x: {shifted.item():.4f} (analytic 0.8571 for 16x4/4x12 T)")

    # --- teleport T onto the goal -> success terminates, reward = 1.0
    env.reset()
    env.step(zero)
    print(f"[probe] after reset: intersection {fmt(maniskill.intersection_area(env))}")
    teleport_t(env, g_pos[:, :2], g_yaw)
    _, rew, term, trunc, extras = env.step(zero)
    print(
        f"[probe] T teleported onto goal: reward {fmt(rew, 6)} terminated {term.tolist()} "
        f"success term {env.termination_manager.get_term('success').tolist()}"
    )
    print(f"[probe] episode log: { {k: round(v, 4) for k, v in extras['log'].items() if 'Metrics' in k} }")
    # partial overlap: 1 cm off along the goal's x -> ~0.857, no success
    env.step(zero)
    off = torch.stack([torch.cos(g_yaw), torch.sin(g_yaw)], -1) * 0.01
    teleport_t(env, g_pos[:, :2] + off, g_yaw)
    _, rew, term, _, _ = env.step(zero)
    print(
        f"[probe] T 1 cm off goal: intersection {fmt(maniskill.intersection_area(env))} reward {fmt(rew[:2], 4)} "
        f"terminated {int(term.sum())}"
    )

    # --- scripted push with damped-least-squares IK on the TCP position
    env.reset()
    env.step(zero)
    tcp_idx = robot.find_bodies(env.cfg.ee_body_name)[0][0]
    t_start, _, _ = mdp.t_pose(env)
    bar_y, _ = t_part_centers(spawn.bar_size, spawn.stem_size)
    top = t_start[:, 1] + bar_y + 0.5 * spawn.bar_size[1]
    push_z = env.cfg.table_top_z + 0.015
    waypoints = [
        torch.stack([t_start[:, 0], top + 0.03, torch.full_like(top, 0.03)], -1),
        torch.stack([t_start[:, 0], top + 0.03, torch.full_like(top, push_z)], -1),
        torch.stack([t_start[:, 0], top - 0.10, torch.full_like(top, push_z)], -1),
    ]
    # 1 + 30 + 15 + 45 = 91 steps, inside the 100-step episode
    for wp, steps in zip(waypoints, (30, 15, 45)):
        for _ in range(steps):
            jac = robot.data.body_link_jacobian_w.torch
            jl = jac[:, tcp_idx - (robot.num_bodies - jac.shape[1]), :, :7]
            tcp_w, _ = mdp.ee_tip_pose_w(env)
            # Jacobian of the stick tip: v_tip = v_hand - [r]x w, r = tip - hand origin
            r = tcp_w - robot.data.body_link_pos_w.torch[:, tcp_idx]
            rx = torch.zeros(n, 3, 3, device=dev)
            rx[:, 0, 1], rx[:, 0, 2], rx[:, 1, 2] = -r[:, 2], r[:, 1], -r[:, 0]
            rx = rx - rx.transpose(1, 2)
            j = jl[:, :3] - rx @ jl[:, 3:]
            err = wp - (tcp_w - origin)
            err = err * (0.02 / err.norm(dim=-1, keepdim=True).clamp_min(0.02))
            jjt = j @ j.transpose(1, 2) + 1e-4 * torch.eye(3, device=dev)
            dq = (j.transpose(1, 2) @ torch.linalg.solve(jjt, err.unsqueeze(-1))).squeeze(-1)
            _, rew, term, trunc, _ = env.step((dq / 0.1).clamp(-1, 1))
        tcp_w, _ = mdp.ee_tip_pose_w(env)
        print(f"[probe] push waypoint {fmt(wp[0])} reached TCP {fmt((tcp_w - origin)[0])}")
    p, y, _ = mdp.t_pose(env)
    print(
        f"[probe] scripted push: T moved {fmt((p - t_start)[:, 1])} m along y (env0 pos {fmt(p[0])}, "
        f"yaw {y[0].item():.3f}); reward {fmt(rew[:4], 4)}; terminated {int(term.sum())}"
    )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
