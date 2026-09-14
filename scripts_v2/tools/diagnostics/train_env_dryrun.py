"""Dry-run the training env with random actions and report term statistics."""
import argparse, torch
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--finite_limits", type=float, default=None, help="if set, give the 4 inf-limit gripper joints +-DEG limits at spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, os, torch
_IFK = 1 if os.environ.get('UWLAB_ROBOT_ASSETS_DIR') else -1  # swapped joints in the patched USD read +q
import isaaclab_tasks, uwlab_tasks  # noqa
from uwlab_tasks.utils.hydra import hydra_task_compose
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as u
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import find_matching_prims

INF_JOINTS = ["right_inner_knuckle_joint", "left_inner_knuckle_joint", "right_inner_finger_knuckle_joint", "left_inner_finger_knuckle_joint"]

def spawn_with_finite_limits(prim_path, cfg, translation=None, orientation=None, **kw):
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kw)
    n = 0
    for j in INF_JOINTS:
        for jp in find_matching_prims(f"{prim_path}/{j}"):
            jp.GetAttribute("physics:lowerLimit").Set(-args_cli.finite_limits)
            jp.GetAttribute("physics:upperLimit").Set(args_cli.finite_limits)
            n += 1
    print(f"[finite_limits] set +-{args_cli.finite_limits} deg on {n} joint prims")
    return prim

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.finite_limits is not None:
        env_cfg.scene.robot.spawn.func = spawn_with_finite_limits
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    jn = list(robot.joint_names)
    vl = robot.data.joint_vel_limits.torch[0]
    pl = robot.data.joint_pos_limits.torch[0]
    print("\n=== robot joints: vel limit / pos limits ===")
    for i, n in enumerate(jn):
        print(f"  {n:34s} vel_lim={vl[i].item():10.3f}  pos_lim=({pl[i,0].item():+.3f},{pl[i,1].item():+.3f})")
    pc = env.reward_manager.get_term_cfg("progress_context").func
    print("\nassembled_offset quats after metadata reorder: insertive", pc.insertive_asset_offset.quat, "receptive", pc.receptive_asset_offset.quat)
    print("success thresholds:", env.command_manager.get_term("task_command").success_position_threshold, env.command_manager.get_term("task_command").success_orientation_threshold)
    act_names = env.action_manager.active_terms
    print("action terms:", act_names, "dim", env.action_manager.total_action_dim)
    g = env.action_manager.get_term("gripper")
    print("gripper open/close:", g._open_command.tolist(), g._close_command.tolist())
    print("obs dims:", {k: tuple(v.shape) for k, v in env.observation_manager.compute().items()})

    term_names = env.termination_manager.active_terms
    rew_names = env.reward_manager.active_terms
    term_counts = {n: 0 for n in term_names}
    rew_sums = {n: 0.0 for n in rew_names}
    abn_steps = 0
    qmin = torch.full((len(jn),), 1e9, device=env.device); qmax = torch.full((len(jn),), -1e9, device=env.device)
    jv_max = 0.0
    fi = jn.index("finger_joint")
    mimic = {"right_outer_knuckle_joint": 1, "left_inner_knuckle_joint": 1, "right_inner_knuckle_joint": -1, "left_inner_finger_knuckle_joint": _IFK, "right_inner_finger_knuckle_joint": _IFK}
    mimic_idx = [(jn.index(k), v) for k, v in mimic.items()]
    broken_steps = 0.0
    broken_ep = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.device)
    ep_total = 0; ep_broken = 0
    for t in range(args_cli.steps):
        a = torch.randn(env.action_space.shape, device=env.device).clamp(-1, 1)
        obs, rew, term, trunc, extras = env.step(a)
        for n in term_names:
            term_counts[n] += env.termination_manager.get_term(n).sum().item()
        for n in rew_names:
            rew_sums[n] += env.reward_manager._episode_sums[n].mean().item() if False else env.reward_manager._step_reward[:, rew_names.index(n)].mean().item()
        abn = (robot.data.joint_vel.torch.abs() > robot.data.joint_vel_limits.torch * 2).any(dim=1)
        abn_steps += abn.float().mean().item()
        q = robot.data.joint_pos.torch
        qmin = torch.minimum(qmin, q.min(0).values); qmax = torch.maximum(qmax, q.max(0).values)
        jv_max = max(jv_max, robot.data.joint_vel.torch.abs().max().item())
        dev = torch.stack([(q[:, i] - sgn * q[:, fi]).abs() for i, sgn in mimic_idx], dim=1).max(dim=1).values
        broken = dev > 0.15
        broken_steps += broken.float().mean().item()
        broken_ep |= broken
        done = term | trunc
        ep_total += done.sum().item(); ep_broken += (broken_ep & done).sum().item(); broken_ep[done] = False
        if not torch.isfinite(rew).all():
            print("NON-FINITE REWARD at step", t)
    N = args_cli.steps * args_cli.num_envs
    print(f"\n=== {args_cli.steps} random-action steps x {args_cli.num_envs} envs ===")
    print("terminations (count over env-steps):", {k: int(v) for k, v in term_counts.items()}, f"of {N}")
    print("mean per-step reward by term:", {k: round(v / args_cli.steps, 5) for k, v in rew_sums.items()})
    print(f"abnormal_robot fraction of env-steps: {abn_steps / args_cli.steps:.4f}   max |joint_vel| seen: {jv_max:.2f}")
    print("joint pos range seen:", {n: (round(qmin[i].item(), 2), round(qmax[i].item(), 2)) for i, n in enumerate(jn)})
    print("success monitor rate:", pc.success_monitor.get_success_rate().tolist())
    print(f"LINKAGE: broken on {100*broken_steps/args_cli.steps:.2f}% of env-steps; {ep_broken}/{ep_total} completed episodes had a break; currently broken: {broken.float().mean().item()*100:.1f}% of envs")
    env.close()
main(); app.close()
