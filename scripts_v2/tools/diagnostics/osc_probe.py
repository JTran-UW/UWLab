"""OSC step-response and EE-observation consistency. Portable across 2.x / 3.0."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--dataset_dir", default=None)
parser.add_argument("--steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch
import isaaclab_tasks, uwlab_tasks  # noqa
import isaaclab.utils.math as mu
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose
def T(x): return x.torch if hasattr(x, "torch") else x

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    p = env_cfg.events.reset_from_reset_states.params
    if args_cli.dataset_dir: p["dataset_dir"] = args_cli.dataset_dir
    p["reset_types"] = ["ObjectAnywhereEEAnywhere"]; p["probs"] = [1.0]
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    robot = env.scene["robot"]; bn = list(robot.body_names); w3 = bn.index("wrist_3_link"); base = bn.index("robotiq_base_link")
    def ee():
        pos = T(robot.data.body_link_pos_w)[:, w3] if hasattr(robot.data, "body_link_pos_w") else T(robot.data.body_pos_w)[:, w3]
        quat = T(robot.data.body_link_quat_w)[:, w3] if hasattr(robot.data, "body_link_quat_w") else T(robot.data.body_quat_w)[:, w3]
        rp, rq = T(robot.data.root_link_pos_w) if hasattr(robot.data, "root_link_pos_w") else T(robot.data.root_pos_w), T(robot.data.root_link_quat_w) if hasattr(robot.data, "root_link_quat_w") else T(robot.data.root_quat_w)
        pb, qb = mu.subtract_frame_transforms(rp, rq, pos, quat)
        return pos.clone(), quat.clone(), pb.clone(), qb.clone()
    # EE obs consistency: compare the policy obs term to our own root-frame computation
    obs = env.observation_manager.compute()["policy"]
    print(f"\nrobot root quat (env0): {[round(v,3) for v in T(robot.data.root_quat_w)[0].tolist()]}   base_link body quat: {[round(v,3) for v in T(robot.data.body_quat_w)[0, bn.index(bn[0])].tolist()]}")
    p0w, q0w, p0b, q0b = ee()
    print(f"EE (wrist_3) in root frame at reset, env0: pos={[round(v,3) for v in p0b[0].tolist()]}  |pos| mean={p0b.norm(dim=1).mean():.3f}")
    print(f"robotiq_base_link - wrist_3 offset (world, env0): {[round(v,4) for v in (T(robot.data.body_pos_w)[0, base]-T(robot.data.body_pos_w)[0, w3]).tolist()]}")
    N = env.num_envs
    tests = {"+x": (0, 1.0), "+y": (1, 1.0), "+z": (2, 1.0), "-z": (2, -1.0), "+rx": (3, 1.0), "+rz": (5, 1.0)}
    print(f"\n=== OSC step response: {args_cli.steps} steps of unit action (scale 0.02 m / 0.02 rad, rz 0.2 rad per step) ===")
    print(f"{'cmd':>4} {'expected':>9} {'d(root-frame pos) mean xyz':>32} {'|d| mean':>9} {'d(rot) axis-angle mean':>26} {'per-env std':>11}")
    for name, (dim, val) in tests.items():
        env.reset()
        p0w, q0w, p0b, q0b = ee()
        a = torch.zeros(env.action_space.shape, device=env.device); a[:, dim] = val; a[:, -1] = -1.0
        for _ in range(args_cli.steps): env.step(a)
        p1w, q1w, p1b, q1b = ee()
        dp = p1b - p0b
        dq = mu.quat_mul(q1b, mu.quat_inv(q0b)); daa = mu.axis_angle_from_quat(dq)
        scale = env_cfg.actions.arm.scale_xyz_axisangle[dim]
        print(f"{name:>4} {val*scale*args_cli.steps:>9.3f} {str([round(v,3) for v in dp.mean(0).tolist()]):>32} {dp.norm(dim=1).mean():>9.3f} {str([round(v,3) for v in daa.mean(0).tolist()]):>26} {dp.norm(dim=1).std():>11.3f}")
    env.close()
main(); app.close()
