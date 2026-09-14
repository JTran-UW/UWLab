"""Identical grasped states; gripper closed; measure peg rotation within the wrist frame under (a) hold, (b) +x push."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
parser.add_argument("--dataset_dir", default="./Datasets/OmniReset")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=20)
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
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    p = env_cfg.events.reset_from_reset_states.params
    p["dataset_dir"] = args_cli.dataset_dir; p["reset_types"] = ["ObjectAnywhereEEGrasped"]; p["probs"] = [1.0]
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import sample_from_nested_dict
    term = env.event_manager.get_term_cfg("reset_from_reset_states").func
    robot, peg = env.scene["robot"], env.scene["insertive_object"]; bn = list(robot.body_names); w3 = bn.index("wrist_3_link")
    def peg_in_wrist():
        _, q = mu.subtract_frame_transforms(T(robot.data.body_pos_w)[:, w3], T(robot.data.body_quat_w)[:, w3], T(peg.data.root_pos_w), T(peg.data.root_quat_w)); return q
    def run(tag, ax):
        env.reset()
        ids = torch.arange(env.num_envs, device=env.device)
        term._reset_to(sample_from_nested_dict(term.datasets[0], ids)["initial_state"], env_ids=ids, is_relative=True)
        for at in env.action_manager._terms.values(): at.reset(None)
        env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(env.physics_dt)
        q0 = peg_in_wrist().clone(); d0 = (T(peg.data.root_pos_w) - T(robot.data.body_pos_w)[:, w3]).norm(dim=1)
        act = torch.zeros(env.action_space.shape, device=env.device); act[:, 0] = ax; act[:, -1] = -1.0
        for t in range(args_cli.steps): env.step(act)
        q1 = peg_in_wrist(); ang = mu.axis_angle_from_quat(mu.quat_mul(q1, mu.quat_inv(q0))).norm(dim=1)
        d1 = (T(peg.data.root_pos_w) - T(robot.data.body_pos_w)[:, w3]).norm(dim=1); held = ((d1 - d0).abs() < 0.02)
        deg = ang * 57.3
        print(f"[{tag}] peg rotation in wrist after {args_cli.steps} steps: p50={deg.median():.1f} deg p90={deg.quantile(0.9):.1f} max={deg.max():.1f} | >10deg: {(deg>10).float().mean()*100:.0f}% | still held: {held.float().mean()*100:.0f}%")
    run("hold", 0.0); run("push +x", 1.0)
    env.close()
main(); app.close()
