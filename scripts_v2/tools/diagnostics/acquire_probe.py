"""Scripted grasp acquisition from dataset reset states: open -> descend -> close -> lift.
Portable across IsaacLab 2.x and 3.0. Reports hold rate per descend depth."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--reset_type", default="ObjectRestingEEGrasped")
parser.add_argument("--dataset_dir", default=None)
parser.add_argument("--descends", default="0,2,4,6", help="cm to descend after opening (comma list)")
parser.add_argument("--no_gain_rand", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, os, torch
_IFK = 1 if os.environ.get('UWLAB_ROBOT_ASSETS_DIR') else -1  # swapped joints in the patched USD read +q, inspect, torch
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose

def T(x):
    return x.torch if hasattr(x, "torch") else x

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    print('[local assets] redirected:', apply_local_object_assets(env_cfg))
    p = env_cfg.events.reset_from_reset_states.params
    if args_cli.dataset_dir: p["dataset_dir"] = args_cli.dataset_dir
    p["reset_types"] = [args_cli.reset_type]; p["probs"] = [1.0]
    if args_cli.no_gain_rand and hasattr(env_cfg.events, "randomize_gripper_actuator_parameters"):
        env_cfg.events.randomize_gripper_actuator_parameters = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    robot, peg = env.scene["robot"], env.scene["insertive_object"]
    jn, bn = list(robot.joint_names), list(robot.body_names)
    fi, base = jn.index("finger_joint"), bn.index("robotiq_base_link")
    mimic = [(jn.index(k), s) for k, s in {"right_outer_knuckle_joint": 1, "left_inner_knuckle_joint": 1, "right_inner_knuckle_joint": -1, "left_inner_finger_knuckle_joint": _IFK, "right_inner_finger_knuckle_joint": _IFK}.items()]
    N = env.num_envs
    def act(z=0.0, g=-1.0):
        a = torch.zeros(env.action_space.shape, device=env.device); a[:, 2] = z; a[:, -1] = g; return a
    def run(a, n, broke):
        for _ in range(n):
            env.step(a)
            q = T(robot.data.joint_pos); broke |= torch.stack([(q[:, i] - sg * q[:, fi]).abs() for i, sg in mimic], 1).max(1).values > 0.15
    print(f"\n=== scripted acquisition on {args_cli.reset_type}, N={N}, gain_rand={'off' if args_cli.no_gain_rand else 'on'} ===")
    print(f"{'descend':>8} {'q_open':>7} {'q_closed':>9} {'pinch%':>7} {'peg dz(lift)':>12} {'grip dz':>8} {'held%':>6} {'linkbreak%':>10} {'peg moved@open%':>15}")
    for d in [float(v) for v in args_cli.descends.split(",")]:
        env.reset()
        broke = torch.zeros(N, dtype=torch.bool, device=env.device)
        peg0 = T(peg.data.root_pos_w).clone()
        run(act(0, -1), 3, broke)
        run(act(0, +1), 8, broke)
        q_open = T(robot.data.joint_pos)[:, fi].mean().item()
        peg_moved_open = ((T(peg.data.root_pos_w) - peg0).norm(dim=1) > 0.01).float().mean().item() * 100
        steps = int(round(d / 1.0))  # z=-0.5 -> 1 cm/step at 0.02 m scale
        run(act(-0.5, +1), steps, broke)
        run(act(0, -1), 12, broke)
        q_closed = T(robot.data.joint_pos)[:, fi]
        pinch = ((q_closed > 0.40) & (q_closed < 0.62)).float().mean().item() * 100
        grip_b = T(robot.data.body_pos_w)[:, base].clone(); peg_b = T(peg.data.root_pos_w).clone()
        rel_b = (peg_b - grip_b).norm(dim=1)
        run(act(+1.0, -1), 30, broke)
        grip_e = T(robot.data.body_pos_w)[:, base]; peg_e = T(peg.data.root_pos_w)
        held = (((peg_e - grip_e).norm(dim=1) - rel_b).abs() < 0.02) & ((grip_e[:, 2] - grip_b[:, 2]) > 0.05)
        print(f"{d:>7.0f}cm {q_open:>7.3f} {q_closed.mean().item():>9.3f} {pinch:>6.1f}% {(peg_e[:,2]-peg_b[:,2]).mean().item():>12.3f} {(grip_e[:,2]-grip_b[:,2]).mean().item():>8.3f} {held.float().mean().item()*100:>5.1f}% {broke.float().mean().item()*100:>9.1f}% {peg_moved_open:>14.1f}%")
    env.close()
main(); app.close()
