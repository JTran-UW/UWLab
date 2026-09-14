"""Fraction of dataset resets that already satisfy the success criterion, per reset type."""
import argparse, torch
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--dataset_dir", default="./Datasets/OmniReset")
parser.add_argument("--steps_after_reset", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose

TYPES = ["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"]

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = None
    env_cfg.events.reset_from_reset_states.params["dataset_dir"] = args_cli.dataset_dir
    env_cfg.events.reset_from_reset_states.params["reset_types"] = TYPES
    env_cfg.events.reset_from_reset_states.params["probs"] = [1.0] * len(TYPES)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    term = env.event_manager.get_term_cfg("reset_from_reset_states").func
    robot = env.scene["robot"]
    fi = robot.find_joints(["finger_joint"])[0][0]
    close_val = env_cfg.actions.gripper.close_command_expr["finger_joint"]
    hits = torch.zeros(len(TYPES)); tot = torch.zeros(len(TYPES))
    hits_hold = torch.zeros(len(TYPES))
    for r in range(args_cli.rounds):
        env.reset()
        tid = term.task_id.clone().cpu()
        q = robot.data.joint_pos.torch[:, fi]
        act = torch.zeros(env.action_space.shape, device=env.device)
        act[:, -1] = torch.where((q.abs() / close_val) > 0.1, -1.0, 1.0)
        for s in range(args_cli.steps_after_reset):
            env.step(act)
        succ0 = env.reward_manager.get_term_cfg("progress_context").func.success.clone().cpu().bool()
        for s in range(20):
            env.step(act)
        succ20 = env.reward_manager.get_term_cfg("progress_context").func.success.clone().cpu().bool()
        for t in range(len(TYPES)):
            m = tid == t
            tot[t] += m.sum(); hits[t] += (succ0 & m).sum(); hits_hold[t] += (succ20 & m).sum()
    print("\n=== fraction of resets already in the success configuration ===")
    print(f"{'reset type':36s} {'N':>6} {'success@reset':>14} {'success@+20 steps':>18}")
    for t, name in enumerate(TYPES):
        print(f"{name:36s} {int(tot[t]):>6} {100*hits[t]/max(tot[t],1):>13.1f}% {100*hits_hold[t]/max(tot[t],1):>17.1f}%")
    print(f"{'ALL (uniform mix)':36s} {int(tot.sum()):>6} {100*hits.sum()/tot.sum():>13.1f}% {100*hits_hold.sum()/tot.sum():>17.1f}%")
    env.close()
main(); app.close()
