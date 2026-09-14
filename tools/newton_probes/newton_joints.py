import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Newton-v0")
AppLauncher.add_app_launcher_args(parser); args_cli, remaining = parser.parse_known_args(); app = AppLauncher(args_cli).app
import gymnasium as gym, torch, isaaclab_tasks, uwlab_tasks  # noqa
from uwlab_tasks.utils.hydra import hydra_task_compose
def T(x): return x.torch if hasattr(x, "torch") else x
@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 2; env_cfg.events.reset_from_reset_states = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    r = env.scene["robot"]
    print("\nJOINTS:", list(r.joint_names)); print("BODIES:", list(r.body_names))
    print("joint_pos_limits:", [(n, [round(v, 2) for v in T(r.data.joint_pos_limits)[0, i].tolist()]) for i, n in enumerate(r.joint_names)])
    print("actuated (actuator groups):", {k: v.joint_names for k, v in r.actuators.items()})
    print("obs policy dim:", env.observation_manager.group_obs_dim)
    env.close()
main(); app.close()
