"""Random-policy rollout on PartiallyAssembled resets (train task): how often does a seated peg stay seated?"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--dataset_dir", default="./Datasets/OmniReset")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--std", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    p = env_cfg.events.reset_from_reset_states.params
    p["dataset_dir"] = args_cli.dataset_dir; p["reset_types"] = ["ObjectPartiallyAssembledEEGrasped"]; p["probs"] = [1.0]
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    pc = env.reward_manager.get_term_cfg("progress_context").func
    env.step(torch.zeros(env.action_space.shape, device=env.device))
    s0 = pc.success.clone(); print(f"backend={type(env_cfg.sim.physics).__name__} seated at reset: {s0.float().mean():.2f}")
    torch.manual_seed(0)
    for t in range(args_cli.steps):
        a = torch.randn(env.action_space.shape, device=env.device) * args_cli.std
        env.step(a)
        if t in (9, 39, 79, 159):
            print(f"  t={t+1:3d}: success now {pc.success.float().mean():.2f} | of initially-seated still seated {(pc.success & s0).float().sum()/max(1,s0.float().sum()):.2f} | abnormal {env.termination_manager.get_term('abnormal_robot').float().mean():.3f}")
    env.close()
main(); app.close()
