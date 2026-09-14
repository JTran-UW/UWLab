import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Newton-v0")
parser.add_argument("--checkpoint", default="expert_seed0_rslrl52.pt")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch, traceback
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    import os as _os
    env_cfg.events.reset_from_reset_states.params["dataset_dir"] = "./Datasets/OmniReset_patched" if _os.environ.get("UWLAB_ROBOT_ASSETS_DIR") else "./Datasets/OmniReset"
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    om = env.observation_manager
    for g in om.active_terms:
        print(f"group {g}: terms={list(zip(om.active_terms[g], [int(d[0]) for d in om.group_obs_term_dim[g]]))} total={om.group_obs_dim[g]}")
    obs = om.compute()
    rm = env.reward_manager
    print("reward term funcs:", {n: (type(tc.func).__name__ if not callable(tc.func) or hasattr(tc.func, "__self__") or not hasattr(tc.func, "__name__") else tc.func.__name__) for n, tc in zip(rm.active_terms, rm._term_cfgs)})
    for g, v in obs.items(): print(f"  obs[{g}] shape={tuple(v.shape)} finite={bool(torch.isfinite(v).all())}")
    ck = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    a = ck["actor_state_dict"]; c = ck.get("critic_state_dict", {})
    print("expert actor in-dim:", a["mlp.0.weight"].shape[1], " critic in-dim:", c["mlp.0.weight"].shape[1] if c else None)
    for t in range(3):
        o, r, term, trunc, info = env.step(torch.zeros(env.action_space.shape, device=env.device))
    print("3 steps ok; reward finite:", bool(torch.isfinite(r).all()), "extras keys:", list(info.keys())[:6])
    env.close()
main(); app.close()
