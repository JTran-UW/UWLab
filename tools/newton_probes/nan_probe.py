import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Newton-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--noise", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch, sys, os
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose
def load_policy(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck["actor_state_dict"]; W = [sd[f"mlp.{i}.weight"] for i in (0, 2, 4, 6, 8)]; b = [sd[f"mlp.{i}.bias"] for i in (0, 2, 4, 6, 8)]
    mean, std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
    def policy(obs):
        x = (obs - mean) / (std + 1e-2)
        for i, (Wi, bi) in enumerate(zip(W, b)):
            x = x @ Wi.T + bi
            if i < 4: x = torch.nn.functional.elu(x)
        return x
    return policy
def T(x): return x.torch if hasattr(x, "torch") else x

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    env_cfg.events.reset_from_reset_states.params["dataset_dir"] = "./Datasets/OmniReset_patched"
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    obs, _ = env.reset()
    om = env.observation_manager; names = om.active_terms["policy"]; dims = [int(d[0]) for d in om.group_obs_term_dim["policy"]]
    import numpy as np; off = np.cumsum([0] + dims)
    robot = env.scene["robot"]; term = env.event_manager.get_term_cfg("reset_from_reset_states").func
    policy = load_policy("expert_seed0_rslrl52.pt", env.device)
    def report(tag, o):
        bad = ~torch.isfinite(o["policy"]).all(dim=1)
        if bad.any():
            e = bad.nonzero().flatten()
            terms = [names[k] for k in range(len(names)) if not torch.isfinite(o["policy"][e][:, off[k]:off[k+1]]).all()]
            jv = T(robot.data.joint_vel)[e]; jp = T(robot.data.joint_pos)[e]
            print(f"[{tag}] NaN in {int(bad.sum())} envs {e[:10].tolist()} terms={terms} task_ids={term.task_id[e][:10].tolist()} |jv|max={jv.abs().nan_to_num(nan=1e9).max():.1f} jp_finite={bool(torch.isfinite(jp).all())} crit_finite={bool(torch.isfinite(o['critic']).all()) if 'critic' in o else None}")
            return True
        return False
    if report("reset", obs): env.close(); return
    print("reset obs finite"); nan_events = 0
    for t in range(args_cli.steps):
        a = policy(obs["policy"]) + args_cli.noise * torch.randn(env.num_envs, 7, device=env.device)
        obs, r, term_, trunc, info = env.step(a)
        jv = T(robot.data.joint_vel)
        if t % 10 == 0: print(f"t={t} |jv|max={jv.abs().nan_to_num(nan=1e9).max():.1f} n_jv>50={int((jv.abs()>50).any(1).sum())} r_finite={bool(torch.isfinite(r).all())} dones={int(term_.sum())}/{int(trunc.sum())}")
        bad = ~torch.isfinite(obs["policy"]).all(dim=1)
        if bad.any():
            e = bad.nonzero().flatten(); nan_events += int(bad.sum())
            print(f"[step {t}] NaN envs {e[:8].tolist()} task_ids={term.task_id[e][:8].tolist()} -> resetting them")
            env._reset_idx(e); obs = env.observation_manager.compute()
            obs["policy"] = torch.nan_to_num(obs["policy"])
    print(f"TOTAL NaN events: {nan_events} over {args_cli.steps} steps x {env.num_envs} envs")
    env.close()
main(); app.close()
