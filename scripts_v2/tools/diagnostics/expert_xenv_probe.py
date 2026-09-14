"""Run a 2.x rsl_rl PPO expert (plain MLP) in whichever env version this runs under."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--expert", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--reset_types", default="ObjectAnywhereEEAnywhere,ObjectRestingEEGrasped,ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEGrasped")
parser.add_argument("--dataset_dir", default=None)
parser.add_argument("--noise_std", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose
def T(x): return x.torch if hasattr(x, "torch") else x

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    print('[local assets] redirected:', apply_local_object_assets(env_cfg))
    rts = args_cli.reset_types.split(",")
    p = env_cfg.events.reset_from_reset_states.params
    if args_cli.dataset_dir: p["dataset_dir"] = args_cli.dataset_dir
    p["reset_types"] = rts; p["probs"] = [1.0] * len(rts)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    dev = env.device
    sd = torch.load(args_cli.expert, map_location=dev, weights_only=False)["model_state_dict"]
    layers = [sd[f"actor.{i}.weight"] for i in (0, 2, 4, 6, 8)]; biases = [sd[f"actor.{i}.bias"] for i in (0, 2, 4, 6, 8)]
    mean, std = sd["actor_obs_normalizer._mean"], sd["actor_obs_normalizer._std"]
    def policy(obs):
        x = (obs - mean) / (std + 1e-8)
        for i, (W, b) in enumerate(zip(layers, biases)):
            x = x @ W.T + b
            if i < 4: x = torch.nn.functional.elu(x)
        return x
    robot = env.scene["robot"]; jn = list(robot.joint_names); fi = jn.index("finger_joint")
    print("joint order:", jn)
    term = env.event_manager.get_term_cfg("reset_from_reset_states").func
    pc = env.reward_manager.get_term_cfg("progress_context").func
    Tmax = int(env.max_episode_length); N = env.num_envs
    agg = {rt: dict(n=0, succ=0, opened=0, open_steps=0.0) for rt in rts}
    obs_dict, _ = env.reset()
    for ep in range(args_cli.episodes):
        obs_dict, _ = env.reset(); tid = term.task_id.clone()
        opened = torch.zeros(N, dtype=torch.bool, device=dev); open_steps = torch.zeros(N, device=dev)
        for t in range(Tmax - 1):
            with torch.inference_mode():
                a = policy(obs_dict["policy"])
                if args_cli.noise_std > 0: a = a + args_cli.noise_std * torch.randn_like(a)
            obs_dict, _, _, _, _ = env.step(a)
            g = a[:, -1]; opened |= g > 0; open_steps += (g > 0).float()
        succ = pc.success.clone()
        for i, rt in enumerate(rts):
            m = tid == i; k = m.sum().item()
            agg[rt]["n"] += k; agg[rt]["succ"] += (succ & m).sum().item(); agg[rt]["opened"] += (opened & m).sum().item(); agg[rt]["open_steps"] += (open_steps[m] / Tmax).sum().item()
    print(f"\n=== 2.x expert {args_cli.expert.split('/')[-1]} in this env (noise_std={args_cli.noise_std}) ===")
    print(f"{'reset type':34s} {'N':>5} {'success':>8} {'ever opened':>12} {'frac steps open':>16}")
    for rt, s in agg.items():
        n = max(s["n"], 1); print(f"{rt:34s} {s['n']:>5} {100*s['succ']/n:>7.1f}% {100*s['opened']/n:>11.1f}% {100*s['open_steps']/n:>15.1f}%")
    env.close()
main(); app.close()
