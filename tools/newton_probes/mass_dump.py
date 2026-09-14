"""Dump per-body mass / COM / inertia as the physics backend sees them."""
import argparse, json
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
parser.add_argument("--dataset_dir", default=None)
parser.add_argument("--out", required=True)
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
    env_cfg.scene.num_envs = 1; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    p = env_cfg.events.reset_from_reset_states.params
    if args_cli.dataset_dir: p["dataset_dir"] = args_cli.dataset_dir
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    out = {}
    for name in ("robot", "insertive_object", "receptive_object"):
        a = env.scene[name]; d = a.data
        mass = T(d.default_mass)[0]
        inertia = T(d.default_inertia)[0] if hasattr(d, "default_inertia") else None
        com = T(d.body_com_pos_b)[0] if hasattr(d, "body_com_pos_b") else (T(d.com_pos_b)[0] if hasattr(d, "com_pos_b") else None)
        names = list(a.body_names)
        out[name] = {}
        for i, b in enumerate(names):
            out[name][b] = {"mass": float(mass[i]),
                            "inertia": [float(v) for v in inertia[i].tolist()] if inertia is not None else None,
                            "com": [float(v) for v in com[i].tolist()] if com is not None else None}
    json.dump({"backend": type(env_cfg.sim.physics).__name__, "bodies": out}, open(args_cli.out, "w"), indent=1)
    for name, bodies in out.items():
        print(f"== {name}")
        for b, v in bodies.items():
            I = v["inertia"]; diag = [round(I[0],5), round(I[4],5), round(I[8],5)] if I else None
            print(f"  {b:22s} m={v['mass']:.4f} Idiag={diag} com={[round(c,4) for c in v['com']] if v['com'] else None}")
    env.close()
main(); app.close()
