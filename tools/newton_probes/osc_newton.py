import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Newton-v0"); parser.add_argument("--steps", type=int, default=12); parser.add_argument("--ax", type=float, default=1.0); parser.add_argument("--no_dataset_reset", action="store_true"); parser.add_argument("--grip", type=float, default=-1.0)
AppLauncher.add_app_launcher_args(parser); args_cli, remaining = parser.parse_known_args(); app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, os, torch, isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
def T(x): return x.torch if hasattr(x, "torch") else x
@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 8; apply_local_object_assets(env_cfg)
    p = env_cfg.events.reset_from_reset_states.params; p["dataset_dir"] = "./Datasets/OmniReset_patched"; p["reset_types"] = ["ObjectAnywhereEEGrasped"]; p["probs"] = [1.0]
    env_cfg.terminations.abnormal_robot = None
    if args_cli.no_dataset_reset:
        env_cfg.events.reset_from_reset_states = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase): tc.func = tc.func(cfg=tc, env=env)
    r = env.scene["robot"]; w3 = list(r.body_names).index("wrist_3_link")
    print(f"CONFIG damping={os.environ.get('UWLAB_NEWTON_ARM_DAMPING')} substeps={getattr(env_cfg.sim.physics, "num_substeps", "physx")} | data fields:", [k for k in ("joint_effort_target", "applied_torque", "computed_torque", "joint_effort") if hasattr(r.data, k)])
    env.reset(); ee0 = T(r.data.body_pos_w)[:, w3].clone()
    a = torch.zeros(env.action_space.shape, device=env.device); a[:, 0] = args_cli.ax; a[:, -1] = args_cli.grip
    nan_at = None
    for t in range(args_cli.steps):
        env.step(a)
        q = T(r.data.joint_pos); jv = T(r.data.joint_vel); ee = T(r.data.body_pos_w)[:, w3]
        tgt = T(r.data.joint_effort_target)[:, :6] if hasattr(r.data, "joint_effort_target") else None
        app_ = T(r.data.applied_torque)[:, :6] if hasattr(r.data, "applied_torque") else None
        if torch.isnan(q).any(): nan_at = t; print(f"  t={t}: NaN"); break
        print(f"  t={t:2d} max|arm jv|={jv[:, :6].abs().max().item():7.2f}  ee dx(mean)={(ee - ee0)[:, 0].mean().item():+.4f} |ee d|={(ee - ee0).norm(dim=1).mean().item():.4f}  effort_tgt(env0)={[round(v,1) for v in tgt[0].tolist()] if tgt is not None else None}  applied(env0)={[round(v,1) for v in app_[0].tolist()] if app_ is not None else None}")
    print(f"RESULT damping={os.environ.get('UWLAB_NEWTON_ARM_DAMPING')} substeps={getattr(env_cfg.sim.physics, "num_substeps", "physx")}: {'NaN at t=%d' % nan_at if nan_at is not None else 'stable'}; final |ee d|={(T(r.data.body_pos_w)[:, w3] - ee0).norm(dim=1).mean().item():.4f} m (PhysX ref 0.038 m / 10 steps)")
    env.close()
main(); app.close()
