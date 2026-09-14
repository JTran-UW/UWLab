"""PartiallyAssembled resets: close gripper + push down; report success-criterion errors (both backends)."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
parser.add_argument("--dataset_dir", default="./Datasets/OmniReset")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--push", type=float, default=-1.0)
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
    p["dataset_dir"] = args_cli.dataset_dir; p["reset_types"] = ["ObjectPartiallyAssembledEEGrasped"]; p["probs"] = [1.0]
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import sample_from_nested_dict
    term = env.event_manager.get_term_cfg("reset_from_reset_states").func
    ids = torch.arange(env.num_envs, device=env.device)
    term._reset_to(sample_from_nested_dict(term.datasets[0], ids)["initial_state"], env_ids=ids, is_relative=True)
    for at in env.action_manager._terms.values(): at.reset(None)
    env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(env.physics_dt)
    pc = env.reward_manager.get_term_cfg("progress_context").func
    peg, hole = env.scene["insertive_object"], env.scene["receptive_object"]
    def rel():
        pp, pq = mu.subtract_frame_transforms(T(hole.data.root_pos_w), T(hole.data.root_quat_w), T(peg.data.root_pos_w), T(peg.data.root_quat_w))
        return pp
    act = torch.zeros(env.action_space.shape, device=env.device); act[:, 2] = args_cli.push; act[:, -1] = -1.0
    r0 = rel(); h0 = T(hole.data.root_pos_w).clone(); pg0 = T(peg.data.root_pos_w).clone()
    for name, a in (("hole", hole), ("peg", peg)):
        d = a.data
        rp, rq = T(d.root_pos_w)[0], T(d.root_quat_w)[0]
        lp = T(d.root_link_pos_w)[0] if hasattr(d, "root_link_pos_w") else None
        cp = T(d.root_com_pos_w)[0] if hasattr(d, "root_com_pos_w") else None
        lq = T(d.root_link_quat_w)[0] if hasattr(d, "root_link_quat_w") else None
        cq = T(d.root_com_quat_w)[0] if hasattr(d, "root_com_quat_w") else None
        print(f"FRAMES {name}: root_pos={rp.cpu().numpy().round(4).tolist()} link={None if lp is None else lp.cpu().numpy().round(4).tolist()} com={None if cp is None else cp.cpu().numpy().round(4).tolist()} | root_quat={rq.cpu().numpy().round(3).tolist()} link_q={None if lq is None else lq.cpu().numpy().round(3).tolist()} com_q={None if cq is None else cq.cpu().numpy().round(3).tolist()}")
    ip, iq = pc.insertive_asset_offset.apply(peg); hp, hq = pc.receptive_asset_offset.apply(hole)
    dp, dq = mu.subtract_frame_transforms(hp, hq, ip, iq)
    print(f"KEYPOINTS env0: peg_kp={ip[0].cpu().numpy().round(4).tolist()} hole_kp={hp[0].cpu().numpy().round(4).tolist()} rel={dp[0].cpu().numpy().round(4).tolist()} |rel| p50 over envs={dp.norm(dim=1).median():.4f}  offsets: peg {pc.insertive_asset_offset.pos} {pc.insertive_asset_offset.quat} hole {pc.receptive_asset_offset.pos} {pc.receptive_asset_offset.quat}")
    print(f"RESET(no step): peg-in-hole xy={r0[:, :2].norm(dim=1).median():.4f} z p10={r0[:,2].quantile(0.1):.4f} p50={r0[:,2].median():.4f}")
    print(f"backend={type(env_cfg.sim.physics).__name__} thresholds pos={pc.cfg.params.get('success_position_threshold', None)}")
    for t in range(args_cli.steps):
        env.step(act)
        if t in (0, 4, 9, 19, 39, args_cli.steps - 1):
            r = rel(); xyz = pc.xyz_distance; exy = pc.euler_xy_distance; succ = pc.success
            hd = (T(hole.data.root_pos_w) - h0).norm(dim=1); pd = (T(peg.data.root_pos_w) - pg0)
            print(f"t={t:2d} hole moved: p50={hd.median():.4f} p90={hd.quantile(0.9):.4f} | peg dxy p50={pd[:, :2].norm(dim=1).median():.4f} dz p50={pd[:,2].median():.4f} | peg-in-hole xy p50={r[:, :2].norm(dim=1).median():.4f}")
            print(f"t={t:2d} peg-in-hole z: mean={r[:,2].mean():.4f} p10={r[:,2].quantile(0.1):.4f} p90={r[:,2].quantile(0.9):.4f} | xyz_dist mean={xyz.mean():.4f} p50={xyz.median():.4f} | euler_xy mean={exy.mean():.4f} p50={exy.median():.4f} | pos_ok={pc.position_aligned.float().mean():.2f} ori_ok={pc.orientation_aligned.float().mean():.2f} success={succ.float().mean():.2f}")
    print(f"at reset: peg-in-hole z mean={r0[:,2].mean():.4f}")
    env.close()
main(); app.close()
