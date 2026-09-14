"""Close-up renders of grasped resets (t=0 and after a short hold) on either backend; prints pad/peg geometry."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
parser.add_argument("--dataset_dir", default="./Datasets/OmniReset")
parser.add_argument("--reset_type", default="ObjectAnywhereEEGrasped")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--out", default="/tmp/jtran_grasp")
parser.add_argument("--renderer", choices=["rtx", "newton"], default="rtx")
parser.add_argument("--hold", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, math, os, torch, numpy as np
from PIL import Image
import isaaclab_tasks, uwlab_tasks  # noqa
import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import CameraCfg
from uwlab_tasks.utils.hydra import hydra_task_compose
def T(x): return x.torch if hasattr(x, "torch") else x

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    p = env_cfg.events.reset_from_reset_states.params
    p["dataset_dir"] = args_cli.dataset_dir; p["reset_types"] = [args_cli.reset_type]; p["probs"] = [1.0]
    kw = {}
    if args_cli.renderer == "newton":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg
        kw["renderer_cfg"] = NewtonWarpRendererCfg(enable_shadows=True)
    env_cfg.scene.cam = CameraCfg(prim_path="{ENV_REGEX_NS}/cam", update_period=0, height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=70.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.02, 10.0)), **kw)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    robot, peg, cam = env.scene["robot"], env.scene["insertive_object"], env.scene["cam"]
    o = env.scene.env_origins
    cam.set_world_poses_from_view(o + torch.tensor([1.05, -0.65, 0.65], device=env.device), o + torch.tensor([0.42, 0.10, 0.05], device=env.device))
    print("   dbg pre-reset cam pos", [round(v,2) for v in T(cam.data.pos_w)[0].tolist()])
    env.reset()
    print("   dbg post-reset cam pos", [round(v,2) for v in T(cam.data.pos_w)[0].tolist()])
    os.makedirs(args_cli.out, exist_ok=True)
    bn = list(robot.body_names); jn = list(robot.joint_names)
    pads = [bn.index("left_inner_finger"), bn.index("right_inner_finger")]
    gj = [jn.index(n) for n in ("finger_joint","right_outer_knuckle_joint","left_inner_knuckle_joint","right_inner_knuckle_joint","left_inner_finger_knuckle_joint","right_inner_finger_knuckle_joint") if n in jn]
    act = torch.zeros(env.action_space.shape, device=env.device); act[:, -1] = -1.0
    import isaaclab.utils.math as mu
    parser_side = 0.30
    def snap(tag):
        ee = T(robot.data.body_pos_w)[:, bn.index("robotiq_base_link")]
        eq = T(robot.data.body_quat_w)[:, bn.index("robotiq_base_link")]
        pp = T(peg.data.root_pos_w)
        pinch = ee + mu.quat_apply(eq, torch.tensor([[0.1345, 0.0, 0.0]], device=env.device).expand(ee.shape[0], -1))
        eye = env.scene.env_origins + torch.tensor([1.05, -0.65, 0.65], device=env.device)
        cam.set_world_poses_from_view(eye, pinch)
        if args_cli.renderer == "newton": cam.reset()
        if args_cli.renderer == "newton":
            env.step(act)
        for k in range(3):
            if args_cli.renderer == "rtx": env.sim.render()
            cam.update(env.step_dt, force_recompute=True)
            r = cam.data.output["rgb"]; r = r.cpu().numpy() if hasattr(r, "cpu") else np.asarray(r)
            print(f"   dbg {tag} upd{k}: rgb mean={r.mean():.1f} shape={r.shape} cam pos={[round(v,2) for v in T(cam.data.pos_w)[0].tolist()]}")
        if args_cli.renderer == "newton":
            env.step(act); r = cam.data.output["rgb"]; r = r.cpu().numpy() if hasattr(r, "cpu") else np.asarray(r)
            print(f"   dbg {tag} after step: rgb mean={r.mean():.1f}")
        rgb = cam.data.output["rgb"]; arr = (rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb))[..., :3].astype(np.uint8)
        n = args_cli.num_envs; cols = 2; rows = math.ceil(n / cols)
        canvas = Image.new("RGB", (cols * 640, rows * 480))
        for i in range(n): canvas.paste(Image.fromarray(arr[i]), ((i % cols) * 640, (i // cols) * 480))
        fn = os.path.join(args_cli.out, f"{args_cli.renderer}_{tag}.png"); canvas.save(fn); print("wrote", fn)
        q = T(robot.data.joint_pos)[:, gj]
        padmid = 0.5 * (T(robot.data.body_pos_w)[:, pads[0]] + T(robot.data.body_pos_w)[:, pads[1]])
        for i in range(n):
            print(f"  [{tag}] env{i} gripper q={[round(v,3) for v in q[i].tolist()]} peg-padmid={(pp[i]-padmid[i]).norm():.3f} peg-base={(pp[i]-ee[i]).norm():.3f} peg z={pp[i,2]:.3f} padgap={(T(robot.data.body_pos_w)[i,pads[0]]-T(robot.data.body_pos_w)[i,pads[1]]).norm():.3f}")
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import sample_from_nested_dict
    rt = env.event_manager.get_term_cfg("reset_from_reset_states").func
    ids = torch.arange(env.num_envs, device=env.device)
    rt._reset_to(sample_from_nested_dict(rt.datasets[0], ids)["initial_state"], env_ids=ids, is_relative=True)
    for at in env.action_manager._terms.values(): at.reset(None)
    snap("t0")
    for t in range(args_cli.hold):
        env.step(act)
        if t + 1 in (5, 10, args_cli.hold): snap(f"t{t+1}")
    env.close()
main(); app.close()
