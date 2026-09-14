"""Evaluate a PPO checkpoint (2.x or rsl_rl-5.2 layout) as a plain MLP on a task, with a tiled per-env video."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--reset_types", default="ObjectAnywhereEEAnywhere,ObjectRestingEEGrasped,ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEGrasped")
parser.add_argument("--dataset_dir", default=None)
parser.add_argument("--video_out", default=None)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--renderer", choices=["rtx", "newton"], default="rtx", help="newton = Warp raytracer (no RTX needed)")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
if args_cli.video_out: args_cli.enable_cameras = True
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, math, os, subprocess, torch, numpy as np
import isaaclab_tasks, uwlab_tasks  # noqa
import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import CameraCfg
from uwlab_tasks.utils.hydra import hydra_task_compose
def T(x): return x.torch if hasattr(x, "torch") else x

def load_policy(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    if "model_state_dict" in ck:  # rsl_rl 3.x
        sd = ck["model_state_dict"]; W = [sd[f"actor.{i}.weight"] for i in (0, 2, 4, 6, 8)]; b = [sd[f"actor.{i}.bias"] for i in (0, 2, 4, 6, 8)]
        mean, std = sd["actor_obs_normalizer._mean"], sd["actor_obs_normalizer._std"]
    else:  # rsl_rl 5.x
        sd = ck["actor_state_dict"]; W = [sd[f"mlp.{i}.weight"] for i in (0, 2, 4, 6, 8)]; b = [sd[f"mlp.{i}.bias"] for i in (0, 2, 4, 6, 8)]
        mean, std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
    def policy(obs):
        x = (obs - mean) / (std + 1e-2)
        for i, (Wi, bi) in enumerate(zip(W, b)):
            x = x @ Wi.T + bi
            if i < 4: x = torch.nn.functional.elu(x)
        return x
    return policy

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    print('[local assets] redirected:', apply_local_object_assets(env_cfg))
    rts = args_cli.reset_types.split(",")
    p = env_cfg.events.reset_from_reset_states.params
    if args_cli.dataset_dir: p["dataset_dir"] = args_cli.dataset_dir
    p["reset_types"] = rts; p["probs"] = [1.0] * len(rts)
    if args_cli.video_out:
        cam_kw = {}
        if args_cli.renderer == "newton":
            from isaaclab_newton.renderers import NewtonWarpRendererCfg
            cam_kw["renderer_cfg"] = NewtonWarpRendererCfg(enable_shadows=True)
        env_cfg.scene.cam = CameraCfg(prim_path="{ENV_REGEX_NS}/cam", update_period=0, height=480, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)), **cam_kw)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    dev = env.device
    policy = load_policy(args_cli.checkpoint, dev)
    print("physics backend:", type(env_cfg.sim.physics).__name__)
    term = env.event_manager.get_term_cfg("reset_from_reset_states").func
    pc = env.reward_manager.get_term_cfg("progress_context").func
    robot = env.scene["robot"]; fi = list(robot.joint_names).index("finger_joint")
    N = env.num_envs; Tmax = int(env.max_episode_length)
    proc = None
    if args_cli.video_out:
        from PIL import Image, ImageDraw
        import imageio_ffmpeg
        cam = env.scene["cam"]; o = env.scene.env_origins
        cam.set_world_poses_from_view(o + torch.tensor([1.05, -0.65, 0.65], device=dev), o + torch.tensor([0.42, 0.10, 0.05], device=dev))
        cols = math.ceil(math.sqrt(N)); rows = math.ceil(N / cols); W_, H_ = cols * 640, rows * 480
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.video_out)), exist_ok=True)
        proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_}x{H_}", "-r", str(args_cli.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args_cli.video_out], stdin=subprocess.PIPE)
    agg = {rt: dict(n=0, succ=0) for rt in rts}
    for ep in range(args_cli.episodes):
        obs_dict, _ = env.reset(); tid = term.task_id.clone()
        for t in range(Tmax - 1):
            with torch.inference_mode():
                a = policy(obs_dict["policy"])
            obs_dict, _, _, _, _ = env.step(a)
            if proc is not None:
                cam.update(env.step_dt, force_recompute=True)
                arr = cam.data.output["rgb"]; arr = (arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr))[..., :3].astype(np.uint8)
                canvas = Image.new("RGB", (W_, H_))
                for i in range(N):
                    tile = Image.fromarray(arr[i]); d = ImageDraw.Draw(tile)
                    s = bool(pc.success[i]); d.rectangle([0, 0, 640, 18], fill=(0, 0, 0))
                    d.text((4, 3), f"env{i} {rts[int(tid[i])]} ep{ep} t={t} grip={a[i,-1].item():+.2f} q={T(robot.data.joint_pos)[i, fi].item():.2f} success={s} [{type(env_cfg.sim.physics).__name__}]", fill=(0, 255, 0) if s else (255, 255, 255))
                    canvas.paste(tile, ((i % cols) * 640, (i // cols) * 480))
                proc.stdin.write(canvas.tobytes())
        succ = pc.success.clone()
        for i, rt in enumerate(rts):
            m = tid == i; agg[rt]["n"] += m.sum().item(); agg[rt]["succ"] += (succ & m).sum().item()
        print(f"  episode {ep}: " + ", ".join(f"env{i}={rts[int(tid[i])]}:{'OK' if succ[i] else 'fail'}" for i in range(N)), flush=True)
    print(f"\n=== {os.path.basename(args_cli.checkpoint)} on {args_cli.task} ({type(env_cfg.sim.physics).__name__}) ===")
    for rt, s in agg.items():
        print(f"  {rt:34s} {s['succ']}/{s['n']}")
    if proc is not None:
        proc.stdin.close(); proc.wait(); print("  wrote", args_cli.video_out)
    env.close()
main(); app.close()
