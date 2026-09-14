"""Run a trained checkpoint on one reset type and characterise grasp acquisition."""
import argparse
from isaaclab.app import AppLauncher
import cli_args  # isort: skip
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--reset_type", default="ObjectRestingEEGrasped")
parser.add_argument("--dataset_dir", default="./Datasets/OmniReset")
parser.add_argument("--video_out", default=None)
parser.add_argument("--video_envs", type=int, default=4)
parser.add_argument("--det_policy", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
if args_cli.video_out: args_cli.enable_cameras = True
import sys; sys.argv = [sys.argv[0]] + remaining
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, math, os, subprocess, torch, numpy as np
import isaaclab_tasks, uwlab_tasks  # noqa
import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import CameraCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from uwlab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    p = env_cfg.events.reset_from_reset_states.params
    p["dataset_dir"] = args_cli.dataset_dir; p["reset_types"] = [args_cli.reset_type]; p["probs"] = [1.0]
    if args_cli.video_out:
        env_cfg.scene.cam = CameraCfg(prim_path="{ENV_REGEX_NS}/cam", update_period=0, height=480, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)))
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    for mode_cfgs in u.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=u)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=u.device)
    if args_cli.det_policy and hasattr(runner.alg.policy, "act_inference"):
        policy = runner.alg.policy.act_inference
    robot, peg = u.scene["robot"], u.scene["insertive_object"]
    jn, bn = list(robot.joint_names), list(robot.body_names)
    fi, base = jn.index("finger_joint"), bn.index("robotiq_base_link")
    pc = u.reward_manager.get_term_cfg("progress_context").func
    proc = None
    if args_cli.video_out:
        cam = u.scene["cam"]; n = args_cli.video_envs
        o = u.scene.env_origins[:n]
        cam.set_world_poses_from_view(o + torch.tensor([1.05, -0.65, 0.65], device=u.device), o + torch.tensor([0.42, 0.10, 0.05], device=u.device), env_ids=torch.arange(n, device=u.device))
        cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols); W, H = cols * 640, rows * 480
        import imageio_ffmpeg
        from PIL import Image, ImageDraw
        proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "10", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args_cli.video_out], stdin=subprocess.PIPE)
    obs = env.get_observations()
    T = int(u.max_episode_length)
    agg = dict(n=0, opened=0, repinch=0, lifted=0, success=0, succ_any=0, abn=0, q0=0.0, open_frac=0.0)
    for ep in range(args_cli.episodes):
        u.reset(); obs = env.get_observations()
        N = u.num_envs
        opened = torch.zeros(N, dtype=torch.bool, device=u.device); repinch = torch.zeros_like(opened); lifted = torch.zeros_like(opened); succ_any = torch.zeros_like(opened); abn = torch.zeros_like(opened)
        open_steps = torch.zeros(N, device=u.device)
        q0 = robot.data.joint_pos.torch[:, fi].clone(); pz0 = peg.data.root_pos_w.torch[:, 2].clone()
        for t in range(T):
            with torch.inference_mode():
                a = policy(obs)
            obs, _, dones, _ = env.step(a)
            g = a[:, -1]
            opened |= g > 0; open_steps += (g > 0).float()
            q = robot.data.joint_pos.torch[:, fi]
            repinch |= opened & (q > 0.40) & (q < 0.62) & (g < 0)
            lifted |= (peg.data.root_pos_w.torch[:, 2] - pz0) > 0.05
            succ_any |= pc.success
            abn |= u.termination_manager.get_term("abnormal_robot")
            if proc is not None:
                cam.update(u.step_dt, force_recompute=True)
                arr = cam.data.output["rgb"]; arr = (arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr))[..., :3].astype(np.uint8)
                canvas = Image.new("RGB", (W, H))
                for i in range(n):
                    tile = Image.fromarray(arr[i]); d = ImageDraw.Draw(tile)
                    d.rectangle([0, 0, 640, 18], fill=(0, 0, 0))
                    d.text((4, 3), f"env{i} {args_cli.reset_type} t={t} grip_act={g[i].item():+.2f} finger_q={q[i].item():.2f} success={bool(pc.success[i])}", fill=(0, 255, 0) if pc.success[i] else (255, 255, 255))
                    canvas.paste(tile, ((i % cols) * 640, (i // cols) * 480))
                proc.stdin.write(canvas.tobytes())
            if t == T - 2:
                succ_end = pc.success.clone()
        agg["n"] += N; agg["opened"] += opened.sum().item(); agg["repinch"] += repinch.sum().item(); agg["lifted"] += lifted.sum().item()
        agg["success"] += succ_end.sum().item(); agg["succ_any"] += succ_any.sum().item(); agg["abn"] += abn.sum().item(); agg["q0"] += q0.sum().item(); agg["open_frac"] += (open_steps / T).sum().item()
    n = agg["n"]
    print(f"\n=== {args_cli.reset_type}: {n} episodes with {os.path.basename(args_cli.checkpoint)} ({'deterministic' if args_cli.det_policy else 'stochastic'}) ===")
    print(f"  finger q at reset (mean)      : {agg['q0']/n:.3f}")
    print(f"  ever commanded OPEN           : {100*agg['opened']/n:.1f}%   (mean fraction of steps open: {100*agg['open_frac']/n:.1f}%)")
    print(f"  re-pinched (q in 0.40-0.62 after opening, while commanding close): {100*agg['repinch']/n:.1f}%")
    print(f"  peg lifted > 5 cm at any time : {100*agg['lifted']/n:.1f}%")
    print(f"  success at any step / at end  : {100*agg['succ_any']/n:.1f}% / {100*agg['success']/n:.1f}%")
    print(f"  abnormal_robot termination    : {100*agg['abn']/n:.1f}%")
    if proc is not None:
        proc.stdin.close(); proc.wait(); print("  wrote", args_cli.video_out)
    env.close()
main(); app.close()
