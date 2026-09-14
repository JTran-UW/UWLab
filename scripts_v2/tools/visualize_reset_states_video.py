# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a video of saved reset states (headless variant of visualize_reset_states.py).

One camera per env, tiled into a grid, with the reset type and success flag
overlaid. Run with ``--enable_cameras``.
"""

from __future__ import annotations

import argparse
import torch
from typing import cast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a video of saved reset states.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--dataset_dir", type=str, default="./Datasets/OmniReset")
parser.add_argument("--reset_type", type=str, default=None)
parser.add_argument("--out", type=str, default="reset_states.mp4")
parser.add_argument("--duration", type=float, default=600.0, help="video length in seconds")
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--hold_steps", type=int, default=20, help="env steps (= frames) per reset state")
parser.add_argument("--cam_res", type=int, nargs=2, default=(640, 480))
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import inspect  # noqa: E402
import math  # noqa: E402
import numpy as np  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import subprocess  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import ManagerTermBase  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402

ALL_RESET_TYPES = [
    "ObjectAnywhereEEAnywhere",
    "ObjectRestingEEGrasped",
    "ObjectAnywhereEEGrasped",
    "ObjectPartiallyAssembledEEGrasped",
]


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    reset_types = [args_cli.reset_type] if args_cli.reset_type else ALL_RESET_TYPES
    env_cfg.events.reset_from_reset_states.params["dataset_dir"] = args_cli.dataset_dir
    env_cfg.events.reset_from_reset_states.params["reset_types"] = reset_types
    env_cfg.events.reset_from_reset_states.params["probs"] = [1.0] * len(reset_types)

    w, h = args_cli.cam_res
    env_cfg.scene.cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam",
        update_period=0,
        height=h,
        width=w,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)),
    )

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()

    cam = env.scene["cam"]
    origins = env.scene.env_origins
    eyes = origins + torch.tensor([1.05, -0.65, 0.65], device=env.device)
    targets = origins + torch.tensor([0.42, 0.10, 0.05], device=env.device)
    cam.set_world_poses_from_view(eyes, targets)

    reset_term = env.event_manager.get_term_cfg("reset_from_reset_states").func
    robot = env.scene["robot"]
    finger_idx = robot.find_joints(["finger_joint"])[0][0]
    close_val = env_cfg.actions.gripper.close_command_expr["finger_joint"]

    n = args_cli.num_envs
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    W, H = cols * w, rows * h
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(args_cli.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args_cli.out],
        stdin=subprocess.PIPE,
    )
    total_frames = int(args_cli.duration * args_cli.fps)
    frame_count = 0
    state_count = 0
    print(f"Recording {total_frames} frames ({args_cli.duration:.0f}s @ {args_cli.fps}fps) to {args_cli.out}")

    while frame_count < total_frames:
        task_ids = reset_term.task_id.clone()
        q = robot.data.joint_pos.torch[:, finger_idx]
        gripper_mask = (q.abs() / close_val) > 0.1
        action = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
        action[gripper_mask, -1] = -1.0
        action[~gripper_mask, -1] = 1.0
        for step in range(args_cli.hold_steps):
            env.step(action)
            success = env.reward_manager.get_term_cfg("progress_context").func.success
            cam.update(env.step_dt, force_recompute=True)
            rgb = cam.data.output["rgb"]
            arr = (rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb))[..., :3].astype(np.uint8)
            canvas = Image.new("RGB", (W, H))
            for i in range(n):
                tile = Image.fromarray(arr[i])
                d = ImageDraw.Draw(tile)
                rt = reset_types[int(task_ids[i])]
                s = bool(success[i]) if success is not None else None
                label = f"env{i}  {rt}  step {step}  success={s}"
                d.rectangle([0, 0, w, 18], fill=(0, 0, 0))
                d.text((4, 3), label, fill=(0, 255, 0) if s else (255, 255, 255))
                canvas.paste(tile, ((i % cols) * w, (i // cols) * h))
            proc.stdin.write(canvas.tobytes())
            frame_count += 1
            if frame_count >= total_frames:
                break
        state_count += 1
        if state_count % 10 == 0:
            print(f"  {frame_count}/{total_frames} frames, {state_count} reset rounds", flush=True)
        env.reset()

    proc.stdin.close()
    proc.wait()
    print(f"Wrote {args_cli.out}: {frame_count} frames, {state_count} reset rounds x {n} envs")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
