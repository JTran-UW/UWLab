"""Env-only throughput and memory of a camera task: steps with small random actions and reports ms/step,
env-steps/s, the cameras' share of the step and GPU memory.

usage: run_in_sif.sh scripts/tools/bench_camera_env.py --task <vision task> --num_envs 4096 <hydra overrides>
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--agent", default="flashsac_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--action_std", type=float, default=0.3)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import warp as wp

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


def gpu_used_gib():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30, total / 2**30


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 1
    start = time.perf_counter()
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    n, dim, dev = base.num_envs, base.action_manager.total_action_dim, base.device
    print(f"[bench] num_envs={n} created in {time.perf_counter() - start:.0f} s", flush=True)

    def run(count):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(count):
            env.step((torch.randn(n, dim, device=dev) * args_cli.action_std).clamp(-1, 1))
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / count

    run(args_cli.warmup)
    step = run(args_cli.steps)
    cameras = {}
    device = wp.get_device(str(dev))
    for name, sensor in base.scene.sensors.items():
        if "rgb" not in getattr(sensor.data, "output", {}):
            continue
        for _ in range(2):
            sensor.update(base.physics_dt, force_recompute=True)
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(10):
            sensor.update(base.physics_dt, force_recompute=True)
        wp.synchronize_device(device)
        cameras[name] = (time.perf_counter() - t0) / 10
        shape = tuple(sensor.data.output["rgb"].shape[1:3])
    used, total = gpu_used_gib()
    render = sum(cameras.values())
    print(
        f"[bench] RESULT num_envs={n} image={shape} step_ms={step * 1000:.1f} env_steps_per_s={n / step:.0f} "
        f"render_ms={render * 1000:.1f} ({', '.join(f'{k} {v * 1000:.1f}' for k, v in cameras.items())}) "
        f"gpu_used_gib={used:.1f} of {total:.0f} torch_peak_gib={torch.cuda.max_memory_allocated() / 2**30:.1f}",
        flush=True,
    )
    import os

    os._exit(0)  # large envs hang in teardown


if __name__ == "__main__":
    main()
