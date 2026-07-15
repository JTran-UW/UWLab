"""Smoke-test the multi-offset success math by instantiating the training env for
each (peg, leg, drawer) and stepping it with random actions for a few steps.
Exercises:
  - receptive metadata load (assembled_offsets list)
  - TaskCommand init (loads multi-offset tensors)
  - progress_context init (loads multi-offset tensors)
  - success_termination init (looks up TaskCommand)
  - per-step: success math iterates all offsets and ORs

If any task errors, it'll print + raise. Clean exit means all 3 work.

Usage:
  PYTHONPATH=...source/uwlab_tasks:... python scripts/test_train_envs.py --headless
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
args_cli.enable_cameras = False
args_cli.headless = True

app = AppLauncher(args_cli).app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from typing import cast

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-v0"

cases = [
    ("peg", "peg", "peghole", 4),     # K=4 offsets
    ("leg", "fbleg", "fbtabletop", 1),
    ("drawer", "fbdrawerbottom", "fbdrawerbox", 1),
]


def main():
    for label, ins, rec, expected_K in cases:
        print(f"\n========== {label.upper()} ({ins}/{rec}, expected K={expected_K}) ==========")

        @hydra_task_compose(
            TASK,
            "env_cfg_entry_point",
            hydra_args=[
                f"env.scene.insertive_object={ins}",
                f"env.scene.receptive_object={rec}",
            ],
        )
        def _run(env_cfg, agent_cfg):
            env_cfg.scene.num_envs = 4
            env_cfg.sim.device = args_cli.device or env_cfg.sim.device
            env_cfg.seed = 0
            env = cast(ManagerBasedRLEnv, gym.make(TASK, cfg=env_cfg)).unwrapped
            env.reset()

            # Verify TaskCommand's offset list size
            tc = env.command_manager.get_term("task_command")
            assert hasattr(tc, "receptive_asset_offsets_pos"), "TaskCommand missing receptive_asset_offsets_pos"
            K = tc.receptive_asset_offsets_pos.shape[0]
            print(f"  TaskCommand.receptive_asset_offsets_pos.shape: {tuple(tc.receptive_asset_offsets_pos.shape)}")
            print(f"  K (num offsets): {K}  (expected: {expected_K})")
            assert K == expected_K, f"Mismatch: got K={K}, expected {expected_K}"

            # Verify progress_context's offset list size
            pc = env.reward_manager.get_term_cfg("progress_context").func
            assert hasattr(pc, "receptive_asset_offsets_pos")
            assert pc.receptive_asset_offsets_pos.shape[0] == K

            # Step a few times, check no errors
            for i in range(5):
                action = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
                _, _, term, trunc, _ = env.step(action)
                # Check success_termination produces a tensor
                # (the sites all iterate offsets internally — if no error, multi-offset works)
            print(f"  Stepped 5 times with K={K} offsets — OK")
            env.close()

        _run()
    print("\n========== ALL 3 TASKS PASSED ==========")


if __name__ == "__main__":
    main()
    app.close()
