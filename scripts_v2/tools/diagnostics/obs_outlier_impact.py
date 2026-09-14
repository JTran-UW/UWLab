# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How much do fallen-peg outliers inflate observation normalization?

At iteration 150 the 3.0 policy's |action| is ~1.4x smaller than 2.x's while the
position-based reward terms are identical, so the gap sits in ``||phi||`` -- the
penultimate activations, which are driven by *normalized* observations.

Policy observations contain insertive-object poses reaching -40 m (pegs that fell
out of the workspace; canonical omnireset has no fall termination). EmpiricalNormalization
divides by a running std, so a heavy outlier tail inflates that std and squashes
typical states toward zero, shrinking ``phi`` and therefore the gSDE exploration
std, which is ``sqrt(phi^2 @ exp(log_std)^2)``.

This measures the per-dimension raw std of the policy observation with and without
the fallen-peg envs, and reports the resulting change in normalized magnitude.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--fall_z", type=float, default=-0.5, help="peg world z below this counts as fallen")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

import sys  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=cfg).unwrapped
env.reset()
om = env.observation_manager
peg = env.scene.rigid_objects["insertive_object"]

act = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
obs_all, fallen_all = [], []
for _ in range(args_cli.steps):
    act.uniform_(-1.0, 1.0)
    env.step(act)
    obs_all.append(om.compute()["policy"].detach().float())
    fallen_all.append(peg.data.root_pos_w[:, 2] < args_cli.fall_z)

X = torch.cat(obs_all, dim=0)          # (steps*envs, dim)
fallen = torch.cat(fallen_all, dim=0)  # (steps*envs,)
keep = ~fallen

print("\n" + "=" * 84)
print("OBSERVATION OUTLIER IMPACT ON NORMALIZATION")
print("=" * 84)
print(f"  samples={X.shape[0]}  obs_dim={X.shape[1]}  fallen (peg z < {args_cli.fall_z}): "
      f"{fallen.float().mean().item()*100:.2f}%")

std_all = X.std(dim=0)
std_keep = X[keep].std(dim=0) if keep.any() else std_all
mean_all = X.mean(dim=0)
mean_keep = X[keep].mean(dim=0) if keep.any() else mean_all

ratio = (std_all / (std_keep + 1e-9))
print(f"\n  per-dim raw std   : with outliers mean={std_all.mean():.4f}  max={std_all.max():.4f}")
print(f"                      without       mean={std_keep.mean():.4f}  max={std_keep.max():.4f}")
print(f"  inflation ratio   : mean={ratio.mean():.3f}  median={ratio.median():.3f}  max={ratio.max():.3f}")
print(f"  dims inflated >2x : {(ratio > 2).sum().item()} / {ratio.numel()}")

# Effect on what the network actually sees: normalized magnitude of typical (non-fallen) states
z_infl = ((X[keep] - mean_all) / (std_all + 1e-9))
z_clean = ((X[keep] - mean_keep) / (std_keep + 1e-9))
print(f"\n  normalized |obs| for NON-fallen states:")
print(f"     using outlier-inflated stats : {z_infl.abs().mean().item():.4f}")
print(f"     using clean stats            : {z_clean.abs().mean().item():.4f}")
shrink = z_clean.abs().mean().item() / max(z_infl.abs().mean().item(), 1e-9)
print(f"     shrinkage factor             : {shrink:.3f}x")
print("\n  gSDE std scales ~linearly with ||phi||, which scales with normalized |obs|.")
if shrink > 1.25:
    print(f"  => Outliers shrink typical normalized observations by {shrink:.2f}x, which would")
    print(f"     depress exploration by a comparable factor. Consistent with the ~1.4x deficit.")
else:
    print(f"  => Outliers do NOT meaningfully shrink normalized observations ({shrink:.2f}x).")
    print("     This mechanism does not explain the exploration deficit.")
print("=" * 84 + "\n", flush=True)
app.close()
