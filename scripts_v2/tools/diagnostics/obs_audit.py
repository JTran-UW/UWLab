# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-term observation statistics for the omnireset policy/critic groups.

gSDE's exploration std is ``sqrt(phi(s)^2 @ exp(log_std)^2)``, so it scales with
the policy's penultimate activations, which are driven entirely by the
observations. The 3.0 policy moves 2-2.5x less than 2.x at matched iterations
(measured on action_magnitude / action_rate / joint_vel penalties). A degenerate
observation term -- constant, all-zero, or non-finite -- would shrink ``phi`` and
throttle exploration exactly that way, while still letting tasks that start in
favourable states succeed.

Slices each observation group by term using the ObservationManager's own
dimensions, so the numbers are the ones the policy actually receives.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=40)
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
act = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)

acc: dict[str, list[torch.Tensor]] = {}
for _ in range(args_cli.steps):
    act.uniform_(-1.0, 1.0)  # vary the state so constant terms are unambiguous
    env.step(act)
    obs = om.compute()
    for grp in ("policy", "critic"):
        if grp in obs:
            acc.setdefault(grp, []).append(obs[grp].detach().float())

print("\n" + "=" * 100)
print("OBSERVATION TERM AUDIT")
print("=" * 100)
print(f"  {'group':>7} {'term':>42} {'dim':>4} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}  flag")

degenerate = []
for grp, chunks in acc.items():
    x = torch.cat(chunks, dim=0)  # (steps*envs, group_dim)
    names = om.active_terms[grp]
    dims = om.group_obs_term_dim[grp]
    off = 0
    for name, d in zip(names, dims):
        width = int(d[0]) if isinstance(d, (tuple, list)) else int(d)
        sl = x[:, off : off + width]
        off += width
        flag = ""
        if not torch.isfinite(sl).all():
            flag = "NON-FINITE"
        elif sl.std().item() < 1e-9:
            flag = "CONSTANT"
        elif sl.abs().max().item() == 0.0:
            flag = "ALL-ZERO"
        if flag:
            degenerate.append((grp, name, flag))
        print(
            f"  {grp:>7} {name[:42]:>42} {width:>4} {sl.mean().item():>10.4f} {sl.std().item():>10.4f} "
            f"{sl.min().item():>10.3f} {sl.max().item():>10.3f}  {flag}"
        )
    print(f"  {'':>7} {'-- group total --':>42} {off:>4} {x.mean().item():>10.4f} {x.std().item():>10.4f} "
          f"{x.min().item():>10.3f} {x.max().item():>10.3f}")

print("-" * 100)
print(f"  degenerate terms: {len(degenerate)}")
for g, n, f in degenerate:
    print(f"    [{g}] {n}: {f}")
print("=" * 100 + "\n", flush=True)
app.close()
