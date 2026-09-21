# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""


from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import importlib
import math
import torch
from types import SimpleNamespace

import pytest
from env_test_utils import _run_environments, setup_environment
from isaaclab.managers import SceneEntityCfg

import uwlab_tasks  # noqa: F401


@pytest.mark.parametrize("num_envs, device", [(32, "cuda"), (1, "cuda")])
@pytest.mark.parametrize("task_name", setup_environment(include_play=False, factory_envs=False, multi_agent=False))
@pytest.mark.isaacsim_ci
def test_environments(task_name, num_envs, device):
    # run environments without stage in memory
    _run_environments(task_name, device, num_envs, create_stage_in_memory=False)


@pytest.mark.parametrize("task_family", ["omnireset", "factory_extension"])
@pytest.mark.parametrize("body_ids", [slice(None), [1]])
@pytest.mark.parametrize("stationary", [True, False])
@pytest.mark.parametrize("translated", [True, False])
@pytest.mark.isaacsim_ci
def test_asset_link_velocity_frame_transform(task_family, body_ids, stationary, translated):
    mdp = importlib.import_module(f"uwlab_tasks.manager_based.manipulation.{task_family}.mdp.observations")
    positions = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 7.0, 9.0], [25.0, -6.0, 0.5]])
    if not translated:
        positions.zero_()
    half_sqrt = math.sqrt(0.5)
    quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half_sqrt, half_sqrt], [1.0, 0.0, 0.0, 0.0]])
    linear = torch.tensor([[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]]).repeat(3, 1, 1)
    angular = torch.tensor([[[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]]).repeat(3, 1, 1)
    if stationary:
        linear.zero_()
        angular.zero_()
    target = SimpleNamespace(
        data=SimpleNamespace(
            body_lin_vel_w=SimpleNamespace(torch=linear), body_ang_vel_w=SimpleNamespace(torch=angular)
        )
    )
    root = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=positions), root_quat_w=SimpleNamespace(torch=quaternions)
        )
    )
    env = SimpleNamespace(scene={"target": target, "robot": root})
    actual = mdp.asset_link_velocity_in_root_asset_frame(env, SceneEntityCfg("target", body_ids=body_ids))
    index = 0 if isinstance(body_ids, slice) else body_ids[0]
    linear_selected, angular_selected = linear[:, index], angular[:, index]
    expected = torch.cat([linear_selected, angular_selected], dim=-1)
    expected[1] = expected[1, [1, 0, 2, 4, 3, 5]] * torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    expected[2] *= torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
