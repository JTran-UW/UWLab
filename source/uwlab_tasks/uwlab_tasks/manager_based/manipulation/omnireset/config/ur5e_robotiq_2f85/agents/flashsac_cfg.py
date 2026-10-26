# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from uwlab_tasks.manager_based.manipulation.push_t.config.panda_stick.agents.flashsac_cfg import (
    PushT_FlashSACRGBRunnerCfg,
    PushT_FlashSACStateMSRunnerCfg,
)

# per action dim: the Newton symmetric PPO expert's p99 |action| (x y z rx ry rz gripper), so ~1% of it clips
EXPERT_P99_ACTION_BOUNDS = (4.77, 8.78, 8.01, 12.78, 14.39, 9.16, 8.89)
# the same for the sysid-finetuned PPO expert (model_2500) under the eval OSC action of the max-sysid tasks
SYSID_EXPERT_P99_ACTION_BOUNDS = (17.04, 16.88, 27.88, 16.03, 20.52, 6.46, 11.53)


@configclass
class OmniReset_FlashSACStateRunnerCfg(PushT_FlashSACStateMSRunnerCfg):
    """Control: the PushT FlashSAC state recipe unchanged except gamma 0.99, 5% expert demos and per-dim action bounds.

    Set ``agent.algorithm.expert_buffer_path``; collect it with scripts/tools/collect_flashsac_expert_buffer.py.
    """

    experiment_name = "ur5e_robotiq_2f85_omnireset_flashsac_state"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    action_bound_scale = EXPERT_P99_ACTION_BOUNDS

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.gamma = 0.99
        self.algorithm.expert_ratio = 0.05


@configclass
class OmniReset_FlashSACDinoRunnerCfg(OmniReset_FlashSACStateRunnerCfg):
    """The state recipe on frozen DINOv2 camera features + 3-frame ``vision_state``; 4096 envs, 1.31M fp16 buffer."""

    experiment_name = "ur5e_robotiq_2f85_omnireset_flashsac_dino"
    obs_groups = {"actor": ["dino", "vision_state"], "critic": ["dino", "vision_state"]}

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.replay_buffer_size = 1_310_720
        self.algorithm.buffer_obs_dtype = "float16"


@configclass
class OmniReset_FlashSACRGBRunnerCfg(PushT_FlashSACRGBRunnerCfg):
    """The PushT FlashSAC-RGB recipe with the state control's changes: gamma 0.99, 5% expert demos, per-dim bounds.

    Agent input: the stacked front + wrist cameras (``image``, stored as uint8) and the 25-D ``vision_state``.
    """

    experiment_name = "ur5e_robotiq_2f85_omnireset_flashsac_rgb"
    obs_groups = {"actor": ["image", "vision_state"], "critic": ["image", "vision_state"]}
    action_bound_scale = EXPERT_P99_ACTION_BOUNDS

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.gamma = 0.99
        self.algorithm.expert_ratio = 0.05
        self.algorithm.image_group = "image"
        self.algorithm.state_group = "vision_state"
        self.algorithm.buffer_obs_dtype = {"image": "uint8"}


@configclass
class OmniReset_FlashSACRGB4096RunnerCfg(OmniReset_FlashSACRGBRunnerCfg):
    """Best vision recipe (2026-10-22 sweep ``fsac_rgb_g_b1024_u16_s1``, ~97% full-reset success in ~1.1 h on one L40):
    the RGB recipe at 4096 envs with a 1.31M replay buffer; batch 1024, 16 updates per 4096-env iteration.

    Run with ``--num_envs 4096 --enable_cameras presets=newton_mjwarp,newton_renderer
    env.sim.physics.solver_cfg.njmax=400 env.scene.insertive_object=peg env.scene.receptive_object=peghole
    env.scene.front_camera.spawn.focal_length=26.4`` and
    ``agent.algorithm.expert_buffer_path=<omnireset_newton_symppo_flashsac_rgb.pt>``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.replay_buffer_size = 1_310_720


@configclass
class OmniReset_FlashSACRGB4096SysidRunnerCfg(OmniReset_FlashSACRGB4096RunnerCfg):
    """The 4096-env pixel recipe at max sysid: action bounds from the sysid-finetuned expert."""

    action_bound_scale = SYSID_EXPERT_P99_ACTION_BOUNDS


@configclass
class OmniReset_FlashSACDinoSysidRunnerCfg(OmniReset_FlashSACDinoRunnerCfg):
    """The DINO recipe at max sysid: action bounds from the sysid-finetuned expert."""

    action_bound_scale = SYSID_EXPERT_P99_ACTION_BOUNDS
