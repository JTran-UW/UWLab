# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

_RUNNER = f"{agents.__name__}.rsl_rl_cfg:PushT_PPORunnerCfg"
_FLASHSAC = f"{agents.__name__}.flashsac_cfg:PushT_FlashSACRunnerCfg"
_FLASHSAC_DEMO = f"{agents.__name__}.flashsac_cfg:PushT_FlashSACDemoRunnerCfg"
_FASTSAC = f"{agents.__name__}.sac_cfg:PushT_FastSACRunnerCfg"
_FLASHSAC_RGB = f"{agents.__name__}.flashsac_cfg:PushT_FlashSACRGBRunnerCfg"
_FLASHSAC_STATE_MS = f"{agents.__name__}.flashsac_cfg:PushT_FlashSACStateMSRunnerCfg"
_FLASHSAC_RGB_PRIV = f"{agents.__name__}.flashsac_cfg:PushT_FlashSACRGBPrivStateRunnerCfg"

gym.register(
    id="PushT-PandaStick-JointDeltaPos-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PandaStickPushTEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
        "flashsac_cfg_entry_point": _FLASHSAC,
        "flashsac_demo_cfg_entry_point": _FLASHSAC_DEMO,
    },
)

gym.register(
    id="PushT-PandaStick-JointDeltaPos-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PandaStickPushTPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
        "flashsac_cfg_entry_point": _FLASHSAC,
        "flashsac_demo_cfg_entry_point": _FLASHSAC_DEMO,
    },
)

gym.register(
    id="PushT-PandaStick-EEDeltaPose-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PandaStickEEPushTEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
        "flashsac_cfg_entry_point": _FLASHSAC,
        "flashsac_demo_cfg_entry_point": _FLASHSAC_DEMO,
    },
)

gym.register(
    id="PushT-PandaStick-EEDeltaPose-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PandaStickEEPushTPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
        "flashsac_cfg_entry_point": _FLASHSAC,
        "flashsac_demo_cfg_entry_point": _FLASHSAC_DEMO,
    },
)

gym.register(
    id="PushT-PandaStick-EEDeltaPose-ManiSkill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PandaStickManiSkillPushTEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
        "flashsac_cfg_entry_point": _FLASHSAC,
        "flashsac_demo_cfg_entry_point": _FLASHSAC_DEMO,
        "rsl_rl_sac_cfg_entry_point": _FASTSAC,
        "flashsac_state_ms_cfg_entry_point": _FLASHSAC_STATE_MS,
    },
)

gym.register(
    id="PushT-PandaStick-EEDeltaPose-ManiSkill-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PandaStickManiSkillPushTPlayEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
        "flashsac_cfg_entry_point": _FLASHSAC,
        "flashsac_demo_cfg_entry_point": _FLASHSAC_DEMO,
        "rsl_rl_sac_cfg_entry_point": _FASTSAC,
        "flashsac_state_ms_cfg_entry_point": _FLASHSAC_STATE_MS,
    },
)

gym.register(
    id="PushT-PandaStick-EEDeltaPose-ManiSkill-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PandaStickManiSkillRGBPushTEnvCfg",
        "rsl_rl_sac_cfg_entry_point": _FASTSAC,
        "flashsac_cfg_entry_point": _FLASHSAC_RGB,
        "flashsac_rgb_privstate_cfg_entry_point": _FLASHSAC_RGB_PRIV,
    },
)

gym.register(
    id="PushT-PandaStick-EEDeltaPose-ManiSkill-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PandaStickManiSkillRGBPushTPlayEnvCfg",
        "rsl_rl_sac_cfg_entry_point": _FASTSAC,
        "flashsac_cfg_entry_point": _FLASHSAC_RGB,
    },
)
