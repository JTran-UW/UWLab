# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset states tasks for IsaacLab."""

import gymnasium as gym

from . import agents

# Register the partial assemblies environment
gym.register(
    id="OmniReset-PartialAssemblies-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.partial_assemblies_cfg:PartialAssembliesCfg"},
    disable_env_checker=True,
)

# Register the grasp sampling environment
gym.register(
    id="OmniReset-Robotiq2f85-GraspSampling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.grasp_sampling_cfg:Robotiq2f85GraspSamplingCfg"},
    disable_env_checker=True,
)

# Register reset states environments
gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-Reaching-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ReachingResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectPartiallyAssembledEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectPartiallyAssembledEEGraspedResetStatesCfg"},
)

# Register SysID env
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sysid_cfg:SysidEnvCfg"},
)

# Register Camera Alignment env
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-CameraAlign-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.camera_align_cfg:CameraAlignEnvCfg"},
)

# Register RL state environments
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Reaching-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainReachingCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Reaching-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainReachingCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-Reaching-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainReachingDepthCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Reaching-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainReachingGrayscaleCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Reaching-FastRender-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainReachingGrayscaleFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)



gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-Success-Termination-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-Success-Termination-Sparse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-Success-Termination-Sparse-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingSparseCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-No-Privileged-Obs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainNoPrivilegedObsCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Success-Termination-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainSuccessTerminationCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-No-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainNoDRCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-No-DR-6bdbe5e-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainNoDR_6bdbe5e_Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Finetune-Suboptimal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainFinetuneSuboptimalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Finetune-Dynamics-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainFinetuneDynamicsCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainEasyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainEasyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-OffPolicy-No-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainEasyNoDRCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Finetune-Reward-Scaling-Success-Termination-Sparse-No-Privileged-Obs-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


# As above but nothing ramps: fixed/maximal sysid + OSC gains, no curriculum, and resets drawn
# uniformly from all four distributions instead of the curriculum-ramped schedule.
gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Finetune-Reward-Scaling-"
        "Success-Termination-Sparse-No-Privileged-Obs-Full-Reset-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsFullResetCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


# As above but the robot USD + sysid metadata.yaml come from yandabao/uwlab-assets.
gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Finetune-Reward-Scaling-"
        "Success-Termination-Sparse-No-Privileged-Obs-Full-Reset-Yanda-Sysid-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsFullResetYandaSysidCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


# As above but without the success termination, matching the
# ...-No-Success-Termination-Yanda-Sysid data-collection task. No curriculum.
gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Finetune-Reward-Scaling-"
        "Sparse-No-Privileged-Obs-No-Success-Termination-Full-Reset-Yanda-Sysid-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSparseNoPrivilegedObsNoSuccessTerminationFullResetYandaSysidCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


# As above but success ends the episode as a time_out-flagged term (bootstrapped truncation
# online), matching buffers recorded with play.py --success_to_truncation.
gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Finetune-Reward-Scaling-"
        "Sparse-No-Privileged-Obs-Success-Truncation-Full-Reset-Yanda-Sysid-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSparseNoPrivilegedObsSuccessTruncationFullResetYandaSysidCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Finetune-Reward-Scaling-Success-Termination-Sparse-No-Privileged-Obs-Play-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsEvalCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Reward-Scaling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Reward-Scaling-Sparse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingSparseCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-Success-Termination-Sparse-No-Privileged-Obs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-Sparse-No-Privileged-Obs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSparseNoPrivilegedObsCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-Success-Termination-Sparse-No-Privileged-Obs-Dynamics-Gap-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsPegMassGapCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-Success-Termination-"
        "Sparse-No-Privileged-Obs-Dynamics-Gap-Full-Reset-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSuccessTerminationSparseNoPrivilegedObsPegMassGapFullResetCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-"
        "Sparse-No-Privileged-Obs-Full-Reset-GC-AutoReset-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCGCAutoResetFinetuneNoGapCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-"
        "Sparse-No-Privileged-Obs-Dynamics-Gap-Full-Reset-GC-AutoReset-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCGCAutoResetFinetuneCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-"
        "Sparse-No-Privileged-Obs-Dynamics-Gap-Full-Reset-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSparseNoPrivilegedObsPegMassGapFullResetCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-"
        "Sparse-No-Privileged-Obs-Dynamics-Gap-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCTrainRewardScalingSparseNoPrivilegedObsPegMassGapCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-Success-Termination-"
        "Sparse-No-Privileged-Obs-Play-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Reward-Scaling-Success-Termination-"
        "Sparse-No-Privileged-Obs-Dynamics-Gap-Play-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingSuccessTerminationSparseNoPrivilegedObsPegMassGapCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-DataCollection-Reward-Scaling-Sparse-No-Privileged-Obs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCDataCollectionRewardScalingSparseNoPrivilegedObsCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


# As above but without the success termination: recorded episodes run past insertion to time-out.
gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-DataCollection-Reward-Scaling-Sparse-No-Privileged-Obs-No-Success-Termination-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCDataCollectionRewardScalingSparseNoPrivilegedObsNoSuccessTerminationCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


# Stage-2 counterpart of the above: same recorded groups, but finetune dynamics (explicit actuator,
# fixed/maximal sysid + OSC gains, no curriculum) so the buffer matches what the finetune task trains
# under. Feeds the ...-Finetune-Reward-Scaling-Success-Termination-Sparse-No-Privileged-Obs task.
gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-DataCollection-Finetune-Reward-Scaling-"
        "Success-Termination-Sparse-No-Privileged-Obs-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCDataCollectionFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Yanda-sysid finetune data collection WITH success termination: episodes end at insertion, so the
# buffer is not dominated by post-success dwell samples. Record with play.py
# --success_to_truncation so the success done is stored as a truncation (bootstrapped).
gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-DataCollection-Finetune-Reward-Scaling-"
        "Success-Termination-Sparse-No-Privileged-Obs-Yanda-Sysid-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCDataCollectionFinetuneRewardScalingSuccessTerminationSparseNoPrivilegedObsYandaSysidCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


# As above but without the success termination and with the Yanda-sysid robot USD, matching the
# ...-Full-Reset-Yanda-Sysid finetune training dynamics.
gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-DataCollection-Finetune-Reward-Scaling-"
        "Sparse-No-Privileged-Obs-No-Success-Termination-Yanda-Sysid-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCDataCollectionFinetuneRewardScalingSparseNoPrivilegedObsNoSuccessTerminationYandaSysidCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Reward-Scaling-No-Privileged-Obs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingNoPrivilegedObsCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-No-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalNoDRCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-OffPolicy-No-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalNoDRCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-No-DR-6bdbe5e-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalNoDR_6bdbe5e_Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-OffPolicy-No-DR-6bdbe5e-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalNoDR_6bdbe5e_Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Easy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalEasyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Easy-No-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalEasyNoDRCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Finetune-Dynamics-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalFinetuneDynamicsCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


# RGB environments for data collection and evaluation
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-DataCollection-FastRender-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


# Asymmetric grayscale: vision+proprio actor, non-privileged full-state critic.
# DataCollection variants keep the expert's state groups so a PPO checkpoint loads; the Train
# variants drop them. FastRender variants disable photorealism and should be paired with each other.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-Asymmetric-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCDataCollectionDepthAsymmetricCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-DataCollection-FastRender-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscaleAsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainGrayscaleAsymmetricCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-Asymmetric-FastRender-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamAsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-NoHist-Asymmetric-FastRender-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistAsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-NoHist-LowRes-Asymmetric-FastRender-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistLowResAsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-NoHist-LowRes-Obs32-"
        "Asymmetric-DataCollection-FastRender-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscale2CamNoHistLowResObs32AsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-NoHist-LowRes-Obs32-Asymmetric-FastRender-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistLowResObs32AsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-NoHist-LowRes-Obs64-Asymmetric-FastRender-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistLowResObs64AsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-NoHist-LowRes-Obs64-Asymmetric-DataCollection-FastRender-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscale2CamNoHistLowResObs64AsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-NoHist-LowRes-"
        "Asymmetric-DataCollection-FastRender-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscale2CamNoHistLowResAsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-NoHist-Obs126-Asymmetric-FastRender-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCTrainGrayscale2CamNoHistObs126AsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-NoHist-Obs126-"
        "Asymmetric-DataCollection-FastRender-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCDataCollectionGrayscale2CamNoHistObs126AsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-Asymmetric-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainDepthAsymmetricCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-FastRender-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainGrayscaleAsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

# Play/eval counterparts of the two asymmetric grayscale training envs. Same observation groups so
# checkpoints load as-is; only the first-episode stagger termination is dropped. Match the render
# variant to the one the checkpoint was trained on.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalGrayscaleAsymmetricCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-FastRender-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalGrayscaleAsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id=(
        "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-2Cam-NoHist-LowRes-Obs32-"
        "Asymmetric-FastRender-Play-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCEvalGrayscale2CamNoHistLowResObs32AsymmetricFastRenderCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

# OOD (out-of-distribution) RGB environments
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-FastSAC-Finetune-State-Expert-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85DataCollectionRGBFastSACFinetuneStateExpertCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)


gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-FastSAC-State-Expert-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85DataCollectionRGBFastSACStateExpertCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-OOD-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCOODCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-OOD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85EvalRGBRelCartesianOSCOODCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GCMRM-Visualization-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85GCMRMVisualizationCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GC-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCGCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GC-Grasped-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCGCGraspedTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCGCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GC-Grasped-Resting-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCGCGraspedRestingTrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GC-Grasped-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCGCGraspedPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GC-Grasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCGCGraspedTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GC-Intermediate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCGCIntermediateTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GC-Intermediate-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCGCIntermediateTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_FastSACRunnerCfg",
    },
)
