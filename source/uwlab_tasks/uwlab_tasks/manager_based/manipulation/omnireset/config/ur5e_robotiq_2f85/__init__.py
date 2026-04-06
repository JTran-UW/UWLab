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
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.partial_assemblies_cfg:PartialAssembliesCfg"},
    disable_env_checker=True,
)

# Register the grasp sampling environment
gym.register(
    id="OmniReset-Robotiq2f85-GraspSampling-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.grasp_sampling_cfg:Robotiq2f85GraspSamplingCfg"},
    disable_env_checker=True,
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-GripperClose-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.gripper_close_cfg:Ur5eRobotiq2f85GripperCloseCfg"},
    disable_env_checker=True,
)

# Register reset states environments
gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectPartiallyAssembledEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ObjectPartiallyAssembledEEGraspedResetStatesCfg"},
)

# Zero-G approach reset state recording
gym.register(
    id="OmniReset-UR5eRobotiq2f85-ZeroGPartialAssembly-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ZeroGPartialAssemblyResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ZeroGAnywhere-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ZeroGAnywhereResetStatesCfg"},
)

# Register SysID env
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-Sysid-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sysid_cfg:SysidEnvCfg"},
)

# Register Camera Alignment env
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-CameraAlign-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.camera_align_cfg:CameraAlignEnvCfg"},
)

# Register RL state environments
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Gravity trick: baseline (success=10.0, fail=-1.0)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GravityTrick-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:Ur5eRobotiq2f85RelCartesianOSCGravityTrickTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# Gravity trick: PA EE lerp (0.9, 1.0)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GravityTrick-PALerp-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:Ur5eRobotiq2f85RelCartesianOSCGravityTrickPALerpTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# Gravity trick: success_reward weight = 50
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GravityTrick-Success50-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:Ur5eRobotiq2f85RelCartesianOSCGravityTrickSuccess50TrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# Gravity trick: uniform random gravity (no curriculum)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GravityTrick-RandomGrav-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:Ur5eRobotiq2f85RelCartesianOSCGravityTrickRandomGravTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# Gravity trick: contact rewards (sanity check — reach + contact + contact-gated dense)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-GravityTrick-Contact-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:Ur5eRobotiq2f85RelCartesianOSCGravityTrickContactTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# ZeroG state-based ablations (load pre-recorded ZeroG states + gravity curriculum)
# A: baseline 50-50 sampling, sparse rewards
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-Baseline-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:Ur5eRobotiq2f85RelCartesianOSCZeroGBaselineTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# B: GPS over all 20K states
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-GPS-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:Ur5eRobotiq2f85RelCartesianOSCZeroGGPSTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# C: 50-50 + dense rewards curricuumed out over time
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-Dense-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:Ur5eRobotiq2f85RelCartesianOSCZeroGDenseTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


# Single-task state baseline ablation (no DR, no history, symmetric obs)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Baseline-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCStateBaselineTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Single-task pointcloud ablation — wrist frame
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-PointCloud-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCPointCloudTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Single-task pointcloud ablation — robot base frame
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-PointCloud-BaseFrame-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCPointCloudBaseFrameTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Multi-task RL state environments
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multitask_cfg:Ur5eRobotiq2f85RelCartesianOSCMultiTaskTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-Simplified-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multitask_cfg:Ur5eRobotiq2f85RelCartesianOSCMultiTaskSimplifiedTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-Curriculum-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.multitask_cfg:Ur5eRobotiq2f85RelCartesianOSCMultiTaskCurriculumTrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-Simplified-Curriculum-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.multitask_cfg:Ur5eRobotiq2f85RelCartesianOSCMultiTaskSimplifiedCurriculumTrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-PerStateCurriculum-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.multitask_cfg:Ur5eRobotiq2f85RelCartesianOSCMultiTaskPerStateCurriculumTrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-Simplified-PerStateCurriculum-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.multitask_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCMultiTaskSimplifiedPerStateCurriculumTrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Multi-task pointcloud — 128pt wrist frame + shared MLP encoder
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-PointCloud-128-SharedEncoder-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.multitask_cfg:Ur5eRobotiq2f85RelCartesianOSCMultiTaskPC128SharedEncTrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# Multi-task pointcloud — 128pt wrist frame + shared MLP encoder + per-state GPS curriculum
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-MultiTask-PointCloud-128-SharedEncoder-PerStateCurriculum-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.multitask_cfg:"
            "Ur5eRobotiq2f85RelCartesianOSCMultiTaskPC128SharedEncPerStateCurriculumTrainCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# Single-task pointcloud ablation — 128pt wrist frame + shared MLP encoder
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-PointCloud-128-SharedEncoder-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCPC128SharedEncTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# RGB environments for data collection and evaluation
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-Play-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)

# OOD (out-of-distribution) RGB environments
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-OOD-DataCollection-v0",
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
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
    entry_point="uwlab.envs:UWLabManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_rgb_cfg:Ur5eRobotiq2f85EvalRGBRelCartesianOSCOODCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_DAggerRunnerCfg",
    },
)
