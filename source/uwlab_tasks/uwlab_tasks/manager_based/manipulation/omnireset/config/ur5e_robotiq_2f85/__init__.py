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
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
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
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)


# Depth DAgger env: distill state RL expert → depth CNN student
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerRunnerCfg",
    },
)

# Depth DAgger with fixed per-env student/teacher pool split
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Split-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerSplitRunnerCfg",
    },
)

# RGB DAgger with fixed per-env student/teacher pool split
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-Split-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85RgbDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Rgb_DAggerSplitRunnerCfg",
    },
)

# 1-camera ResNet18 variants (depth + rgb)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-ResNet18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerResNet18RunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-ResNet18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85RgbDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Rgb_DAggerResNet18RunnerCfg",
    },
)

# Depth ResNet18 + aux pose head (ablation: does aux loss bootstrap CNN?)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Aux-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerAuxRunnerCfg",
    },
)

# Depth ResNet18 + ImageNet pretrained (ablation: does ImageNet prior help depth?)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerPretrainedRunnerCfg",
    },
)

# 2-camera depth DAgger (side + front)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-2Cam-Split-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAgger2CamCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAgger2CamSplitRunnerCfg",
    },
)

# 2-camera RGB DAgger (side + front)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-2Cam-Split-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85RgbDAgger2CamCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Rgb_DAgger2CamSplitRunnerCfg",
    },
)

# 2-camera RGB DAgger (side + front) + ImageNet-pretrained ResNet18
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-2Cam-Pretrained-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85RgbDAgger2CamCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Rgb_DAgger2CamPretrainedRunnerCfg",
    },
)

# 2-camera RGB DAgger (side + wrist) + ImageNet-pretrained ResNet18
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-WristSide-Pretrained-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85RgbDAggerWristSideCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Rgb_DAggerWristSidePretrainedRunnerCfg",
    },
)

# 2-camera depth DAgger (side + wrist) + ImageNet-pretrained ResNet18
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSideCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerWristSidePretrainedRunnerCfg",
    },
)

# 1-cam depth + ImageNet ResNet18 + aux head @ aux_coeff=10.0 (BC:aux parity)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-Aux10x-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerPretrainedAux10xRunnerCfg",
    },
)

# 1-cam depth + ImageNet ResNet18 + recurrent (LSTM) student
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-Recurrent-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerRecurrentCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerPretrainedRecurrentRunnerCfg",
    },
)

# 1-cam depth + ImageNet ResNet18 + DEXTRAH-style weighted-L2 loss on (μ, σ)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-Pretrained-Weighted-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerPretrainedWeightedRunnerCfg",
    },
)

# State→state DAgger sanity check: same obs for student and teacher (no cameras)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-DAgger-Split-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.state_dagger_cfg:Ur5eRobotiq2f85StateDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:State_DAggerSplitRunnerCfg",
    },
)

# State→state DAgger with DEXTRAH-matched cadence (1 env step → 1 gradient step)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-DAgger-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.state_dagger_cfg:Ur5eRobotiq2f85StateDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:State_DAggerFastRunnerCfg",
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

# OOD (out-of-distribution) RGB environments
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
