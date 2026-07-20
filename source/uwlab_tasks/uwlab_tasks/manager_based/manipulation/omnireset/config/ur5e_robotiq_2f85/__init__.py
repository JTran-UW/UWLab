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

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-GripperClose-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.gripper_close_cfg:Ur5eRobotiq2f85GripperCloseCfg"},
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

# Zero-G approach reset state recording
gym.register(
    id="OmniReset-UR5eRobotiq2f85-ZeroGPartialAssembly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ZeroGPartialAssemblyResetStatesCfg"},
)

gym.register(
    id="OmniReset-UR5eRobotiq2f85-ZeroGAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:ZeroGAnywhereResetStatesCfg"},
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

# Register open-loop action-playback env (scripts_v2/tools/sim2real/playback_actions.py)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Playback-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.playback_cfg:Ur5eRobotiq2f85PlaybackCfg"},
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


# Diversity ablation: same as State-v0 but reset states are drawn by a deterministic
# interleaved round-robin over all reset types (MultiResetManager sampling_mode=
# "sequential") instead of i.i.d. random sampling. Holds the reset stream fixed so any
# behavioral diversity across seeds is attributable to random network init, not resets.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Sequential-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCTrainSequentialCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# Video-eval variants for the seed-diversity study: subclass the env each policy was
# trained on, but reset ONLY from ObjectAnywhereEEAnywhere (probs=[1.0]) and terminate
# on success. With a fixed --seed every seed of a configuration sees identical reset
# states. See video_eval_cfg.py and analysis/record_seed_videos.py.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-VideoEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.video_eval_cfg:StateVideoEvalCfg",
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
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Sequential-VideoEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.video_eval_cfg:StateSequentialVideoEvalCfg",
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
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-Uniform-VideoEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.video_eval_cfg:ZeroGScenePCUniformVideoEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
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
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-Train-VideoEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.video_eval_cfg:ZeroGScenePCSysidVideoEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ScenePC-VideoEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.video_eval_cfg:ScenePCVideoEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

# Point-cloud pre-training WITHOUT gravity curriculum: base Stage-1 train cfg
# (full DR + EXPLICIT actuator + normal gravity) with state obs replaced by 128-pt
# wrist-frame PCs (shared encoder). Decouples PC obs from the ZeroG recipe that all
# registered PC envs in gravity_cfg.py are bound to.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ScenePC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCPCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

# Point-cloud pre-training WITHOUT gravity curriculum: base Stage-1 train cfg
# (full DR + EXPLICIT actuator + normal gravity) with state obs replaced by 128-pt
# wrist-frame PCs (shared encoder). Decouples PC obs from the ZeroG recipe that all
# registered PC envs in gravity_cfg.py are bound to.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ScenePC-OffPolicy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCPCTrainRewardScalingSuccessTerminationCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePC_FastSACRunnerCfg",
    },
)
# Point-cloud pre-training WITHOUT gravity curriculum: base Stage-1 train cfg
# (full DR + EXPLICIT actuator + normal gravity) with state obs replaced by 128-pt
# wrist-frame PCs (shared encoder). Decouples PC obs from the ZeroG recipe that all
# registered PC envs in gravity_cfg.py are bound to.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ScenePC-OffPolicy-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCPCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePC_FastSACRunnerCfg",
    },
)

# Point-cloud pre-training WITHOUT gravity curriculum: base Stage-1 train cfg
# (full DR + EXPLICIT actuator + normal gravity) with state obs replaced by 128-pt
# wrist-frame PCs (shared encoder). Decouples PC obs from the ZeroG recipe that all
# registered PC envs in gravity_cfg.py are bound to.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ScenePC-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCPCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)


# Same as ScenePC-v0 but the scene pointcloud is expressed in the end-effector
# (wrist_3_link) frame instead of the robot base frame.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ScenePC-EECentric-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCPCEECentricTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

# ===========================================================================
# ZeroG (self-contained in gravity_cfg.py, GPS + truncated success)
# ===========================================================================

# ScenePC 512pt + GPS + terminate on success (single-task)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-GPS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGGPSScenePCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
    },
)

# State-only (no PC) + uniform routing (no GPS) — gravity-reduction ablations.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:StatePPORunnerCfg",
    },
)

# ScenePC + uniform routing — peg RL retrain to break yaw multimodality
# via PC obs (cylindrical peg PC ~ SO(2)-symmetric) + 4-canonical reward.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-Uniform-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGScenePCUniformTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

# Sim2Real DR variant of ScenePC-Uniform-v0: wide arm DR (0-1.5x sysid, friction
# included) + narrow contact DR (material/mass/gripper-actuator) on top of the
# same scene/obs/curriculum. Used to train teachers that transfer 0-shot to the
# real robot regardless of where its true sysid lands inside the wide DR window.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-Uniform-Sim2RealDR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGScenePCUniformSim2RealDRTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)


# Do not need to randomize over full dynamics space, since low gravity already provides "easy" dynamics for learning signal
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGScenePCSysidSim2RealTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

# Same as ZeroG-ScenePC-SysID-Train-v0 but the scene pointcloud is expressed in the
# end-effector (wrist_3_link) frame instead of the robot base frame.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-Train-EECentric-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGScenePCSysidSim2RealTrainEECentricCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

# we use these to finetune on the new sysid parameters
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGScenePCSysidFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGScenePCSysidSim2RealEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

# RMA: same Sysid Sim2Real dynamics, with an added privileged_rma obs group and
# a runner that trains a history-encoder transformer to match phi(privileged_rma).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-RMA-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rma_cfg:ZeroGScenePCSysidRMATrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCRMARunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-RMA-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rma_cfg:ZeroGScenePCSysidRMAEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCRMARunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ScenePC-Real-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ScenePCRealEvalTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

# Ablation: same as above but drops `prev_actions` from proprio.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-Uniform-NoPrevAct-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGScenePCUniformNoPrevActTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

# Same as ZeroG-State-v0 but adds arm sysid + OSC gain DR (widened friction)
# for sim-to-real-robust state teachers we'll DAgger from.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateSysidTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:StatePPORunnerCfg",
    },
)

# Leg-DR-isolation envs: Sysid env minus one DR term each. Used to identify
# which DR specifically prevents leg from training.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-Sysid-NoArmFric-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateSysidNoArmFricTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:StatePPORunnerCfg",
    },
)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-Sysid-NoOSCGain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateSysidNoOSCGainTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:StatePPORunnerCfg",
    },
)

# Long-term sim2sim alignment: ZeroG-State-Sysid plus the 9 BaseEventCfg DRs
# (mass + material + gripper) that depth-DAgger envs apply. Narrowed arm
# friction (0.8-1.2) and ZeroGAnywhere-only resets so DAgger envs become a
# strict SUBSET of this training distribution.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-Sysid-FullDR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateSysidFullDRTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:StatePPORunnerCfg",
    },
)

# DAgger-identical baseline: bit-for-bit ZeroGStateSysidTrainCfg, but obs group
# renamed `teacher` for State_DAggerFastRunnerCfg routing. Curriculum floor=1.0
# (full gravity from iter 0). If teacher rate << ~95% here, the gap is NOT in
# the DAgger scaffolding (cameras/curtains/event/actuator) — re-investigate.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-Sysid-DAggerIdentical-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateSysidDAggerIdenticalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:State_DAggerFastRunnerCfg",
    },
)

# Sim2sim debug variant: events swapped to FinetuneEvalEventCfg (matches
# 4Canonical's events base) on the ZeroG scene. Tests whether the events delta
# (sysid arm friction widening + GPS routing + ZeroGPartialAssembly resets)
# accounts for the teacher rate gap.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-Sysid-DAggerEvalEvents-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateSysidDAggerEvalEventsCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:State_DAggerFastRunnerCfg",
    },
)

# Sim2sim debug variant: V0 + DAgger scene additions (curtains + 2 cameras +
# wrist-cam pose reset event). Keeps ZeroG events/curriculum/terminations.
# Tests whether visual entities perturb physics enough to drop teacher rate.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-Sysid-DAggerWithScene-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateSysidDAggerWithSceneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:State_DAggerFastRunnerCfg",
    },
)

# State-only (no PC) + empirical GPS — GPS + gravity-reduction ablations.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-State-GPS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateGPSTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:StatePPORunnerCfg",
    },
)

# Multi-task ZeroG: peg + leg + ScenePC 512pt + GPS + terminate on success
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-ZeroG-MultiTask-GPS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGMultiTaskGPSScenePCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SharedEncoder128PPORunnerCfg",
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
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Reward-Scaling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85RelCartesianOSCEvalRewardScalingCfg",
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


# ============================================================================
# Depth DAgger envs — ImageNet-pretrained ResNet18 + DEXTRAH weighted-L2 loss
# (μ, σ). Wrist+side depth obs. 50-50 student/teacher split via Hydra override
# `agent.student_fraction=0.5` on the same task; default = 100/0 pure student.
# ============================================================================

# Base wrist+side weighted DAgger (used as parent of 4Canonical variants).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSideCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# 4-canonical peg DAgger paired with pat/gravity's 4-yaw symmetric state teacher.
# Runtime success math uses the cylindrical-symmetry OR via merged ProgressContext.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-4Canonical-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# A/B test env: 4-canonical with cameras + curtains removed. For diagnosing
# whether camera/curtain sim presence is the source of the teacher sim2sim gap
# (~0.25 in DAgger vs ~0.98 in training). Uses State_DAggerFastRunnerCfg
# (state-only MLP student over teacher obs) so no vision groups are required.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-4Canonical-NoCam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalNoCamCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:State_DAggerFastRunnerCfg",
    },
)

# Drawer variant of the 4-canonical pipeline (drawer has one assembled_offset
# so multi-offset code falls through to single-canonical path).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-4Canonical-Drawer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalDrawerCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# Lean 4-canonical DAgger envs: training events only (no BaseEventCfg DR
# superset). Use with existing peg_sysid_mean.pt / drawer_sysid_full.pt
# teachers — those trained on ZeroGGPSSysidEventCfg, so this env feeds them
# the same dynamics distribution. Sim2sim ladder (2026-04-29 commits) showed
# the standard 4Canonical's BaseEventCfg DRs (mass/material/gripper) drop
# teacher rate from 0.985 to 0.224.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-4Canonical-Lean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalLeanCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# Lean DAgger paired with the new ScenePC peg teacher (pat/dagger-symmetry).
# Tests whether yaw-symmetric teacher (trained on cylindrical-peg PC obs +
# 8-canonical reward) lifts peg DAgger past the ~50% ceiling.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-PCTeacher-Lean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherLeanCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# Same as above but with full sysid DR
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-PCTeacher-Lean-FullSysidDR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherSysidTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# RGB variant of the sysid PC-teacher DAgger env above (wrist + side, full visual DR).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-WristSide-Pretrained-Weighted-PCTeacher-FullSysidDR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgb_dagger_cfg:Ur5eRobotiq2f85RGBDAggerWristSidePCTeacherSysidTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:RGB_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# Direct JIT ScenePC-teacher eval: FinetuneEval dynamics (fixed sysid/OSC,
# ObjectAnywhereEEAnywhere resets) with PC obs only — no cameras.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PCTeacher-FinetuneEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgb_dagger_cfg:Ur5eRobotiq2f85PCTeacherFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:RGB_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# Evaling the PC Expert, debug eng
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PCTeacher-DebugEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgb_dagger_cfg:DebugEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:RGB_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# Eval cfg for students trained on the RGB PC-teacher sysid DAgger env above.
# Inherits Stage-2 finetune-eval semantics (fixed sysid + OSC gains, 10-consecutive
# success, EXPLICIT actuator) and layers on the wrist + side RGB cameras + matching
# obs groups so the vision student can run at inference.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-WristSide-Pretrained-Weighted-PCTeacher-FullSysidDR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgb_dagger_cfg:Ur5eRobotiq2f85RGBDAggerWristSidePCTeacherSysidEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:RGB_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# Same as above but student + teacher proprio both drop prev_actions
# (paired with Tillicum 107906, the NoPrevAct teacher).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-PCTeacher-Lean-NoPrevAct-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSidePCTeacherLeanNoPrevActCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# BCPPO peg env: same Lean scene/teacher/cameras, but runs PPO+BC instead of
# DAgger. Reward-bearing PPO loop with BC auxiliary toward JIT teacher
# pulls the depth student past the DAgger-only ceiling (~50% on peg).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-BCPPO-Lean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalLeanCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_BCPPORunnerCfg",
    },
)

# BCPPO peg env (Sysid-rewards): mirrors teacher's RL training cfg
# (ZeroGStateSysidTrainCfg) with sparse rewards + cameras. Avoids the
# dense-reward hacking pathology seen on Lean env.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-BCPPO-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85BCPPOSysidCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_BCPPORunnerCfg",
    },
)
# State→state BCPPO sanity check. Same Sysid env state PPO trained on; student
# reads 43d teacher obs with no encoder. Confirms BCPPO machinery itself works
# before pinning depth-specific failure modes.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-BCPPO-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateSysidTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:State_BCPPORunnerCfg",
    },
)
# State→state PPO + PBRS using V_expert from state RL expert.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-PPOPBRS-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGStateSysidTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:State_PPOPBRSRunnerCfg",
    },
)

# GRPO finetune for depth peg student (Doorman Phase 3). Reuses the same env
# as BCPPO-Sysid (sparse rewards + cameras + 4 obs groups). Pair with a DAgger
# checkpoint via agent.algorithm.init_from_dagger_path=...
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-GRPO-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85BCPPOSysidCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_GRPORunnerCfg",
    },
)
# Real grouped GRPO. Same env as Depth-GRPO-Sysid; pair with the GRPOGroupedRunner
# (forces full-env reset each iter) + ``env.events.reset_from_states.params.group_size=K``
# so the reset manager replicates each leader's state to its group followers.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-GRPO-Grouped-Sysid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85BCPPOSysidCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_GRPOGroupedRunnerCfg",
    },
)
# Same as above but with GPS curriculum re-enabled (target=0.5). Pairs grouped
# GRPO's group-baseline mechanism with reset states routed to ~50% success → the
# K envs in each group get within-group return variance → meaningful advantages.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-GRPO-Grouped-Sysid-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85BCPPOSysidCurriculumCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_GRPOGroupedRunnerCfg",
    },
)
# GRPO using the *DAgger native env* (FinetuneEvalEventCfg). Mean ep len ~40,
# DAgger ckpt has ~50% success here. The right env for GRPO finetune from
# DAgger because BCPPOSysid env's PA reset states satisfy success in 1 step
# (mean ep len ~1) which yields no useful rollout data.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-GRPO-Grouped-DAggerNative-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthGRPOGroupedDAggerNativeCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_GRPOGroupedRunnerCfg",
    },
)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-4Canonical-Lean-Drawer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_dagger_cfg:Ur5eRobotiq2f85DepthDAggerWristSide4CanonicalLeanDrawerCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Depth_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# State→state DAgger orthogonal sanity check (MLP student reads teacher obs;
# DEXTRAH-cadence 1 env step → 1 grad step).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-DAgger-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.state_dagger_cfg:Ur5eRobotiq2f85StateDAggerRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:State_DAggerFastRunnerCfg",
    },
)


# RGB DAgger from data-collection scene + state teacher (seed42 finetuned expert).
# Env: DataCollection scene (224x224, 4-reset-type sampling, FinetuneEval sysid).
# Runner: RGB_DAggerWristSidePretrainedWeightedRunnerCfg (unchanged).
# NOTE: teacher must be exported with --std (teacher_returns_std=True in runner).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-DataCollection-StateTeacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgb_dagger_cfg:Ur5eRobotiq2f85RGBDAggerDataCollectionStateCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:RGB_DAggerWristSidePretrainedWeightedRunnerCfg",
    },
)

# DataCollection scene + PC teacher (seed23_sysidenv.pt): broader reset distribution
# with the same ScenePC teacher used in the sysid DAgger runs.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DAgger-DataCollection-PCTeacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rgb_dagger_cfg:Ur5eRobotiq2f85RGBDAggerDataCollectionPCTeacherCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:RGB_DAggerDataCollectionPCTeacherRunnerCfg",
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

# ===========================================================================
# BAMDP-over-expert-failure-rates (uses the seeds 20/21 ScenePC experts +
# their discriminator from UWLab-ICL/EXPERT_DISCRIMINATOR.md). Wire the
# action-side injector at rollout time via uwlab_rl.wrappers.bamdp_failures.
# See BAMDP_FAILURES.md for the launch template.
# ===========================================================================
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-BAMDPFailures-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bamdp_failures_cfg:Ur5eRobotiq2f85BAMDPFailuresTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
        # Critic-only V_success fitting for a frozen expert (BAMDP value
        # feedback); select with --agent rsl_rl_success_critic_cfg_entry_point.
        "rsl_rl_success_critic_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCSuccessCriticOnlyRunnerCfg",
    },
)

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-BAMDPFailures-StudentEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bamdp_failures_cfg:Ur5eRobotiq2f85BAMDPFailuresStudentEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:ScenePCPPORunnerCfg",
    },
)

# Sim2real point-cloud augmentation sanity-check env (no objects, robot-only PC)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-Sim2RealPC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85Sim2RealPCCfg"},
)

# Data-collection env: Train cfg + calibrated/DR'd sim2real point-cloud obs group
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCRelCartesianOSCCfg"},
)

# Data-collection env + peg-mounted ContactSensor recording two ground-truth contact
# flags/step (gripper<->peg, peg<->hole). Same distribution as DataCollectionPC-v0.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-Contact-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCContactCfg"},
)

# Per-prim occluded data-collection env: scene_pc is the normal ratio-enforced occluded cloud,
# but each point carries its SOURCE prim id (robot link / insertive / receptive). No per-prim
# budget or padding -- the collector peels the prim id into a parallel obs/scene_pc_prim_id
# dataset (cloud stays pure xyz). Selecting prims (pc_parts) and zero-padding to a fixed size
# happen at TRAIN time. Same distribution as DataCollectionPC-Contact-v0 (per_prim=True, seg off).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-PerPrim-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCPerPrimCfg"},
)

# Data-collection env recording the CLEAN vanilla ScenePointCloud (no sim2real occlusion /
# noise), resample_on_reset=True. Same distribution as DataCollectionPC-v0; only the recorded
# cloud's realism differs (clean full cloud vs occluded + noisy).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCCleanCfg"},
)

# CLEAN data collection WITH the per-point segmentation channel (scene_pc -> point_dim=4:
# xyz + robot=0/insertive=-1/receptive=+1). Same clean cloud / 4-path reset distribution as
# DataCollectionPC-Clean; only the extra seg channel differs.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-CleanSeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCCleanSegCfg"},
)

# CANONICAL clean+seg data collection: same as DataCollectionPC-CleanSeg but with the FPS
# subset FIXED for the run (resample_on_reset=False) -> the same 512 points every episode.
# Ablates the per-reset resampling stochasticity to test perception-vs-data-coverage.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-CleanSeg-Canonical-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCCleanSegCanonicalCfg"},
)

# ONLINE point-cloud DAgger: PointNet student <- ScenePC JIT teacher, distilled online via
# DistillationRunnerSplit. Student obs = flat proprio + CleanSeg cloud; teacher obs = ScenePC
# expert input. Launch with agent.policy.teacher_jit_path=<scenepc expert jit> and set the
# student arch (agent.policy.encoder_dims / action_dims) to match the BC run being distilled.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PC-DAgger-CleanSeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85PCDAggerCleanSegCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PC_DAggerSplitRunnerCfg",
    },
)

# Weighted (DEXTRAH inverse-variance) online PC DAgger, XL-residual student matching the BC baseline.
# Same env as PC-DAgger-CleanSeg-v0; the runner bakes in weighted loss + XL sizing + predict_std, so
# launch only needs a --std teacher JIT (agent.policy.teacher_jit_path) and an optional BC warm-start
# (agent.policy.bc_init_path).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PC-DAgger-CleanSeg-WeightedXL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85PCDAggerCleanSegCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PC_DAggerSplitWeightedXLRunnerCfg",
    },
)

# Apples-to-apples EVAL env for DAgger checkpoints (play.py --checkpoint): same flat
# proprio/scene_pc/teacher groups as the DAgger training env, but the BC canonical-eval
# protocol (canonical cloud + single hard reset) so success rates are directly comparable
# to the offline-BC CleanSegCanonicalEval numbers.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PC-DAgger-CleanSeg-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85PCDAggerCleanSegEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PC_DAggerSplitWeightedXLRunnerCfg",
    },
)

# FROM-SCRATCH online DAgger on the DEPLOYABLE cloud: occluded (frustum + z-buffer from the
# calibrated extrinsic, FoundationStereo noise, extrinsics DR), robot points restricted to the
# GRIPPER links (no arm, no wrist D415 mesh), NO seg channel -> (1024, 3) student cloud.
# Weighted XL runner with point_dim=3; launch with agent.policy.teacher_jit_path=<--std teacher>
# and NO bc_init_path (from scratch).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PC-DAgger-OccludedGripper-WeightedXL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85PCDAggerOccludedGripperCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PC_DAggerSplitWeightedXLOccGripRunnerCfg",
    },
)

# HARD-reset-only occgrip DAgger variants: train on the SAME single hard reset path
# (ObjectAnywhereEEAnywhere) the honest eval uses, killing the 4-path reset-mix
# train/eval mismatch. Both zero-pad fully occluded classes (explicit absence signal).
# 3D variant (deployable, no seg) — from-scratch OR BC-init finetune (bc_init_path):
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PC-DAgger-OccludedGripper-HardOnly-WeightedXL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85PCDAggerOccGripHardOnlyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PC_DAggerSplitWeightedXLOccGripHardOnlyRunnerCfg",
    },
)

# 4D variant (per-point seg label, sim-only probe of how much the label is worth):
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PC-DAgger-OccludedGripperSeg-HardOnly-WeightedXL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85PCDAggerOccGripSegHardOnlyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PC_DAggerSplitWeightedXLOccGripSegHardOnlyRunnerCfg",
    },
)

# Expert-demo collection env matching the occgrip hard-only DAgger student cloud exactly
# (occluded gripper-only 1024x3 EE-frame, zero-pad, hard reset path only). For the
# offline-BC arm of the hard-only ablation (collect_pc_demos.py with a --std teacher JIT).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-OccGripHardOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCOccGripHardOnlyCfg"},
)

# Same occgrip collection cloud, broad 4-path reset mix — BC data covering ALL reset
# distributions (the DAgger finetune afterwards stays hard-only).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-OccGrip4Path-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCOccGrip4PathCfg"},
)

# FINGERS-ONLY variants: output points only from finger/knuckle/pad links (no wrist body,
# no camera mount), with ALL excluded robot bodies sampled as occluder-only points so the
# arm/wrist still hide what's behind them (proper self-occlusion).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PC-DAgger-OccludedFingers-HardOnly-WeightedXL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85PCDAggerOccFingersHardOnlyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PC_DAggerSplitWeightedXLOccFingersHardOnlyRunnerCfg",
    },
)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-OccFingers4Path-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCOccFingers4PathCfg"},
)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-OccFingersHardOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCOccFingersHardOnlyCfg"},
)

# OBJECTS-ONLY variants: zero robot points in the cloud (512/512 ins/rec), whole robot
# (arm + gripper + D415 mount) is occluder-only. Ablates seeing vs feeling the gripper.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PC-DAgger-OccludedObjects-HardOnly-WeightedXL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85PCDAggerOccObjectsHardOnlyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PC_DAggerSplitWeightedXLOccObjectsHardOnlyRunnerCfg",
    },
)
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-OccObjectsHardOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85DataCollectionPCOccObjectsHardOnlyCfg"},
)
# Arm-only-proprio (real-robot) variant: joint_pos = 6 arm joints, no gripper mimic joints.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PC-DAgger-OccludedObjects-HardOnly-Arm6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85PCDAggerOccObjectsHardOnlyArm6Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PC_DAggerSplitWeightedXLOccObjectsHardOnlyArm6RunnerCfg",
    },
)
# History-16 Transformer student on the same objects-only hard-only env (same env cfg; only the
# runner/policy differ — the student keeps its own rolling window, env obs stay single-frame).
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-PC-DAgger-OccludedObjects-HardOnly-History16-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85PCDAggerOccObjectsHardOnlyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:PC_DAggerSplitWeightedHist16OccObjectsHardOnlyRunnerCfg",
    },
)

# BC-PointNet eval env: roll out a trained PointNet (play.py --bc_checkpoint) on the
# sim2real PC obs. Same scene/action/reset as DataCollectionPC, minus the teacher group.
# The rsl_rl agent cfg is only used for seed/clip_actions -- the --bc_checkpoint path
# bypasses the runner entirely.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85BCPointNetEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# BC-PointNet eval for SEGMENTED-cloud models (point_dim=4: xyz + per-point class label).
# Same as BCPointNetEval but scene_pc carries the segmentation channel (matches the
# contact_seg dataset). Use for models trained with point_dim=4.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetSegEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85BCPointNetSegEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# BC-PointNet eval for PER-PRIM models. scene_pc is the normal occluded cloud with a trailing
# per-point prim-id channel; bc_utils peels that channel, keeps only the checkpoint's pc_parts,
# and zero-pads to the trained size at inference. per_prim config must match the collection env.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetPerPrimEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85BCPointNetPerPrimEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# BC-PointNet eval for CLEAN-cloud models (point_dim=3, 512 pts, no sim2real occlusion).
# Same as BCPointNetEval but scene_pc is the clean 512-pt ScenePointCloud (matches the
# clean_scenepc dataset). Use for models trained on clean_scenepc_100k.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetCleanEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85BCPointNetCleanEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# BC-PointNet eval for CLEAN + SEG models (point_dim=4, 512 pts). Same single-path HARD reset as
# BCPointNetCleanEval; the live clean cloud carries the matching per-point seg channel.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetCleanSegEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85BCPointNetCleanSegEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# BC-PointNet eval for CANONICAL (non-resampled) clean+seg models. Same single-path HARD reset
# as BCPointNetCleanSegEval; the live cloud's FPS subset is fixed for the run
# (resample_on_reset=False) to match the canonical collection env.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetCleanSegCanonicalEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85BCPointNetCleanSegCanonicalEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)

# BC-PointNet eval for CLEAN-cloud models trained with the scene_pc NOT EE-centric
# (ref_cfg unset -> robot frame). Same as BCPointNetCleanEval but the clean ScenePointCloud
# is expressed in the robot frame rather than the EE frame. Use for models trained that way.
gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetCleanNotEECentricEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pc_cfg:Ur5eRobotiq2f85BCPointNetCleanNotEECentricEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
)
