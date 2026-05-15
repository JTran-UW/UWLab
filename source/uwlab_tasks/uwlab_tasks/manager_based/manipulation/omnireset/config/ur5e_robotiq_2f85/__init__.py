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

gym.register(
    id="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-SysID-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gravity_cfg:ZeroGScenePCSysidSim2RealEValCfg",
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
