# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""BAMDP-over-expert-failure-rates env configs.

The forced-failure injection lives **inside the environment** (see
:class:`task_mdp.BAMDPLatentSampler`) — it monkey-patches
``action_manager.process_action`` so every caller of ``env.step`` gets the
BAMDP semantics for free. Past_action obs are kept on the learner's
intended action so the BAMDP latent doesn't leak via prev_actions
(action-perturbation is dynamics; the policy shouldn't see whether it was
overridden).

Two cfg classes + two gym registrations:

  * ``…-BAMDPFailures-Train-v0``       — data-collection env.
  * ``…-BAMDPFailures-StudentEval-v0`` — eval target (no success termination).

Both inherit :class:`ZeroGScenePCSysidSim2RealTrainCfg` (gravity ScenePC +
sysid DR — the same env the seeds 20/21 experts were trained on).

Expert checkpoints + discriminator paths come in via cfg.params on the
BAMDP latent event. Override at launch with Hydra, e.g.::

    env.events.bamdp_latent_sampler.params.expert_specs='[{...}, {...}]' \\
    env.events.bamdp_latent_sampler.params.discriminator_ckpt='/abs/path/best.pt'

For convenience, ``DEFAULT_EXPERT_SPECS`` and ``DEFAULT_DISCRIMINATOR_CKPT``
below carry the seeds 20/21 setup so the env IDs work out of the box. The
two strategy slots correspond to class indices the discriminator was
trained with: index 0 = seed_21 (label A in the EXPERT_DISCRIMINATOR.md
data), index 1 = seed_20 (label B). DO NOT reorder.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .gravity_cfg import (
    ScenePCObsCfg,
    ZeroGScenePCSysidSim2RealTrainCfg,
)


# Default seeds 20/21 setup. Class order MUST match the discriminator's
# training labels: 0 = seed_21 (A), 1 = seed_20 (B). See
# UWLab-ICL/EXPERT_DISCRIMINATOR.md and analysis/train_expert_classifier.py:53-54.
DEFAULT_EXPERT_SPECS = [
    {
        "label": "seed_21",
        "checkpoint": "/mnt/storage/lti/UWLab-patrick-private/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-05-13_04-11-11/model_4000.pt",
        "proprio_dim": 25,
        "pc_dim": 1536,
    },
    {
        "label": "seed_20",
        "checkpoint": "/mnt/storage/lti/UWLab-patrick-private/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-05-13_04-19-11/model_3100.pt",
        "proprio_dim": 25,
        "pc_dim": 1536,
    },
]
DEFAULT_DISCRIMINATOR_CKPT = (
    "/mnt/storage/lti/UWLab-patrick-private/analysis/expert_classifier_runs/seeds20_21_v2_unbiased/best.pt"
)


# ---------------------------------------------------------------------------
# Cost-signal obs group (theta stays hidden).
# ---------------------------------------------------------------------------


@configclass
class BAMDPMetaObsGroupCfg(ObsGroup):
    """Per-env cost signals the BAMDP solver is allowed to see.

    Theta itself is NOT here — the solver must infer the rescue from cost.
    Add a separate ``theta`` term in your own cfg if you want an oracle
    baseline.
    """

    stall = ObsTerm(func=task_mdp.bamdp_stall_bit)
    rescue = ObsTerm(func=task_mdp.bamdp_rescue_bit)
    failure_count = ObsTerm(func=task_mdp.bamdp_failure_count)
    steps_since_failure = ObsTerm(func=task_mdp.bamdp_steps_since_failure)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


# ---------------------------------------------------------------------------
# Data-collection obs cfg (adds bamdp_meta + the rescue-action label).
# ---------------------------------------------------------------------------


@configclass
class _PolicyObsGroupCfg(ObsGroup):
    """Per-step obs the trained diffusion student consumes at eval / inference.

    Matches the diffusion_policy task config (``bamdp_scene_pc.yaml``) shape_meta:
    ``prev_actions`` (7), ``joint_pos`` (12), ``end_effector_pose`` (6), ``scene_pc`` (1536).
    ``concatenate_terms=False`` so the DiffusionPolicyWrapper sees one tensor per
    key in ``obs_buf["policy"]``.
    """

    prev_actions = ObsTerm(func=task_mdp.last_action)
    joint_pos = ObsTerm(func=task_mdp.joint_pos)
    end_effector_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "rotation_repr": "axis_angle",
        },
    )
    scene_pc = ObsTerm(
        func=task_mdp.ScenePointCloud,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_cfg": SceneEntityCfg("insertive_object"),
            "receptive_cfg": SceneEntityCfg("receptive_object"),
            "visualize": False,
            "num_points": 512,
            "resample_on_reset": False,
            "resample_on_reset_robot": False,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class _DataCollectionGroupCfg(ObsGroup):
    """Combined obs group recorded for BAMDP-ASTEROID training.

    Carries:
      * the student's input obs (proprio terms + scene pointcloud),
      * the rescue expert's intended action as the supervision label
        (``expert_action_mean`` — counterfactual: what the rescue would
        have done at every step regardless of who actually executed),
      * the action that physically drove the env (``executed_action``)
        for debug + sanity-check logging,
      * the BAMDP cost signals (``stall``, ``rescue``, ``failure_count``,
        ``steps_since_failure``) so the BAMDP solver can condition on
        them.

    ``concatenate_terms=False`` so each term lands as its own key in the
    recorded zarr (matches how the diffusion-policy dataset loader pulls
    per-key observations out of the dataset).
    """

    # --- student inputs: proprio terms ---
    # NOTE: ``task_mdp.last_action`` returns the action manager's aggregate
    # _action buffer. After the BAMDP injector's prev_action swap, that's
    # the learner's pre-override action — exactly what we want the student
    # to see in training too.
    prev_actions = ObsTerm(func=task_mdp.last_action)
    joint_pos = ObsTerm(func=task_mdp.joint_pos)
    end_effector_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "rotation_repr": "axis_angle",
        },
    )

    # --- student input: scene pointcloud (shared with the parent ScenePC obs) ---
    scene_pc = ObsTerm(
        func=task_mdp.ScenePointCloud,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_cfg": SceneEntityCfg("insertive_object"),
            "receptive_cfg": SceneEntityCfg("receptive_object"),
            "visualize": False,
            "num_points": 512,
            "resample_on_reset": False,
            "resample_on_reset_robot": False,
        },
    )

    # --- supervision target ---
    expert_action_mean = ObsTerm(func=task_mdp.bamdp_rescue_action)

    # --- debug / context ---
    executed_action = ObsTerm(func=task_mdp.bamdp_executed_action)
    stall_bit = ObsTerm(func=task_mdp.bamdp_stall_bit)
    rescue_bit = ObsTerm(func=task_mdp.bamdp_rescue_bit)
    failure_count = ObsTerm(func=task_mdp.bamdp_failure_count)
    steps_since_failure = ObsTerm(func=task_mdp.bamdp_steps_since_failure)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class BAMDPScenePCObsCfg(ScenePCObsCfg):
    """ScenePC obs + bamdp_meta cost-signal group + data_collection labels.

    Don't add a ``theta`` term here. Theta-leakage defeats the BAMDP.

    Groups:
      * ``proprio`` / ``pointcloud`` / ``time_left`` / ``success_classifier``
        — inherited from ScenePCObsCfg (used for legacy PPO / value heads).
      * ``policy`` — the student's input view, one tensor per shape_meta key.
        The DiffusionPolicyWrapper reads from this group at eval time.
      * ``bamdp_meta`` — cost signals (stall/rescue/failure_count/etc.).
      * ``data_collection`` — what the recorder writes to the zarr at
        training-data-collection time. Combines the policy obs + the
        rescue-action supervision label + BAMDP cost signals.
    """

    policy: _PolicyObsGroupCfg = _PolicyObsGroupCfg()
    bamdp_meta: BAMDPMetaObsGroupCfg = BAMDPMetaObsGroupCfg()
    data_collection: _DataCollectionGroupCfg = _DataCollectionGroupCfg()


# ---------------------------------------------------------------------------
# BAMDP latent EventTerm builder.
# ---------------------------------------------------------------------------


def _make_bamdp_latent_event(
    *,
    expert_specs=None,
    discriminator_ckpt: str = DEFAULT_DISCRIMINATOR_CKPT,
    num_strategies: int = 2,
    p_min: float = 0.0,
    p_max: float = 1.0,
    n_avg: float = 50.0,
    stall_steps: int = 4,
    warmup_steps: int = 1,
    discriminator_temperature: float = 1.0,
    discriminator_argmax: bool = False,
    use_hazard_budget: bool = True,
    bamdp_disabled: bool = False,
    force_theta=None,
    force_rescue_idx=None,
) -> EventTerm:
    """Build the reset-mode BAMDP latent event term with sensible defaults.

    Hydra-overridable at launch via
    ``env.events.bamdp_latent_sampler.params.<key>=<value>``.
    """
    if expert_specs is None:
        expert_specs = DEFAULT_EXPERT_SPECS

    return EventTerm(
        func=task_mdp.BAMDPLatentSampler,
        mode="reset",
        params={
            "num_strategies": num_strategies,
            "p_min": p_min,
            "p_max": p_max,
            "n_avg": n_avg,
            "expert_specs": expert_specs,
            "discriminator_ckpt": discriminator_ckpt,
            "proprio_obs_group": "proprio",
            "pointcloud_obs_group": "pointcloud",
            "stall_steps": stall_steps,
            "warmup_steps": warmup_steps,
            "discriminator_temperature": discriminator_temperature,
            "discriminator_argmax": discriminator_argmax,
            "arm_action_dims": (0, 1, 2, 3, 4, 5),
            "gripper_action_dim": -1,
            "open_gripper_value": 1.0,
            "fail_only_when_learner_in_control": True,
            "use_hazard_budget": use_hazard_budget,
            "bamdp_disabled": bamdp_disabled,
            "force_theta": force_theta,
            "force_rescue_idx": force_rescue_idx,
        },
    )


# ---------------------------------------------------------------------------
# Train cfg (= data-collection cfg for ASTEROID-style demos).
# ---------------------------------------------------------------------------


@configclass
class Ur5eRobotiq2f85BAMDPFailuresTrainCfg(ZeroGScenePCSysidSim2RealTrainCfg):
    """BAMDP-over-expert-failure-rates env, ScenePC observations.

    Pairs with the seeds 20/21 experts and their classifier (see
    ``UWLab-ICL/EXPERT_DISCRIMINATOR.md``). The BAMDP latent is sampled
    every reset and the forced-failure injection runs inside the env's
    ``action_manager.process_action`` — no script-side wrapping required.
    """

    observations: BAMDPScenePCObsCfg = BAMDPScenePCObsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Add the BAMDP latent / injection event. Layered on top of the
        # parent's reset events (gravity GPS resets + sysid DR) so they
        # all fire each reset.
        if not hasattr(self.events, "bamdp_latent_sampler") or self.events.bamdp_latent_sampler is None:
            self.events.bamdp_latent_sampler = _make_bamdp_latent_event()

        # Replace the parent's first-frame `success` termination with the
        # ASTEROID-style "≥ N consecutive success steps after a minimum
        # episode length" pattern, plus an `early_success` companion. The
        # gravity task's reset distribution includes states that are *already*
        # in the success zone (`ZeroGPartialAssembly`), so a first-frame
        # termination would cut data-collection episodes to length 1 — they'd
        # be useless as DAgger demos. Pattern + thresholds copied from
        # `UWLab-ICL/.../asteroid_env_cfg.py`.
        self.terminations.success = DoneTerm(
            func=task_mdp.consecutive_success_state_with_min_length,
            params={"num_consecutive_successes": 5, "min_episode_length": 10},
        )
        self.terminations.early_success = DoneTerm(
            func=task_mdp.early_success_termination,
            params={"num_consecutive_successes": 5, "min_episode_length": 10},
        )

        # ZeroGPartialAssembly spawns the object already in the success zone —
        # the rescue expert hits 5 consecutive successes before
        # min_episode_length=10 → early_success fires → episode terminated as
        # failure. About half of all demos get wasted that way. Restrict to
        # ZeroGAnywhere only for data collection so every reset requires the
        # expert to actually drive insertion (genuine ASTEROID demos).
        if (
            getattr(self.events, "reset_from_states", None) is not None
            and "reset_types" in self.events.reset_from_states.params
        ):
            self.events.reset_from_states.params["reset_types"] = ["ZeroGAnywhere"]
            self.events.reset_from_states.params["probs"] = [1.0]


# ---------------------------------------------------------------------------
# Student-eval cfg — same env, no success termination.
# ---------------------------------------------------------------------------


@configclass
class Ur5eRobotiq2f85BAMDPFailuresStudentEvalCfg(Ur5eRobotiq2f85BAMDPFailuresTrainCfg):
    """Eval cfg: same env, no success termination so we report
    end-of-episode metrics in the UWLab-ICL ASTEROID convention.
    """

    def __post_init__(self):
        super().__post_init__()

        if hasattr(self.terminations, "success"):
            self.terminations.success = None
        if hasattr(self.terminations, "early_success"):
            self.terminations.early_success = None
