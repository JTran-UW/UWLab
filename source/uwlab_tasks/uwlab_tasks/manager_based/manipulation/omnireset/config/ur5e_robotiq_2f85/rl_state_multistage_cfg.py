# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-stage round table (trunk -> top, then base -> trunk) in one episode.

Scene: insertive_object = trunk, receptive_object = top, insertive_object_2 = base. Reset types ``Trunk_<type>`` /
``Base_<type>`` come from ``logs/more_tasks/tools/compose_round_multistage.py``. Every object has a fixed observation
slot; the per-pair shaping terms are always on; success (reward, SGS, metrics) means both stages seated.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_JTRAN_ASSETS_DIR

from ... import mdp as task_mdp
from ...mdp.utils import read_metadata_from_usd_directory
from .rl_state_cfg import RlStateSceneCfg, make_insertive_object, make_receptive_object
from .rl_state_sgs_cfg import Ur5eRobotiq2f85RelCartesianOSCSGSTrainCfg

RESET_TYPES = ["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped"]
ROUND_MULTI_RESET_TYPES = [f"Trunk_{t}" for t in RESET_TYPES] + [f"Base_{t}" for t in RESET_TYPES]


def _base_target_in_trunk_frame() -> dict:
    """The base's seat on the trunk, in the trunk frame: the merged top+trunk asset's target (top frame) seen from the
    trunk, which sits at the top's own seat (trunk assembled_offset is the identity)."""
    fb = f"{UWLAB_JTRAN_ASSETS_DIR}/Props/FurnitureBench"
    seat = read_metadata_from_usd_directory(f"{fb}/RoundTableTop/round_table_top.usd")["assembled_offset"]
    tgt = read_metadata_from_usd_directory(f"{fb}/RoundTableTopLeg/round_table_top_leg.usd")["assembled_offset"]
    sp, sq = torch.tensor([seat["pos"]]), torch.tensor([seat["quat"]])
    tp, tq = torch.tensor([tgt["pos"]]), torch.tensor([tgt["quat"]])
    pos = math_utils.quat_apply_inverse(sq, tp - sp)[0]
    quat = math_utils.quat_mul(math_utils.quat_inv(sq), tq)[0]
    return {"pos": [round(float(v), 7) for v in pos], "quat": [round(float(v), 9) for v in quat]}


FULL_SUCCESS = ("(env.command_manager.get_term('task_command').orientation_aligned"
                " & env.command_manager.get_term('task_command').position_aligned)")


@configclass
class RoundMultiSceneCfg(RlStateSceneCfg):
    """Objects spawn without their USD physics materials (3 per env would cap PhysX at ~21.8k envs); the material
    events set every object's friction at startup, so the simulated physics is unchanged."""

    insertive_object = make_insertive_object(
        f"{UWLAB_JTRAN_ASSETS_DIR}/Props/FurnitureBench/RoundLeg/round_leg.usd", strip_materials=True
    )
    receptive_object = make_receptive_object(
        f"{UWLAB_JTRAN_ASSETS_DIR}/Props/FurnitureBench/RoundTableTop/round_table_top.usd", strip_materials=True
    )
    insertive_object_2 = make_insertive_object(
        f"{UWLAB_JTRAN_ASSETS_DIR}/Props/FurnitureBench/RoundBase/round_base.usd", strip_materials=True
    ).replace(prim_path="{ENV_REGEX_NS}/InsertiveObject2")


def _retarget(term, **assets):
    return term.replace(params={**term.params, **{k: SceneEntityCfg(v) for k, v in assets.items()}})


@configclass
class Ur5eRobotiq2f85RelCartesianOSCRoundMultiTrainCfg(Ur5eRobotiq2f85RelCartesianOSCSGSTrainCfg):
    scene: RoundMultiSceneCfg = RoundMultiSceneCfg(num_envs=32, env_spacing=1.5)

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 32.0

        params = self.events.reset_from_reset_states.params
        params["reset_types"] = list(ROUND_MULTI_RESET_TYPES)
        params["probs"] = [1.0 / len(ROUND_MULTI_RESET_TYPES)] * len(ROUND_MULTI_RESET_TYPES)
        params["success"] = FULL_SUCCESS
        self.events.insertive_object_2_material = _retarget(self.events.insertive_object_material, asset_cfg="insertive_object_2")
        self.events.randomize_insertive_object_2_mass = _retarget(
            self.events.randomize_insertive_object_mass, asset_cfg="insertive_object_2"
        )

        base_target = _base_target_in_trunk_frame()
        cmd = self.commands.task_command
        self.commands.task_command = task_mdp.MultiStageTaskCommandCfg(
            asset_cfg=cmd.asset_cfg,
            resampling_time_range=cmd.resampling_time_range,
            insertive_asset_cfg=cmd.insertive_asset_cfg,
            receptive_asset_cfg=cmd.receptive_asset_cfg,
            extra_stages=[{"insertive": "insertive_object_2", "receptive": "insertive_object",
                           "receptive_offset": base_target}],
        )

        pol = self.observations.policy
        pol.base_pose = _retarget(pol.insertive_asset_pose, target_asset_cfg="insertive_object_2")
        pol.base_in_trunk_frame = _retarget(
            pol.insertive_asset_in_receptive_asset_frame, target_asset_cfg="insertive_object_2", root_asset_cfg="insertive_object"
        )

        r = self.rewards
        r.ee_asset_distance_2 = _retarget(r.ee_asset_distance, target_asset_cfg="insertive_object_2")
        r.progress_context_2 = RewTerm(
            func=task_mdp.ProgressContext,
            weight=0.1,
            params={"insertive_asset_cfg": SceneEntityCfg("insertive_object_2"),
                    "receptive_asset_cfg": SceneEntityCfg("insertive_object"),
                    "receptive_offset": base_target},
        )
        r.dense_success_reward_2 = RewTerm(func=task_mdp.dense_success_reward, weight=0.1,
                                           params={"std": 1.0, "context": "progress_context_2"})
        r.success_reward = RewTerm(func=task_mdp.command_success_reward, weight=1.0)
