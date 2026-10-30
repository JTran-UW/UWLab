# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision RL task for the FlashSAC-RGB port: two 64x64 grayscale cameras plus a non-privileged state.

The ``policy`` group keeps the privileged state so the PPO state expert can drive expert-buffer collection;
the vision agent reads only ``camera`` and ``vision_state``. Terminations are removed: the vision wrapper owns
the fixed horizon so every env's episode stays synchronized (required by the episodic image replay buffer).
"""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from ... import mdp as task_mdp
from .physics_presets import joint_pos_obs
from .rl_state_symmetry_cfg import HOLE_SYMMETRY, PEG_SYMMETRY, SymmetryObsCfg
from .rl_camera_bench_cfg import _RGB_SCENE, _camera, _static_curtain, apply_isaac_rtx_settings_if_kit, grayscale_image
from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85

from .actions import Ur5eRobotiq2f85RelativeOSCEvalAction
from .rl_state_cfg import (
    FinetuneEvalEventCfg,
    POSE_OBS_CLIP,
    ObservationsCfg,
    RlStateSceneCfg,
    Ur5eRobotiq2f85RelCartesianOSCTrainCfg,
    variants,
)

VISION_RES = 64
VISION_HORIZON = 160


@configclass
class VisionSceneCfg(RlStateSceneCfg):
    curtain_left = _static_curtain(_RGB_SCENE.curtain_left)
    curtain_back = _static_curtain(_RGB_SCENE.curtain_back)
    curtain_right = _static_curtain(_RGB_SCENE.curtain_right)
    front_camera = _camera(
        "{ENV_REGEX_NS}/Robot/rgb_front_camera",
        (1.0770121, -0.1679045, 0.4486344),
        (0.46613815, 0.25072644, 0.47107948, 0.70564552),
        13.20,
        res=VISION_RES,
    )
    wrist_camera = _camera(
        "{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        (0.0182505, -0.00408447, -0.0689107),
        (-0.61819255, -0.6160212, 0.347879, 0.34254336),
        24.55,
        moving=True,
        res=VISION_RES,
    )


@configclass
class VisionImageObsCfg(ObsGroup):
    front = ObsTerm(func=grayscale_image, params={"sensor_cfg": SceneEntityCfg("front_camera")})
    wrist = ObsTerm(func=grayscale_image, params={"sensor_cfg": SceneEntityCfg("wrist_camera")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class VisionStateObsCfg(ObsGroup):
    """Non-privileged state (no object poses): 7 + 12 + 6 = 25-D."""

    prev_actions = ObsTerm(func=task_mdp.last_action)
    joint_pos = joint_pos_obs()
    end_effector_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "rotation_repr": "axis_angle",
        },
        clip=POSE_OBS_CLIP,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class VisionObservationsCfg(ObservationsCfg):
    camera: VisionImageObsCfg = VisionImageObsCfg()
    vision_state: VisionStateObsCfg = VisionStateObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionCfg(Ur5eRobotiq2f85RelCartesianOSCTrainCfg):
    scene: VisionSceneCfg = VisionSceneCfg(num_envs=1024, env_spacing=1.5)
    observations: VisionObservationsCfg = VisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.scene.insertive_object = variants["scene.insertive_object"]["peg"].replace()
        self.scene.receptive_object = variants["scene.receptive_object"]["peghole"].replace()
        for name in ("front_camera", "wrist_camera"):
            cam = getattr(self.scene, name)
            cam.renderer_cfg = MultiBackendRendererCfg(isaacsim_rtx=cam.renderer_cfg)
        self.events.render_settings.func = apply_isaac_rtx_settings_if_kit
        # the vision wrapper resets all envs together every VISION_HORIZON steps
        self.terminations.time_out = None
        self.episode_length_s = 1.0e6
        self.events.reset_from_reset_states.params["max_states"] = None
        self.events.reset_from_reset_states.params["max_joint_vel"] = -1.0
        # symmetric success is opt-in (env.commands.task_command.symmetric_success=true)
        self.commands.task_command.insertive_symmetry = PEG_SYMMETRY
        self.commands.task_command.receptive_symmetry = HOLE_SYMMETRY


@configclass
class Vision3CamSceneCfg(VisionSceneCfg):
    side_camera = _camera(
        "{ENV_REGEX_NS}/Robot/rgb_side_camera",
        (0.8323904, 0.5877843, 0.2805111),
        (0.22122445, 0.51336143, 0.77676798, 0.29008842),
        20.10,
        res=VISION_RES,
    )


@configclass
class Vision3CamImageObsCfg(VisionImageObsCfg):
    side = ObsTerm(func=grayscale_image, params={"sensor_cfg": SceneEntityCfg("side_camera")})


@configclass
class Vision3CamObservationsCfg(VisionObservationsCfg):
    camera: Vision3CamImageObsCfg = Vision3CamImageObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVision3CamCfg(Ur5eRobotiq2f85RelCartesianOSCVisionCfg):
    """Front + side + wrist cameras (the RGB data-collection set) at 64x64 grayscale."""

    scene: Vision3CamSceneCfg = Vision3CamSceneCfg(num_envs=1024, env_spacing=1.5)
    observations: Vision3CamObservationsCfg = Vision3CamObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        cam = self.scene.side_camera
        cam.renderer_cfg = MultiBackendRendererCfg(isaacsim_rtx=cam.renderer_cfg)


# Symmetry declarations of the jtran-uw main-branch assets the symmetry-invariant PPO expert was trained with.


# the symmetric state expert's policy obs, exactly as it was trained (State-Symmetry-v0)
SymExpertPolicyObsCfg = SymmetryObsCfg


@configclass
class VisionSymObservationsCfg(VisionObservationsCfg):
    policy: SymExpertPolicyObsCfg = SymExpertPolicyObsCfg()


@configclass
class Vision3CamSymObservationsCfg(Vision3CamObservationsCfg):
    policy: SymExpertPolicyObsCfg = SymExpertPolicyObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionSymCfg(Ur5eRobotiq2f85RelCartesianOSCVisionCfg):
    """Vision task whose ``policy`` group is the symmetry-invariant expert observation (expert collection only)."""

    observations: VisionSymObservationsCfg = VisionSymObservationsCfg()


def stacked_gray_image(env, sensor_cfgs: list[SceneEntityCfg]) -> torch.Tensor:
    """(N, n_cams, H, W) grayscale in 0..255, rounded so uint8 replay storage is exact."""
    return torch.cat([grayscale_image(env, c) for c in sensor_cfgs], dim=-1).permute(0, 3, 1, 2).mul(255.0).round()


@configclass
class VisionStackedImageObsCfg(ObsGroup):
    """Front and wrist cameras as one channel-first image, the rsl_rl FlashSAC-RGB input."""

    image = ObsTerm(
        func=stacked_gray_image,
        params={"sensor_cfgs": [SceneEntityCfg("front_camera"), SceneEntityCfg("wrist_camera")]},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class VisionFlashSACObservationsCfg(VisionSymObservationsCfg):
    camera: VisionImageObsCfg | None = None
    image: VisionStackedImageObsCfg = VisionStackedImageObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACCfg(Ur5eRobotiq2f85RelCartesianOSCVisionSymCfg):
    """Vision task for rsl_rl FlashSAC-RGB: ``image`` + 25-D ``vision_state`` for the agent, the symmetric expert's
    ``policy`` obs for collection, and the env's own 160-step time-out (per-env resets, so frames re-render on reset)."""

    observations: VisionFlashSACObservationsCfg = VisionFlashSACObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
        self.episode_length_s = VISION_HORIZON * self.sim.dt * self.decimation
        self.num_rerenders_on_reset = 1
        self.commands.task_command.symmetric_success = True


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACFinetuneCfg(Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACCfg):
    """Sysid finetune of the pixel FlashSAC policy at the final Stage-2 setting from the start (no ADR / curriculum):
    explicit arm actuator, the eval OSC action (end-of-curriculum scales and gains), fixed full-range sysid + delay and
    OSC-gain randomization (``FinetuneEvalEventCfg`` terms), all four reset types."""

    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        apply_max_sysid(self)


def apply_max_sysid(cfg) -> None:
    """Final Stage-2 dynamics without a curriculum: explicit arm actuator + fixed full-range sysid/delay and OSC-gain
    randomization. Pair with the eval OSC action (``Ur5eRobotiq2f85RelativeOSCEvalAction``)."""
    cfg.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
    fixed = FinetuneEvalEventCfg()
    cfg.events.randomize_arm_sysid = fixed.randomize_arm_sysid
    cfg.events.randomize_osc_gains = fixed.randomize_osc_gains


DINO_RES = 112
DINO3_RES = 128
DINO_MODEL_DIR = "/workspace/uwlab/checkpoints/dinov2"
DINO3_MODEL_DIR = "/workspace/uwlab/checkpoints/dinov3"
TIMM_DIR = "/workspace/uwlab/_timm"


class dino_features(ManagerTermBase):
    """Frozen DINO features per camera (RGB, normalized): a global token + a ``pool_grid`` x ``pool_grid`` average of
    the spatial features (0 = global only), concatenated over cameras.

    ``backend="hf"``: a transformers DINOv2 folder (global = CLS). ``backend="timm"``: a folder named after the timm
    pretrained tag (e.g. ``vit_small_patch16_dinov3.lvd1689m``) holding ``model.safetensors``; ViTs drop their prefix
    (CLS + register) tokens from the grid, ConvNeXts use the final feature map and its mean as the global token.
    """

    def __init__(self, cfg: ObsTerm, env):
        super().__init__(cfg, env)
        import os
        import sys

        path, self.backend = cfg.params["model_path"], cfg.params.get("backend", "hf")
        if self.backend == "hf":
            from transformers import Dinov2Model

            self.model = Dinov2Model.from_pretrained(path, torch_dtype=torch.bfloat16)
            mean, std, dim = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), self.model.config.hidden_size
        else:
            if os.path.isdir(TIMM_DIR) and TIMM_DIR not in sys.path:
                sys.path.insert(0, TIMM_DIR)
            import timm

            arch = os.path.basename(path.rstrip("/")).split(".")[0]
            self.model = timm.create_model(
                arch, pretrained=True, num_classes=0, pretrained_cfg_overlay=dict(file=os.path.join(path, "model.safetensors"))
            ).to(torch.bfloat16)
            mean, std, dim = self.model.pretrained_cfg["mean"], self.model.pretrained_cfg["std"], self.model.num_features
        self.model = self.model.to(env.device).eval().requires_grad_(False)
        self.mean = torch.tensor(mean, device=env.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=env.device).view(1, 3, 1, 1)
        print(f"[dino] {path} ({self.backend}): dim {dim}, pool_grid {cfg.params['pool_grid']}", flush=True)

    def _global_and_grid(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, D) global token and (B, D, g, g) spatial features."""
        if self.backend == "hf":
            h = self.model(pixel_values=x).last_hidden_state.float()
            g = x.shape[-1] // self.model.config.patch_size
            return h[:, 0], h[:, 1 : 1 + g * g].reshape(h.shape[0], g, g, -1).permute(0, 3, 1, 2)
        h = self.model.forward_features(x).float()
        if h.ndim == 4:
            return h.mean(dim=(2, 3)), h
        prefix = self.model.num_prefix_tokens
        g = int(round((h.shape[1] - prefix) ** 0.5))
        return h[:, 0], h[:, prefix:].reshape(h.shape[0], g, g, -1).permute(0, 3, 1, 2)

    @torch.no_grad()
    def __call__(self, env, sensor_cfgs: list[SceneEntityCfg], model_path: str, pool_grid: int, chunk: int = 2048,
                 backend: str = "hf"):
        feats = []
        for sensor_cfg in sensor_cfgs:
            rgb = env.scene.sensors[sensor_cfg.name].data.output["rgb"][..., :3].float().div(255.0).permute(0, 3, 1, 2)
            x = ((rgb - self.mean) / self.std).to(torch.bfloat16)
            out = []
            for i in range(0, x.shape[0], chunk):
                glob, grid = self._global_and_grid(x[i : i + chunk])
                parts = [glob]
                if pool_grid > 0:
                    parts.append(torch.nn.functional.adaptive_avg_pool2d(grid, pool_grid).flatten(2).transpose(1, 2).flatten(1))
                out.append(torch.cat(parts, dim=-1))
            feats.append(torch.cat(out, dim=0))
        return torch.cat(feats, dim=-1)


@configclass
class VisionDinoObsCfg(ObsGroup):
    features = ObsTerm(
        func=dino_features,
        params={
            "sensor_cfgs": [SceneEntityCfg("front_camera"), SceneEntityCfg("wrist_camera")],
            "model_path": f"{DINO_MODEL_DIR}/small",
            "pool_grid": 2,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class VisionDinoObservationsCfg(VisionSymObservationsCfg):
    camera: VisionImageObsCfg | None = None
    dino: VisionDinoObsCfg = VisionDinoObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDinoCfg(Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACCfg):
    """Vision-FlashSAC with frozen DINOv2 features instead of pixels: RGB 112x112 front + wrist cameras -> ``dino``,
    plus ``vision_state`` with a 3-frame history (75-D)."""

    observations: VisionDinoObservationsCfg = VisionDinoObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        for cam in (self.scene.front_camera, self.scene.wrist_camera):
            cam.height = cam.width = DINO_RES
        for term in ("prev_actions", "joint_pos", "end_effector_pose"):
            getattr(self.observations.vision_state, term).history_length = 3


OOD_VISUAL_EVENTS = (
    "randomize_wrist_mount_appearance", "randomize_inner_finger_appearance", "randomize_insertive_object_appearance",
    "randomize_receptive_object_appearance", "randomize_table_appearance", "randomize_curtain_left_appearance",
    "randomize_curtain_back_appearance", "randomize_curtain_right_appearance", "randomize_sky_light",
    "randomize_front_camera", "randomize_front_camera_focal_length", "randomize_wrist_camera",
    "randomize_wrist_camera_focal_length",
)


class _render_only_visual_appearance(task_mdp.randomize_visual_appearance_multiple_meshes):
    """The OmniPBR appearance randomizer for render-only eval on replicated Newton envs: its replicate_physics guard
    protects PhysX property parsing, which material edits seen only by RTX do not touch. Static scene decorations
    (``AssetBaseCfg`` curtains) are looked up as their cfg, so they get an ``asset.cfg`` shim for the prim path."""

    def __init__(self, cfg, env):
        import types

        name = cfg.params["asset_cfg"].name
        extras = env.scene._extras
        static = name in extras and not hasattr(extras[name], "cfg")
        if static:
            extras[name] = types.SimpleNamespace(cfg=extras[name])
        replicate = env.cfg.scene.replicate_physics
        env.cfg.scene.replicate_physics = False
        try:
            super().__init__(cfg, env)
        finally:
            env.cfg.scene.replicate_physics = replicate
            if static:
                extras[name] = extras[name].cfg
        import re

        import omni.replicator.core as rep

        import isaaclab.sim as sim_utils

        stage = sim_utils.SimulationContext.instance().stage
        prims = rep.functional.get.prims(path_pattern=self.prim_path_pattern, stage=stage)
        self._prim_env = [int(m.group(1)) if (m := re.search(r"/env_(\d+)/", str(p.GetPath()))) else -1 for p in prims]
        assert len(self._prim_env) == len(self._shader_prims), "mesh / material count mismatch"
        self._all_shader_prims = list(self._shader_prims)

    def __call__(
        self,
        env,
        env_ids,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str] | None = None,
        texture_config_path: str | None = None,
        mesh_names: list[str] = [],
        texture_prob: float = 1.0,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]] | None = None,
        diffuse_tint_range: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        texture_scale_range: tuple[float, float] | None = None,
        roughness_range: tuple[float, float] | None = None,
        metallic_range: tuple[float, float] | None = None,
        specular_range: tuple[float, float] | None = None,
        per_env: bool = False,
    ):
        """With ``per_env``, only the meshes of ``env_ids`` are re-randomized (the base class redraws every env's)."""
        kwargs = dict(
            texture_paths=texture_paths, texture_config_path=texture_config_path, mesh_names=mesh_names,
            texture_prob=texture_prob, colors=colors, diffuse_tint_range=diffuse_tint_range,
            texture_scale_range=texture_scale_range, roughness_range=roughness_range, metallic_range=metallic_range,
            specular_range=specular_range,
        )
        if not per_env or env_ids is None or not hasattr(self, "_all_shader_prims"):
            return super().__call__(env, env_ids, event_name, asset_cfg, **kwargs)
        ids = set(env_ids.tolist())
        self._shader_prims = [p for p, e in zip(self._all_shader_prims, self._prim_env) if e in ids]
        try:
            return super().__call__(env, env_ids, event_name, asset_cfg, **kwargs)
        finally:
            self._shader_prims = self._all_shader_prims


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDinoOODCfg(Ur5eRobotiq2f85RelCartesianOSCVisionDinoCfg):
    """Eval-only: the DINO task as trained plus the OmniReset RGB OOD visual randomization (``OODRGBEventCfg``: AmbientCG
    textures/colors on robot mount, fingers, peg, hole, table and curtains, OOD HDRI dome light, camera pose/focal
    jitter). USD-material/light edits only render under RTX (``presets=...,rtx``)."""

    def __post_init__(self):
        super().__post_init__()
        add_ood_visual_events(self)


def add_ood_visual_events(cfg, source: str = "ood", cameras: bool = True) -> None:
    """Add the OmniReset RGB visual randomization (render-only, RTX) to a DINO cfg: ``source="ood"`` is the eval set
    (``OODRGBEventCfg``, AmbientCG), ``"train"`` the training set (``RGBEventCfg``, Poly Haven / vMaterials + HDRIs)."""
    from .data_collection_rgb_cfg import OODRGBEventCfg, RGBEventCfg

    events = OODRGBEventCfg() if source == "ood" else RGBEventCfg()
    for name in OOD_VISUAL_EVENTS:
        if not cameras and "camera" in name:
            continue
        term = getattr(events, name)
        if term.func is task_mdp.randomize_visual_appearance_multiple_meshes:
            term.func = _render_only_visual_appearance
        setattr(cfg.events, name, term)
    if cameras:
        # same +-15% focal jitter as the RGB eval, around our zoomed front camera (26.4) instead of 13.2
        cfg.events.randomize_front_camera_focal_length.params["focal_length_range"] = (22.4, 30.4)


def add_train_visual_events(cfg, per_episode: bool = False, hdri_interval_s: float = 160.0) -> None:
    """The in-distribution RGB visual randomization without camera jitter, on a fixed subset of the training textures /
    HDRIs (RTX keeps them all resident, and the full sets do not fit 24 GB).

    The data-collection terms redraw every env's materials every 4 s, so appearance changes mid-episode. With
    ``per_episode`` each env's meshes are redrawn only on its own reset; the dome light is one light for all envs, so
    it moves to a single global timer of ``hdri_interval_s`` (160 s = 10 episodes)."""
    from pathlib import Path

    add_ood_visual_events(cfg, source="train", cameras=False)
    res = Path(__file__).parent / "resources"
    for name in OOD_VISUAL_EVENTS:
        term = getattr(cfg.events, name, None)
        if term is None:
            continue
        if "texture_config_path" in term.params:
            term.params["texture_config_path"] = str(res / "texture_paths_train_subset150.yaml")
        if per_episode and term.func is _render_only_visual_appearance:
            term.mode = "reset"
            term.interval_range_s = None
            term.params["per_env"] = True
    sky = cfg.events.randomize_sky_light
    sky.params["hdri_config_path"] = str(res / "hdri_paths_train_subset100.yaml")
    if per_episode:
        sky.interval_range_s = (hdri_interval_s, hdri_interval_s)
        sky.is_global_time = True


def add_full_visual_events(cfg, sky_intensity_range: tuple[float, float] = (1000.0, 4000.0), subset: bool = False) -> None:
    """The original OmniReset RGB training randomization: full texture / HDRI sets redrawn every 4 s (``RGBEventCfg``)
    plus the side and wrist camera pose / focal-length jitter on reset. ``sky_intensity_range`` rescales the HDRI dome
    (the original range assumed the robot USD's per-env dome lights were on). ``subset`` swaps in the 150-texture / 100-HDRI
    training subset (the full texture set can render no geometry in some RTX setups)."""
    from .data_collection_rgb_cfg import RGBEventCfg

    add_ood_visual_events(cfg, source="train", cameras=False)
    cfg.events.randomize_sky_light.params["intensity_range"] = sky_intensity_range
    if subset:
        from pathlib import Path

        res = Path(__file__).parent / "resources"
        for name in OOD_VISUAL_EVENTS:
            term = getattr(cfg.events, name, None)
            if term is not None and "texture_config_path" in term.params:
                term.params["texture_config_path"] = str(res / "texture_paths_train_subset150.yaml")
        cfg.events.randomize_sky_light.params["hdri_config_path"] = str(res / "hdri_paths_train_subset100.yaml")
    events = RGBEventCfg()
    for name in ("randomize_side_camera", "randomize_side_camera_focal_length", "randomize_wrist_camera",
                 "randomize_wrist_camera_focal_length"):
        setattr(cfg.events, name, getattr(events, name))
    # +-1 cm wrist translations that combine axes put the camera inside the gripper in ~4% of draws (black image);
    # +-3 mm keeps every env clear
    cfg.events.randomize_wrist_camera.params["position_deltas"] = {k: (-0.003, 0.003) for k in ("x", "y", "z")}


# surfaces the RTX randomization re-textures, as Newton shape-label patterns under each env root
WARP_COLOR_SURFACES = (
    "Robot/robotiq_base_link/visuals/D415_to_Robotiq_Mount", "Robot/left_inner_finger/visuals/mesh_1",
    "Robot/right_inner_finger/visuals/mesh_1", "InsertiveObject/visuals", "ReceptiveObject/visuals",
    "Table/visuals/vention_mat", "CurtainLeft/geometry", "CurtainBack/geometry", "CurtainRight/geometry",
)


class randomize_newton_shape_colors(ManagerTermBase):
    """Random solid colors on Newton visual shapes (``model.shape_color``): the part of the RGB randomization the Newton
    Warp renderer can show (it has no textures, HDRI or PBR). One uniform color per matching shape and env."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ids = None

    def _shape_ids(self, env, patterns):
        from isaaclab_newton.physics import NewtonManager

        labels = list(NewtonManager.get_model().shape_label)
        ids = []
        for i in range(env.num_envs):
            root = f"/World/envs/env_{i}/"
            ids.append([s for s, label in enumerate(labels) if label.startswith(root) and any(label[len(root):].startswith(p) for p in patterns)])
        assert ids[0] and all(len(x) == len(ids[0]) for x in ids), "shape-label patterns matched no / uneven shapes"
        return torch.tensor(ids, device=env.device)

    def __call__(self, env, env_ids, patterns=WARP_COLOR_SURFACES, color_range=(0.0, 1.0)):
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        if self._ids is None:
            self._ids = self._shape_ids(env, patterns)
        env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device)
        ids = self._ids[env_ids].reshape(-1)
        colors = wp.to_torch(NewtonManager.get_model().shape_color)
        colors[ids] = torch.empty(len(ids), 3, device=colors.device).uniform_(*color_range)


class randomize_camera_world_pose(ManagerTermBase):
    """Camera pose jitter through the sensor (``Camera.set_world_poses``), which every renderer sees: the USD-prim edit of
    ``randomize_tiled_cameras`` never reaches Newton Warp, whose static cameras render from the pose cached at startup.
    Deltas are applied to the startup world pose (positions in m, Euler angles in degrees)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._base = None

    def __call__(self, env, env_ids, sensor_name, position_deltas, euler_deltas):
        import isaaclab.utils.math as math_utils

        cam = env.scene.sensors[sensor_name]
        if self._base is None:
            pos, quat = cam.data.pos_w, cam.data.quat_w_opengl
            self._base = ((pos.torch if hasattr(pos, "torch") else pos).clone(), (quat.torch if hasattr(quat, "torch") else quat).clone())
        env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device)
        n = len(env_ids)
        u = lambda lo_hi: torch.empty(n, device=env.device).uniform_(*lo_hi)
        pos = self._base[0][env_ids] + torch.stack([u(position_deltas[k]) for k in ("x", "y", "z")], -1)
        rad = [torch.deg2rad(u(euler_deltas[k])) for k in ("roll", "pitch", "yaw")]
        quat = math_utils.quat_mul(math_utils.quat_from_euler_xyz(*rad), self._base[1][env_ids])
        cam.set_world_poses(pos, quat, env_ids=env_ids, convention="opengl")


def add_warp_visual_events(cfg) -> None:
    """The Warp-renderable part of ``add_full_visual_events``: random solid colors on the same surfaces (redrawn on
    each reset and every 4 s) plus the side camera pose jitter set through the sensor. No textures, HDRI, focal or wrist
    jitter (Warp has no env-map lighting, builds its camera rays once from env 0's focal length, and re-derives the
    moving wrist camera's pose from the gripper link every step)."""
    from .data_collection_rgb_cfg import RGBEventCfg

    cfg.events.warp_colors_reset = EventTerm(func=randomize_newton_shape_colors, mode="reset")
    cfg.events.warp_colors_interval = EventTerm(func=randomize_newton_shape_colors, mode="interval", interval_range_s=(4.0, 4.0))
    side = RGBEventCfg().randomize_side_camera.params
    cfg.events.randomize_side_camera = EventTerm(func=randomize_camera_world_pose, mode="reset", params={
        "sensor_name": "side_camera", "position_deltas": side["position_deltas"], "euler_deltas": side["euler_deltas"]})


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDino3Cfg(Ur5eRobotiq2f85RelCartesianOSCVisionDinoCfg):
    """The DINO task with RGB 128x128 cameras and a timm DINOv3 backbone (ViT-S/16 by default; 8x8 patches)."""

    def __post_init__(self):
        super().__post_init__()
        for cam in (self.scene.front_camera, self.scene.wrist_camera):
            cam.height = cam.width = DINO3_RES
        self.observations.dino.features.params["backend"] = "timm"
        self.observations.dino.features.params["model_path"] = f"{DINO3_MODEL_DIR}/vit_small_patch16_dinov3.lvd1689m"


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDino3FinetuneCfg(Ur5eRobotiq2f85RelCartesianOSCVisionDino3Cfg):
    """The DINOv3 task at max sysid (see ``Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACFinetuneCfg``)."""

    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        apply_max_sysid(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDino3VisRandCfg(Ur5eRobotiq2f85RelCartesianOSCVisionDino3Cfg):
    """DINOv3 training task with the in-distribution RGB visual randomization (textures/colors on robot mount, fingers,
    peg, hole, table, curtains + HDRI dome light); RTX only. Camera pose/focal jitter is left out."""

    def __post_init__(self):
        super().__post_init__()
        add_train_visual_events(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDino3OODCfg(Ur5eRobotiq2f85RelCartesianOSCVisionDino3Cfg):
    """Eval-only: the DINOv3 task plus the OOD visual randomization of ``Ur5eRobotiq2f85RelCartesianOSCVisionDinoOODCfg``."""

    def __post_init__(self):
        super().__post_init__()
        add_ood_visual_events(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVision3CamSymCfg(Ur5eRobotiq2f85RelCartesianOSCVision3CamCfg):
    observations: Vision3CamSymObservationsCfg = Vision3CamSymObservationsCfg()


WRIST_SIDE = ("side_camera", "wrist_camera")
WRIST_SIDE_FRONT = ("front_camera", "side_camera", "wrist_camera")


def use_cameras(cfg, names: tuple[str, ...]) -> None:
    """Agent image / DINO input from ``names`` in order; the side camera matches the wrist camera's resolution and an
    unused front camera is removed (the front camera keeps its unzoomed 13.2 focal length)."""
    side = cfg.scene.side_camera
    side.height, side.width = cfg.scene.wrist_camera.height, cfg.scene.wrist_camera.width
    side.renderer_cfg = MultiBackendRendererCfg(isaacsim_rtx=side.renderer_cfg)
    if "front_camera" not in names:
        cfg.scene.front_camera = None
    term = cfg.observations.dino.features if getattr(cfg.observations, "dino", None) else cfg.observations.image.image
    term.params["sensor_cfgs"] = [SceneEntityCfg(n) for n in names]


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideCfg(Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACCfg):
    """Pixel FlashSAC task on the side + wrist cameras (no front camera)."""

    scene: Vision3CamSceneCfg = Vision3CamSceneCfg(num_envs=1024, env_spacing=1.5)

    def __post_init__(self):
        super().__post_init__()
        use_cameras(self, WRIST_SIDE)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideFrontCfg(Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACCfg):
    """Pixel FlashSAC task on the unzoomed front + side + wrist cameras."""

    scene: Vision3CamSceneCfg = Vision3CamSceneCfg(num_envs=1024, env_spacing=1.5)

    def __post_init__(self):
        super().__post_init__()
        use_cameras(self, WRIST_SIDE_FRONT)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDino3WristSideCfg(Ur5eRobotiq2f85RelCartesianOSCVisionDino3Cfg):
    """DINOv3 task on the side + wrist cameras (no front camera)."""

    scene: Vision3CamSceneCfg = Vision3CamSceneCfg(num_envs=1024, env_spacing=1.5)

    def __post_init__(self):
        super().__post_init__()
        use_cameras(self, WRIST_SIDE)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDino3WristSideWarpRandCfg(Ur5eRobotiq2f85RelCartesianOSCVisionDino3WristSideCfg):
    """DINOv3 side + wrist task with the Warp-renderable visual randomization (``add_warp_visual_events``)."""

    def __post_init__(self):
        super().__post_init__()
        add_warp_visual_events(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDino3WristSideVisRandCfg(Ur5eRobotiq2f85RelCartesianOSCVisionDino3WristSideCfg):
    """DINOv3 side + wrist task with the RTX visual randomization of the pixel finetunes (``add_full_visual_events``,
    training subset); RTX only."""

    def __post_init__(self):
        super().__post_init__()
        add_full_visual_events(self, subset=True)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDino3WristSideFrontCfg(Ur5eRobotiq2f85RelCartesianOSCVisionDino3Cfg):
    """DINOv3 task on the unzoomed front + side + wrist cameras."""

    scene: Vision3CamSceneCfg = Vision3CamSceneCfg(num_envs=1024, env_spacing=1.5)

    def __post_init__(self):
        super().__post_init__()
        use_cameras(self, WRIST_SIDE_FRONT)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidCfg(Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideCfg):
    """Pixel side + wrist task at max sysid (see ``Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACFinetuneCfg``)."""

    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        apply_max_sysid(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidVisRandCfg(
    Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidCfg
):
    """The side + wrist max-sysid task with the in-distribution RGB visual randomization (``add_train_visual_events``);
    RTX only."""

    def __post_init__(self):
        super().__post_init__()
        add_train_visual_events(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidVisRandEpCfg(
    Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidCfg
):
    """The side + wrist max-sysid task with the visual randomization redrawn once per episode per env
    (``add_train_visual_events(per_episode=True)``); RTX only."""

    def __post_init__(self):
        super().__post_init__()
        add_train_visual_events(self, per_episode=True)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACOODCfg(Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACCfg):
    """Eval-only: the pixel FlashSAC task plus the OOD visual randomization of
    ``Ur5eRobotiq2f85RelCartesianOSCVisionDinoOODCfg``; RTX only."""

    def __post_init__(self):
        super().__post_init__()
        add_ood_visual_events(self)


def stacked_rgb_image(env, sensor_cfgs: list[SceneEntityCfg]) -> torch.Tensor:
    """(N, 3 * n_cams, H, W) RGB in 0..255, the colour counterpart of ``stacked_gray_image``."""
    return torch.cat([env.scene.sensors[c.name].data.output["rgb"][..., :3].float() for c in sensor_cfgs], dim=-1).permute(
        0, 3, 1, 2
    )


_SYM_OBS = SymmetryObsCfg()


@configclass
class VisionAuxStateObsCfg(ObsGroup):
    """Auxiliary state-prediction targets (stored for the aux loss, never an actor / critic input): peg and hole poses
    and the peg-minus-hole position, exactly as the symmetry expert sees them (wrist frame, symmetry-reduced)."""

    insertive_asset_pose = _SYM_OBS.insertive_asset_pose.replace()
    receptive_asset_pose = _SYM_OBS.receptive_asset_pose.replace()
    insertive_minus_receptive_position = _SYM_OBS.insertive_minus_receptive_position.replace()

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class VisionFlashSACAuxObservationsCfg(VisionFlashSACObservationsCfg):
    aux_state: VisionAuxStateObsCfg = VisionAuxStateObsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidRGBCfg(Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidCfg):
    """The side + wrist max-sysid pixel task with RGB instead of grayscale cameras (6 image channels)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.image.image.func = stacked_rgb_image


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxCfg(Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidCfg):
    """The side + wrist max-sysid pixel task plus the ``aux_state`` group for the auxiliary state-prediction loss."""

    observations: VisionFlashSACAuxObservationsCfg = VisionFlashSACAuxObservationsCfg()


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxVisRandFullCfg(
    Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxCfg
):
    """The side + wrist max-sysid aux task with the full original RGB randomization (``add_full_visual_events``); RTX
    only."""

    def __post_init__(self):
        super().__post_init__()
        add_full_visual_events(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxRGBCfg(
    Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxCfg
):
    """The side + wrist max-sysid aux task with RGB instead of grayscale cameras (6 image channels)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.image.image.func = stacked_rgb_image


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidRGBVisRandCfg(
    Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidRGBCfg
):
    """The side + wrist max-sysid RGB task with the original RGB randomization on the training subset
    (``add_full_visual_events(subset=True)``: textures / HDRI every 4 s, side + wrist camera jitter); RTX only."""

    def __post_init__(self):
        super().__post_init__()
        add_full_visual_events(self, subset=True)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxRGBVisRandCfg(
    Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxRGBCfg
):
    """The side + wrist max-sysid aux RGB task with the original RGB randomization on the training subset
    (``add_full_visual_events(subset=True)``); RTX only."""

    def __post_init__(self):
        super().__post_init__()
        add_full_visual_events(self, subset=True)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxRGBWarpRandCfg(
    Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxRGBCfg
):
    """The side + wrist max-sysid aux RGB task with the Warp-renderable visual randomization (``add_warp_visual_events``:
    random surface colors + camera pose jitter), for Newton + Warp pretraining."""

    def __post_init__(self):
        super().__post_init__()
        add_warp_visual_events(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxRGBWarpRandPerStepCfg(
    Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxRGBWarpRandCfg
):
    """The WarpRand task with the Warp colours redrawn on every env step instead of every 4 s."""

    def __post_init__(self):
        super().__post_init__()
        self.events.warp_colors_interval.interval_range_s = (0.0, 0.0)


def real2sim_matched_groups() -> dict:
    """Yanda's real2sim matched appearance groups (base colours / materials of the real cell, newton-flow-distill d817fcc7)."""

    def group(asset: str, color: tuple, roughness: float, **kwargs) -> dict:
        return dict(asset=asset, color=color, roughness=roughness, **kwargs)

    return {
        "robot": dict(asset="robot", roughness=0.4, metallic=0.25, color_delta=0.015),
        "table_frame": group("table", (0.018, 0.015, 0.019), 0.42, metallic=0.65, color_delta=0.008),
        "tabletop": group("table", (0.30, 0.37, 0.30), 0.58, filter="vention_mat", texture=0),
        "curtain_left": group("curtain_left", (0.68, 0.58, 0.48), 0.95, texture=1),
        "curtain_back": group("curtain_back", (0.75, 0.65, 0.55), 0.95, texture=1),
        "curtain_right": group("curtain_right", (0.75, 0.59, 0.56), 0.95, texture=1),
        "insertive_object": group("insertive_object", (0.025, 0.17, 0.03), 0.5, texture=2, color_delta=0.015),
        "receptive_object": group("receptive_object", (0.025, 0.17, 0.03), 0.5, texture=2, color_delta=0.015),
        "mount": group("robot", (0.8, 0.68, 0.45), 0.7, filter="D415_to_Robotiq_Mount"),
        "support": group("ur5_metal_support", (0.48, 0.45, 0.42), 0.38, metallic=0.7),
    }


def add_real2sim_appearance(cfg) -> None:
    """Yanda's real2sim look on the policy cameras (newton-flow-distill ac0fa455): patched Newton-Warp renderer (GGX
    materials with normal / roughness maps, shadows, matched textures, key + fill lights, precomputed studio IBL, ACES
    tone mapping, camera response) and bounded per-reset appearance / exposure / white-balance / sensor-noise / IBL
    jitter (``RandomizeMatchedAppearance``), plus the side camera pose jitter set through the sensor. Needs Newton 1.5.2."""
    from pathlib import Path

    from uwlab.utils.newton_warp_renderer_cfg import UWLabNewtonWarpRendererCfg

    from ...mdp.real2sim_visual import RandomizeMatchedAppearance
    from .data_collection_rgb_cfg import RGBEventCfg

    tex = Path(__file__).parent / "resources" / "real2sim"
    for name in ("side_camera", "wrist_camera"):
        camera = getattr(cfg.scene, name)
        camera.renderer_cfg = UWLabNewtonWarpRendererCfg(
            texture_paths=tuple(str(tex / f"{n}.png") for n in ("smooth", "cloth", "printed")),
            enable_patches=True, enable_textures=True, enable_shadows=True,
            enable_pbr=True, enable_reflections=False, enable_dome_light=False,
            enable_precomputed_ibl=True, create_default_light=False,
            environment_paths=tuple(str(tex / f"studio_{n}.npy") for n in ("neutral", "warm", "cool")),
            surface_maps={name: {kind: str(tex / f"{name}_{kind}.png") for kind in ("normal", "roughness")}
                          for name in ("smooth", "cloth", "printed", "metal")},
            tone_mapper="aces", exposure_range=(-0.2, 0.2), enable_camera_response=True,
            ambient_sky=(0.65, 0.68, 0.72), ambient_ground=(0.24, 0.23, 0.22), ambient_intensity=0.7,
            shadow_floor=0.15, samples_per_pixel=1, shadow_samples=1,
        )
        camera.update_latest_camera_pose = True
    cfg.events.matched_appearance = EventTerm(
        func=RandomizeMatchedAppearance, mode="reset", params={"groups": real2sim_matched_groups()}
    )
    side = RGBEventCfg().randomize_side_camera.params
    cfg.events.randomize_side_camera = EventTerm(func=randomize_camera_world_pose, mode="reset", params={
        "sensor_name": "side_camera", "position_deltas": side["position_deltas"], "euler_deltas": side["euler_deltas"]})


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxRGBReal2SimCfg(
    Ur5eRobotiq2f85RelCartesianOSCVisionFlashSACWristSideSysidAuxRGBCfg
):
    """The side + wrist max-sysid aux RGB task rendered with Yanda's real2sim Newton-Warp look (``add_real2sim_appearance``)."""

    def __post_init__(self):
        super().__post_init__()
        add_real2sim_appearance(self)


@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionDino3WristSideReal2SimCfg(Ur5eRobotiq2f85RelCartesianOSCVisionDino3WristSideCfg):
    """DINOv3 side + wrist task rendered with Yanda's real2sim Newton-Warp look (``add_real2sim_appearance``)."""

    def __post_init__(self):
        super().__post_init__()
        add_real2sim_appearance(self)
