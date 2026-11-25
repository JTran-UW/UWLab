# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export an rsl_rl FlashSAC-RGB checkpoint for deployment: a standalone TorchScript policy, an obs/action/camera spec,
and a sim rollout (obs + the original policy's actions) to check the export against on the target machine."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export a FlashSAC-RGB checkpoint (TorchScript + spec + parity rollout).")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="flashsac_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--save_envs", type=int, default=4, help="Envs whose obs/actions go into parity_rollout.npz.")
parser.add_argument("--use_ema", action="store_true", help="Export the checkpoint's EMA policy instead of the live one.")
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import copy
import json
import os
import shutil
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from isaaclab_flashsac.wrapper import FlashSACVecEnvWrapper
from rsl_rl_flashsac.runners import OffPolicyRunner

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_rl.flashsac import make_flashsac_env
from uwlab_tasks.utils.hydra import hydra_task_config


class DinoFrontEnd(nn.Module):
    """Frozen DINOv3 on the 0..255 channel-stacked cameras -> the token obs the policy was trained on, camera-major:
    ``mode="patch"``: final patch tokens (``dino_patch_tokens``); ``mode="head"``: prefix + patch tokens after the first
    ``split_block`` blocks (``dino_block_tokens``)."""

    def __init__(self, model_dir: str, mode: str, split_block: int, num_cameras: int):
        super().__init__()
        from uwlab_rl.flashsac.rgb import load_timm_dino

        self.model = load_timm_dino(model_dir).eval().requires_grad_(False)
        self.mode, self.split_block, self.num_cameras = mode, split_block, num_cameras
        cfg = self.model.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        from uwlab_rl.flashsac.rgb import dino_head_tokens

        b = image.shape[0]
        x = (image.float() / 255.0).unflatten(1, (self.num_cameras, 3)).flatten(0, 1)
        x = (x - self.mean) / self.std
        if self.mode == "patch":
            t = self.model.forward_features(x)[:, self.model.num_prefix_tokens :]
        else:
            t = dino_head_tokens(self.model, x, self.split_block)
        return t.unflatten(0, (b, self.num_cameras)).flatten(1, 2)


class ExportedPolicy(nn.Module):
    """(image uint8 [B, C, H, W], state float [B, S]) -> deterministic action in env units: tanh(mean) * scale + bias.
    With ``front_end`` the encoder sees front_end(image) (frozen DINO tokens) instead of the image."""

    def __init__(self, encoder: nn.Module, actor: nn.Module, front_end: nn.Module | None = None):
        super().__init__()
        self.encoder, self.actor, self.front_end = encoder, actor, front_end

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = self.front_end(image) if self.front_end is not None else image.float()
        mean, _ = self.actor.get_mean_and_std(self.encoder(x, state.float()), training=False)
        return torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias


def _plain(m: nn.Module) -> nn.Module:
    return getattr(m, "_orig_mod", m)


def _jsonable(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if hasattr(v, "name") and hasattr(v, "joint_names"):  # SceneEntityCfg
        return {"name": v.name, "joint_names": v.joint_names, "body_names": v.body_names}
    if callable(v):
        return f"{getattr(v, '__module__', '')}.{getattr(v, '__qualname__', getattr(v, '__name__', type(v).__name__))}"
    return repr(v)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    tok_term = getattr(getattr(env_cfg.observations, "dino_tokens", None), "tokens", None)
    if tok_term is not None:
        # token policies: no augmentation, and a raw-image group (the exported policy's input) next to the tokens
        from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_vision_cfg import (
            VisionStackedImageObsCfg, stacked_rgb_image)

        tok_term.params["image_aug"] = False
        env_cfg.observations.image = VisionStackedImageObsCfg()
        env_cfg.observations.image.image.func = stacked_rgb_image
        env_cfg.observations.image.image.params["sensor_cfgs"] = list(tok_term.params["sensor_cfgs"])
    agent_cfg.algorithm.expert_buffer_path = ""
    agent_cfg.algorithm.replay_buffer_size = 4 * args_cli.num_envs
    agent_cfg.algorithm.buffer_min_length = args_cli.num_envs
    env = make_flashsac_env(
        gym.make(args_cli.task, cfg=env_cfg), FlashSACVecEnvWrapper, agent_cfg.clip_actions, agent_cfg.action_bound,
        agent_cfg.action_bound_scale,
    )
    runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
    alg = runner.alg
    if args_cli.use_ema:
        ckpt = torch.load(args_cli.checkpoint, map_location=agent_cfg.device, weights_only=False)
        _plain(alg.actor).load_state_dict(ckpt["ema_actor_state_dict"])
        _plain(alg.encoder).load_state_dict(ckpt["ema_encoder_state_dict"])
    img_g, st_g = alg.image_group, alg.state_group
    front, front_spec, tok_g = None, None, None
    if tok_term is not None:
        mode = "head" if "split_block" in tok_term.params else "patch"
        split = int(tok_term.params.get("split_block", 0))
        front = DinoFrontEnd(tok_term.params["model_path"], mode, split, len(tok_term.params["sensor_cfgs"])).to(agent_cfg.device)
        front_spec = {"type": "frozen DINOv3 (timm)", "model_path": tok_term.params["model_path"], "mode": mode,
                      "split_block": split, "tokens_group": img_g, "note": "run inside policy_ts.pt; input is the raw image"}
        tok_g, img_g = img_g, "image"
    policy = ExportedPolicy(_plain(alg.encoder), _plain(alg.actor), front).eval()
    os.makedirs(args_cli.out_dir, exist_ok=True)

    # sim rollout with the original modules (eval mode, deterministic)
    base = env.unwrapped
    obs = env.get_observations()
    n = args_cli.save_envs
    rec = {"image": [], "state": [], "action_env": [], "action_norm": [], "reward": [], "done": []}
    if tok_g is not None:
        rec["action_env_env_tokens"] = []
    with torch.no_grad():
        for _ in range(args_cli.steps):
            a_env = policy(obs[img_g], obs[st_g])
            if tok_g is not None:  # the training-time path: tokens from the env's own (bf16, GPU) DINO term
                m, _ = policy.actor.get_mean_and_std(policy.encoder(obs[tok_g].float(), obs[st_g].float()), training=False)
                rec["action_env_env_tokens"].append((torch.tanh(m) * policy.actor.action_scale + policy.actor.action_bias)[:n].cpu())
            a_norm = ((a_env - env.action_bias) / env.action_scale).clamp(-1.0, 1.0)
            rec["image"].append(obs[img_g][:n].to(torch.uint8).cpu())
            rec["state"].append(obs[st_g][:n].float().cpu())
            rec["action_env"].append(a_env[:n].cpu())
            rec["action_norm"].append(a_norm[:n].cpu())
            obs, rew, dones, extras = env.step(a_norm)
            rec["reward"].append(rew[:n].cpu())
            rec["done"].append(dones[:n].cpu())
    rec = {k: torch.stack(v, 0).numpy() for k, v in rec.items()}

    # standalone TorchScript copy on CPU, checked against the original on the recorded obs
    cpu_policy = copy.deepcopy(policy).cpu().eval()
    ex = (torch.from_numpy(rec["image"][0]), torch.from_numpy(rec["state"][0]))
    traced = torch.jit.trace(cpu_policy, ex, check_trace=False)
    traced = torch.jit.freeze(traced.eval())
    ts_path = os.path.join(args_cli.out_dir, "policy_ts.pt")
    traced.save(ts_path)
    reload = torch.jit.load(ts_path)
    T = rec["image"].shape[0]
    with torch.no_grad():
        out = reload(torch.from_numpy(rec["image"].reshape(T * n, *rec["image"].shape[2:])),
                     torch.from_numpy(rec["state"].reshape(T * n, -1))).numpy().reshape(T, n, -1)
    err = float(np.abs(out - rec["action_env"]).max())
    err_env = float(np.abs(out - rec["action_env_env_tokens"]).max()) if tok_g is not None else None
    rec["action_env_torchscript_cpu"] = out
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    x1 = (torch.from_numpy(rec["image"][0, :1]), torch.from_numpy(rec["state"][0, :1]))
    with torch.no_grad():
        for _ in range(5):
            reload(*x1)
        t0 = time.perf_counter()
        for _ in range(30):
            reload(*x1)
    cpu_ms = (time.perf_counter() - t0) / 30 * 1000
    np.savez_compressed(os.path.join(args_cli.out_dir, "parity_rollout.npz"), **rec)

    # spec: obs terms, action term, scaling, timing, cameras
    om, am = base.observation_manager, base.action_manager
    groups = {}
    for g in (img_g, st_g, getattr(alg, "aux_group", None)):
        if not g or g not in om.active_terms:
            continue
        terms, start = [], 0
        cfgs = om._group_obs_term_cfgs[g]
        for name, dims, cfg in zip(om.active_terms[g], om.group_obs_term_dim[g], cfgs):
            size = int(np.prod(dims))
            terms.append({"name": name, "shape": list(dims), "slice": [start, start + size], "func": _jsonable(cfg.func),
                          "params": _jsonable(cfg.params), "scale": _jsonable(cfg.scale), "clip": _jsonable(cfg.clip),
                          "noise": repr(cfg.noise) if cfg.noise is not None else None})
            start += size
        groups[g] = {"terms": terms, "concatenate": bool(om.group_obs_concatenate[g]), "total": start}
    actions = []
    for name, term in am._terms.items():
        c = term.cfg
        actions.append({"name": name, "class": type(term).__name__, "dim": int(term.action_dim),
                        "cfg": {k: _jsonable(getattr(c, k)) for k in c.__dataclass_fields__ if k not in ("class_type",)}})
    cams = {}
    cam_group = getattr(env_cfg.observations, img_g)
    for cname in [s.name for s in cam_group.image.params["sensor_cfgs"]] if hasattr(cam_group, "image") else []:
        cam = base.scene.sensors[cname]
        cc = cam.cfg
        robot = base.scene["robot"]
        cams[cname] = {
            "prim_path": cc.prim_path, "height": cc.height, "width": cc.width, "data_types": list(cc.data_types),
            "offset": {"pos": list(cc.offset.pos), "rot_wxyz": list(cc.offset.rot), "convention": cc.offset.convention},
            "spawn": {k: _jsonable(getattr(cc.spawn, k)) for k in ("focal_length", "horizontal_aperture", "vertical_aperture", "clipping_range", "focus_distance") if hasattr(cc.spawn, k)},
            "intrinsic_matrix_env0": _jsonable(cam.data.intrinsic_matrices[0]),
            "pos_w_env0_minus_robot_root": _jsonable(cam.data.pos_w[0] - robot.data.root_pos_w[0]),
            "quat_w_ros_env0": _jsonable(cam.data.quat_w_ros[0]),
            "renderer": type(cc.renderer_cfg).__name__ if getattr(cc, "renderer_cfg", None) is not None else None,
        }
    spec = {
        "task": args_cli.task, "checkpoint": os.path.abspath(args_cli.checkpoint), "weights": "ema" if args_cli.use_ema else "live",
        "policy": ("action_env = tanh(mean(encoder(front_end(image), state))) * action_scale + action_bias (deterministic)"
                   if front is not None else "action_env = tanh(mean(encoder(image / 255, state))) * action_scale + action_bias (deterministic)"),
        "encoder_type": getattr(agent_cfg.algorithm, "encoder_type", "pixel"), "front_end": front_spec,
        "cpu_latency_ms_batch1": round(cpu_ms, 1), "cpu_threads": torch.get_num_threads(),
        "inputs": {"image": {"group": img_g, "dtype": "uint8", "shape": list(rec["image"].shape[2:])},
                   "state": {"group": st_g, "dtype": "float32", "shape": list(rec["state"].shape[2:])}},
        "observation_groups": groups, "action_terms": actions,
        "wrapper": {"action_scale": _jsonable(env.action_scale), "action_bias": _jsonable(env.action_bias),
                    "clip_actions": _jsonable(agent_cfg.clip_actions), "action_bound": _jsonable(agent_cfg.action_bound)},
        "actor_action_scale": _jsonable(policy.actor.action_scale), "actor_action_bias": _jsonable(policy.actor.action_bias),
        "timing": {"sim_dt": float(base.cfg.sim.dt), "decimation": int(base.cfg.decimation), "step_dt": float(base.step_dt),
                   "episode_length_s": float(base.cfg.episode_length_s)},
        "cameras": cams,
        "robot_joint_names": list(base.scene["robot"].joint_names),
        "parity": {"steps": T, "envs": n, "max_abs_err_torchscript_cpu_vs_original": err,
                   "max_abs_err_torchscript_cpu_vs_env_tokens_bf16": err_env,
                   "mean_reward": float(rec["reward"].mean()), "dones": int(rec["done"].sum())},
    }
    with open(os.path.join(args_cli.out_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    shutil.copy(args_cli.checkpoint, os.path.join(args_cli.out_dir, "checkpoint.pt"))
    print(f"[export] {ts_path}: parity max |TorchScript(cpu) - original| = {err:.2e} over {T} x {n}"
          + (f"; vs the env's bf16 token path {err_env:.2e}" if err_env is not None else "")
          + f"; CPU batch-1 forward {cpu_ms:.1f} ms ({torch.get_num_threads()} threads)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
