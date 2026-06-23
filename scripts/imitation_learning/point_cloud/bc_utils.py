# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-time helpers for a behavior-cloned :class:`PointNet`.

Reconstructs the raw :class:`PointNet` (+ the proprio/action z-scoring stats) from a
Lightning checkpoint written by ``train_point_net.py``, and runs it on a live
observation group -- so rolling out a BC PointNet (e.g. in ``play.py``) needs no
Lightning dependency, only ``torch`` + ``point_net.py``.
"""

from __future__ import annotations

import os
import sys

import torch

# Policy modules live next to this file; make them importable regardless of caller cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flatten_mlp import FlattenMLP  # noqa: E402
from point_net import PointNet  # noqa: E402
from residual_point_net import ResidualPointNet  # noqa: E402

# Must match train_point_net.py's _ARCHITECTURES so a checkpoint loads into the class it trained.
_ARCHITECTURES = {"point_net": PointNet, "flatten_mlp": FlattenMLP, "residual_point_net": ResidualPointNet}

# Obs terms that are point clouds (stored flat as (N, num_points*point_dim)) -> the PointNet's set input.
# Mirrors train_point_net.py so eval proprio is built exactly as the model was trained.
_BC_PC_TERMS = {"scene_pc"}
# Proprio is an explicit ALLOWLIST mirroring train_point_net._PROPRIO_TERMS: the model sees
# only these terms (in the obs-group's declaration order) as the 18d proprio it was trained on.
# Any other term in the live group (privileged poses, contact flags, ... recorded for aux
# losses) is ignored at deploy, exactly as the trainer ignored it.
_BC_PROPRIO_TERMS = {"joint_pos", "end_effector_pose"}


def load_bc_pointnet(path: str, device: str) -> dict:
    """Reconstruct a PointNet (+ saved normalization stats) from a Lightning BC checkpoint.

    The checkpoint is the plain ``torch.save`` dict Lightning writes: ``hyper_parameters``
    holds the PointNet shape, ``state_dict`` holds ``model.*`` weights plus the
    proprio/action z-scoring buffers. We rebuild the raw :class:`PointNet` so eval needs
    no Lightning dependency.

    Returns a dict with ``model`` (eval-mode PointNet on ``device``), ``hp`` (the saved
    hyper-parameters), and the four normalization tensors (``proprio_mean/std``,
    ``action_mean/std``).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp = ckpt["hyper_parameters"]
    sd = ckpt["state_dict"]
    arch = hp.get("architecture", "point_net")  # older ckpts predate the field -> PointNet
    kwargs = dict(
        encoder_hidden_dims=hp["encoder_dims"],
        action_hidden_dims=hp["action_dims"],
        proprio_dim=hp["proprio_dim"],
        action_dim=hp["action_dim"],
        predict_std=hp["predict_std"],
        point_dim=hp.get("point_dim", 3),  # 3 = xyz; 4 = xyz + segmentation label
    )
    if arch == "flatten_mlp":
        kwargs["num_points"] = hp["num_points"]  # FlattenMLP needs the fixed cloud size
    model = _ARCHITECTURES[arch](**kwargs)
    model.load_state_dict({k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")})
    model.to(device).eval()
    norm = {k: sd[k].to(device) for k in ("proprio_mean", "proprio_std", "action_mean", "action_std")}
    # PER-PRIM deploy layout (None for legacy flat models). The live env emits ONE flat
    # scene_pc term holding every prim's padded block concatenated in `pc_all_prim_names`
    # order; we slice it by `pc_all_prim_caps` and keep only `pc_parts`, in that order --
    # exactly the blocks the model trained on. See bc_actions.
    pc_parts = hp.get("pc_parts", None)
    pc_all_prim_names = hp.get("pc_all_prim_names", None)
    pc_all_prim_caps = hp.get("pc_all_prim_caps", None)
    return {
        "model": model, "hp": hp, **norm,
        # point_dim / proprio_dim are surfaced here (not just on the model) so bc_actions can
        # assemble inputs identically for an eager model OR a baked JIT policy (load_bc_jit).
        "point_dim": int(model.point_dim),
        "proprio_dim": int(norm["proprio_mean"].shape[0]),
        "pc_parts": pc_parts,
        "pc_all_prim_names": pc_all_prim_names,
        "pc_all_prim_caps": pc_all_prim_caps,
    }


def bc_actions(bc: dict, obs, group_name: str) -> torch.Tensor:
    """Run the BC PointNet on the ``group_name`` obs group and return denormalized actions.

    The group is a per-term dict (``concatenate_terms=False``): the point cloud term is
    reshaped to ``(N, num_points, point_dim)`` and the allowlisted proprio terms
    (``_BC_PROPRIO_TERMS``) are concatenated, in declaration order, into proprio, exactly as
    ``train_point_net.py`` did. Any other term in the group (privileged poses, contact flags
    recorded for aux losses) is ignored. Proprio is z-scored with the saved stats; the
    predicted (normalized) action is denormalized back to env action units.
    """
    g = obs[group_name]
    keys = list(g.keys())
    pc_key = next(k for k in keys if k in _BC_PC_TERMS)
    # point_dim (3 or 4) is baked into the policy; the live scene_pc must match.
    pc = g[pc_key].reshape(g[pc_key].shape[0], -1, bc["point_dim"])
    # PER-PRIM models: the live cloud is every prim's padded block concatenated in
    # `pc_all_prim_names` order. Slice by the stored caps and keep only `pc_parts`, in the
    # SAME order the model trained on. (Legacy flat models leave pc_all_prim_names None.)
    if bc.get("pc_all_prim_names"):
        names = list(bc["pc_all_prim_names"])
        caps = [int(c) for c in bc["pc_all_prim_caps"]]
        parts = list(bc["pc_parts"]) if bc.get("pc_parts") else names
        total = sum(caps)
        assert pc.shape[1] == total, (
            f"live scene_pc has {pc.shape[1]} points but per-prim layout expects {total} "
            f"(prims={names}, caps={caps}); eval env per_prim config must match training."
        )
        off = {}
        acc = 0
        for n, c in zip(names, caps):
            off[n] = (acc, acc + c)
            acc += c
        pc = torch.cat([pc[:, off[p][0]:off[p][1], :] for p in parts], dim=1)
    proprio_keys = [k for k in keys if k in _BC_PROPRIO_TERMS]
    # Real-robot proprio trim: the model's proprio_dim (== saved proprio_mean length) may be SMALLER
    # than the live obs provides. That deficit is the trailing joint_pos entries -- the last 6 of the
    # 12 joint_pos dims are Robotiq gripper mimic joints that DO NOT EXIST on the real robot, so a
    # sim2real policy was trained on joint_pos[:6] (see train_point_net.py joint_pos_dims). Drop the
    # same trailing joint_pos dims here so eval/deploy proprio matches what the model trained on.
    target = bc["proprio_dim"]
    deficit = sum(g[k].shape[-1] for k in proprio_keys) - target
    parts = []
    for k in proprio_keys:
        a = g[k]
        if k == "joint_pos" and deficit > 0:
            a = a[..., : a.shape[-1] - deficit]
        parts.append(a)
    proprio = torch.cat(parts, dim=-1)
    assert proprio.shape[-1] == target, (
        f"proprio dim {proprio.shape[-1]} != model proprio_dim {target}; proprio terms {proprio_keys} "
        f"(joint_pos trimmed by {max(deficit, 0)} for real-robot gripper dims)"
    )
    # JIT policy (load_bc_jit): normalization + denormalization are baked into the scripted
    # module, so feed RAW (points, proprio) and return its output directly. Eager path z-scores
    # proprio, runs the model, and denormalizes here.
    if bc.get("jit"):
        return bc["model"](pc, proprio)
    proprio = (proprio - bc["proprio_mean"]) / bc["proprio_std"]
    out = bc["model"](pc, proprio)
    mean = out[0] if isinstance(out, tuple) else out  # predict_std -> (mean, log_std)
    return mean * bc["action_std"] + bc["action_mean"]


# ---------------------------------------------------------------------------
# JIT export / load -- fast inference for real eval
# ---------------------------------------------------------------------------
class _BCPolicyJIT(torch.nn.Module):
    """Self-contained deploy policy: ``forward(points, proprio) -> action`` with the
    proprio z-scoring and action de-normalization BAKED IN as buffers.

    ``points`` is the (already-assembled, already-selected-for-per-prim) cloud
    ``(B, N, point_dim)`` and ``proprio`` is the raw ``(B, proprio_dim)`` proprio (already
    joint-trimmed). Output is the action in env units -- the deterministic mean (a
    ``predict_std`` head's std is dropped, matching :func:`bc_actions`). Traced + saved by
    :func:`export_bc_jit`; reloaded by :func:`load_bc_jit`."""

    def __init__(self, model, proprio_mean, proprio_std, action_mean, action_std, predict_std: bool):
        super().__init__()
        self.model = model
        self.predict_std = predict_std
        self.register_buffer("proprio_mean", proprio_mean)
        self.register_buffer("proprio_std", proprio_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)

    def forward(self, points: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        out = self.model(points, proprio)
        mean = out[0] if self.predict_std else out  # predict_std head -> (mean, log_std)
        return mean * self.action_std + self.action_mean


def export_bc_jit(bc: dict, out_path: str, device: str = "cpu") -> torch.jit.ScriptModule:
    """Trace a loaded BC policy (from :func:`load_bc_pointnet`) into a self-contained JIT
    module and save it to ``out_path`` (+ a ``<out_path>.meta.json`` sidecar describing the
    expected input layout). Returns the scripted module.

    Tracing (not scripting) is used because the PointNet's ``forward`` has a
    ``Union[Tensor, Tuple]`` return (``return_pooled``) that scripts poorly; there is no
    data-dependent control flow in the actual compute, so a trace is exact. Verified against
    the eager policy before saving.
    """
    import json

    hp = bc["hp"]
    point_dim = bc["point_dim"]
    proprio_dim = bc["proprio_dim"]
    num_points = int(hp["num_points"])  # total cloud size the model expects (per-prim: sum of caps)
    action_dim = int(bc["action_mean"].shape[0])
    predict_std = bool(hp.get("predict_std", False))

    wrapper = _BCPolicyJIT(
        bc["model"], bc["proprio_mean"], bc["proprio_std"], bc["action_mean"], bc["action_std"],
        predict_std=predict_std,
    ).to(device).eval()

    ex_points = torch.randn(2, num_points, point_dim, device=device)
    ex_proprio = torch.randn(2, proprio_dim, device=device)
    with torch.no_grad():
        eager = wrapper(ex_points, ex_proprio)
        scripted = torch.jit.trace(wrapper, (ex_points, ex_proprio))
        # freeze() inlines the buffers/params as constants and enables inference graph
        # optimizations (constant folding, conv/linear fusion); optimize_for_inference adds
        # more. This is where the actual speedup comes from -- a bare trace barely helps.
        # Both are best-effort: fall back to the plain trace if a torch version balks.
        try:
            scripted = torch.jit.freeze(scripted)
            scripted = torch.jit.optimize_for_inference(scripted)
        except Exception as e:  # noqa: BLE001
            print(f"[export_bc_jit] WARNING: freeze/optimize_for_inference failed ({type(e).__name__}: {e}); "
                  f"saving the un-frozen trace.")
        traced = scripted(ex_points, ex_proprio)
    diff = (eager - traced).abs().max().item()
    # Tolerance is loose because freeze/optimize_for_inference fuses ops, perturbing the
    # output at the ~1e-5 level -- negligible in action units, not a correctness issue.
    assert diff < 1e-3, f"traced JIT diverges from eager by {diff:.2e}"
    assert eager.shape == (2, action_dim), f"unexpected action shape {tuple(eager.shape)}"

    scripted.save(out_path)
    meta = {
        "architecture": hp.get("architecture", "point_net"),
        "point_dim": point_dim,
        "num_points": num_points,
        "proprio_dim": proprio_dim,
        "action_dim": action_dim,
        "predict_std": predict_std,
        "pc_parts": bc.get("pc_parts"),
        "pc_all_prim_names": bc.get("pc_all_prim_names"),
        "pc_all_prim_caps": bc.get("pc_all_prim_caps"),
    }
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export_bc_jit] saved {out_path} (jit) + .meta.json | "
          f"in=(points[B,{num_points},{point_dim}], proprio[B,{proprio_dim}]) -> action[B,{action_dim}] "
          f"| trace-vs-eager max diff {diff:.2e}")
    return scripted


def load_bc_jit(path: str, device: str) -> dict:
    """Load a JIT policy saved by :func:`export_bc_jit` into the same ``bc`` dict shape that
    :func:`bc_actions` consumes (with ``jit=True`` so it feeds RAW inputs). The
    ``<path>.meta.json`` sidecar supplies the input layout (point_dim / proprio_dim / per-prim
    layout) that the eager path otherwise reads from the model + norm buffers."""
    import json

    model = torch.jit.load(path, map_location=device).eval()
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    return {
        "model": model, "jit": True, "meta": meta,
        "point_dim": meta["point_dim"],
        "proprio_dim": meta["proprio_dim"],
        "pc_parts": meta.get("pc_parts"),
        "pc_all_prim_names": meta.get("pc_all_prim_names"),
        "pc_all_prim_caps": meta.get("pc_all_prim_caps"),
    }
