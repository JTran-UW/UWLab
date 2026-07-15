# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy-time helpers for a behavior-cloned :class:`PointNet`.

Reconstructs the raw :class:`PointNet` (+ the proprio/action z-scoring stats) from a
Lightning checkpoint written by ``train_point_net.py`` -- or the student of an rsl_rl
DAgger distillation checkpoint (``StudentTeacherPointCloud`` ``model_*.pt``) -- and runs
it on a live observation group, so rolling out a point-cloud policy (e.g. in ``play.py``)
needs no Lightning / rsl_rl dependency, only ``torch`` + ``point_net.py``.
"""

from __future__ import annotations

import os
import sys

import torch

# Policy modules live next to this file; make them importable regardless of caller cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion_point_net import DiffusionActionPointNet  # noqa: E402
from flatten_mlp import FlattenMLP  # noqa: E402
from history_point_net import HistoryPointNet  # noqa: E402
from pc_signature import signature_from_run_params  # noqa: E402
from point_net import PointNet  # noqa: E402
from residual_point_net import ResidualPointNet  # noqa: E402

# Must match pc_bc_module._ARCHITECTURES so a checkpoint loads into the class it trained.
_ARCHITECTURES = {
    "point_net": PointNet,
    "flatten_mlp": FlattenMLP,
    "residual_point_net": ResidualPointNet,
    "diffusion_point_net": DiffusionActionPointNet,
    "history_point_net": HistoryPointNet,
}

# Obs terms that are point clouds (stored flat as (N, num_points*point_dim)) -> the PointNet's set input.
# Mirrors train_point_net.py so eval proprio is built exactly as the model was trained.
_BC_PC_TERMS = {"scene_pc"}
# Proprio is an explicit ALLOWLIST mirroring train_point_net._PROPRIO_TERMS: the model sees
# only these terms (in the obs-group's declaration order) as the 18d proprio it was trained on.
# Any other term in the live group (privileged poses, contact flags, ... recorded for aux
# losses) is ignored at deploy, exactly as the trainer ignored it.
_BC_PROPRIO_TERMS = {"joint_pos", "end_effector_pose"}

# Prim-name -> segmentation label, mirroring train_point_net._SEG_LABEL (robot=0, insertive=-1,
# receptive=+1). When a model was trained with append_prim_semantic, the seg channel was DERIVED from
# the prim id; deploy re-derives the identical channel here from the live prim ids.
_BC_OBJECT_PRIMS = ("insertive", "receptive")
_BC_SEG_LABEL = {"robot": 0.0, "insertive": -1.0, "receptive": 1.0}


def _prim_seg_lookup(names, device) -> torch.Tensor:
    """(n_prims,) seg label indexed by prim id, from the checkpoint's pc_all_prim_names."""
    return torch.tensor(
        [_BC_SEG_LABEL[n if n in _BC_OBJECT_PRIMS else "robot"] for n in names],
        device=device, dtype=torch.float32,
    )


def _mlp_layer_dims(sd: dict, prefix: str) -> tuple[int, list[int]]:
    """(in_dim, [out_dim per Linear, in order]) of the MLP whose weights live under ``prefix``.

    Works for both key layouts our PointNets produce: torchvision ``MLP`` (Sequential ->
    ``<prefix>0.weight``) and ``ResidualMLP`` (``<prefix>linears.0.weight``). Linears are the
    2-D weights; LayerNorm weights are 1-D and skipped."""
    ws = [(k, v) for k, v in sd.items() if k.startswith(prefix) and k.endswith(".weight") and v.ndim == 2]
    ws.sort(key=lambda kv: [int(p) for p in kv[0].split(".") if p.isdigit()])
    return ws[0][1].shape[1], [v.shape[0] for _, v in ws]


def _sum_num_points(node) -> int:
    """Sum every ``num_points`` value in a nested yaml config subtree."""
    if isinstance(node, dict):
        return sum(v if k == "num_points" and isinstance(v, int) else _sum_num_points(v) for k, v in node.items())
    if isinstance(node, (list, tuple)):
        return sum(_sum_num_points(v) for v in node)
    return 0


def _dagger_num_points(ckpt_path: str) -> int | None:
    """Best-effort cloud size for a DAgger ckpt: sum the ``num_points`` of the student's
    ``pointcloud_groups`` terms from the run's ``params/{agent,env}.yaml`` sidecars.

    A set network doesn't store its point count, so this is the only place it exists. Returns
    None (with a warning) if the sidecars are missing/unparseable -- pass ``num_points``
    explicitly then (``convert_bc_to_jit.py --num_points``)."""
    import yaml

    params = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "params")
    try:
        with open(os.path.join(params, "agent.yaml")) as f:
            groups = list(yaml.unsafe_load(f)["policy"]["pointcloud_groups"])
        with open(os.path.join(params, "env.yaml")) as f:
            obs_cfg = yaml.unsafe_load(f)["observations"]
        return sum(_sum_num_points(obs_cfg[g]) for g in groups) or None
    except Exception as e:  # noqa: BLE001
        print(f"[load_bc_pointnet] WARNING: could not derive num_points from {params} ({type(e).__name__}: {e})")
        return None


def _from_dagger_ckpt(ckpt: dict, path: str, device: str, num_points: int | None) -> tuple:
    """Rebuild the PointNet student of an rsl_rl DAgger/distillation checkpoint.

    The ckpt is what ``OnPolicyRunner.save`` writes for a :class:`StudentTeacherPointCloud`:
    ``model_state_dict`` holds ``student.*`` (a PointNet/ResidualPointNet -- the SAME classes
    the BC trainer uses), the ``student_obs_normalizer`` proprio stats, an exploration-noise
    ``std``/``log_std`` param, and frozen ``teacher.*`` weights (ignored). No hyper_parameters
    are stored, so the architecture is inferred from the weight shapes (self-contained: works
    on a ckpt copied away from its run dir); only ``num_points`` needs the run's params/
    sidecars (or an explicit override). Returns ``(model, synthetic hp, norm)`` shaped exactly
    like the Lightning path so everything downstream (export/eval) is checkpoint-agnostic.
    """
    msd = ckpt["model_state_dict"]
    sd = {k[len("student.") :]: v for k, v in msd.items() if k.startswith("student.")}
    if not sd:
        raise ValueError(f"no 'student.*' keys in {path}; not a StudentTeacherPointCloud distillation ckpt")
    # ResidualMLP registers layers under `.linears.`; torchvision MLP uses bare Sequential indices.
    arch = "residual_point_net" if any(k.startswith("set_encoder.linears.") for k in sd) else "point_net"
    point_dim, encoder_dims = _mlp_layer_dims(sd, "set_encoder.")
    _, head_dims = _mlp_layer_dims(sd, "action_head.")
    out_dim, action_dims = head_dims[-1], head_dims[:-1]
    proprio_dim = sd["proprio_encoder.0.weight"].shape[1]
    noise = msd.get("std", msd.get("log_std"))  # exploration-noise param, always (num_actions,)
    if noise is None:
        raise ValueError(f"no 'std'/'log_std' exploration param in {path}; cannot infer action_dim")
    action_dim = noise.shape[0]
    if out_dim not in (action_dim, 2 * action_dim):
        raise ValueError(f"action head emits {out_dim}d but action_dim={action_dim}; unrecognized head layout")
    predict_std = out_dim == 2 * action_dim

    model = _ARCHITECTURES[arch](
        encoder_hidden_dims=encoder_dims,
        action_hidden_dims=action_dims,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        predict_std=predict_std,
        point_dim=point_dim,
    )
    model.load_state_dict(sd)
    model.to(device).eval()

    # Proprio norm: StudentTeacherPointCloud normalizes proprio with rsl_rl's EmpiricalNormalization,
    # whose forward is (x - mean) / (std + eps) with eps=1e-2 -- bake the eps into the saved std so the
    # shared (x - mean) / std deploy path is bit-identical. Keys absent => normalization was disabled.
    if "student_obs_normalizer._mean" in msd:
        proprio_mean = msd["student_obs_normalizer._mean"].squeeze(0).to(device)
        proprio_std = msd["student_obs_normalizer._std"].squeeze(0).to(device) + 1e-2
    else:
        proprio_mean = torch.zeros(proprio_dim, device=device)
        proprio_std = torch.ones(proprio_dim, device=device)
    # DAgger students regress the teacher's env-unit actions directly -> identity action norm.
    norm = {
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_mean": torch.zeros(action_dim, device=device),
        "action_std": torch.ones(action_dim, device=device),
    }

    num_points = num_points or _dagger_num_points(path)
    if num_points is None:
        raise ValueError(
            f"could not determine num_points for DAgger ckpt {path} (no params/ sidecars next to it); "
            "pass it explicitly (convert_bc_to_jit.py --num_points N)"
        )
    # PC observation signature (classes / per-class budget / frame / proprio layout) from
    # the run's params/ sidecars -- best-effort None when the ckpt was copied away from
    # its run dir. The env.yaml is the same source of truth the student trained on.
    pc_sig = signature_from_run_params(os.path.join(os.path.dirname(os.path.abspath(path)), "params"))
    if pc_sig is not None and pc_sig["proprio"]["dim"] != proprio_dim:
        print(f"[pc_signature] WARNING: sidecar proprio dim {pc_sig['proprio']['dim']} != "
              f"student proprio_dim {proprio_dim}; keeping the student's value")
        pc_sig["proprio"]["dim"] = proprio_dim
    hp = {
        "source": "rsl_rl_distillation",
        "architecture": arch,
        "encoder_dims": encoder_dims,
        "action_dims": action_dims,
        "proprio_dim": proprio_dim,
        "action_dim": action_dim,
        "predict_std": predict_std,
        "point_dim": point_dim,
        "num_points": int(num_points),
        "pc_signature": pc_sig,
    }
    return model, hp, norm


def _from_lightning_ckpt(ckpt: dict, device: str) -> tuple:
    """Rebuild the PointNet of a Lightning BC checkpoint (``train_point_net.py``).

    ``hyper_parameters`` holds the PointNet shape, ``state_dict`` holds ``model.*`` weights
    plus the proprio/action z-scoring buffers. Returns ``(model, hp, norm)``."""
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
    if arch == "diffusion_point_net":  # DDPM schedule + DDIM sample steps baked into the head
        kwargs["num_train_timesteps"] = hp.get("num_train_timesteps", 100)
        kwargs["num_sample_steps"] = hp.get("num_sample_steps", 10)
    if arch == "history_point_net":  # causal-Transformer geometry (must match training)
        kwargs["history_len"] = hp.get("history_len", 8)
        kwargs["d_model"] = hp.get("d_model", 256)
        kwargs["n_heads"] = hp.get("n_heads", 4)
        kwargs["n_layers"] = hp.get("n_layers", 4)
        kwargs["transformer_dropout"] = hp.get("transformer_dropout", 0.1)
    model = _ARCHITECTURES[arch](**kwargs)
    model.load_state_dict({k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")})
    model.to(device).eval()
    norm = {k: sd[k].to(device) for k in ("proprio_mean", "proprio_std", "action_mean", "action_std")}
    return model, hp, norm


def load_bc_pointnet(path: str, device: str, num_points: int | None = None) -> dict:
    """Reconstruct a PointNet (+ normalization stats) from a BC OR DAgger checkpoint.

    Accepts either checkpoint flavor and returns the same ``bc`` dict, so everything
    downstream (``bc_actions``, ``export_bc_jit``, eval scripts) is checkpoint-agnostic:

    * **Lightning BC** (``train_point_net.py``): ``hyper_parameters`` + ``state_dict`` with
      ``model.*`` weights and proprio/action z-scoring buffers.
    * **rsl_rl DAgger distillation** (``model_*.pt`` with ``model_state_dict``): the
      ``StudentTeacherPointCloud`` student is rebuilt from its weight shapes; proprio norm
      comes from the ``student_obs_normalizer``, action norm is identity (see
      :func:`_from_dagger_ckpt`). ``num_points`` (unused by the set network itself, but
      recorded in the JIT meta) is read from the run's params/ sidecars unless given here.

    Returns a dict with ``model`` (eval-mode PointNet on ``device``), ``hp`` (saved --
    or, for DAgger, reconstructed -- hyper-parameters), and the four normalization tensors
    (``proprio_mean/std``, ``action_mean/std``).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "hyper_parameters" in ckpt:
        model, hp, norm = _from_lightning_ckpt(ckpt, device)
    elif "model_state_dict" in ckpt:
        model, hp, norm = _from_dagger_ckpt(ckpt, path, device, num_points)
    else:
        raise KeyError(
            f"{path} has neither 'hyper_parameters' (Lightning BC) nor 'model_state_dict' (rsl_rl DAgger); "
            f"top-level keys: {list(ckpt)}"
        )
    # PER-PRIM deploy (None for legacy flat models). The live env emits scene_pc with a trailing
    # per-point prim-id channel; bc_actions peels it, keeps only `pc_parts`' prims (mapped via
    # `pc_all_prim_names`), and zero-pads to `pc_pad_target` -- the size the model trained on.
    return {
        "model": model, "hp": hp, **norm,
        # point_dim / proprio_dim are surfaced here (not just on the model) so bc_actions can
        # assemble inputs identically for an eager model OR a baked JIT policy (load_bc_jit).
        "point_dim": int(model.point_dim),
        "proprio_dim": int(norm["proprio_mean"].shape[0]),
        "pc_parts": hp.get("pc_parts", None),
        "pc_all_prim_names": hp.get("pc_all_prim_names", None),
        "pc_pad_target": hp.get("pc_pad_target", None),
        # If set, the model's last cloud channel is a seg label DERIVED from the prim id (not present
        # in the live cloud); bc_actions re-derives and appends it before select+pad.
        "append_prim_semantic": bool(hp.get("append_prim_semantic", False)),
        # History-conditioned sequence policy: bc_actions keeps a rolling per-env (state, action)
        # buffer and reads the last state token. is_sequence routes bc_actions; history_len is the
        # window; action_dim sizes the action buffer. bc_reset() clears history for terminated envs.
        "is_sequence": bool(getattr(model, "is_sequence", False)),
        "history_len": int(getattr(model, "history_len", 1)),
        "action_dim": int(norm["action_mean"].shape[0]),
        # PC observation signature (see pc_signature.py): how the cloud + proprio were built
        # (classes, per-class budget, frame, seg labels, proprio term layout). None for old
        # checkpoints that predate it.
        "pc_signature": hp.get("pc_signature"),
    }


def _select_pad_batched(
    xyz: torch.Tensor, prim_id: torch.Tensor, selected_ids: torch.Tensor, target: int
) -> torch.Tensor:
    """Per-prim deploy: keep points whose prim id is in ``selected_ids`` and zero-pad to a fixed
    ``target`` size, batched over envs. ``xyz`` is ``(N, M, C)``, ``prim_id`` ``(N, M)``.

    Mirrors the trainer's ``_select_and_pad`` but deterministic (no random subset) and batched:
    selected points come first (stable argsort of the mask), the rest of each row is zeroed.
    When an env has more than ``target`` selected points the extra are dropped (rare; only if
    pc_pad_target < the selection's max count)."""
    N, M, C = xyz.shape
    mask = torch.isin(prim_id.long(), selected_ids)  # (N, M)
    counts = mask.sum(1)  # (N,)
    order = torch.argsort(mask.to(torch.int8), dim=1, descending=True, stable=True)  # selected first
    take = order[:, :target]  # (N, target)
    gathered = torch.gather(xyz, 1, take.unsqueeze(-1).expand(-1, -1, C))  # (N, target, C)
    pad = torch.arange(target, device=xyz.device).unsqueeze(0) >= counts.unsqueeze(1)  # (N, target)
    return gathered.masked_fill(pad.unsqueeze(-1), 0.0)


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
    if bc.get("pc_all_prim_names"):
        # PER-PRIM: the live term is (N, M, raw + 1) -- the raw cloud channels plus a trailing
        # prim-id channel. Peel the prim id, keep only points whose prim is in pc_parts, and zero-pad
        # to the trained size (see _select_pad_batched). (Legacy flat models leave this None.)
        seg_appended = bc.get("append_prim_semantic", False)
        # When the model was trained with append_prim_semantic, its point_dim INCLUDES that derived
        # seg channel, which is NOT in the live cloud -> the live cloud has point_dim-1 raw channels.
        raw = bc["point_dim"] - (1 if seg_appended else 0)
        full = g[pc_key].reshape(g[pc_key].shape[0], -1, raw + 1)
        names = list(bc["pc_all_prim_names"])
        parts = list(bc["pc_parts"]) if bc.get("pc_parts") else names
        selected_ids = torch.tensor([names.index(p) for p in parts], device=full.device)
        xyz, pid = full[..., :raw], full[..., raw]
        if seg_appended:  # re-derive the seg label from prim id and append: [xyz, seg]
            seg = _prim_seg_lookup(names, full.device)[pid.long()]
            xyz = torch.cat([xyz, seg.unsqueeze(-1)], dim=-1)
        pc = _select_pad_batched(xyz, pid, selected_ids, int(bc["pc_pad_target"]))
    else:
        # point_dim (3 or 4) is baked into the policy; the live scene_pc must match.
        pc = g[pc_key].reshape(g[pc_key].shape[0], -1, bc["point_dim"])
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
    # History-conditioned policy: push this step's (cloud, proprio) into a rolling per-env buffer and
    # read the action from the last state token (see _bc_actions_history). Stateful across calls.
    if bc.get("is_sequence"):
        return _bc_actions_history(bc, pc, proprio)
    # JIT policy (load_bc_jit): normalization + denormalization are baked into the scripted
    # module, so feed RAW (points, proprio) and return its output directly. Eager path z-scores
    # proprio, runs the model, and denormalizes here.
    if bc.get("jit"):
        out = bc["model"](pc, proprio)
        return out[0] if isinstance(out, tuple) else out  # returns_std export -> (action, std)
    proprio = (proprio - bc["proprio_mean"]) / bc["proprio_std"]
    out = bc["model"](pc, proprio)
    mean = out[0] if isinstance(out, tuple) else out  # predict_std -> (mean, log_std)
    return mean * bc["action_std"] + bc["action_mean"]


def _bc_actions_history(bc: dict, pc: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    """Rolling-buffer inference for a :class:`HistoryPointNet`.

    Maintains a per-env history of the last ``history_len`` states (raw cloud + z-scored proprio) and
    the z-scored actions executed at those states, mirroring the training layout. Each call shifts the
    newest state in, runs the causal Transformer, reads the action at the LAST state token, records it
    as this step's action (so it becomes a past action-token next step), and returns it in env units.

    ``pc`` is the assembled ``(B, N, point_dim)`` cloud and ``proprio`` the raw ``(B, proprio_dim)``
    proprio (both exactly as the feed-forward path builds them). Buffers live in ``bc`` under
    ``_hist_*`` and are cleared per-env by :func:`bc_reset` on episode termination."""
    H = bc["history_len"]
    B, N, C = pc.shape
    proprio_n = (proprio - bc["proprio_mean"]) / bc["proprio_std"]
    # Lazily allocate the buffers once B / shapes / device are known (they can't be sized before the
    # first observation). _hist_count tracks how many real (non-pad) frames each env has accumulated.
    if bc.get("_hist_pc") is None or bc["_hist_pc"].shape[0] != B:
        dev = pc.device
        bc["_hist_pc"] = torch.zeros(B, H, N, C, device=dev)
        bc["_hist_proprio"] = torch.zeros(B, H, proprio.shape[-1], device=dev)
        bc["_hist_action"] = torch.zeros(B, H, bc["action_dim"], device=dev)
        bc["_hist_count"] = torch.zeros(B, dtype=torch.long, device=dev)
    # Shift the window left by one and append the newest state (actions get a placeholder last slot;
    # the model only consumes actions[:, :H-1], and we overwrite it below with the predicted action).
    bc["_hist_pc"] = torch.cat([bc["_hist_pc"][:, 1:], pc.unsqueeze(1)], dim=1)
    bc["_hist_proprio"] = torch.cat([bc["_hist_proprio"][:, 1:], proprio_n.unsqueeze(1)], dim=1)
    bc["_hist_action"] = torch.cat(
        [bc["_hist_action"][:, 1:], torch.zeros(B, 1, bc["action_dim"], device=pc.device)], dim=1
    )
    bc["_hist_count"] = (bc["_hist_count"] + 1).clamp_max(H)
    # valid[:, i] True for the newest `count` slots (real frames sit at the END; older = left-pad).
    idx = torch.arange(H, device=pc.device).unsqueeze(0)                 # (1, H)
    valid = idx >= (H - bc["_hist_count"].unsqueeze(1))                  # (B, H)
    pred_n = bc["model"].predict(bc["_hist_pc"], bc["_hist_proprio"], bc["_hist_action"], valid)  # (B, ad)
    bc["_hist_action"][:, -1] = pred_n  # record the executed (z-scored) action for the next step
    return pred_n * bc["action_std"] + bc["action_mean"]


def bc_reset(bc: dict, dones: torch.Tensor) -> None:
    """Clear the history buffer for terminated envs (feed-forward policies: no-op).

    Call after ``env.step`` with the step's ``dones`` (bool ``(B,)``). Resetting ``_hist_count`` to 0
    for done envs makes their next :func:`bc_actions` call start from an empty (fully left-padded)
    window, so a new episode never attends to the previous one's states/actions."""
    if not bc.get("is_sequence") or bc.get("_hist_count") is None:
        return
    d = dones.to(bc["_hist_count"].device).bool()
    if not bool(d.any()):
        return
    bc["_hist_count"][d] = 0
    bc["_hist_pc"][d] = 0.0
    bc["_hist_proprio"][d] = 0.0
    bc["_hist_action"][d] = 0.0


# ---------------------------------------------------------------------------
# JIT export / load -- fast inference for real eval
# ---------------------------------------------------------------------------
class _BCPolicyJIT(torch.nn.Module):
    """Self-contained deploy policy: ``forward(points, proprio) -> action`` with the
    proprio z-scoring and action de-normalization BAKED IN as buffers.

    ``points`` is the (already-assembled, already-selected-for-per-prim) cloud
    ``(B, N, point_dim)`` and ``proprio`` is the raw ``(B, proprio_dim)`` proprio (already
    joint-trimmed). Output is the action in env units -- the deterministic mean (a
    ``predict_std`` head's std is dropped, matching :func:`bc_actions`), or
    ``(action, std)`` when ``export_std=True`` (std also in env units, from the
    ``predict_std`` head's log-std, clamped like ``StudentTeacherPointCloud``). Traced +
    saved by :func:`export_bc_jit`; reloaded by :func:`load_bc_jit`."""

    # Same clamp StudentTeacherPointCloud.act_inference_with_std applies to the head's log-std.
    LOG_STD_LIMITS = (-5.0, 2.0)

    def __init__(self, model, proprio_mean, proprio_std, action_mean, action_std, predict_std: bool,
                 export_std: bool = False):
        super().__init__()
        self.model = model
        self.predict_std = predict_std
        self.export_std = export_std
        self.register_buffer("proprio_mean", proprio_mean)
        self.register_buffer("proprio_std", proprio_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)

    def forward(self, points: torch.Tensor, proprio: torch.Tensor):
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        out = self.model(points, proprio)
        if self.predict_std:
            mean, log_std = out
        else:
            mean, log_std = out, None
        action = mean * self.action_std + self.action_mean
        if self.export_std:
            # log_std is in normalized-action units (BC trains the NLL on z-scored actions;
            # DAgger action norm is identity) -> scale by action_std to get env units.
            std = torch.exp(log_std.clamp(*self.LOG_STD_LIMITS)) * self.action_std
            return action, std
        return action


def export_bc_jit(bc: dict, out_path: str, device: str = "cpu", export_std: bool = False) -> torch.jit.ScriptModule:
    """Trace a loaded BC policy (from :func:`load_bc_pointnet`) into a self-contained JIT
    module and save it to ``out_path`` (+ a ``<out_path>.meta.json`` sidecar describing the
    expected input layout). Returns the scripted module.

    ``export_std=True`` makes the JIT ``forward`` return ``(action, std)`` instead of just
    the action (both in env units; requires a ``predict_std`` checkpoint). This is the
    teacher format ``StudentTeacherPointCloud(teacher_returns_std=True)`` consumes for
    DEXTRAH-style weighted distillation; the sidecar records it as ``returns_std``.

    Tracing (not scripting) is used because the PointNet's ``forward`` has a
    ``Union[Tensor, Tuple]`` return (``return_pooled``) that scripts poorly; there is no
    data-dependent control flow in the actual compute, so a trace is exact. Verified against
    the eager policy before saving.
    """
    import json

    hp = bc["hp"]
    if hp.get("architecture") == "history_point_net" or bc.get("is_sequence"):
        # The history policy is stateful (a rolling per-env buffer maintained by bc_actions), so a
        # single-frame (points, proprio) -> action trace can't capture it. Run it eager via bc_actions.
        raise NotImplementedError(
            "export_bc_jit does not support history_point_net (stateful rolling-buffer policy); "
            "evaluate it eagerly through bc_actions instead."
        )
    point_dim = bc["point_dim"]
    proprio_dim = bc["proprio_dim"]
    num_points = int(hp["num_points"])  # total cloud size the model expects (per-prim: sum of caps)
    action_dim = int(bc["action_mean"].shape[0])
    predict_std = bool(hp.get("predict_std", False))
    if export_std and not predict_std:
        raise ValueError(
            "export_std=True requires a predict_std checkpoint (action head emitting (mean, log_std)); "
            "this checkpoint's head is mean-only, so there is no std to export."
        )

    wrapper = _BCPolicyJIT(
        bc["model"], bc["proprio_mean"], bc["proprio_std"], bc["action_mean"], bc["action_std"],
        predict_std=predict_std, export_std=export_std,
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
    eager_outs = eager if isinstance(eager, tuple) else (eager,)
    traced_outs = traced if isinstance(traced, tuple) else (traced,)
    diff = max((e - t).abs().max().item() for e, t in zip(eager_outs, traced_outs))
    # Tolerance is loose because freeze/optimize_for_inference fuses ops, perturbing the
    # output at the ~1e-5 level -- negligible in action units, not a correctness issue.
    assert diff < 1e-3, f"traced JIT diverges from eager by {diff:.2e}"
    for e in eager_outs:
        assert e.shape == (2, action_dim), f"unexpected output shape {tuple(e.shape)}"

    meta = {
        "architecture": hp.get("architecture", "point_net"),
        "point_dim": point_dim,
        "num_points": num_points,
        "proprio_dim": proprio_dim,
        "action_dim": action_dim,
        "predict_std": predict_std,
        "returns_std": export_std,  # forward -> (action, std) instead of action
        "pc_parts": bc.get("pc_parts"),
        "pc_all_prim_names": bc.get("pc_all_prim_names"),
        "pc_pad_target": bc.get("pc_pad_target"),
        "append_prim_semantic": bc.get("append_prim_semantic", False),
        # Full PC observation signature so the real-eval harness can configure its
        # perception pipeline (classes/budgets/frame/proprio) from the export alone.
        "pc_signature": bc.get("pc_signature"),
    }
    # The meta rides INSIDE the .pt (TorchScript extra file) so deploy needs a single file;
    # the .meta.json sidecar is written too, purely for auditing without loading the archive.
    meta_json = json.dumps(meta, indent=2)
    scripted.save(out_path, _extra_files={"meta.json": meta_json})
    with open(out_path + ".meta.json", "w") as f:
        f.write(meta_json)
    out_desc = f"(action[B,{action_dim}], std[B,{action_dim}])" if export_std else f"action[B,{action_dim}]"
    print(f"[export_bc_jit] saved {out_path} (jit) + .meta.json | "
          f"in=(points[B,{num_points},{point_dim}], proprio[B,{proprio_dim}]) -> {out_desc} "
          f"| trace-vs-eager max diff {diff:.2e}")
    return scripted


def load_bc_jit(path: str, device: str) -> dict:
    """Load a JIT policy saved by :func:`export_bc_jit` into the same ``bc`` dict shape that
    :func:`bc_actions` consumes (with ``jit=True`` so it feeds RAW inputs). The input layout
    (point_dim / proprio_dim / per-prim layout) that the eager path otherwise reads from the
    model + norm buffers comes from the meta embedded in the ``.pt`` itself (so only one file
    needs transferring); older exports without it fall back to the ``<path>.meta.json``
    sidecar."""
    import json

    extra = {"meta.json": ""}
    model = torch.jit.load(path, map_location=device, _extra_files=extra).eval()
    if extra["meta.json"]:  # missing in pre-embed exports -> torch leaves it empty
        meta = json.loads(extra["meta.json"])
    else:
        with open(path + ".meta.json") as f:
            meta = json.load(f)
    return {
        "model": model, "jit": True, "meta": meta,
        "returns_std": meta.get("returns_std", False),  # forward -> (action, std)
        "point_dim": meta["point_dim"],
        "proprio_dim": meta["proprio_dim"],
        "pc_parts": meta.get("pc_parts"),
        "pc_all_prim_names": meta.get("pc_all_prim_names"),
        "pc_pad_target": meta.get("pc_pad_target"),
        "append_prim_semantic": meta.get("append_prim_semantic", False),
        "pc_signature": meta.get("pc_signature"),
    }
