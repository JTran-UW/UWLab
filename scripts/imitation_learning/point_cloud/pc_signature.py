# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Point-cloud observation SIGNATURE: a JSON-able description of how a policy's cloud +
proprio observations are built, saved with every BC/DAgger checkpoint and JIT export so
the real-robot eval harness (``eval_real_robot_pc.py``) can configure its perception
pipeline (which classes to segment, per-class point budgets, seg channel on/off, output
frame) without guessing.

The signature is extracted from the SAME source of truth the sim used:

* **BC (collection time)** -- the live env cfg's obs group (``signature_from_obs_group``),
  stamped into the demo HDF5 (``data.attrs['pc_signature']``), threaded by
  ``train_point_net.py`` into the Lightning hparams (with trainer-side adjustments:
  cloud subsampling, ``joint_pos_dims`` trim, per-prim selection).
* **DAgger (load time)** -- the run's ``params/{env,agent}.yaml`` sidecars
  (``signature_from_run_params``), read by ``bc_utils._from_dagger_ckpt``.

Either way ``bc_utils.export_bc_jit`` writes it into the ``<jit>.meta.json`` sidecar.

Schema (all keys always present; unknown/inapplicable -> None)::

    {
      "version": 1,
      "pc_term": "ScenePointCloud" | "OccludedScenePointCloud" | ...,
      "frame": "wrist_3_link" | "base" | "camera",   # cloud output frame
      "num_points": 1024,
      "point_dim": 3 | 4,                             # 4 = xyz + seg label channel
      "include_segmentation": bool,
      "segmentation_labels": {"robot": 0.0, "insertive": -1.0, "receptive": 1.0} | None,
      "classes": ["robot", "insertive", "receptive"], # order matches class_ratios
      "class_ratios": [0.0, 0.5, 0.5] | None,
      "class_points": {"robot": 0, "insertive": 512, "receptive": 512} | None,
      "robot_body_names": [...] | None,   # regex filters; [] = robot excluded entirely,
                                          # None = whole robot (clean ScenePointCloud)
      "zero_pad_missing_class": bool | None,          # occluded term only
      # per-prim deploy layout (None for flat models; mirrors the hp fields)
      "pc_parts": [...] | None, "pc_all_prim_names": [...] | None,
      "pc_pad_target": int | None, "append_prim_semantic": bool,
      # proprio layout, in the trained declaration order
      "proprio": {"terms": ["joint_pos", "end_effector_pose"],
                  "joint_pos_dims": 6 | 12, "dim": 12},
    }
"""

from __future__ import annotations

import json
import os

# Class order is FIXED across the sim terms: class_ratios is always (robot, insertive,
# receptive) -- see ScenePointCloud / OccludedScenePointCloud.
PC_CLASSES = ("robot", "insertive", "receptive")
# Default per-class seg labels (mirrors observations.py / train seg-channel convention).
DEFAULT_SEG_LABELS = {"robot": 0.0, "insertive": -1.0, "receptive": 1.0}
# Proprio terms a BC/DAgger student consumes, and their dims (joint_pos is dynamic).
_EE_POSE_DIMS = {"axis_angle": 6, "quat": 7}


def ratio_to_counts(ratios, total: int) -> tuple[int, ...]:
    """Per-class ratios -> integer counts summing EXACTLY to ``total`` (largest-remainder
    rounding). Torch-free copy of ``omnireset.mdp.observations.ratio_to_counts`` so the
    real-eval side reproduces the sim's per-class budget bit-for-bit."""
    raw = [r * total for r in ratios]
    counts = [int(x) for x in raw]  # floor (ratios are non-negative)
    rem = int(total - sum(counts))
    order = sorted(range(len(ratios)), key=lambda i: raw[i] - counts[i], reverse=True)
    for k in range(max(rem, 0)):
        counts[order[k % len(counts)]] += 1
    return tuple(counts)


def _get(node, key, default=None):
    """Read ``key`` from a dict (yaml sidecar) OR an attribute (live cfg object)."""
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _term_class_name(term_cfg) -> str | None:
    """The obs-term class/function name: from a live cfg's ``func`` or the yaml's
    ``func: module:Name`` string."""
    func = _get(term_cfg, "func")
    if func is None:
        return None
    if isinstance(func, str):  # yaml sidecar: "pkg.mod:ScenePointCloud"
        return func.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
    return getattr(func, "__name__", type(func).__name__)


def _frame_from_params(params, pc_term: str | None) -> str:
    """Cloud output frame from the term's ``ref_cfg``: a body name (EE frame), 'base'
    (robot root), or 'camera' (occluded term's default when no ref_cfg)."""
    ref = _get(params, "ref_cfg")
    body = _get(ref, "body_names")
    if body:
        return body[0] if isinstance(body, (list, tuple)) else str(body)
    # ScenePointCloud falls back to the robot ROOT frame; OccludedScenePointCloud stays
    # in the camera optical frame when no ref body is given.
    return "camera" if pc_term == "OccludedScenePointCloud" else "base"


def _norm_str_list(v) -> list | None:
    if v is None:
        return None
    return [str(x) for x in v]


def _proprio_spec(group_cfg, term_names: list[str]) -> dict:
    """Proprio layout from an obs group: the allowlisted terms in declaration order.

    ``joint_pos`` dims come from the term's explicit ``joint_names`` list (6 = arm-only,
    real-robot convention) or default to 12 (all UR5e+Robotiq joints). ``end_effector_pose``
    dims follow its ``rotation_repr``.
    """
    terms, jp_dims, total = [], None, 0
    for name in term_names:
        term = _get(group_cfg, name)
        if term is None:
            continue
        if name == "joint_pos":
            names = _get(_get(_get(term, "params"), "asset_cfg"), "joint_names")
            jp_dims = len(names) if isinstance(names, (list, tuple)) else 12
            terms.append(name)
            total += jp_dims
        elif name == "end_effector_pose":
            repr_ = _get(_get(term, "params"), "rotation_repr", "quat")
            terms.append(name)
            total += _EE_POSE_DIMS.get(str(repr_), 7)
    return {"terms": terms, "joint_pos_dims": jp_dims, "dim": total}


def _term_names(group_cfg) -> list[str]:
    """Obs-term names of a group, in declaration order (dict order for yaml; live
    ObsGroup cfgs iterate their annotated fields via __dict__ order)."""
    if isinstance(group_cfg, dict):
        skip = {"concatenate_terms", "concatenate_dim", "enable_corruption",
                "history_length", "flatten_history_dim"}
        return [k for k, v in group_cfg.items() if k not in skip and isinstance(v, dict) and "func" in v]
    return [k for k, v in vars(group_cfg).items() if _get(v, "func") is not None]


def signature_from_obs_group(pc_group_cfg, proprio_group_cfg=None) -> dict | None:
    """Build the signature from obs-group cfg(s): the group holding the ``scene_pc`` term,
    plus the group holding the proprio terms (same group for the BC ``data_collect``
    layout; a separate ``proprio`` group for the DAgger 3-group layout).

    Accepts live cfg objects (collection scripts) or plain dicts (yaml sidecars).
    Returns None if no point-cloud term is found.
    """
    proprio_group_cfg = proprio_group_cfg if proprio_group_cfg is not None else pc_group_cfg
    pc_term_cfg = _get(pc_group_cfg, "scene_pc")
    if pc_term_cfg is None or _get(pc_term_cfg, "func") is None:
        return None
    params = _get(pc_term_cfg, "params") or {}
    pc_term = _term_class_name(pc_term_cfg)

    num_points = _get(params, "num_points", 512)
    ratios = _get(params, "class_ratios")
    ratios = [float(r) for r in ratios] if ratios is not None else None
    counts = dict(zip(PC_CLASSES, ratio_to_counts(ratios, int(num_points)))) if ratios else None
    include_seg = bool(_get(params, "include_segmentation", False))
    seg_labels = _get(params, "segmentation_labels")
    if include_seg and seg_labels is None:
        seg_labels = dict(DEFAULT_SEG_LABELS)
    if seg_labels is not None:
        seg_labels = {str(k): float(v) for k, v in dict(seg_labels).items()}

    return {
        "version": 1,
        "pc_term": pc_term,
        "frame": _frame_from_params(params, pc_term),
        "num_points": int(num_points),
        "point_dim": 4 if include_seg else 3,
        "include_segmentation": include_seg,
        "segmentation_labels": seg_labels,
        "classes": list(PC_CLASSES),
        "class_ratios": ratios,
        "class_points": counts,
        "robot_body_names": _norm_str_list(_get(params, "robot_body_names")),
        "zero_pad_missing_class": _get(params, "zero_pad_missing_class"),
        "pc_parts": None,
        "pc_all_prim_names": None,
        "pc_pad_target": None,
        "append_prim_semantic": False,
        "proprio": _proprio_spec(proprio_group_cfg, _term_names(proprio_group_cfg)),
    }


def _load_yaml_loose(path: str):
    """Load an IsaacLab params yaml WITHOUT reconstructing python objects: every
    ``!!python/...`` tag collapses to its plain mapping/sequence/scalar, so no uwlab/
    isaaclab import is needed to read a run's sidecars."""
    import yaml

    class _Loose(yaml.SafeLoader):
        pass

    def _any(loader, tag_suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    _Loose.add_multi_constructor("tag:yaml.org,2002:python/", _any)
    _Loose.add_multi_constructor("", _any)
    with open(path) as f:
        return yaml.load(f, _Loose)


def signature_from_run_params(params_dir: str) -> dict | None:
    """Signature for an rsl_rl DAgger run from its ``params/{env,agent}.yaml`` sidecars.

    ``agent.yaml`` names the student's cloud group(s) (``policy.pointcloud_groups``) and
    the policy obs groups; the remaining policy groups hold the proprio terms. Returns
    None (best-effort) if the sidecars are missing or don't describe a scene_pc term.
    """
    try:
        agent = _load_yaml_loose(os.path.join(params_dir, "agent.yaml"))
        env = _load_yaml_loose(os.path.join(params_dir, "env.yaml"))
        pc_groups = list(agent["policy"]["pointcloud_groups"])
        policy_groups = list(agent["obs_groups"]["policy"])
        obs = env["observations"]
    except Exception as e:  # noqa: BLE001
        print(f"[pc_signature] WARNING: could not read {params_dir} sidecars "
              f"({type(e).__name__}: {e}); no pc_signature for this ckpt")
        return None
    if len(pc_groups) > 1:
        print(f"[pc_signature] WARNING: multiple pointcloud groups {pc_groups}; using {pc_groups[0]}")
    proprio_groups = [g for g in policy_groups if g not in pc_groups]
    proprio_cfg = obs.get(proprio_groups[0]) if proprio_groups else None
    return signature_from_obs_group(obs.get(pc_groups[0]), proprio_cfg)


def apply_training_adjustments(sig: dict | None, *, num_points: int | None = None,
                               point_dim: int | None = None, joint_pos_dims: int | None = None,
                               proprio_dim: int | None = None, pc_parts=None,
                               pc_all_prim_names=None, pc_pad_target=None,
                               append_prim_semantic: bool = False) -> dict | None:
    """Return a copy of the collection-time signature updated with what TRAINING changed:
    cloud subsampling (``num_points``), a derived seg channel (``point_dim`` /
    ``append_prim_semantic``), the arm-only ``joint_pos_dims`` trim, and the per-prim
    selection (``pc_parts`` + zero-pad target). No-op on None."""
    if sig is None:
        return None
    sig = json.loads(json.dumps(sig))  # deep copy, keeps it JSON-able
    if num_points and num_points != sig["num_points"]:
        sig["num_points"] = int(num_points)
        # Subsampling is uniform over the stored cloud -> ratios hold, counts shrink.
        if sig.get("class_ratios"):
            sig["class_points"] = dict(zip(sig["classes"], ratio_to_counts(sig["class_ratios"], int(num_points))))
    if point_dim:
        sig["point_dim"] = int(point_dim)
        sig["include_segmentation"] = int(point_dim) == 4
        if sig["include_segmentation"] and sig.get("segmentation_labels") is None:
            sig["segmentation_labels"] = dict(DEFAULT_SEG_LABELS)
    if joint_pos_dims and sig["proprio"].get("joint_pos_dims") != joint_pos_dims:
        old = sig["proprio"].get("joint_pos_dims") or 12
        sig["proprio"]["joint_pos_dims"] = int(joint_pos_dims)
        sig["proprio"]["dim"] = sig["proprio"]["dim"] - old + int(joint_pos_dims)
    if proprio_dim is not None and sig["proprio"]["dim"] != proprio_dim:
        print(f"[pc_signature] WARNING: signature proprio dim {sig['proprio']['dim']} != "
              f"model proprio_dim {proprio_dim}; keeping the model's value")
        sig["proprio"]["dim"] = int(proprio_dim)
    if pc_parts is not None:
        sig["pc_parts"] = [str(p) for p in pc_parts]
    if pc_all_prim_names is not None:
        sig["pc_all_prim_names"] = [str(p) for p in pc_all_prim_names]
    if pc_pad_target is not None:
        sig["pc_pad_target"] = int(pc_pad_target)
    if append_prim_semantic:
        sig["append_prim_semantic"] = True
    return sig
