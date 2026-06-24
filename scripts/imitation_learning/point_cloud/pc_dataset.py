# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-RAM point-cloud BC dataset (:class:`PCDemoDataset`) loaded from an HDF5 demo file.

Split out of ``train_point_net.py`` so eval/analysis tooling can import the dataset (and the
PC term / proprio allowlists) without pulling in Lightning or W&B. The matching deploy-time
input assembly lives in ``bc_utils.py``.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Obs terms that are point clouds (stored (T, num_points, point_dim)) -> the PointNet set input.
_PC_TERMS = {"scene_pc"}
# Proprio is an explicit ALLOWLIST (not "everything that isn't a PC"). This guarantees that
# recording extra obs terms in the collection env -- privileged poses, contact flags, etc.,
# kept "on the side" for future aux losses -- never silently changes the policy input. Any
# stored obs term not in _PC_TERMS or _PROPRIO_TERMS is aux metadata and is ignored by the
# model. {joint_pos, end_effector_pose} reproduces every existing dataset's 18d proprio.
_PROPRIO_TERMS = {"joint_pos", "end_effector_pose"}
# Privileged scene-state targets for the OPTIONAL auxiliary linear-probe loss (predicted from the
# pooled PC feature; NOT a policy input). Ordered list -> deterministic concat. Terms absent from a
# dataset are skipped; if none are present (or aux_weight=0) the aux loss is silently disabled.
_AUX_TERMS = ["insertive_asset_pose", "receptive_asset_pose", "insertive_in_receptive"]

# Prim-name -> semantic category, and category -> segmentation-label value. The label mirrors the
# collector's seg convention (robot=0, insertive=-1, receptive=+1) so a per-prim cloud whose seg
# channel is DERIVED FROM the prim id is byte-identical in format to a natively-segmented cloud.
# Used both to key per-prim dropout and (with append_prim_semantic) to append the seg channel.
_OBJECT_PRIMS = ("insertive", "receptive")
_SEG_LABEL = {"robot": 0.0, "insertive": -1.0, "receptive": 1.0}


def _prim_category(name: str) -> str:
    return name if name in _OBJECT_PRIMS else "robot"


class PCDemoDataset(Dataset):
    """All (point_cloud, proprio, action) timesteps from an HDF5 demo file, in RAM.

    ``proprio`` is the concatenation of every non-PC obs term, in the dataset's
    ``obs_keys`` order. If ``num_points`` is smaller than the stored cloud size, each
    ``__getitem__`` randomly subsamples that many points (cheap PC augmentation).
    """

    def __init__(self, path: str, num_points: int | None = None, success_only: bool = True,
                 joint_pos_dims: int | None = None, pc_parts: list[str] | None = None,
                 append_prim_semantic: bool = False):
        super().__init__()
        # append_prim_semantic (per-prim datasets only): derive a per-point segmentation label from
        # each point's source prim (robot=0 / insertive=-1 / receptive=+1) and APPEND it as an extra
        # cloud channel, so the model trains on [xyz, seg] (point_dim 3 -> 4). Giving the network the
        # category explicitly typically helps a lot. Deploy (bc_utils) re-derives the same channel.
        self.append_prim_semantic = append_prim_semantic
        self._prim_seg: torch.Tensor | None = None  # (n_prims,) seg label indexed by prim id
        # pc_parts: for PER-PRIM datasets (a ratio-enforced occluded cloud where each point
        # carries its source prim id in obs/<term>_prim_id), the prim names to KEEP -- this is
        # how a run drops individual robot links, all robot links, or an object. Points whose
        # prim is selected are gathered and zero-PADDED to a fixed size at training time. None =
        # keep every prim. Ignored for legacy flat datasets.
        self.pc_parts: list[str] | None = None
        # Full prim id->name table as collected. Deploy maps the live prim-id channel through
        # this, then keeps pc_parts -- so it must travel in the checkpoint.
        self.pc_all_prim_names: list[str] | None = None
        # Fixed output size the selected points are padded/subsampled to (per-prim only).
        self.pc_pad_target: int | None = None
        # joint_pos is stored as 12 dims = 6 UR5e ARM joints + 6 Robotiq gripper mimic joints. The
        # last 6 (gripper) DO NOT EXIST on the real robot, so a sim2real policy must train on only the
        # first `joint_pos_dims` (arm) entries. None = keep all 12 (sim-only).
        self.joint_pos_dims = joint_pos_dims

        def _ncols(g_or_data, k):
            w = g_or_data["obs"][k].shape[1] if hasattr(g_or_data, "__getitem__") else g_or_data
            return min(self.joint_pos_dims, w) if (k == "joint_pos" and self.joint_pos_dims) else w

        def _slice(arr, k):  # keep only the first joint_pos_dims columns of joint_pos
            return arr[:, : self.joint_pos_dims] if (k == "joint_pos" and self.joint_pos_dims) else arr

        with h5py.File(path, "r") as f:
            data = f["data"]
            obs_keys = list(data.attrs["obs_keys"])
            pc_key = next(k for k in obs_keys if k in _PC_TERMS)
            # Allowlist + preserve stored order (eval in bc_utils rebuilds the same order).
            self.proprio_keys = [k for k in obs_keys if k in _PROPRIO_TERMS]
            # Aux-probe targets present in this dataset, in the fixed _AUX_TERMS order.
            self.aux_keys = [k for k in _AUX_TERMS if k in obs_keys]

            demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
            if success_only:
                demos = [d for d in demos if bool(data[d].attrs.get("success", True))]

            total = sum(data[d]["actions"].shape[0] for d in demos)

            # PER-PRIM detection: a ratio-enforced occluded cloud whose per-point source prim is
            # stored in a parallel obs/<term>_prim_id dataset (int). We load the full cloud +
            # prim ids, and at __getitem__ keep only the points whose prim is in pc_parts,
            # zero-padding (or subsampling) to a fixed `pc_pad_target` size.
            per_prim = "pc_prim_names" in data.attrs
            prim_id_key = f"{pc_key}_prim_id"
            pc_shape = data[demos[0]]["obs"][pc_key].shape  # (T, n_load, point_dim)
            n_load = pc_shape[1]
            point_dim = pc_shape[2]  # 3 = xyz, 4 = xyz + segmentation label
            if per_prim:
                all_names = [n.decode() if isinstance(n, bytes) else str(n) for n in data.attrs["pc_prim_names"]]
                parts = list(pc_parts) if pc_parts else all_names
                missing = [p for p in parts if p not in all_names]
                if missing:
                    raise KeyError(f"pc_parts {missing} not in dataset prims {all_names}")
                self.pc_parts = parts
                self.pc_all_prim_names = all_names
                self._selected_ids = torch.tensor([all_names.index(p) for p in parts], dtype=torch.long)
                # Per-prim-id seg label (for append_prim_semantic), indexed by prim id.
                self._prim_seg = torch.tensor(
                    [_SEG_LABEL[_prim_category(n)] for n in all_names], dtype=torch.float32
                )
                # Output size after selection: `num_points` if given, else the full stored cloud.
                self.pc_pad_target = int(num_points) if num_points else int(n_load)
                n_pts = self.pc_pad_target

            proprio_dim = sum(_ncols(data[demos[0]], k) for k in self.proprio_keys)
            aux_dim = sum(data[demos[0]]["obs"][k].shape[1] for k in self.aux_keys)
            action_dim = data[demos[0]]["actions"].shape[1]

            # Preallocate once and fill (avoids the 2x peak of np.concatenate).
            self.points = np.empty((total, n_load, point_dim), dtype=np.float32)
            self.prim_id = np.empty((total, n_load), dtype=np.int16) if per_prim else None
            self.proprio = np.empty((total, proprio_dim), dtype=np.float32)
            self.actions = np.empty((total, action_dim), dtype=np.float32)
            self.aux = np.empty((total, aux_dim), dtype=np.float32) if aux_dim else None

            i = 0
            for d in demos:
                g = data[d]
                t = g["actions"].shape[0]
                self.points[i : i + t] = g["obs"][pc_key][:]
                if per_prim:
                    self.prim_id[i : i + t] = g["obs"][prim_id_key][:]
                self.proprio[i : i + t] = np.concatenate(
                    [_slice(g["obs"][k][:], k) for k in self.proprio_keys], axis=1
                )
                self.actions[i : i + t] = g["actions"][:]
                if self.aux is not None:
                    self.aux[i : i + t] = np.concatenate([g["obs"][k][:] for k in self.aux_keys], axis=1)
                i += t

            self.expert_success_rate = float(data.attrs.get("expert_success_rate", float("nan")))

        self.points = torch.from_numpy(self.points)
        self.prim_id = torch.from_numpy(self.prim_id) if self.prim_id is not None else None
        self.proprio = torch.from_numpy(self.proprio)
        self.actions = torch.from_numpy(self.actions)
        self.aux = torch.from_numpy(self.aux) if self.aux is not None else None
        self.per_prim = per_prim
        self.n_load = n_load
        self.n_pts = n_pts if per_prim else n_load
        # append_prim_semantic adds a derived seg channel -> the model's input is one wider.
        self._append_seg = per_prim and append_prim_semantic
        self.point_dim = point_dim + (1 if self._append_seg else 0)
        self.proprio_dim = proprio_dim
        self.aux_dim = aux_dim
        self.action_dim = action_dim
        # Flat-dataset point subsample (per-prim handles its own selection+pad instead).
        self.subsample = num_points if (not per_prim and num_points and num_points < n_load) else None
        # Placeholder prim-id for flat datasets so every __getitem__ returns the same 5-tuple shape.
        self._empty_pid = torch.zeros(0, dtype=torch.long)

    def __len__(self):
        return self.points.shape[0]

    def _select_and_pad(self, pts: torch.Tensor, pid: torch.Tensor):
        """Per-prim: keep points whose prim id is selected, then zero-pad (or random-subsample)
        to ``pc_pad_target``. ``pts`` is (n_load, point_dim), ``pid`` is (n_load,). Returns the
        padded points AND their (original) prim ids -- pad slots get id ``-1`` -- so train-time
        per-prim dropout can run on the GPU (val keeps the clean cloud)."""
        mask = torch.isin(pid.long(), self._selected_ids)
        sel = pts[mask]  # (k, point_dim)
        sel_pid = pid[mask].long()
        k, target = sel.shape[0], self.pc_pad_target
        if k == target:
            return sel, sel_pid
        if k > target:  # more selected points than the budget -> random distinct subset
            perm = torch.randperm(k)[:target]
            return sel[perm], sel_pid[perm]
        out = pts.new_zeros(target, pts.shape[1])  # zero-pad the remainder
        out[:k] = sel
        out_pid = pid.new_full((target,), -1, dtype=torch.long)  # -1 = padding (never dropped)
        out_pid[:k] = sel_pid
        return out, out_pid

    def __getitem__(self, idx):
        pts = self.points[idx]
        if self.per_prim:
            pts, pid = self._select_and_pad(pts, self.prim_id[idx])
            if self._append_seg:
                seg = self._prim_seg[pid.clamp_min(0)]      # (target,) seg label per point
                seg[pid < 0] = 0.0                          # pad slots -> 0 (already all-zero points)
                pts = torch.cat([pts, seg.unsqueeze(-1)], dim=-1)  # [xyz, seg] -> point_dim 4
        else:
            if self.subsample is not None:
                sel = torch.randint(0, self.n_pts, (self.subsample,))
                pts = pts[sel]
            pid = self._empty_pid  # flat datasets carry no per-point prim id
        # aux is an empty (0,) tensor when the dataset has no aux-probe targets.
        aux = self.aux[idx] if self.aux is not None else self.points.new_zeros(0)
        return pts, pid, self.proprio[idx], self.actions[idx], aux
