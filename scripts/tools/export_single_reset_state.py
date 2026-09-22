# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export ONE reset state from an eval_critic reset-state dump
({resets: {idx: {state, init_peg_xyz}}, is_relative}) into the reset-dataset format consumed by
``SingleResetManager`` ({initial_state: nested lists of per-state tensors}), plus a JSON sidecar
with provenance. No Isaac Sim needed.

    python scripts/tools/export_single_reset_state.py \
        --states videos/gap_3ckpt_sidebyside/reset_states.pt --index 0 \
        --out reset_states/single_reset_seed42_r0.pt
"""

import argparse
import json
import os

import torch


def _to_lists(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = _to_lists(v)
        else:
            assert torch.is_tensor(v) and v.shape[0] == 1, f"{k}: expected [1, D] tensor, got {getattr(v, 'shape', v)}"
            out[k] = [v[0].detach().cpu().clone()]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    dump = torch.load(a.states, map_location="cpu", weights_only=False)
    assert dump.get("is_relative", True), "SingleResetManager resets with is_relative=True; dump must be env-relative"
    entry = dump["resets"][a.index]
    dataset = {"initial_state": _to_lists(entry["state"])}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(dataset, a.out)
    meta = {
        "source_file": os.path.abspath(a.states),
        "source_index": a.index,
        "source_task": dump.get("task"),
        "source_seed": dump.get("seed"),
        "init_peg_xyz": entry.get("init_peg_xyz"),
        "robot_joint_position": dataset["initial_state"]["articulation"]["robot"]["joint_position"][0].tolist(),
        "insertive_object_root_pose": dataset["initial_state"]["rigid_object"]["insertive_object"]["root_pose"][0].tolist(),
        "receptive_object_root_pose": dataset["initial_state"]["rigid_object"]["receptive_object"]["root_pose"][0].tolist(),
    }
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {a.out} (+ .json)\n{json.dumps(meta, indent=1)}")


if __name__ == "__main__":
    main()
