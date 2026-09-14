#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert OmniReset datasets to the patched-gripper joint convention.

tools/fix_robotiq_usd.py swaps body0/body1 on left/right_inner_finger_knuckle_joint, which
negates those two joint angles. Reset states and grasp files recorded with the original USD
store the old sign; writing them into the patched articulation explodes the linkage.
This negates those two joints' positions/velocities and stamps
``robot_joint_convention: "swapped_ifk"`` so the loaders can verify the pairing.

Usage: convert_datasets_for_patched_gripper.py --src Datasets/OmniReset --dst Datasets/OmniReset_patched
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch

SWAPPED = ("left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint")
ROBOT_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "finger_joint", "right_outer_knuckle_joint", "right_inner_knuckle_joint", "left_inner_knuckle_joint",
    "right_inner_finger_knuckle_joint", "left_inner_finger_knuckle_joint",
]
MARKER = "robot_joint_convention"


def convert_reset_states(src: str, dst: str) -> None:
    d = torch.load(src, map_location="cpu", weights_only=False)
    if d.get(MARKER) == "swapped_ifk":
        print(f"  already converted: {src}")
        shutil.copy(src, dst)
        return
    idx = [ROBOT_JOINTS.index(j) for j in SWAPPED]
    robot = d["initial_state"]["articulation"]["robot"]
    n = 0
    for key in ("joint_position", "joint_velocity"):
        seq = robot[key]
        for i, t in enumerate(seq):
            t = t.clone()
            t[..., idx] *= -1
            seq[i] = t
            n += 1
    d[MARKER] = "swapped_ifk"
    torch.save(d, dst)
    print(f"  {os.path.basename(src)}: negated {SWAPPED} in {n // 2} states -> {dst}")


def convert_grasps(src: str, dst: str) -> None:
    d = torch.load(src, map_location="cpu", weights_only=False)
    if d.get(MARKER) == "swapped_ifk":
        print(f"  already converted: {src}")
        shutil.copy(src, dst)
        return
    gj = d["grasp_relative_pose"]["gripper_joint_positions"]
    for j in SWAPPED:
        gj[j] = [(-v if torch.is_tensor(v) else -v) for v in gj[j]]
    d[MARKER] = "swapped_ifk"
    torch.save(d, dst)
    print(f"  {os.path.basename(src)}: negated {SWAPPED} in {len(gj[SWAPPED[0]])} grasps -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    for root, _, files in os.walk(args.src):
        rel = os.path.relpath(root, args.src)
        out = os.path.join(args.dst, rel)
        os.makedirs(out, exist_ok=True)
        for f in files:
            s, t = os.path.join(root, f), os.path.join(out, f)
            if f.startswith("resets_") and f.endswith(".pt"):
                convert_reset_states(s, t)
            elif f == "grasps.pt":
                convert_grasps(s, t)
            else:
                shutil.copy(s, t)
    print("done")


if __name__ == "__main__":
    main()
