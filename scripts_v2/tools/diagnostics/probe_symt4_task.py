"""Probe a SymT4 task: per-term layout of the requested groups, reset event order / offsets (train task),
and in-sim invariance of the tensor group under the 8 peg symmetry elements (45-deg yaw as negative control).
Also checks the tensor terms' position part equals the peg centre in the hole-root / wrist frame (drop-in)."""
import argparse, json, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--group", default="policy", help="tensor group to test for invariance")
parser.add_argument("--expect_dims", type=int, default=307)
parser.add_argument("--train", action="store_true", help="also check relabel-last and 8 offsets")
parser.add_argument("--layout_out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app

import torch, gymnasium as gym
import isaaclab.utils.math as mu
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import symmetric_obs as so

N = 16
cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=N)
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset(seed=0)
for _ in range(3):
    env.step(torch.zeros(env.action_space.shape, device=env.device))
om = env.observation_manager
fails = []; ok = lambda c, m: fails.append(m) if not c else None
layouts = {}
for g in om._group_obs_term_names:
    dims = list(zip(om._group_obs_term_names[g], [tuple(int(x) for x in d) for d in om._group_obs_term_dim[g]]))
    layouts[g] = dims
    print(f"[dims] {g}: total {sum(d[0] for _, d in dims)} :: {dims}")
G = args.group
tot = sum(d[0] for _, d in layouts[G]); pa = dict(layouts[G]).get("prev_actions")
ok(tot == args.expect_dims and pa == (7,), f"{G} dims {tot} / prev_actions {pa}")
if args.train:
    ok(sum(d[0] for _, d in layouts["critic"]) == args.expect_dims and layouts["critic"] == layouts["policy"], "critic layout != policy layout")
    order = env.event_manager._mode_term_names["reset"]; print(f"[events] reset order: {order}")
    ok(order[-1] == "peg_symmetry_relabel", "relabel not last")
    pc = env.reward_manager.get_term_cfg("progress_context").func; tc = env.command_manager.get_term("task_command")
    print(f"[offsets] rewards {pc.receptive_offsets_pos.shape[0]}, task_command {tc.receptive_offsets_pos.shape[0]}")
    ok(pc.receptive_offsets_pos.shape[0] == 8 and tc.receptive_offsets_pos.shape[0] == 8, "offsets != 8")
if args.layout_out:
    json.dump(layouts[G], open(args.layout_out, "w"))

def raw(g):
    return torch.cat([tc.func(env, **tc.params) for tc in om._group_obs_term_cfgs[g]], dim=-1)

peg = env.scene["insertive_object"]; hole = env.scene["receptive_object"]; robot = env.scene["robot"]
pos0, quat0 = peg.data.root_pos_w.clone(), peg.data.root_quat_w.clone()
base = raw(G)
Gq = so.symmetry_group_local().to(env.device)
worst = 0.0
for k in range(8):
    peg.write_root_pose_to_sim(torch.cat([pos0, mu.quat_mul(quat0, Gq[k].expand(N, 4))], dim=-1)); peg.update(0.0)
    worst = max(worst, (raw(G) - base).abs().max().item())
a = torch.tensor([0.0, 0.0, math.pi / 4], device=env.device)
peg.write_root_pose_to_sim(torch.cat([pos0, mu.quat_mul(quat0, mu.quat_from_angle_axis(a.norm().view(1), (a / a.norm()).view(1, 3)).expand(N, 4))], dim=-1)); peg.update(0.0)
d45 = (raw(G) - base).abs().max().item()
peg.write_root_pose_to_sim(torch.cat([pos0, quat0], dim=-1)); peg.update(0.0)
# drop-in check of the position parts (raw term outputs are single frames: pos = first 3 dims of each tensor term)
outs = {n: tc.func(env, **tc.params) for n, tc in zip(om._group_obs_term_names[G], om._group_obs_term_cfgs[G])}
ph = outs["peg_in_hole_t4"][:, :3]; pg = outs["peg_in_gripper_t4"][:, :3]
wi = robot.find_bodies("wrist_3_link")[0][0]
ph_ref, _ = mu.subtract_frame_transforms(hole.data.root_pos_w, hole.data.root_quat_w, peg.data.root_pos_w, peg.data.root_quat_w)
pg_ref, _ = mu.subtract_frame_transforms(robot.data.body_link_pos_w[:, wi], robot.data.body_link_quat_w[:, wi], peg.data.root_pos_w, peg.data.root_quat_w)
eph, epg = (ph - ph_ref).abs().max().item(), (pg - pg_ref).abs().max().item()
print(f"[sym] {G}: max |delta| over 8 group elements = {worst:.2e}; yaw45 control delta = {d45:.3f}")
print(f"[pos] peg centre in hole-root frame err {eph:.2e}; in wrist frame err {epg:.2e}")
ok(worst < 1e-4, f"not invariant ({worst:.2e})"); ok(d45 > 1e-2, "45-deg control did not move"); ok(eph < 1e-4 and epg < 1e-4, "position part mismatch")
print("RESULT:", "PASS" if not fails else "FAIL: " + " | ".join(fails))
env.close(); app.close()
