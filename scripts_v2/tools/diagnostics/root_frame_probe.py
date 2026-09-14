"""Compare articulation root pose vs root-link pose vs USD prim transforms (IsaacLab 3.0)."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from uwlab_assets.robots.ur5e_robotiq_gripper import ROBOTIQ_2F85, IMPLICIT_UR5E_ROBOTIQ_2F85
from pxr import UsdGeom, Usd
import isaaclab.sim.utils.stage as stage_utils

@configclass
class SceneCfg(InteractiveSceneCfg):
    grip = ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Grip")
    ur = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/UR")

def fmt(t): return [round(v, 3) for v in t.tolist()]

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
sim.reset()
for _ in range(5):
    scene.write_data_to_sim(); sim.step(); scene.update(1/120)
st = stage_utils.get_current_stage()
xc = UsdGeom.XformCache()
for name, base in (("grip", "robotiq_base_link"), ("ur", None)):
    a = scene[name]
    bn = list(a.body_names)
    if base is None:
        base = bn[0]
    print(f"\n== {name}: root link (body 0) = {bn[0]}; default rot cfg = {a.cfg.init_state.rot}")
    print("  root_pos_w      ", fmt(a.data.root_pos_w.torch[0]),      "root_quat_w     ", fmt(a.data.root_quat_w.torch[0]))
    print("  root_link_pos_w ", fmt(a.data.root_link_pos_w.torch[0]), "root_link_quat_w", fmt(a.data.root_link_quat_w.torch[0]))
    print("  root_com_pos_w  ", fmt(a.data.root_com_pos_w.torch[0]),  "root_com_quat_w ", fmt(a.data.root_com_quat_w.torch[0]))
    bi = bn.index(base)
    print(f"  body_pos_w[{base}]", fmt(a.data.body_pos_w.torch[0, bi]), "body_quat_w", fmt(a.data.body_quat_w.torch[0, bi]))
    for pth in (a.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0"), a.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0") + "/" + base):
        prim = st.GetPrimAtPath(pth)
        m = xc.GetLocalToWorldTransform(prim)
        q = m.ExtractRotationQuat()
        print(f"  USD {pth}: T={[round(v,3) for v in m.ExtractTranslation()]} Rxyzw={[round(v,3) for v in list(q.GetImaginary())+[q.GetReal()]]}")
    # the fingers: where do the inner fingers sit relative to the root link, in world?
    for b in ("left_inner_finger", "right_inner_finger"):
        if b in bn:
            print(f"  {b} rel root_link (world axes):", fmt(a.data.body_pos_w.torch[0, bn.index(b)] - a.data.root_link_pos_w.torch[0]))

print("\n== WRITE TEST on grip: pose (0.3,0.2,0.7) rot 90deg about z (xyzw 0,0,0.707,0.707)")
a = scene["grip"]
pose = torch.tensor([[0.3, 0.2, 0.7, 0.0, 0.0, 0.7071, 0.7071]], device=sim.device)
a.write_root_pose_to_sim(pose)
a.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim.device))
for _ in range(3):
    scene.write_data_to_sim(); sim.step(); scene.update(1/120)
print("  root_link_pos_w ", fmt(a.data.root_link_pos_w.torch[0]), "root_link_quat_w", fmt(a.data.root_link_quat_w.torch[0]))
print("  body_pos_w[base]", fmt(a.data.body_pos_w.torch[0, 0]), "body_quat_w", fmt(a.data.body_quat_w.torch[0, 0]))
print("  via write_root_link_pose_to_sim:")
a.write_root_link_pose_to_sim(pose)
for _ in range(3):
    scene.write_data_to_sim(); sim.step(); scene.update(1/120)
print("  root_link_pos_w ", fmt(a.data.root_link_pos_w.torch[0]), "root_link_quat_w", fmt(a.data.root_link_quat_w.torch[0]))
print("  via write_root_state_to_sim:")
a.write_root_state_to_sim(torch.cat([pose, torch.zeros(1, 6, device=sim.device)], dim=-1))
for _ in range(3):
    scene.write_data_to_sim(); sim.step(); scene.update(1/120)
print("  root_link_pos_w ", fmt(a.data.root_link_pos_w.torch[0]), "root_link_quat_w", fmt(a.data.root_link_quat_w.torch[0]))
print("  is_fixed_base:", getattr(a, "is_fixed_base", None))
simulation_app.close()
