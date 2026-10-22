"""Print robot/camera world poses and robot mesh counts for one env (backend debugging)."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-v0")
AppLauncher.add_app_launcher_args(parser)
args, rest = parser.parse_known_args()
args.enable_cameras = True
app = AppLauncher(args).app
import gymnasium as gym, torch
from pxr import Usd, UsdGeom
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

@hydra_task_compose(args.task, "env_cfg_entry_point", hydra_args=rest)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 2
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    print("[dbg] env_origins", env.scene.env_origins[0].tolist())
    print("[dbg] robot root_pos_w", robot.data.root_pos_w[0].tolist(), "root_quat_w", robot.data.root_quat_w[0].tolist())
    for n in ("front_camera", "wrist_camera"):
        c = env.scene.sensors[n]
        print(f"[dbg] {n} pos_w", [round(v, 3) for v in c.data.pos_w[0].tolist()], "quat_w_opengl", [round(v, 3) for v in c.data.quat_w_opengl[0].tolist()])
    stage = env.sim.stage
    for n in ("front_camera", "wrist_camera"):
        path = env.scene.sensors[n].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        p = stage.GetPrimAtPath(path)
        m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        q = m.ExtractRotationQuat(); t = m.ExtractTranslation()
        print(f"[dbg] {n} USD prim {path} valid {p.IsValid()} world t {[round(v,3) for v in t]} quat(w; xyz) {round(q.GetReal(),3)} {[round(v,3) for v in q.GetImaginary()]}")
        print(f"[dbg] {n} xformOps {[(op.GetOpName(), op.Get()) for op in UsdGeom.Xformable(p).GetOrderedXformOps()]}")
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    meshes = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
    vis = [p for p in meshes if UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible]
    inactive = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)) if not p.IsActive()]
    print(f"[dbg] robot prim valid {root.IsValid()} active {root.IsActive()}  meshes {len(meshes)} visible {len(vis)}  inactive prims {len(inactive)} e.g. {inactive[:4]}")
    xf = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    print("[dbg] robot USD world translation", list(xf))
    env.close()

if __name__ == "__main__":
    main(); app.close()
