import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Newton-v0")
parser.add_argument("--dataset_dir", default="./Datasets/OmniReset_patched")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch, numpy as np
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 2; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    env_cfg.events.reset_from_reset_states.params["dataset_dir"] = args_cli.dataset_dir
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    from isaaclab_newton.physics.newton_manager import NewtonManager
    m = NewtonManager._model
    sl = list(m.shape_label); bl = list(m.body_label)
    sb = m.shape_body.numpy(); sg = m.shape_collision_group.numpy(); sf = m.shape_flags.numpy(); sw = m.shape_world.numpy()
    bw = m.body_world.numpy() if hasattr(m, "body_world") else None
    print("shape_collision_filter_pairs:", len(m.shape_collision_filter_pairs) if hasattr(m, "shape_collision_filter_pairs") and m.shape_collision_filter_pairs is not None else None)
    print("shape_contact_pairs:", None if getattr(m, "shape_contact_pairs", None) is None else m.shape_contact_pairs.shape)
    for i, k in enumerate(sl):
        if sf[i] & 0x2 == 0: continue  # colliders only
        if ("env_0/" in k) and any(t in k for t in ("inner_finger", "Insertive", "Receptive", "Table", "table", "hole", "peg", "base_link/")):
            print(f"  shape {i:4d} world={sw[i]:3d} group={sg[i]:3d} flags={sf[i]:#x} body={bl[sb[i]] if sb[i]>=0 else 'static':50s} {k}")
    # static / world -1 shapes
    for i, k in enumerate(sl):
        if sw[i] < 0 and (sf[i] & 0x2): print(f"  GLOBAL shape {i:4d} world={sw[i]} group={sg[i]} {k}")
    peg=[i for i,k in enumerate(sl) if k.endswith("env_0/InsertiveObject/collisions/peg")][0]
    pads=[i for i,k in enumerate(sl) if "env_0/" in k and k.endswith("inner_finger/collisions/mesh_1")]
    table=[i for i,k in enumerate(sl) if "env_0/Table/collisions/mesh_0" in k][0]
    pairs=set(map(tuple, m.shape_contact_pairs.numpy().tolist())) if getattr(m,"shape_contact_pairs",None) is not None else set()
    filt=set(map(tuple, m.shape_collision_filter_pairs)) if getattr(m,"shape_collision_filter_pairs",None) else set()
    def has(a,b,S): return (a,b) in S or (b,a) in S
    print("peg",peg,"pads",pads,"table",table)
    for a in pads: print(f"  pad{a}-peg allowed={has(a,peg,pairs)} filtered={has(a,peg,filt)}   pad{a}-table allowed={has(a,table,pairs)}")
    print(f"  peg-table allowed={has(peg,table,pairs)} filtered={has(peg,table,filt)}")
    robot_shapes=[i for i,k in enumerate(sl) if "env_0/Robot/" in k and (sf[i]&0x2)]
    n_rob_obj=sum(1 for a in robot_shapes if has(a,peg,pairs)); print(f"  robot collider shapes in env0: {len(robot_shapes)}, of which allowed to touch peg: {n_rob_obj}")
    print("  sample pairs involving peg:", [pr for pr in pairs if peg in pr][:20])
    st_ = m.shape_transform.numpy(); ss = m.shape_scale.numpy()
    src = m.shape_source
    import numpy as np
    for body in ("left_inner_finger","right_inner_finger"):
        print("==", body)
        for i,k in enumerate(sl):
            if f"env_0/Robot/{body}/" in k:
                tf = st_[i]; pos = tf[:3]; q = tf[3:]
                mesh = src[i] if src is not None else None
                ext = None
                try:
                    v = np.asarray(mesh.vertices); ext = (v.min(0).round(4).tolist(), v.max(0).round(4).tolist())
                except Exception: pass
                print(f"   shape {i:3d} flags={sf[i]:#x} pos={pos.round(4).tolist()} quat={q.round(3).tolist()} scale={ss[i].round(3).tolist()} verts_bbox={ext} {k.split(body)[-1]}")
    mj = getattr(m, "mujoco", None); cd = getattr(mj, "condim", None) if mj is not None else None
    cdn = cd.numpy() if cd is not None else None
    mt = m.shape_material_mu_torsional.numpy()
    for i,k in enumerate(sl):
        if "env_0/" in k and (k.endswith("inner_finger/collisions/mesh_1") or k.endswith("InsertiveObject/collisions/peg") or "Table/collisions/mesh_0" in k):
            print(f"CONDIM shape {i} condim={None if cdn is None else cdn[i]} mu_torsional={mt[i]:.4f} {k}")
    print("== peg shapes env0")
    for i,k in enumerate(sl):
        if "env_0/InsertiveObject" in k:
            tf = st_[i]; v = None
            try: vv = np.asarray(src[i].vertices); v = (vv.min(0).round(4).tolist(), vv.max(0).round(4).tolist())
            except Exception: pass
            print(f"   shape {i:3d} flags={sf[i]:#x} pos={tf[:3].round(4).tolist()} quat={tf[3:].round(3).tolist()} scale={ss[i].round(4).tolist()} verts_bbox={v} {k}")
    env.close()
main(); app.close()
