"""Newton: list collision shapes on pad bodies + peg, then hold-close on grasped resets and count pad-peg contacts."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Newton-v0")
parser.add_argument("--dataset_dir", default="./Datasets/OmniReset_patched")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch, numpy as np
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose
def T(x): return x.torch if hasattr(x, "torch") else x

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    p = env_cfg.events.reset_from_reset_states.params
    p["dataset_dir"] = args_cli.dataset_dir; p["reset_types"] = ["ObjectAnywhereEEGrasped"]; p["probs"] = [1.0]
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    from isaaclab_newton.physics.newton_manager import NewtonManager
    m = NewtonManager._model
    body_key = list(m.body_label); shape_key = list(m.shape_label)
    sb = m.shape_body.numpy(); sg = m.shape_collision_group.numpy(); sf = m.shape_flags.numpy()
    mu = m.shape_material_mu.numpy(); st = m.shape_type.numpy()
    sw = m.shape_world.numpy() if hasattr(m, "shape_world") else None
    print("num shapes", len(shape_key), "num bodies", len(body_key))
    for i, k in enumerate(shape_key):
        bk = body_key[sb[i]] if sb[i] >= 0 else "static"
        if ("env_0/" in k or "env_0/" in bk) and ("inner_finger" in k or "Insertive" in k or "Receptive" in k or "inner_finger" in bk):
            print(f"  shape {i:4d} {k:80s} body={bk.split('/')[-1]:20s} type={st[i]} group={sg[i]} flags={sf[i]:#x} mu={mu[i]:.2f}")
    # hold closed and count contacts pad<->peg in env_0
    robot, peg = env.scene["robot"], env.scene["insertive_object"]
    bn = list(robot.body_names); fi = robot.find_joints(["finger_joint"])[0][0]
    pads = [bn.index("left_inner_finger"), bn.index("right_inner_finger")]
    act = torch.zeros(env.action_space.shape, device=env.device); act[:, -1] = -1.0
    peg_shapes = [i for i, k in enumerate(shape_key) if "env_0/InsertiveObject" in k]
    pad_shapes = [i for i, k in enumerate(shape_key) if "env_0/" in k and "inner_finger/" in k and "knuckle" not in k]
    print("env0 peg shapes", peg_shapes, "pad shapes", pad_shapes)
    for t in range(args_cli.steps):
        env.step(act)
        c = NewtonManager._contacts
        n = int(c.rigid_contact_count.numpy()[0]) if hasattr(c, "rigid_contact_count") else -1
        s0 = c.rigid_contact_shape0.numpy()[:n]; s1 = c.rigid_contact_shape1.numpy()[:n]
        pp = sum(1 for a, b in zip(s0, s1) if (a in peg_shapes and b in pad_shapes) or (b in peg_shapes and a in pad_shapes))
        pany = sum(1 for a, b in zip(s0, s1) if a in peg_shapes or b in peg_shapes)
        padany = sum(1 for a, b in zip(s0, s1) if a in pad_shapes or b in pad_shapes)
        if t == 0: print('   rigid_contact_max', c.rigid_contact_max)
        # detail of env0 pad-peg contacts: normal (world), pad inward direction, force
        import isaaclab.utils.math as mu
        nrm = c.rigid_contact_normal.numpy()[:n]; frc = c.rigid_contact_force.numpy()[:n]
        p0 = c.rigid_contact_point0.numpy()[:n]; p1 = c.rigid_contact_point1.numpy()[:n]
        bq = T(robot.data.body_quat_w); bp = T(robot.data.body_pos_w)
        # pad-to-pad direction (world) as reference "inward" axis for the left pad
        inward = (bp[0, pads[1]] - bp[0, pads[0]]); inward = (inward / inward.norm()).cpu().numpy()
        for k, (a, b) in enumerate(zip(s0, s1)):
            if (a in peg_shapes and b in pad_shapes) or (b in peg_shapes and a in pad_shapes):
                pad = b if b in pad_shapes else a
                side = "L" if pad in (99,) or "left" in shape_key[pad] else "R"
                nn = nrm[k]; sgn = 1.0 if side == "L" else -1.0
                print(f"      contact shapes ({a},{b}) side={side} normal={nn.round(3).tolist()} dot(normal, L->R inward)={float(np.dot(nn, inward))*sgn:+.3f} |F|={float(np.linalg.norm(frc[k])):.3f} p0={p0[k].round(4).tolist()} p1={p1[k].round(4).tolist()}")
        q = T(robot.data.joint_pos)[:, fi]
        gap = (T(robot.data.body_pos_w)[:, pads[0]] - T(robot.data.body_pos_w)[:, pads[1]]).norm(dim=1)
        d = (T(peg.data.root_pos_w) - 0.5 * (T(robot.data.body_pos_w)[:, pads[0]] + T(robot.data.body_pos_w)[:, pads[1]])).norm(dim=1)
        print(f"t={t:2d} contacts total={n:5d} env0 peg-pad={pp} peg-any={pany} pad-any={padany} | q env0={q[0]:.3f} mean={q.mean():.3f} gap0={gap[0]:.3f} peg-padmid dist env0={d[0]:.3f} mean={d.mean():.3f} peg z0={T(peg.data.root_pos_w)[0,2]:.3f}")
    env.close()
main(); app.close()
