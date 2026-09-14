"""Reset from grasped dataset states in the TRAINING env, hold gripper closed, lift with the OSC.
Reports whether the peg comes with the gripper, per reset type. Works on IsaacLab 2.x and 3.0."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rounds", type=int, default=2)
parser.add_argument("--reset_types", default="ObjectRestingEEGrasped,ObjectAnywhereEEGrasped")
parser.add_argument("--dataset_dir", default=None)
parser.add_argument("--hold_steps", type=int, default=10)
parser.add_argument("--lift_steps", type=int, default=30)
parser.add_argument("--lift_z", type=float, default=1.0, help="OSC z action (scaled by 0.02 m/step)")
parser.add_argument("--no_gain_rand", action="store_true")
parser.add_argument("--gripper_action", type=float, default=-1.0)
parser.add_argument("--flip_ifk", action="store_true", help="negate the two inner_finger_knuckle joints after each reset (swapped-joint USD with old-convention datasets)")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, os, torch
_IFK = 1 if os.environ.get("UWLAB_ROBOT_ASSETS_DIR") else -1
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose

def T(x):
    return x.torch if hasattr(x, "torch") else x

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    print('[local assets] redirected:', apply_local_object_assets(env_cfg))
    import os as _os
    _sr = _os.environ.get("UWLAB_NEWTON_EQ_SOLREF")
    env_cfg.seed = 0
    rts = args_cli.reset_types.split(",")
    p = env_cfg.events.reset_from_reset_states.params
    if args_cli.dataset_dir: p["dataset_dir"] = args_cli.dataset_dir
    p["reset_types"] = rts; p["probs"] = [1.0] * len(rts)
    if args_cli.no_gain_rand:
        env_cfg.events.randomize_gripper_actuator_parameters = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    if _sr:
        from isaaclab_newton.physics.newton_manager import NewtonManager
        import warp as wp
        mjw = NewtonManager._solver.mjw_model
        tcst, dr = [float(v) for v in _sr.split(",")]
        arr = mjw.eq_solref.numpy(); print("eq_solref before", arr.shape, arr.reshape(-1, 2)[:3].tolist())
        arr[..., 0] = tcst; arr[..., 1] = dr
        mjw.eq_solref.assign(wp.array(arr, dtype=mjw.eq_solref.dtype, device=mjw.eq_solref.device))
        print("eq_solref set to", tcst, dr, "on", arr.shape)
    env.reset()
    term = env.event_manager.get_term_cfg("reset_from_reset_states").func
    robot, peg = env.scene["robot"], env.scene["insertive_object"]
    jn = list(robot.joint_names); bn = list(robot.body_names)
    fi = jn.index("finger_joint"); base = bn.index("robotiq_base_link")
    pads = [bn.index("left_inner_finger"), bn.index("right_inner_finger")]
    mimic = [(jn.index(k), s) for k, s in {"right_outer_knuckle_joint": 1, "left_inner_knuckle_joint": 1, "right_inner_knuckle_joint": -1, "left_inner_finger_knuckle_joint": _IFK, "right_inner_finger_knuckle_joint": _IFK}.items()]
    print(f"\nfinger stiffness/damping (env0): {T(robot.data.joint_stiffness)[0, fi].item():.2f} / {T(robot.data.joint_damping)[0, fi].item():.2f}")
    stats = {rt: dict(n=0, held_hold=0, held_lift=0, broke=0, abn=0, peg_dz=0.0, grip_dz=0.0, q0=0.0, q_end=0.0, gap0=0.0, gap_end=0.0) for rt in rts}
    ifk = [jn.index("left_inner_finger_knuckle_joint"), jn.index("right_inner_finger_knuckle_joint")]
    for r in range(args_cli.rounds):
        env.reset()
        if args_cli.flip_ifk:
            q = T(robot.data.joint_pos).clone(); q[:, ifk] *= -1
            robot.write_joint_state_to_sim(q, torch.zeros_like(q)); robot.set_joint_position_target(q)
            env.sim.step(); env.scene.update(env.physics_dt) if hasattr(env, "physics_dt") else None
        tid = term.task_id.clone()
        peg0 = T(peg.data.root_pos_w).clone(); grip0 = T(robot.data.body_pos_w)[:, base].clone()
        q0 = T(robot.data.joint_pos)[:, fi].clone()
        pad0 = (T(robot.data.body_pos_w)[:, pads[0]] - T(robot.data.body_pos_w)[:, pads[1]]).norm(dim=1)
        rel0 = (peg0 - grip0).norm(dim=1)
        broke = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        abn = torch.zeros_like(broke)
        act = torch.zeros(env.action_space.shape, device=env.device); act[:, -1] = args_cli.gripper_action
        for s in range(args_cli.hold_steps):
            env.step(act)
            q = T(robot.data.joint_pos); broke |= torch.stack([(q[:, i] - sg * q[:, fi]).abs() for i, sg in mimic], 1).max(1).values > 0.15
            abn |= env.termination_manager.get_term("abnormal_robot")
        peg_h = T(peg.data.root_pos_w).clone(); grip_h = T(robot.data.body_pos_w)[:, base].clone()
        held_hold = ((peg_h - grip_h).norm(dim=1) - rel0).abs() < 0.02
        act[:, 2] = args_cli.lift_z
        for s in range(args_cli.lift_steps):
            env.step(act)
            q = T(robot.data.joint_pos); broke |= torch.stack([(q[:, i] - sg * q[:, fi]).abs() for i, sg in mimic], 1).max(1).values > 0.15
            abn |= env.termination_manager.get_term("abnormal_robot")
        peg1 = T(peg.data.root_pos_w); grip1 = T(robot.data.body_pos_w)[:, base]
        rel1 = (peg1 - grip1).norm(dim=1)
        held_lift = ((rel1 - rel0).abs() < 0.02) & ((grip1[:, 2] - grip0[:, 2]) > 0.05)
        q1 = T(robot.data.joint_pos)[:, fi]
        pad1 = (T(robot.data.body_pos_w)[:, pads[0]] - T(robot.data.body_pos_w)[:, pads[1]]).norm(dim=1)
        for t, rt in enumerate(rts):
            m = tid == t; k = m.sum().item()
            if k == 0: continue
            st = stats[rt]; st["n"] += k
            st["held_hold"] += (held_hold & m).sum().item(); st["held_lift"] += (held_lift & m).sum().item()
            st["broke"] += (broke & m).sum().item(); st["abn"] += (abn & m).sum().item()
            st["peg_dz"] += (peg1[:, 2] - peg0[:, 2])[m].sum().item(); st["grip_dz"] += (grip1[:, 2] - grip0[:, 2])[m].sum().item()
            st["q0"] += q0[m].sum().item(); st["q_end"] += q1[m].sum().item(); st["gap0"] += pad0[m].sum().item(); st["gap_end"] += pad1[m].sum().item()
    print(f"\n=== lift probe: hold {args_cli.hold_steps} steps (gripper cmd {args_cli.gripper_action}), then lift z={args_cli.lift_z} for {args_cli.lift_steps} steps; gain_rand={'off' if args_cli.no_gain_rand else 'on'} ===")
    print(f"{'reset type':34s} {'N':>5} {'held@hold':>10} {'held@lift':>10} {'linkbreak':>10} {'abnormal':>9} {'peg dz':>7} {'grip dz':>8} {'q0':>6} {'q_end':>6} {'gap0':>6} {'gap_end':>7}")
    for rt, st in stats.items():
        n = max(st["n"], 1)
        print(f"{rt:34s} {st['n']:>5} {100*st['held_hold']/n:>9.1f}% {100*st['held_lift']/n:>9.1f}% {100*st['broke']/n:>9.1f}% {100*st['abn']/n:>8.1f}% {st['peg_dz']/n:>7.3f} {st['grip_dz']/n:>8.3f} {st['q0']/n:>6.3f} {st['q_end']/n:>6.3f} {st['gap0']/n:>6.3f} {st['gap_end']/n:>7.3f}")
    env.close()
main(); app.close()
