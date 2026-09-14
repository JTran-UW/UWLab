"""Apply a unit torque to each arm joint from rest for one physics step; compare dqd with mass-matrix diag."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0")
parser.add_argument("--dataset_dir", default=None)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--nsteps", type=int, default=1)
parser.add_argument("--finger", type=float, default=None)
parser.add_argument("--pose", default="reset", help="reset | default | comma list of 6 arm angles")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose
def T(x): return x.torch if hasattr(x, "torch") else x
ARM = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 6; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    p = env_cfg.events.reset_from_reset_states.params
    if args_cli.dataset_dir: p["dataset_dir"] = args_cli.dataset_dir
    p["reset_types"] = ["ObjectAnywhereEEAnywhere"]; p["probs"] = [1.0]
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    robot = env.scene["robot"]; ids = robot.find_joints(ARM, preserve_order=True)[0]
    dt = env.physics_dt
    print("backend:", type(env_cfg.sim.physics).__name__, "dt", dt, "joint ids", ids)
    print("armature:", [round(v,5) for v in T(robot.data.joint_armature)[0, ids].tolist()])
    print("friction:", [round(v,5) for v in T(robot.data.joint_friction_coeff)[0, ids].tolist()] if hasattr(robot.data, "joint_friction_coeff") else "n/a")
    print("damping :", [round(v,3) for v in T(robot.data.joint_damping)[0, ids].tolist()])
    print("stiff   :", [round(v,3) for v in T(robot.data.joint_stiffness)[0, ids].tolist()])
    print("masses  :", {n: round(m,3) for n, m in zip(robot.body_names, T(robot.data.default_mass)[0].tolist())})
    try:
        M = T(robot.data.mass_matrix)[:, ids][:, :, ids]
        print("M diag  :", [[round(M[e,i,i].item(),4) for i in range(6)] for e in range(6)])
    except Exception as ex:
        print("mass_matrix unavailable:", ex)
    q0 = T(robot.data.joint_pos).clone()
    if args_cli.pose == "default":
        q0[:] = T(robot.data.default_joint_pos)[0:1]
    elif args_cli.pose != "reset":
        q0[:, ids] = torch.tensor([float(v) for v in args_cli.pose.split(",")], device=env.device)
    if args_cli.finger is not None:
        fj = robot.find_joints(["finger_joint"])[0][0]; q0[:, fj] = args_cli.finger
        # settle the passive gripper joints onto the mimic couplings
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        for _ in range(60):
            robot.set_joint_position_target(q0); env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt)
        q0 = T(robot.data.joint_pos).clone()
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    print("q0 env0:", [round(v,3) for v in q0[0, ids].tolist()])
    def step_with(tau):
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        for _ in range(args_cli.nsteps):
            robot.set_joint_effort_target(tau)
            env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt)
        return T(robot.data.joint_vel)[:, ids].clone()
    base = step_with(torch.zeros(6, robot.num_joints, device=env.device))
    tau = torch.zeros(6, robot.num_joints, device=env.device)
    for e in range(6): tau[e, ids[e]] = args_cli.tau
    qd = step_with(tau) - base
    for e in range(6):
        d = qd[e, e].item(); Meff = args_cli.tau * dt * args_cli.nsteps / d if abs(d) > 1e-9 else float("inf")
        print(f"tau={args_cli.tau} on {ARM[e]:20s}: dqd_self={d:+.5f} M_eff={Meff:.4f}  drift={[round(v,4) for v in base[e].tolist()]}  dqd_all={[round(v,4) for v in qd[e].tolist()]}")
    env.close()
main(); app.close()
