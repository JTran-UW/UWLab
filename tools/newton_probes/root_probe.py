import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Newton-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch, numpy as np, warp as wp
import isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose
def T(x): return x.torch if hasattr(x, "torch") else x

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 2; env_cfg.seed = 0
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
    apply_local_object_assets(env_cfg)
    env_cfg.events.reset_from_reset_states.params["dataset_dir"] = "./Datasets/OmniReset_patched"
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()
    from isaaclab_newton.physics.newton_manager import NewtonManager
    from newton.solvers import SolverNotifyFlags
    robot = env.scene["robot"]; m = NewtonManager._model; bn = list(robot.body_names)
    print("is_fixed_base:", getattr(robot, "is_fixed_base", "n/a"), "root_view fixed:", robot.root_view.is_fixed_base)
    jl = list(m.joint_label); jt = m.joint_type.numpy(); jc = m.joint_child.numpy(); jp = m.joint_parent.numpy()
    root_joints = [i for i, l in enumerate(jl) if "env_0/Robot" in l and jp[i] < 0]
    print("env0 robot root joints (parent=-1):", [(i, jl[i], int(jt[i])) for i in root_joints])
    def show(tag):
        jxp = m.joint_X_p.numpy()
        bq = NewtonManager.get_state_0().body_q.numpy()
        bl = list(m.body_label); bi = bl.index("/World/envs/env_0/Robot/base_link")
        print(f"[{tag}] root_pos_w={T(robot.data.root_pos_w)[0].cpu().numpy().round(4).tolist()} joint_X_p[root]={[jxp[i][:3].round(4).tolist() for i in root_joints]} body_q[base_link]={bq[bi][:3].round(4).tolist()} data.body_pos_w[base]={T(robot.data.body_pos_w)[0, bn.index('base_link')].cpu().numpy().round(4).tolist()}")
    show("after reset")
    pose = torch.tensor([[0.80, -0.70, 0.02, 0, 0, 0, 1.0], [0.80 - 1.5, -0.70, 0.02, 0, 0, 0, 1.0]], device=env.device)
    robot.write_root_pose_to_sim(pose)
    show("after write (no step)")
    env.sim.step(render=False); env.scene.update(env.physics_dt)
    show("after 1 step, no notify")
    robot.write_root_pose_to_sim(pose)
    NewtonManager.add_model_change(SolverNotifyFlags.JOINT_PROPERTIES)
    env.sim.step(render=False); env.scene.update(env.physics_dt)
    show("after write + notify JOINT_PROPERTIES + step")
    robot.write_root_pose_to_sim(pose)
    NewtonManager._solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES | SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.MODEL_PROPERTIES)
    env.sim.step(render=False); env.scene.update(env.physics_dt)
    show("after write + direct notify ALL + step")
    # explicit: write joint_X_p rows of the per-env root joints, then notify
    root_ids = torch.tensor([i for i, l in enumerate(jl) if l.endswith("/Robot/root_joint")], device=env.device)
    print("root joint ids:", root_ids.tolist())
    jxp_t = wp.to_torch(m.joint_X_p)
    jxp_t[root_ids] = pose
    NewtonManager.add_model_change(SolverNotifyFlags.JOINT_PROPERTIES)
    show("after set_root_transforms (no step)")
    env.sim.step(render=False); env.scene.update(env.physics_dt)
    show("after set_root_transforms + notify + step")
    for _ in range(12): env.sim.step(render=False)
    env.scene.update(env.physics_dt); show("after 12 more steps")
    env.close()
main(); app.close()
