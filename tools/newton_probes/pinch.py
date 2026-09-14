"""Newton: close the full-arm gripper on a kinematic peg placed between the pads; report pad gap vs finger q."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Newton-v0")
AppLauncher.add_app_launcher_args(parser); args_cli, remaining = parser.parse_known_args(); app = AppLauncher(args_cli).app
import gymnasium as gym, inspect, torch, isaaclab_tasks, uwlab_tasks  # noqa
from isaaclab.managers import ManagerTermBase
from uwlab_tasks.utils.hydra import hydra_task_compose
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.utils import apply_local_object_assets
def T(x): return x.torch if hasattr(x, "torch") else x
@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 8; apply_local_object_assets(env_cfg)
    p = env_cfg.events.reset_from_reset_states.params; p["dataset_dir"] = "./Datasets/OmniReset_patched"; p["reset_types"] = ["ObjectAnywhereEEGrasped"]; p["probs"] = [1.0]
    env_cfg.terminations.abnormal_robot = None
    env_cfg.scene.insertive_object.spawn.rigid_props.kinematic_enabled = True
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase): tc.func = tc.func(cfg=tc, env=env)
    r, peg = env.scene["robot"], env.scene["insertive_object"]; bn = list(r.body_names); jn = list(r.joint_names)
    pads = [bn.index("left_inner_finger"), bn.index("right_inner_finger")]; fi = jn.index("finger_joint")
    env.reset()
    a = torch.zeros(env.action_space.shape, device=env.device)
    print(f"{'t':>3} {'grip cmd':>8} {'finger q':>9} {'pad gap(m)':>10} {'l_ik':>6} {'l_ifk':>6} {'r_ik':>6} {'r_ifk':>6} {'r_ok':>6}   (env0; peg kinematic between pads)")
    for t in range(30):
        a[:, -1] = 1.0 if t < 10 else -1.0
        env.step(a)
        q = T(r.data.joint_pos)[0]; bp = T(r.data.body_pos_w)[0]
        gap = (bp[pads[0]] - bp[pads[1]]).norm().item()
        if t % 3 == 0 or t == 29:
            print(f"{t:>3} {a[0,-1].item():>8.0f} {q[fi].item():>9.3f} {gap:>10.4f} " + " ".join(f"{q[jn.index(n)].item():>6.2f}" for n in ("left_inner_knuckle_joint","left_inner_finger_knuckle_joint","right_inner_knuckle_joint","right_inner_finger_knuckle_joint","right_outer_knuckle_joint")))
    env.close()
main(); app.close()
