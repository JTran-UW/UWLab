import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); parser.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-Newton-v0")
AppLauncher.add_app_launcher_args(parser); args_cli, remaining = parser.parse_known_args(); app = AppLauncher(args_cli).app
import gymnasium as gym, torch, isaaclab_tasks, uwlab_tasks  # noqa
from uwlab_tasks.utils.hydra import hydra_task_compose
def T(x): return x.torch if hasattr(x, "torch") else x
@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1; env_cfg.events.reset_from_reset_states = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    r = env.scene["robot"]; jn = list(r.joint_names); fi = jn.index("finger_joint"); g = [i for i, n in enumerate(jn) if i >= 6]
    a = torch.zeros(env.action_space.shape, device=env.device)
    print("\nq_cmd | " + " ".join(f"{jn[i][:22]:>22s}" for i in g))
    for q in (0.0, 0.2, 0.4, 0.6, 0.785):
        # drive finger via the binary action's underlying joint target: use open(+1)/close(-1)? use direct target instead
        for _ in range(40):
            tgt = T(r.data.joint_pos).clone(); tgt[:, fi] = q
            r.set_joint_position_target(tgt[:, [fi]], joint_ids=[fi])
            env.scene.write_data_to_sim(); env.sim.step(); env.scene.update(env.physics_dt)
        qs = T(r.data.joint_pos)[0]
        print(f"{q:5.3f} | " + " ".join(f"{qs[i].item():>22.3f}" for i in g))
    env.close()
main(); app.close()
