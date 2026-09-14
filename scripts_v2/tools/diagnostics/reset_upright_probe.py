"""After a dataset reset, print each asset's world +Z axis (upright = (0,0,1))."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0")
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()
app = AppLauncher(args_cli).app
import gymnasium as gym, torch
import isaaclab_tasks, uwlab_tasks  # noqa
import isaaclab.utils.math as mu
from uwlab_tasks.utils.hydra import hydra_task_compose

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(env.action_space.shape, device=env.device))
    z = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
    print("\n=== asset up-vectors after dataset reset (mean over envs) ===")
    for name in ("robot", "table", "ur5_metal_support", "receptive_object", "insertive_object"):
        if name not in env.scene.keys():
            continue
        a = env.scene[name]
        q = a.data.root_quat_w.torch
        up = mu.quat_apply(q, z.expand(q.shape[0], 3))
        pos = a.data.root_pos_w.torch - env.scene.env_origins
        print(f"  {name:18s} +Z={[round(v,3) for v in up.mean(0).tolist()]}  pos={[round(v,3) for v in pos.mean(0).tolist()]}")
    env.close()

main()
app.close()
