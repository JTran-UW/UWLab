"""Frame-free check: does the peg stay up after a grasped reset, or fall?

The grasp-retention numbers relied on finger body_pos_w, which is suspect given
the 0.66 m disagreement with the calibrated model. Peg height needs no frame at
all: if the object is truly held, z stays put; if not, it falls to the surface.
"""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument("--num_envs",type=int,default=256); p.add_argument("--steps",type=int,default=40)
AppLauncher.add_app_launcher_args(p); a,h=p.parse_known_args(); a.headless=True
import sys; sys.argv=[sys.argv[0]]+h  # noqa: E702
app=AppLauncher(a).app
import gymnasium as gym, torch  # noqa: E402
import uwlab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
T="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"
print("\n"+"="*80); print("PEG HEIGHT AFTER GRASPED RESET (gripper commanded CLOSED)"); print("="*80)
print(f"  {'reset type':>36} {'z t=0':>9} {'z end':>9} {'drop':>9} {'fell>10cm':>10}")
for rt in ("ObjectRestingEEGrasped","ObjectAnywhereEEGrasped","ObjectPartiallyAssembledEEGrasped"):
    c=parse_env_cfg(T,device=a.device or "cuda:0",num_envs=a.num_envs)
    c.events.reset_from_reset_states.params["reset_types"]=[rt]
    c.events.reset_from_reset_states.params["probs"]=[1.0]
    e=gym.make(T,cfg=c).unwrapped; e.reset()
    peg=e.scene.rigid_objects["insertive_object"]; org=e.scene.env_origins
    z0=(peg.data.root_pos_w.torch-org)[:,2].clone()
    act=torch.zeros((e.num_envs,e.action_space.shape[1]),device=e.device); act[:,-1]=1.0
    for _ in range(a.steps): e.step(act)
    z1=(peg.data.root_pos_w.torch-org)[:,2]
    d=z0-z1
    print(f"  {rt:>36} {z0.mean():>9.4f} {z1.mean():>9.4f} {d.mean():>9.4f} {(d>0.10).float().mean()*100:>9.1f}%")
    e.close(); del e
print("="*80+"\n",flush=True)
app.close()
