## Docs for experiments that we run to generate diverse policies

#### One of the commands we run to train multiple seeds on distributed computing:

```
MEM_PER_GPU=32G ./docker/cluster/sweep_seeds.sh --n 5 --start 0 --name peg_experts -- --cluster hyak job base --gpus 4 --partition ckpt --account weirdlab-ckpt --qos ckpt-gpu --time 24:00:00 --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ScenePC-v0 --num_envs 16384 --logger wandb --headless --distributed env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop
```