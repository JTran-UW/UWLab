# Point-Cloud BC Policy: Collect → Train → Eval

End-to-end pipeline for a PointNet behavior-cloning policy on the sim2real point cloud.
Collection + eval run in the **Isaac container** (`/isaac-sim/python.sh`); training runs in
**host `patlab` conda** (has Lightning). `logs/` is **not** bind-mounted into the container —
`docker cp` a checkpoint in before eval.

## 1. Collect demos (container)

Rolls out the JIT expert in the data-collection env and records `(scene_pc, proprio) → expert action`
for successful episodes into a robomimic-style HDF5.

```bash
/isaac-sim/python.sh scripts/tools/sim2real/collect_pc_demos.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-DataCollectionPC-v0 \
    --expert teachers/patrick_jit_expert.pt \
    --num_envs 64 --num_demos 100000 --out demos/pc_demos.hdf5 --headless
```

## 2. Train PointNet (host `patlab`)

Configs are dataclass-backed YAML in `scripts/imitation_learning/point_cloud/configs/`.
CLI flags override YAML. Logs to wandb (project `pc_bc`); checkpoints → `logs/pc_bc/<run>/`.

```bash
/home/ubuntu/miniforge3/envs/patlab/bin/python \
    scripts/imitation_learning/point_cloud/train_point_net.py \
    --dataset demos/pc_demos.hdf5 \
    --config scripts/imitation_learning/point_cloud/configs/base.yaml \
    --run_name base
```

Sweep several configs over GPUs: `bash scripts/imitation_learning/point_cloud/run_sweep.sh`

## 3. Eval in sim (container)

`play.py --bc_checkpoint` loads the Lightning `.ckpt` (PointNet + saved norm stats) and rolls it
out in the BC-eval env. Success rate is reported per terminated episode.

```bash
docker cp logs/pc_bc/<run>/<ckpt>.ckpt uw-lab-base:/tmp/bc.ckpt   # logs not mounted

/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-BCPointNetEval-v0 \
    --num_envs 512 --num_episodes 1024 --headless \
    --bc_checkpoint /tmp/bc.ckpt
```

Use `--num_episodes` (not `--num_steps`) for an unbiased success rate.
