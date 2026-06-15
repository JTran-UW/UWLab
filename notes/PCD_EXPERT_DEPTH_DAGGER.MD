Status (as of d4129a4 on pat/dagger-symmetry)

Peg DAgger success_student_eval on Anywhere-only resets:
- Old 4-canonical pose teacher → plateaus at ~0.55 over 70K iters.
- New 8-canonical PC teacher + seed-locked PC → ~0.98 within 6K iters (~3h on cse L40S).

Branch + key commits

git fetch && git checkout pat/dagger-symmetry  # head: d4129a4

Critical commits (in chronological order):
- 5e8d970 ZeroG-ScenePC-Uniform RL teacher task registrations
- 08de609 DAgger pipeline: encoder-aware JIT + new env variants
- 80cc469 DistillationRunnerSplit: split eval pool into student/teacher halves
- 2547e12 Hydra-declare teacher_eval_fraction
- 39e8859 The fix: deterministic PC selection (per-body temporary_seed + per-source first-N selection). Old unseeded code made encoder overfit to run-specific points → fresh-load eval got ~30% vs
training-time 99%.
- abcaa38 Anywhere-only resets in the PC-teacher DAgger env (apples-to-apples vs old run).

Teacher checkpoints (deployable)

On Hyak:
- prev_act: /gscratch/weirdlab/patyin/uwlab/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/35149924/model_1600.pt
- NoPrevAct: /gscratch/weirdlab/patyin/uwlab/logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/35141064/model_1800.pt

Local copies + rsync'd JIT exports in the repo:
- teachers/peg_pc_prevact_v2_jit.pt (proprio_dim=25)
- teachers/peg_pc_noprevact_v2_jit.pt (proprio_dim=18)

To re-export from a checkpoint:
python scripts_v2/tools/convert_pc_expert_to_jit.py --std \
    --input  <ckpt.pt> --output teachers/<name>_jit.pt \
    --proprio-dim 25   # or 18 for NoPrevAct

Gym tasks

Teacher RL (4×H200 or 4×A40, ~24h to converge):
OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-Uniform-v0           # prev_act
OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-Uniform-NoPrevAct-v0  # NoPrevAct
Launch:
./docker/cluster/cluster_interface.sh --cluster hyak job base \
    --gpus 4 --partition gpu-a40 --account weirdlab --time 24:00:00 \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-Uniform-v0 \
    --num_envs 16384 --logger wandb --headless --distributed \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole \
    env.curriculum.gravity_curriculum.params.reduction=monitor_mean \
    env.curriculum.gravity_curriculum.params.floor=0.1

DAgger student (1× L40S, ~3h to converge):
OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-PCTeacher-Lean-v0           # prev_act
OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-PCTeacher-Lean-NoPrevAct-v0  # NoPrevAct
Launch:
HYAK_MEM_PER_GPU=5G ./docker/cluster/cluster_interface.sh --cluster hyak job base \
    --gpus 1 --partition gpu-l40s --account cse --time 16:00:00 \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Depth-DAgger-WristSide-Pretrained-Weighted-PCTeacher-Lean-v0 \
    --num_envs 256 --logger wandb --headless --enable_cameras \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole \
    agent.policy.teacher_jit_path=teachers/peg_pc_prevact_v2_jit.pt \
    agent.teacher_eval_fraction=0.5
The teacher_eval_fraction=0.5 splits the 26-env eval pool into 13 student / 13 teacher, both no-grad — gives you Metrics/success_student_eval and Metrics/success_teacher_eval to verify teacher is at
~99% (sanity check) while student learns.

Verifying a teacher checkpoint deploys correctly (locally)

python scripts_v2/eval_teacher_native.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-Uniform-v0 \
    --num_envs 64 --num_steps 500 --headless \
    --checkpoint <path/to/model_X.pt> \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole \
    env.curriculum.gravity_curriculum.params.reduction=monitor_mean \
    env.curriculum.gravity_curriculum.params.floor=0.1
Should report ~99% over ~300+ episodes. Anything below ~90% means a regression in PC determinism.

Plotting

scripts/plot_dagger_pc_vs_pose.py reads /tmp/{jobid}.dagger.tsv files. To pull a TSV from a Hyak run log:
ssh klone-login "awk 'BEGIN{OFS=\"\t\"} /Learning iteration/{...}' \
    /gscratch/weirdlab/patyin/slurm_logs/*{jobid}.out > /tmp/{jobid}.dagger.tsv"
scp klone-login:/tmp/{jobid}.dagger.tsv /tmp/

What was the bug

ScenePointCloud._sample_body_points called pytorch3d.sample_points_from_meshes and sample_farthest_points without temporary_seed, so the specific 285-of-pool robot points selected at __init__ depended
on global torch.random state — which differs across runs based on num_envs, sim init order, distributed rank. The trained teacher's encoder then overfit to its run-specific selection. Fix is in commit
39e8859: (a) per-body temporary_seed(42 + body_idx) lock, (b) replace global-FPS-in-world-frame (depended on physics body_pos_w) with deterministic per-source first-N selection.

Verification: same env, num_envs=16 and num_envs=64 produce identical points_hash (printed by ScenePC at init).