#!/usr/bin/env bash
# Regenerate the OmniReset peg/peghole datasets on IsaacLab 3.0, in dependency order.
set -u
cd ~/research/UWLab
export TMPDIR=/home/jtran/.cache/isaaclab_tmp OMNI_KIT_ACCEPT_EULA=YES
PY=~/miniconda3/envs/env_isaaclab3/bin/python
LOGDIR=~/research/UWLab/logs/regen_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
echo "logdir $LOGDIR"
run() {
  local name=$1; shift
  echo "=== [$(date +%T)] START $name"
  "$PY" -u "$@" > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  echo "=== [$(date +%T)] END $name rc=$rc"
  if [ $rc -ne 0 ]; then echo "FAILED at $name"; exit $rc; fi
}
run partial_assemblies scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 --num_envs 10 --num_trajectories 10 --headless env.scene.insertive_object=peg env.scene.receptive_object=peghole
run grasps scripts_v2/tools/record_grasps.py --task OmniReset-Robotiq2f85-GraspSampling-v0 --num_envs 8192 --num_grasps 1000 --headless env.scene.object=peg
run reaching scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 --num_envs 4096 --num_reset_states 10000 --headless env.scene.insertive_object=peg env.scene.receptive_object=peghole
run resting_grasped scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless env.scene.insertive_object=peg env.scene.receptive_object=peghole env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=./Datasets/OmniReset env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset
run anywhere_grasped scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless env.scene.insertive_object=peg env.scene.receptive_object=peghole env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset
run partially_assembled_grasped scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless env.scene.insertive_object=peg env.scene.receptive_object=peghole env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=./Datasets/OmniReset env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=./Datasets/OmniReset
echo "ALL DONE"
