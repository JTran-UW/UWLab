Single env training:
```
python scripts/reinforcement_learning/holosoma/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-v0 \
    --num_envs 1 --logger wandb --headless \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

Multi-GPU training:
```
python -m torch.distributed.run     --nnodes 1     --nproc_per_node 4     scripts/reinforcement_learning/holosoma/train.py     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-v0     --num_envs 4096     --logger wandb     --headless     --distributed     env.scene.insertive_object=peg     env.scene.receptive_object=peghole
```

Multi-GPU training easy:
```
python -m torch.distributed.run     --nnodes 1     --nproc_per_node 4     scripts/reinforcement_learning/holosoma/train.py     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-OffPolicy-v0     --num_envs 4096     --logger wandb     --headless     --distributed     env.scene.insertive_object=peg     env.scene.receptive_object=peghole
```


Multi-GPU training w/expert buffer:
```
python -m torch.distributed.run     --nnodes 1     --nproc_per_node 2     scripts/reinforcement_learning/holosoma/train.py     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-v0     --num_envs 2048     --logger wandb     --headless     --distributed     env.scene.insertive_object=peg     env.scene.receptive_object=peghole --expert_transitions logs/rsl_rl/expert_buffer.pt
```

Collect expert replay buffer ~1M samples
```
python scripts/reinforcement_learning/holosoma/play.py \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-v0 \
      --num_envs 1024 \
      --checkpoint peg_state_rl_expert_seed42.pt \
      --record_transitions 1000 \
      --transitions_output expert_rb/expert_transitions.pt \
      --headless
```

Eval
```
python scripts/reinforcement_learning/holosoma/play.py \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-OffPolicy-v0 \
      --num_envs 1 \
      --checkpoint logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/4096-Env-4-GPU-32k-BS-2026-04-16_11-45-13/model_0014100.pt
```

Eval critic
```
python scripts/reinforcement_learning/holosoma/eval_critic.py \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-OffPolicy-v0 \
      --num_envs 1 \
      --checkpoint logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/Expert-Data-1M-2026-04-16_01-13-04/model_0118300.pt --expert_checkpoint peg_state_rl_expert_seed42.pt --video
```

Inspect replay buffer
```
python scripts/reinforcement_learning/holosoma/inspect_replay_buffer.py fast_sac_transitions.pt --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-OffPolicy-v0
```

Train reaching PPO
```
python -m torch.distributed.run --nnodes 1 --nproc_per_node 2 scripts/reinforcement_learning/rsl_rl/train.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Reaching-v0 --num_envs 16384 --distributed --headless --logger wandb --log_project_name omnireset_fastsac
```

Train reaching FastSAC
```
python -m torch.distributed.run --nnodes 1 --nproc_per_node 2 scripts/reinforcement_learning/holosoma/train.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Reaching-OffPolicy-v0 --num_envs 8192 --distributed --headless --logger wandb --log_project_name omnireset_fastsac
```

Train with expert data 4-29:
```
python -m torch.distributed.run     --nnodes 1     --nproc_per_node 2     scripts/reinforcement_learning/holosoma/train.py     --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-v0     --num_envs 1024     --logger wandb     --headless     --distributed    --expert_transitions expert_rb/expert_transitions.pt  env.scene.insertive_object=peg     env.scene.receptive_object=peghole env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset
```
