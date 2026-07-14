# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_name",
    type=str,
    default=None,
    help="Filename prefix for the recorded video. Defaults to the checkpoint name (e.g. model_0025000).",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default=None,
    help=(
        "Directory of model_*.pt checkpoints to sweep: records one video per checkpoint, "
        "reusing a single Isaac Sim instance. Implies video recording; ignores --checkpoint."
    ),
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--record_transitions",
    type=int,
    default=None,
    help=(
        "If set, record this many environment steps into a replay buffer and save it, then exit. "
        "Total transitions saved = num_envs * record_transitions. OnPolicyRunner checkpoints only."
    ),
)
parser.add_argument(
    "--transitions_output",
    type=str,
    default=None,
    help="Path to save the recorded replay buffer. Defaults to <log_dir>/play_transitions.pt.",
)
parser.add_argument(
    "--record_n_steps",
    "--num_steps",
    dest="record_n_steps",
    type=int,
    default=1,
    help=(
        "n-step return horizon to tag the recorded replay buffer with. Transitions are "
        "always stored as single steps in temporal order; SimpleReplayBuffer builds the "
        "n-step returns at sample time. Set this to match the FastSAC training agent.num_steps."
    ),
)
parser.add_argument(
    "--single_env_rb",
    action="store_true",
    default=False,
    help=(
        "Save in FastSAC online-replay-buffer format with n_env=1, collapsing the env "
        "dimension into the time dimension. Output has shape (num_envs*num_steps, 1, dim) "
        "and top-level keys (no `buffer_tensors` wrapper) so it can be loaded directly via "
        "train.py --replay_buffer_path. Use agent.buffer_size=<num_envs*num_steps> and "
        "--num_envs 1 in the training command."
    ),
)
parser.add_argument(
    "--eval",
    type=int,
    default=None,
    help="If set, run evaluation for this many episodes (across all envs) and report success rate, then exit.",
)
parser.add_argument(
    "--plot_ee",
    type=int,
    default=None,
    help="If set, run for this many steps collecting EE pose for env 0 (robot base frame), then plot and exit.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video or args_cli.checkpoint_dir:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import glob
import os
import time
import torch
import tqdm
from tensordict import TensorDict

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from holosoma.agents.fast_sac.fast_sac_utils import SimpleReplayBuffer

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from uwlab_rl.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx
from vecenv_wrapper import HolosomaVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def record_transitions_to_replay_buffer(
    env,
    policy,
    actor_obs_keys: list[str],
    critic_obs_keys: list[str],
    num_steps: int,
    output_path: str,
    task_name: str,
    device: str,
    gamma: float = 0.99,
    n_steps: int = 1,
    single_env_rb: bool = False,
) -> None:
    """Run a policy for ``num_steps`` and save transitions in SimpleReplayBuffer format.

    Works with both ``RslRlVecEnvWrapper`` (PPO) and ``HolosomaVecEnvWrapper`` (FastSAC).
    The policy callable should accept whatever ``env.get_observations()`` returns
    (TensorDict) and return an actions tensor.

    Transitions are stored so the output can be reloaded for off-policy training
    (e.g. warm-starting FastSAC from a PPO expert).

    N-step returns: single-step transitions are recorded in temporal order per env
    (with ``dones``/``truncations``), which is exactly the layout ``SimpleReplayBuffer``
    walks to build n-step returns at *sample* time (see the ``n_steps > 1`` branch of
    ``SimpleReplayBuffer.sample``). ``n_steps`` is therefore recorded for provenance and
    to construct the recording buffer consistently; it does not change the stored bytes.
    The consumer (``FastSACAgent.load_expert_replay_buffer``) must build the expert buffer
    with the agent's ``num_steps`` for the n-step returns to actually take effect at train
    time.

    Caveat: on a truncated/terminated step IsaacLab auto-resets and returns the *post-reset*
    observation, and the raw-tensor step path here does not expose the true terminal
    observation. The sampler stops the n-step accumulation at the first done/truncation, so
    the terminal transition bootstraps from the post-reset obs — a pre-existing limitation
    shared with 1-step recording, not introduced by n-step.
    """
    n_env = env.num_envs
    n_act = env.num_actions

    # Initial obs
    obs_td = env.get_observations().to(device)
    actor_obs = torch.cat([obs_td[g] for g in actor_obs_keys], dim=-1)
    critic_obs = torch.cat([obs_td[g] for g in critic_obs_keys], dim=-1)
    n_obs = actor_obs.shape[-1]
    n_critic_obs = critic_obs.shape[-1]

    rb = SimpleReplayBuffer(
        n_env=n_env,
        buffer_size=num_steps,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        n_steps=n_steps,
        gamma=gamma,
        device=device,
    )

    pbar = tqdm.tqdm(total=num_steps, desc="Recording transitions")
    for i in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs_td)
            # 1% per-env chance to replace with uniform random action in [-1, 1]
            # random_mask = torch.rand(n_env, 1, device=actions.device) < 0.01
            # random_actions = torch.rand_like(actions) * 2.0 - 1.0
            # actions = torch.where(random_mask, random_actions, actions)
            # actions = torch.ones_like(actions)

            next_obs_td, rewards, dones, extras = env.step(actions.to(env.device))
            next_obs_td = next_obs_td.to(device)
            rewards = rewards.to(device=device, dtype=torch.float)
            dones = dones.to(device=device, dtype=torch.long)
            truncations = extras.get(
                "time_outs", torch.zeros(n_env, dtype=torch.bool, device=device)
            )
            truncations = truncations.to(device).long()

            next_actor_obs = torch.cat([next_obs_td[g] for g in actor_obs_keys], dim=-1)
            next_critic_obs = torch.cat([next_obs_td[g] for g in critic_obs_keys], dim=-1)

            transition = TensorDict(
                {
                    "observations": actor_obs,
                    "actions": actions.to(device=device, dtype=torch.float),
                    "next": {
                        "observations": next_actor_obs,
                        "rewards": rewards,
                        "truncations": truncations,
                        "dones": dones,
                    },
                },
                batch_size=(n_env,),
                device=device,
            )
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = next_critic_obs
            rb.extend(transition)

            obs_td = next_obs_td
            actor_obs = next_actor_obs
            critic_obs = next_critic_obs
        pbar.update(1)
    pbar.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if single_env_rb:
        # Collapse env dim into time dim → shape (1, num_steps * n_env, ...). Save in
        # FastSAC online-RB format (top-level keys) so train.py --replay_buffer_path can
        # load it directly via FastSACAgent.load_replay_buffer.
        # SimpleReplayBuffer stores tensors as (n_env, buffer_size, ...) — confirmed
        # against the agent-side rb shape at load time.
        def _collapse(t: torch.Tensor) -> torch.Tensor:
            if t.dim() <= 1:
                return t
            return t.detach().cpu().reshape(1, -1, *t.shape[2:])

        total = num_steps * n_env
        payload = {
            "observations": _collapse(rb.observations),
            "actions": _collapse(rb.actions),
            "rewards": _collapse(rb.rewards),
            "dones": _collapse(rb.dones),
            "truncations": _collapse(rb.truncations),
            "next_observations": _collapse(rb.next_observations),
            "critic_observations": _collapse(rb.critic_observations),
            "next_critic_observations": _collapse(rb.next_critic_observations),
            "ptr": total,
            "n_env": 1,
            "buffer_size": total,
            "n_steps": n_steps,
            "global_step": 0,
        }
        torch.save(payload, output_path)
        print(
            f"[INFO] Saved {total} transitions (single-env RB, online format) to: {output_path}"
        )
        return

    payload = {
        "buffer_tensors": {
            "observations": rb.observations.detach().cpu(),
            "actions": rb.actions.detach().cpu(),
            "rewards": rb.rewards.detach().cpu(),
            "dones": rb.dones.detach().cpu(),
            "truncations": rb.truncations.detach().cpu(),
            "next_observations": rb.next_observations.detach().cpu(),
            "critic_observations": rb.critic_observations.detach().cpu(),
            "next_critic_observations": rb.next_critic_observations.detach().cpu(),
            "ptr": rb.ptr,
        },
        "metadata": {
            "n_env": n_env,
            "buffer_size": num_steps,
            "n_obs": n_obs,
            "n_act": n_act,
            "n_critic_obs": n_critic_obs,
            "n_steps": n_steps,
            "gamma": gamma,
            "actor_obs_keys": actor_obs_keys,
            "critic_obs_keys": critic_obs_keys,
            "task": task_name,
            "total_transitions": n_env * num_steps,
        },
    }

    torch.save(payload, output_path)
    print(f"[INFO] Saved {n_env * num_steps} transitions to: {output_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # make config compatible with installed rsl-rl version
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    checkpoint_list: list[str] = []
    if args_cli.checkpoint_dir:
        checkpoint_list = sorted(glob.glob(os.path.join(args_cli.checkpoint_dir, "model_*.pt")))
        if not checkpoint_list:
            print(f"[INFO] No model_*.pt checkpoints found in: {args_cli.checkpoint_dir}")
            return
        resume_path = checkpoint_list[0]
    elif args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # pull the viewport camera back for eval/video (task default eye is (2.0, 0.0, 0.75))
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = (3.5, 0.0, 1.3)

    # create isaac environment
    record_video = args_cli.video or bool(args_cli.checkpoint_dir)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if record_video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording (single-checkpoint mode only; sweep captures frames manually)
    if args_cli.video and not args_cli.checkpoint_dir:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": args_cli.video_name or os.path.splitext(os.path.basename(resume_path))[0],
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # determine runner type and wrap env accordingly
    is_fastsac = agent_cfg.class_name == "OnPolicyRunner" and hasattr(agent_cfg, "actor_obs_keys")

    if is_fastsac:
        env = HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create runner and load checkpoint
    if is_fastsac:
        runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
        runner.setup()
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        critic = runner.get_inference_critic(device=env.unwrapped.device)
        actor_obs_keys = agent_cfg.actor_obs_keys
        critic_obs_keys = agent_cfg.critic_obs_keys

        # Optional: record transitions into a replay buffer and exit
        if args_cli.record_transitions is not None and args_cli.record_transitions > 0:
            # FastSAC policy expects {"actor_obs": tensor}; wrap for compatibility
            def fastsac_sample_policy(obs_td):
                a_obs = torch.cat([obs_td[k] for k in actor_obs_keys], dim=-1)
                return policy({"actor_obs": a_obs})

            output_path = args_cli.transitions_output or os.path.join(log_dir, "play_transitions.pt")
            record_transitions_to_replay_buffer(
                env=env,
                policy=fastsac_sample_policy,
                actor_obs_keys=actor_obs_keys,
                critic_obs_keys=critic_obs_keys,
                num_steps=args_cli.record_transitions,
                output_path=output_path,
                task_name=args_cli.task,
                device=agent_cfg.device,
                gamma=getattr(agent_cfg.algorithm, "gamma", 0.99) if hasattr(agent_cfg, "algorithm") else 0.99,
                n_steps=args_cli.record_n_steps,
                single_env_rb=args_cli.single_env_rb,
            )
            env.close()
            return
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs_groups = runner.cfg["obs_groups"]
        ppo_actor_keys = list(obs_groups["policy"])
        ppo_critic_keys = list(obs_groups.get("critic", ppo_actor_keys))

        # Optional: record transitions into a replay buffer and exit
        if args_cli.record_transitions is not None and args_cli.record_transitions > 0:
            # policy = runner.get_inference_policy_sample(device=env.unwrapped.device)
            output_path = args_cli.transitions_output or os.path.join(log_dir, "play_transitions.pt")
            record_transitions_to_replay_buffer(
                env=env,
                policy=policy,
                actor_obs_keys=ppo_actor_keys,
                critic_obs_keys=ppo_critic_keys,
                num_steps=args_cli.record_transitions,
                output_path=output_path,
                task_name=args_cli.task,
                device=agent_cfg.device,
                gamma=float(runner.alg_cfg.get("gamma", 0.99)),
                n_steps=args_cli.record_n_steps,
                single_env_rb=args_cli.single_env_rb,
            )
            env.close()
            return
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # --checkpoint_dir sweep: record one video per checkpoint, reusing this Isaac Sim instance.
    if args_cli.checkpoint_dir:
        import imageio.v2 as imageio

        fps = int(env.unwrapped.metadata.get("render_fps", 30))
        video_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)
        for ckpt in checkpoint_list:
            name = args_cli.video_name or os.path.splitext(os.path.basename(ckpt))[0]
            print(f"[INFO] Recording {name} from checkpoint: {ckpt}")
            runner.load(ckpt)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            env.unwrapped.reset()
            obs = env.get_observations()
            frames = []
            for _ in range(args_cli.video_length):
                with torch.inference_mode():
                    if is_fastsac:
                        actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=1)
                        actions = policy({"actor_obs": actor_obs})
                    else:
                        actions = policy(obs)
                    obs, _, _, _ = env.step(actions)
                frame = env.unwrapped.render()
                if frame is not None:
                    frames.append(frame)
            out_path = os.path.join(video_dir, f"{name}.mp4")
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"[INFO] Saved {out_path} ({len(frames)} frames)")
        env.close()
        return

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()

    # --eval: run for a fixed number of episodes, report success rate, then exit.
    # Success isn't a termination (only timeout / abnormal_robot are), so we replicate
    # ProgressContext's check from the newest `insertive_asset_in_receptive_asset_frame`
    # frame: pos_norm < pos_thresh AND |aa_x|+|aa_y| < orient_thresh (approximates the
    # exact Euler-XY L1 used by the reward; matches it well near success). An episode
    # counts as success if any of its steps satisfied the check.
    if args_cli.eval is not None:
        obs_mgr = env.unwrapped.observation_manager
        term_names = list(obs_mgr.active_terms["policy"])
        term_dims = list(obs_mgr.group_obs_term_dim["policy"])
        offsets = {}
        cursor = 0
        for name, shape in zip(term_names, term_dims):
            dim = shape[0] if isinstance(shape, tuple) else int(shape)
            offsets[name] = (cursor, cursor + dim)
            cursor += dim
        if "insertive_asset_in_receptive_asset_frame" not in offsets:
            raise RuntimeError(
                "Task has no `insertive_asset_in_receptive_asset_frame` obs term; "
                "cannot compute success from obs for this task."
            )
        ins_block_start, ins_block_end = offsets["insertive_asset_in_receptive_asset_frame"]
        ins_newest_start = ins_block_end - 6  # last 6 dims of the block = newest frame
        POS_THRESH = 0.03
        ORIENT_THRESH = 0.2
        print(f"[eval] success check: pos_norm < {POS_THRESH}  AND  |aa_x|+|aa_y| < {ORIENT_THRESH}")
        print(f"[eval] ins_in_rec newest slice: [{ins_newest_start}:{ins_block_end}]")

        target_episodes = args_cli.eval
        total_episodes = 0
        successful_episodes = 0
        device = env.unwrapped.device
        ever_success = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

        # Per-episode initial peg (insertive_object) xyz in env-local frame.
        # Captured at episode start; refreshed after each done.
        insertive = env.unwrapped.scene["insertive_object"]
        receptive = env.unwrapped.scene["receptive_object"]
        env_origins = env.unwrapped.scene.env_origins  # [num_envs, 3]
        init_peg_xyz = (insertive.data.root_pos_w - env_origins)[:, :3].clone()
        peghole_xyz_env0 = (receptive.data.root_pos_w[0] - env_origins[0])[:3].cpu().numpy()
        # Collected per completed episode: (x, y, z, success_bool)
        episode_init_xyzs: list[torch.Tensor] = []
        episode_successes: list[bool] = []

        pbar = tqdm.tqdm(total=target_episodes, desc="Eval episodes", unit="ep")
        while total_episodes < target_episodes:
            with torch.inference_mode():
                if is_fastsac:
                    actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=1)
                    actions = policy({"actor_obs": actor_obs})
                else:
                    actor_obs = obs["policy"] if isinstance(obs, dict) else obs
                    actions = policy(obs)
                # Accumulate success at the pre-step state (obs at iter start).
                obs, _, dones, _ = env.step(actions)

                context_term = env.env.env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
                orientation_aligned = getattr(context_term, "orientation_aligned")
                position_aligned = getattr(context_term, "position_aligned")
                before_es = ever_success.clone()
                ever_success |= torch.where(orientation_aligned & position_aligned, True, False)
                new_success = torch.argwhere(~before_es & ever_success)
                if len(new_success) > 0:
                    print(f"new success at: {torch.argwhere(~before_es & ever_success)}")
                
            done_mask = dones.bool()
            successes = ever_success & done_mask
            n_done = int(done_mask.sum().item())
            n_success = int(successes.sum().item())
            total_episodes += n_done
            successful_episodes += n_success
            pbar.update(n_done)

            # Record (init_xy, success) for each env that just finished an episode,
            # then refresh init_peg_xy for those envs from the new post-reset peg pose.
            if n_done > 0:
                done_idx = torch.nonzero(done_mask, as_tuple=False).flatten()
                xyz_cpu = init_peg_xyz[done_idx].cpu()
                succ_cpu = successes[done_idx].cpu()
                for k in range(done_idx.numel()):
                    episode_init_xyzs.append(xyz_cpu[k])
                    episode_successes.append(bool(succ_cpu[k]))
                new_xyz = (insertive.data.root_pos_w - env_origins)[:, :3]
                init_peg_xyz[done_mask] = new_xyz[done_mask]
            # Reset success flag for envs that just finished an episode.
            ever_success[done_mask] = False
        
        pbar.close()
        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0
        print(f"\n[EVAL] Episodes: {total_episodes}  |  Successes: {successful_episodes}  |  Success rate: {success_rate:.2%}")

        # Plot 3D initial peg positions colored by episode outcome.
        if len(episode_init_xyzs) > 0:
            import matplotlib.pyplot as plt
            xyzs = torch.stack(episode_init_xyzs).numpy()
            successes_np = torch.tensor(episode_successes).numpy()
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection="3d")
            # ax.scatter(xyzs[successes_np, 0], xyzs[successes_np, 1], xyzs[successes_np, 2],
            #            c="green", s=18, alpha=0.7, label=f"success ({int(successes_np.sum())})")
            ax.scatter(xyzs[~successes_np, 0], xyzs[~successes_np, 1], xyzs[~successes_np, 2],
                       c="red", s=18, alpha=0.7, label=f"failure ({int((~successes_np).sum())})")
            ax.scatter(peghole_xyz_env0[0], peghole_xyz_env0[1], peghole_xyz_env0[2],
                       c="blue", s=200, marker="*", label="peghole (env 0)", zorder=10)
            ax.set_xlabel("x (m, env-local)")
            ax.set_ylabel("y (m, env-local)")
            ax.set_zlabel("z (m, env-local)")

            import numpy as np
            from matplotlib.animation import FuncAnimation

            # 3. Define the animation update function
            def rotate(angle):
                # Set the camera view (elevation, azimuth)
                # azimuth loops from 0 to 360 degrees for a full spin
                ax.view_init(elev=30, azim=angle)

            # 4. Create the slow 360 rotation
            ani = FuncAnimation(
                fig, 
                rotate, 
                frames=np.arange(0, 360, 1), # 1 frame per degree for a slow, smooth spin
                interval=50                  # 50 milliseconds pause between frames (~20 FPS)
            )

            ax.set_title(f"Initial peg xyz by outcome  ({successful_episodes}/{total_episodes} = {success_rate:.1%})")
            ax.legend(loc="best")
            ckpt_name = os.path.splitext(os.path.basename(resume_path))[0]
            plots_dir = os.path.join(log_dir, "plots", ckpt_name)
            os.makedirs(plots_dir, exist_ok=True)
            out_path = os.path.join(plots_dir, "eval_initial_peg_xyz.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print(f"[EVAL] Saved initial-pose scatter plot to: {out_path}")
            plt.show()

        env.close()
        return

    # --plot_ee: collect EE pose from actor obs for env 0 over N steps and plot, then exit.
    # Isaac Lab's ObservationManager builds a dict keyed by term name (insertion order, NOT
    # the order terms are declared in the config), then concatenates list(dict.values()).
    # For OmniReset PolicyCfg the dict iteration order observed at runtime is:
    #   insertive_asset_in_receptive_asset_frame (30) | prev_actions (35) | joint_pos (60)
    #   | end_effector_pose (30) | insertive_asset_pose (30) | receptive_asset_pose (30)
    # Within each term, history is flattened oldest→newest, so the newest EE frame is at
    # the last 6 dims of the end_effector_pose block: [149:155] in the 215-dim policy obs.
    if args_cli.plot_ee is not None:
        import matplotlib.pyplot as plt

        obs_mgr = env.unwrapped.observation_manager
        term_names = list(obs_mgr.active_terms["policy"])
        term_dims = list(obs_mgr.group_obs_term_dim["policy"])
        # Build offsets for each term in concatenation order.
        offsets = {}
        cursor = 0
        for name, shape in zip(term_names, term_dims):
            dim = shape[0] if isinstance(shape, tuple) else int(shape)
            offsets[name] = (cursor, cursor + dim)
            cursor += dim

        EE_DIM = 6
        ee_block_start, ee_block_end = offsets["end_effector_pose"]
        history = (ee_block_end - ee_block_start) // EE_DIM
        ee_start = ee_block_end - EE_DIM   # newest frame is the last EE_DIM dims of the block
        ee_end = ee_block_end

        print(f"[plot_ee] obs['policy'] shape: {obs['policy'].shape}")
        print(f"[plot_ee] term order: {term_names}")
        print(f"[plot_ee] term offsets: {offsets}")
        print(f"[plot_ee] history={history}  EE block=[{ee_block_start}:{ee_block_end}]  newest slice=[{ee_start}:{ee_end}]")

        ee_list = []
        for _ in range(args_cli.plot_ee):
            with torch.inference_mode():
                if is_fastsac:
                    actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=1)
                    actions = policy({"actor_obs": actor_obs})
                else:
                    actor_obs = obs if isinstance(obs, torch.Tensor) else torch.cat(list(obs.values()), dim=-1)
                    actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            ee_list.append(actor_obs[0, ee_start:ee_end].cpu())

        env.close()

        ee = torch.stack(ee_list).numpy()   # [steps, 6]
        steps = range(args_cli.plot_ee)
        labels   = ["pos_x", "pos_y", "pos_z", "aa_x", "aa_y", "aa_z"]
        ylabels  = ["m",     "m",     "m",     "rad",  "rad",  "rad"]

        axes_flat = plt.subplots(2, 3, figsize=(14, 6))[1].flatten()
        for i, (lbl, ylab) in enumerate(zip(labels, ylabels)):
            axes_flat[i].plot(steps, ee[:, i])
            axes_flat[i].set_title(f"EE {lbl} (obs, robot base frame)")
            axes_flat[i].set_xlabel("step")
            axes_flat[i].set_ylabel(ylab)
        plt.tight_layout()
        out_path = os.path.join(log_dir, "ee_pose_over_time.png")
        plt.savefig(out_path)
        print(f"[INFO] Saved EE pose plot to: {out_path}")
        plt.show()
        return

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if is_fastsac:
                # FastSACAgent policy expects {"actor_obs": tensor}
                actor_obs = torch.cat([obs[k] for k in actor_obs_keys], dim=1)
                actions = policy({"actor_obs": actor_obs})
                obs, _, dones, _ = env.step(actions)
            else:
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

            critic_obs = torch.cat([obs[k] for k in critic_obs_keys], dim=1)
            print(critic(critic_obs, actions)[0])

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
