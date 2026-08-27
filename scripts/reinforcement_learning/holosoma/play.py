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
    "--record_actor_obs_keys",
    type=str,
    nargs="+",
    default=None,
    help=(
        "Env observation group(s) to store as the recorded actor observations. Defaults to the "
        "groups the loaded policy acts from. Set this to record data for a task whose policy "
        "obs differ from the expert's (e.g. expert acts on state, record depth/grayscale)."
    ),
)
parser.add_argument(
    "--record_critic_obs_keys",
    type=str,
    nargs="+",
    default=None,
    help=(
        "Env observation group(s) to store as the recorded critic observations. Defaults to the "
        "loaded policy's critic groups. Keep the env's `critic` group matching the checkpoint so "
        "it loads, and point this at an extra group (e.g. `critic_no_priv`) to record something else."
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
    "--stochastic",
    action="store_true",
    default=False,
    help=(
        "Sample actions from the policy distribution instead of taking its mean. This is what the "
        "TRAINING rollout does (fast_sac_agent sets policy = actor.explore), whereas play.py "
        "otherwise evaluates the deterministic mean -- so a policy whose mean is fine but whose "
        "log_std has blown up looks healthy here and destabilises during training. Also essential "
        "for dither-dependent experts whose mean action jams. FastSAC and OnPolicyRunner (PPO)."
    ),
)
parser.add_argument(
    "--success_to_truncation",
    action="store_true",
    default=False,
    help=(
        "When recording transitions, store dones caused by the 'success' termination as truncations "
        "so the consumer bootstraps through them (FastSAC with handle_truncations bootstraps where "
        "truncations | ~dones). Use with a Success-Termination task so episodes end at insertion "
        "without the success step being treated as a zero-value terminal."
    ),
)
parser.add_argument(
    "--eval_episodes_per_env",
    type=int,
    default=None,
    help=(
        "If set, run evaluation until EVERY env has completed this many episodes, then report the "
        "success rate and exit. Counted per env rather than as a global total: success is a "
        "termination, so successful episodes are shorter and recur more often within a fixed "
        "stepping window -- a global target fills up with them and inflates the success rate."
    ),
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


def _flatten_obs_groups(obs_td, keys: list[str]) -> torch.Tensor:
    """Concatenate observation groups into the flat (n_env, D) layout the FastSAC agent expects.

    Vector groups are already (n_env, D) and pass through untouched. Image groups arrive with rank
    > 2 -- the grayscale rig gives (n_env, history, cameras, H, W) -- and a plain ``cat(..., dim=-1)``
    would concatenate along W and then read ``shape[-1]`` as the width, sizing the buffer wrong.

    Flattening each group and concatenating in ``keys`` order reproduces exactly the layout
    ``FastSACAgent.setup`` assumes when it builds ``actor_obs_indices``: groups laid out
    contiguously in ``actor_obs_keys`` order, each of size ``obs_dim * history_length``. ``CNNActor``
    then slices its ``encoder_obs_key`` range back out and ``.view(N, *encoder_obs_shape)`` it, so
    the flatten here and the view there are inverses as long as the key order matches.
    """
    return torch.cat([obs_td[k].reshape(obs_td[k].shape[0], -1) for k in keys], dim=-1)


def resolve_record_obs_keys(
    env, policy_actor_keys: list[str], policy_critic_keys: list[str]
) -> tuple[list[str], list[str]]:
    """Pick which env observation groups get *recorded*, independent of which the policy *acts* from.

    Defaults to the policy's own groups (unchanged behaviour). ``--record_actor_obs_keys`` /
    ``--record_critic_obs_keys`` override them, which is what lets one expert generate data for a
    different task: the expert keeps consuming the groups it was trained on while the buffer stores
    the target task's groups. Requires a single env exposing both sets of groups.
    """
    actor_keys = args_cli.record_actor_obs_keys or policy_actor_keys
    critic_keys = args_cli.record_critic_obs_keys or policy_critic_keys

    # Validate against manager metadata, not get_observations(): ObservationManager.compute()
    # appends to each term's history CircularBuffer, so an extra call here would push a duplicate
    # frame into history_length>1 groups and skew the first recorded transitions.
    obs_manager = getattr(env.unwrapped, "observation_manager", None)
    if obs_manager is not None:
        available = list(obs_manager.active_terms.keys())
        missing = [k for k in (*actor_keys, *critic_keys) if k not in available]
        if missing:
            raise ValueError(
                f"Requested observation group(s) {missing} are not produced by task '{args_cli.task}'. "
                f"Available groups: {available}. Add them to the task's ObservationsCfg (see the "
                f"data-collection configs) or pick from the available groups."
            )
    if actor_keys != policy_actor_keys or critic_keys != policy_critic_keys:
        print(
            f"[INFO] Recording obs groups actor={actor_keys}, critic={critic_keys} "
            f"(policy acts from actor={policy_actor_keys}, critic={policy_critic_keys})"
        )
    return list(actor_keys), list(critic_keys)


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
    success_to_truncation: bool = False,
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

    if success_to_truncation:
        term_mgr = env.unwrapped.termination_manager
        if "success" not in list(term_mgr.active_terms):
            raise ValueError(
                "--success_to_truncation requires a task with a 'success' termination term; "
                f"active terms: {list(term_mgr.active_terms)}"
            )
        print("[INFO] --success_to_truncation: success-caused dones will be recorded as truncations.")

    # Initial obs
    obs_td = env.get_observations().to(device)
    actor_obs = _flatten_obs_groups(obs_td, actor_obs_keys)
    critic_obs = _flatten_obs_groups(obs_td, critic_obs_keys)
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
            if success_to_truncation:
                success = term_mgr.get_term("success").to(device).long()
                truncations = (truncations | success).long()

            next_actor_obs = _flatten_obs_groups(next_obs_td, actor_obs_keys)
            next_critic_obs = _flatten_obs_groups(next_obs_td, critic_obs_keys)

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
            "success_to_truncation": success_to_truncation,
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

        if args_cli.stochastic:
            # Mirror get_inference_policy but sample instead of taking the mean, so eval matches the
            # training rollout (fast_sac_agent.py: self.policy = self.actor.explore). Same frozen
            # normalizer -- update=False -- so the only difference is mean vs sample.
            _dev = env.unwrapped.device
            _actor = runner.actor.to(_dev)
            _actor.eval()
            _obs_norm = runner.obs_normalizer.to(_dev)
            _obs_norm.eval()

            def _stochastic_policy(obs: dict) -> torch.Tensor:
                x = obs["actor_obs"]
                nx = _obs_norm(x, update=False) if runner.obs_normalization else x
                return _actor.explore(nx, deterministic=False)

            policy = _stochastic_policy
            print("[INFO] --stochastic: sampling actions from the policy distribution (matches training rollout).")

        # Optional: record transitions into a replay buffer and exit
        if args_cli.record_transitions is not None and args_cli.record_transitions > 0:
            # FastSAC policy expects {"actor_obs": tensor}; wrap for compatibility
            def fastsac_sample_policy(obs_td):
                a_obs = torch.cat([obs_td[k] for k in actor_obs_keys], dim=-1)
                return policy({"actor_obs": a_obs})

            rec_actor_keys, rec_critic_keys = resolve_record_obs_keys(env, actor_obs_keys, critic_obs_keys)
            output_path = args_cli.transitions_output or os.path.join(log_dir, "play_transitions.pt")
            record_transitions_to_replay_buffer(
                env=env,
                policy=fastsac_sample_policy,
                actor_obs_keys=rec_actor_keys,
                critic_obs_keys=rec_critic_keys,
                num_steps=args_cli.record_transitions,
                output_path=output_path,
                task_name=args_cli.task,
                device=agent_cfg.device,
                gamma=getattr(agent_cfg.algorithm, "gamma", 0.99) if hasattr(agent_cfg, "algorithm") else 0.99,
                n_steps=args_cli.record_n_steps,
                single_env_rb=args_cli.single_env_rb,
                success_to_truncation=args_cli.success_to_truncation,
            )
            env.close()
            return
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        try:
            runner.load(resume_path)
        except RuntimeError as e:
            # Tolerate critic-side size mismatches only (e.g. the env's privileged critic group has a
            # different dim than the checkpoint's): the critic is unused at play/recording time.
            loaded_dict = torch.load(resume_path, weights_only=False, map_location=agent_cfg.device)
            ckpt_sd = loaded_dict["model_state_dict"]
            model_sd = runner.alg.policy.state_dict()
            mismatched = [
                k for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape != v.shape
            ]
            if any(not k.startswith(("critic.", "critic_obs_normalizer.")) for k in mismatched):
                raise e
            filtered = {k: v for k, v in ckpt_sd.items() if k not in mismatched}
            runner.alg.policy.load_state_dict(filtered, strict=False)
            print(
                f"[WARN] Strict checkpoint load failed; skipped critic-side size-mismatched tensors: {mismatched}. "
                "Critic outputs are untrained/meaningless for this run."
            )
        if args_cli.stochastic:
            policy = runner.get_inference_policy_stochastic(device=env.unwrapped.device)
            print("[INFO] --stochastic: sampling actions from the policy distribution (matches training rollout).")
        else:
            policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs_groups = runner.cfg["obs_groups"]
        ppo_actor_keys = list(obs_groups["policy"])
        ppo_critic_keys = list(obs_groups.get("critic", ppo_actor_keys))

        # Optional: record transitions into a replay buffer and exit
        if args_cli.record_transitions is not None and args_cli.record_transitions > 0:
            # policy = runner.get_inference_policy_sample(device=env.unwrapped.device)
            rec_actor_keys, rec_critic_keys = resolve_record_obs_keys(env, ppo_actor_keys, ppo_critic_keys)
            output_path = args_cli.transitions_output or os.path.join(log_dir, "play_transitions.pt")
            record_transitions_to_replay_buffer(
                env=env,
                policy=policy,
                actor_obs_keys=rec_actor_keys,
                critic_obs_keys=rec_critic_keys,
                num_steps=args_cli.record_transitions,
                output_path=output_path,
                task_name=args_cli.task,
                device=agent_cfg.device,
                gamma=float(runner.alg_cfg.get("gamma", 0.99)),
                n_steps=args_cli.record_n_steps,
                single_env_rb=args_cli.single_env_rb,
                success_to_truncation=args_cli.success_to_truncation,
            )
            env.close()
            return
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    if args_cli.stochastic and not is_fastsac and agent_cfg.class_name != "OnPolicyRunner":
        print("[WARN] --stochastic is only wired for FastSAC and OnPolicyRunner; evaluating the deterministic policy instead.")

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
            if args_cli.stochastic and agent_cfg.class_name == "OnPolicyRunner":
                policy = runner.get_inference_policy_stochastic(device=env.unwrapped.device)
            else:
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
    if args_cli.eval_episodes_per_env is not None:
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

        # Episode outcome codes, shared by the printed breakdown and the scatter plot.
        OUTCOME_SUCCESS, OUTCOME_TIMEOUT, OUTCOME_ABNORMAL, OUTCOME_OTHER = 0, 1, 2, 3
        OUTCOME_STYLE = {
            OUTCOME_SUCCESS: ("green", "success"),
            OUTCOME_TIMEOUT: ("red", "timeout"),
            OUTCOME_ABNORMAL: ("orange", "abnormal_robot"),
            OUTCOME_OTHER: ("gray", "other"),
        }

        episodes_per_env = args_cli.eval_episodes_per_env
        device = env.unwrapped.device
        ever_success = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        # Per-env tallies, not global sums. Envs that finish early keep stepping, but their surplus
        # episodes are discarded, so every env contributes exactly `episodes_per_env` to the rate.
        ep_count = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        succ_count = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        # Episode length, counted here rather than read from env.episode_length_buf: the env
        # auto-resets on done and zeroes that buffer before we get to look at it.
        ep_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        episode_durations: list[int] = []
        # Local timestep (1-indexed within the episode) at which each env FIRST met the success
        # condition; -1 while it has not. Success is not a termination here, so the episode keeps
        # running afterwards -- this is time-to-success, not episode duration.
        first_success_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)
        episode_success_steps: list[int] = []
        step_dt = float(getattr(env.unwrapped, "step_dt", 0.0))
        # Failure breakdown. Which termination actually fired is read per-term off the
        # TerminationManager rather than inferred, so an episode that ended for some third reason
        # (e.g. `first_episode_termination` left enabled) shows up as "other" instead of being
        # silently lumped into timeouts.
        timeout_count = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        abnormal_count = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        other_count = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        term_mgr = env.unwrapped.termination_manager
        term_names = list(term_mgr.active_terms)
        has_timeout = "time_out" in term_names
        has_abnormal = "abnormal_robot" in term_names
        print(f"[eval] termination terms: {term_names}")
        if not has_timeout or not has_abnormal:
            print(
                "[eval] WARNING: expected 'time_out' and 'abnormal_robot' terms; missing ones will "
                "be reported under 'other'."
            )
        zeros_b = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

        # Per-episode initial peg (insertive_object) xyz in env-local frame.
        # Captured at episode start; refreshed after each done.
        insertive = env.unwrapped.scene["insertive_object"]
        receptive = env.unwrapped.scene["receptive_object"]
        env_origins = env.unwrapped.scene.env_origins  # [num_envs, 3]
        init_peg_xyz = (insertive.data.root_pos_w - env_origins)[:, :3].clone()
        peghole_xyz_env0 = (receptive.data.root_pos_w[0] - env_origins[0])[:3].cpu().numpy()
        # Collected per completed episode: (x, y, z) and an outcome code from OUTCOME_*.
        episode_init_xyzs: list[torch.Tensor] = []
        episode_outcomes: list[int] = []

        pbar = tqdm.tqdm(total=episodes_per_env * env.num_envs, desc="Eval episodes", unit="ep")
        while bool((ep_count < episodes_per_env).any()):
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
                newly_succeeded = ~before_es & ever_success
                # ep_steps is still the pre-step count, so the step just taken is ep_steps + 1.
                # Same 1-indexed convention as episode_durations below.
                first_success_step[newly_succeeded] = ep_steps[newly_succeeded] + 1
                new_success = torch.argwhere(newly_succeeded)
                if len(new_success) > 0:
                    print(f"new success at: envs {new_success.flatten().tolist()} "
                          f"step {first_success_step[newly_succeeded].tolist()}")

            ep_steps += 1
            done_mask = dones.bool()
            # Only an env's first `episodes_per_env` episodes count. Dropping the surplus is what
            # removes the length bias: without this gate, envs whose episodes end early (i.e.
            # succeed) recycle faster and contribute more samples than envs that run to timeout.
            counting = done_mask & (ep_count < episodes_per_env)
            successes = ever_success & counting
            ep_count += counting.long()
            succ_count += successes.long()
            n_counted = int(counting.sum().item())
            pbar.update(n_counted)

            # Classify each counted episode into exactly one bucket. Read the per-term flags now:
            # they describe the step that just ended, before the next compute() overwrites them.
            # Success wins over any termination flag, and abnormal_robot wins over time_out so a
            # robot that blows up on the very last step is not filed as a clean timeout.
            failed = counting & ~successes
            timed_out = term_mgr.get_term("time_out").bool() if has_timeout else zeros_b
            abnormal = term_mgr.get_term("abnormal_robot").bool() if has_abnormal else zeros_b
            abnormal_f = failed & abnormal
            timeout_f = failed & timed_out & ~abnormal
            abnormal_count += abnormal_f.long()
            timeout_count += timeout_f.long()
            other_count += (failed & ~abnormal_f & ~timeout_f).long()

            # Record (init_xyz, outcome) for each episode that counted. Outcome uses the same
            # buckets as the printed breakdown, so plot and report can never disagree.
            if n_counted > 0:
                outcome = torch.full_like(ep_count, OUTCOME_OTHER)
                outcome[successes] = OUTCOME_SUCCESS
                outcome[timeout_f] = OUTCOME_TIMEOUT
                outcome[abnormal_f] = OUTCOME_ABNORMAL
                counted_idx = torch.nonzero(counting, as_tuple=False).flatten()
                xyz_cpu = init_peg_xyz[counted_idx].cpu()
                out_cpu = outcome[counted_idx].cpu()
                dur_cpu = ep_steps[counted_idx].cpu()
                fss_cpu = first_success_step[counted_idx].cpu()
                for k in range(counted_idx.numel()):
                    episode_init_xyzs.append(xyz_cpu[k])
                    episode_outcomes.append(int(out_cpu[k]))
                    episode_durations.append(int(dur_cpu[k]))
                    episode_success_steps.append(int(fss_cpu[k]))
            # Refresh on EVERY done, not just counted ones: the env resets regardless, so a stale
            # init pose would otherwise be attributed to a later episode that does count.
            if bool(done_mask.any()):
                new_xyz = (insertive.data.root_pos_w - env_origins)[:, :3]
                init_peg_xyz[done_mask] = new_xyz[done_mask]
            # Reset success flag and step counter for envs that just finished an episode.
            ever_success[done_mask] = False
            ep_steps[done_mask] = 0
            first_success_step[done_mask] = -1

        pbar.close()
        total_episodes = int(ep_count.sum().item())
        successful_episodes = int(succ_count.sum().item())
        n_timeout = int(timeout_count.sum().item())
        n_abnormal = int(abnormal_count.sum().item())
        n_other = int(other_count.sum().item())
        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0
        pct = lambda n: (n / total_episodes if total_episodes > 0 else 0.0)
        print(f"\n[EVAL] Episodes: {total_episodes}  |  Successes: {successful_episodes}  |  Success rate: {success_rate:.2%}")
        print(f"[EVAL] Failure breakdown ({total_episodes - successful_episodes} failures):")
        print(f"[EVAL]   timeout        : {n_timeout:6d}  ({pct(n_timeout):.2%} of episodes)")
        print(f"[EVAL]   abnormal_robot : {n_abnormal:6d}  ({pct(n_abnormal):.2%} of episodes)")
        print(f"[EVAL]   other          : {n_other:6d}  ({pct(n_other):.2%} of episodes)")
        if n_other:
            print(
                "[EVAL]   NOTE: 'other' means an episode ended on neither time_out nor "
                "abnormal_robot -- most often first_episode_termination left enabled. Disable it "
                "with env.terminations.first_episode_termination=null."
            )

        # Episode duration, overall and split by outcome. Reported per outcome because the pooled
        # mean is a mixture: successes end early, timeouts sit at the episode cap, so the average
        # over all episodes mostly tracks the success rate rather than any property of the policy.
        def _dur_stats(sel_codes):
            vals = [d for d, o in zip(episode_durations, episode_outcomes) if o in sel_codes]
            if not vals:
                return None
            m = sum(vals) / len(vals)
            return m, min(vals), max(vals), len(vals)

        _fmt = lambda s: f"{s:.1f} steps" + (f" ({s * step_dt:.2f} s)" if step_dt > 0 else "")
        allstats = _dur_stats({OUTCOME_SUCCESS, OUTCOME_TIMEOUT, OUTCOME_ABNORMAL, OUTCOME_OTHER})
        if allstats:
            print(f"[EVAL] Mean episode duration: {_fmt(allstats[0])}  "
                  f"[min {allstats[1]}, max {allstats[2]} steps over {allstats[3]} episodes]")
            for code in (OUTCOME_SUCCESS, OUTCOME_TIMEOUT, OUTCOME_ABNORMAL, OUTCOME_OTHER):
                st = _dur_stats({code})
                if st:
                    print(f"[EVAL]   {OUTCOME_STYLE[code][1]:<15}: mean {_fmt(st[0])}  (n={st[3]})")
        # Time to success: mean over SUCCESSFUL episodes only of the local step at which the
        # success condition was first met. Distinct from the success-episode duration above --
        # success is not a termination, so the episode runs on to timeout after reaching it.
        succ_steps = [s for s, o in zip(episode_success_steps, episode_outcomes)
                      if o == OUTCOME_SUCCESS and s > 0]
        assert len(succ_steps) == successful_episodes, (
            f"time-to-success mismatch: {len(succ_steps)} recorded vs {successful_episodes} successes"
        )
        if succ_steps:
            mean_tts = sum(succ_steps) / len(succ_steps)
            print(f"[EVAL] Mean time to success: {_fmt(mean_tts)}  "
                  f"[min {min(succ_steps)}, max {max(succ_steps)} steps over {len(succ_steps)} successes]")
        else:
            print("[EVAL] Mean time to success: n/a (no successful episodes)")

        # Buckets partition the counted episodes; a mismatch means an episode was double-counted or
        # missed, which would silently corrupt every rate above.
        assert successful_episodes + n_timeout + n_abnormal + n_other == total_episodes, (
            f"bucket mismatch: {successful_episodes}+{n_timeout}+{n_abnormal}+{n_other} "
            f"!= {total_episodes}"
        )

        # Plot 3D initial peg positions colored by episode outcome.
        if len(episode_init_xyzs) > 0:
            import matplotlib.pyplot as plt
            xyzs = torch.stack(episode_init_xyzs).numpy()
            outcomes_np = torch.tensor(episode_outcomes).numpy()
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection="3d")
            # One scatter per outcome. Abnormal is drawn last so the rare orange points land on top
            # of the dense success/timeout clouds instead of being hidden inside them.
            for code in (OUTCOME_SUCCESS, OUTCOME_TIMEOUT, OUTCOME_OTHER, OUTCOME_ABNORMAL):
                sel = outcomes_np == code
                if not sel.any():
                    continue
                color, label = OUTCOME_STYLE[code]
                ax.scatter(xyzs[sel, 0], xyzs[sel, 1], xyzs[sel, 2],
                           c=color, s=18, alpha=0.7, label=f"{label} ({int(sel.sum())})")
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

            # Episode-duration histogram, stacked by outcome. Kept as its own figure rather than a
            # subplot so the 3D scatter above keeps its layout and rotation animation.
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            durs = np.asarray(episode_durations, dtype=float)
            codes = np.asarray(episode_outcomes)
            present = [c for c in (OUTCOME_SUCCESS, OUTCOME_TIMEOUT, OUTCOME_ABNORMAL, OUTCOME_OTHER)
                       if (codes == c).any()]
            bins = np.histogram_bin_edges(durs, bins=min(40, max(10, len(set(episode_durations)))))
            ax2.hist([durs[codes == c] for c in present], bins=bins, stacked=True,
                     color=[OUTCOME_STYLE[c][0] for c in present],
                     label=[f"{OUTCOME_STYLE[c][1]} (n={int((codes == c).sum())}, "
                            f"mean {durs[codes == c].mean():.0f})" for c in present])
            mean_all = durs.mean()
            ax2.axvline(mean_all, color="black", ls="--", lw=1.5,
                        label=f"overall mean {mean_all:.1f} steps"
                              + (f" = {mean_all * step_dt:.2f} s" if step_dt > 0 else ""))
            ax2.set_xlabel("episode duration (steps)" + (f"   [1 step = {step_dt:.3f} s]" if step_dt > 0 else ""))
            ax2.set_ylabel("episodes")
            ax2.set_title(f"Episode duration by outcome  (mean {mean_all:.1f} steps, "
                          f"success {success_rate:.1%})")
            ax2.legend(loc="best", fontsize=8)
            dur_path = os.path.join(plots_dir, "eval_episode_duration.png")
            fig2.tight_layout()
            fig2.savefig(dur_path, dpi=150)
            print(f"[EVAL] Saved episode-duration histogram to: {dur_path}")
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
                critic_obs = torch.cat([obs[k] for k in critic_obs_keys], dim=1)
                print(critic(critic_obs, actions)[0])
            else:
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

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
