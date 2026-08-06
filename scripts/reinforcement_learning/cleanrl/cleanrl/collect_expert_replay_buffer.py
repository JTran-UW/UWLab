# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect an expert replay buffer into the CleanRL AsymmetricReplayBuffer.

Loads a holosoma/rsl_rl expert checkpoint the same way scripts/reinforcement_learning/holosoma/play.py
does (FastSACAgent for FastSAC runners, OnPolicyRunner for PPO), runs it on the task it was trained on,
and records transitions into cleanrl_utils.buffers.AsymmetricReplayBuffer. The buffer stores both the
policy (actor) and critic obs streams, so the output can warm-start a CleanRL off-policy agent.

The saved payload holds the raw buffer tensors plus metadata and can be reloaded into a fresh
AsymmetricReplayBuffer (see load_expert_replay_buffer at the bottom of this file for the round-trip).

Cross-task collection: the groups the expert *acts* from and the groups the buffer *stores* are
independent. Point ``--record_actor_obs_keys`` at another task's observation group and a state-based
expert will generate demonstrations carrying, say, camera observations -- the standard way to seed a
vision student from a state teacher. This needs a task exposing both sets of groups; the
DataCollection configs in rl_state_cfg.py exist for exactly that (e.g.
``Ur5eRobotiq2f85RlStateDataCollectionGrayscaleCfg`` keeps ``policy``/``critic`` state groups for the
expert and adds a ``grayscale`` group to record). Same flags as holosoma/play.py.

Example (state PPO expert, recording grayscale observations for a vision student):
    python scripts/reinforcement_learning/cleanrl/cleanrl/collect_expert_replay_buffer.py \
        --task <DataCollection-Grayscale-task-id> \
        --num_envs 64 --checkpoint <state_ppo_expert.pt> \
        --record_transitions 1000 --num_steps 3 \
        --record_actor_obs_keys grayscale \
        --output expert_rb/grayscale_from_state_expert.pt --headless --enable_cameras

Example (FastSAC state peg expert, same task it trained on):
    python scripts/reinforcement_learning/cleanrl/cleanrl/collect_expert_replay_buffer.py \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-Play-v0 \
        --num_envs 1024 --checkpoint peg_state_rl_expert_seed42.pt \
        --record_transitions 1000 --num_steps 3 \
        --output expert_rb/peg_state_expert_rb.pt --headless \
        env.scene.insertive_object=peg env.scene.receptive_object=peghole
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect an expert replay buffer (CleanRL AsymmetricReplayBuffer).")
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
parser.add_argument(
    "--record_transitions",
    type=int,
    default=1000,
    help="Number of environment steps to record (per env). Total transitions = num_envs * record_transitions.",
)
parser.add_argument(
    "--num_steps",
    "--record_n_steps",
    dest="num_steps",
    type=int,
    default=1,
    help=(
        "n-step return horizon to tag the buffer with. Transitions are stored as single steps in "
        "temporal order; AsymmetricReplayBuffer builds the n-step returns at sample time. Match this "
        "to the training agent's num_steps."
    ),
)
parser.add_argument(
    "--output",
    "--transitions_output",
    dest="output",
    type=str,
    default=None,
    help="Path to save the recorded replay buffer. Defaults to <log_dir>/expert_replay_buffer.pt.",
)
parser.add_argument(
    "--record_actor_obs_keys",
    type=str,
    nargs="+",
    default=None,
    help=(
        "Observation group(s) stored as the buffer's policy stream. Defaults to the groups the "
        "expert acts from. Set this to have one expert generate data for a different task -- e.g. a "
        "state-trained PPO expert acting on `policy` while the buffer records `grayscale`. Requires "
        "a task exposing both sets of groups (see the DataCollection configs in rl_state_cfg.py)."
    ),
)
parser.add_argument(
    "--record_critic_obs_keys",
    type=str,
    nargs="+",
    default=None,
    help="Observation group(s) stored as the buffer's critic stream. Defaults to the expert's own critic groups.",
)
parser.add_argument(
    "--record_proprio_obs_keys",
    type=str,
    nargs="+",
    default=None,
    help=(
        "Observation group(s) stored as the buffer's proprio stream. Off by default. Needed by the "
        "asymmetric vision tasks, whose actor consumes image AND proprioception -- these cannot "
        "share one stream because the recorder concatenates on the last dim and an image "
        "(N,3,3,84,84) will not concat with a vector (N,D)."
    ),
)
parser.add_argument(
    "--buffer_on_cpu",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Allocate the replay buffer in host RAM instead of on the GPU (default on). Collection only "
        "writes to the buffer and then saves it -- nothing is sampled -- so GPU residency buys "
        "nothing and an image buffer will not fit: 128 envs x 1000 steps of grayscale is ~31 GiB "
        "against 24 GiB of VRAM. Use --no-buffer_on_cpu to force it onto the compute device."
    ),
)
parser.add_argument(
    "--share_policy_critic_obs",
    action="store_true",
    default=False,
    help=(
        "Store ONE observation stream instead of separate policy/critic copies (~2x smaller). Only "
        "valid when both streams are bitwise identical -- i.e. the actor and critic record keys "
        "resolve to the same groups. The buffer verifies this on the first add() and raises if not."
    ),
)
parser.add_argument(
    "--store_next_obs",
    action="store_true",
    default=False,
    help=(
        "Materialize next_* observation streams. Off by default: they are recovered at sample time "
        "as observations[t+1], with terminal observations for truncated steps kept in a small side "
        "store, which is exact and ~2x smaller. Pass this flag to force the old dense layout."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import tqdm
from gymnasium import spaces
from tensordict import TensorDict

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cleanrl_utils.buffers import AsymmetricReplayBuffer

from rsl_rl.runners import OnPolicyRunner
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
# NOTE: HolosomaVecEnvWrapper is deliberately NOT imported here. This directory has its own
# vecenv_wrapper.py (the CleanRL IsaacLabVectorEnv), which shadows holosoma's on sys.path, so a
# top-level import fails with ImportError before anything runs -- even for PPO experts that never
# touch it. It is imported lazily in the FastSAC branch instead.

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def resolve_record_obs_keys(
    env, policy_actor_keys: list[str], policy_critic_keys: list[str]
) -> tuple[list[str], list[str]]:
    """Pick which observation groups get *recorded*, independent of which the policy *acts* from.

    Defaults to the policy's own groups. ``--record_actor_obs_keys`` / ``--record_critic_obs_keys``
    override them, which is what lets one expert generate data for a different task: the expert keeps
    consuming the groups it was trained on while the buffer stores the target task's groups. Requires
    a single env exposing both sets of groups.

    Mirrors ``holosoma/play.py::_resolve_record_obs_keys`` so both collectors take the same flags.
    """
    actor_keys = args_cli.record_actor_obs_keys or policy_actor_keys
    critic_keys = args_cli.record_critic_obs_keys or policy_critic_keys
    proprio_keys = list(args_cli.record_proprio_obs_keys or [])

    # Validate against manager metadata, not get_observations(): ObservationManager.compute()
    # appends to each term's history CircularBuffer, so an extra call here would push a duplicate
    # frame into history_length>1 groups and skew the first recorded transitions.
    obs_manager = getattr(env.unwrapped, "observation_manager", None)
    if obs_manager is not None:
        available = list(obs_manager.active_terms.keys())
        missing = [k for k in (*actor_keys, *critic_keys, *proprio_keys) if k not in available]
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
    if proprio_keys:
        print(f"[INFO] Recording proprio obs groups {proprio_keys}")
    return list(actor_keys), list(critic_keys), proprio_keys


def record_expert_replay_buffer(
    env,
    policy,
    actor_obs_keys: list[str],
    critic_obs_keys: list[str],
    proprio_obs_keys: list[str],
    record_transitions: int,
    output_path: str,
    task_name: str,
    checkpoint_path: str,
    device: str,
    gamma: float = 0.99,
    n_steps: int = 1,
) -> None:
    """Run ``policy`` for ``record_transitions`` steps and save transitions in AsymmetricReplayBuffer format.

    ``policy`` accepts the TensorDict returned by ``env.get_observations()`` and returns an actions tensor
    (mirrors holosoma/play.py's ``record_transitions_to_replay_buffer``). The buffer's ``policy`` /``critic``
    obs streams are the concatenations of ``actor_obs_keys`` / ``critic_obs_keys`` obs groups.

    N-step: single-step transitions are recorded in temporal order per env (with terminations/truncations),
    which is the layout AsymmetricReplayBuffer walks to build n-step returns at *sample* time. ``n_steps`` is
    recorded for provenance / to tag the buffer; it does not change the stored bytes.

    Caveat (shared with play.py): on a truncated/terminated step IsaacLab auto-resets and the raw-tensor step
    path returns the *post-reset* observation, so the terminal transition's next-obs is post-reset. The n-step
    sampler stops accumulation at the first done/truncation, so this only affects the bootstrap of terminal
    transitions -- a pre-existing limitation, not introduced here.
    """
    n_env = env.num_envs

    # Initial obs; discover the concatenated policy/critic obs dims at runtime.
    obs_td = env.get_observations().to(device)
    actor_obs = torch.cat([obs_td[g] for g in actor_obs_keys], dim=-1)
    critic_obs = torch.cat([obs_td[g] for g in critic_obs_keys], dim=-1)
    # Per-env element count, NOT shape[-1]: image groups keep their structure, so a grayscale group
    # arrives as (N, history, cams, H, W) and shape[-1] would be the image width. The buffer stores
    # everything flattened, so it must be sized by the full per-env numel.
    n_obs = actor_obs[0].numel()
    n_critic_obs = critic_obs[0].numel()
    has_proprio = bool(proprio_obs_keys)
    proprio_obs = torch.cat([obs_td[g] for g in proprio_obs_keys], dim=-1) if has_proprio else None
    n_proprio = proprio_obs[0].numel() if has_proprio else 0
    n_act = env.num_actions

    # Build synthetic obs/action spaces so the buffer's storage shapes match exactly what we store.
    obs_space = spaces.Dict(
        {
            "policy": spaces.Box(low=-float("inf"), high=float("inf"), shape=(n_obs,)),
            "critic": spaces.Box(low=-float("inf"), high=float("inf"), shape=(n_critic_obs,)),
        }
    )
    if has_proprio:
        # The buffer allocates its proprio stream from this key's presence.
        obs_space.spaces["proprio"] = spaces.Box(low=-float("inf"), high=float("inf"), shape=(n_proprio,))
    action_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(n_act,))

    # buffer_size is TOTAL and divided by n_envs internally, so pass record_transitions * n_env to get
    # exactly record_transitions rows per env.
    # Collection never samples, so the buffer lives wherever it is cheapest to hold; host RAM by
    # default, since an image buffer dwarfs VRAM. add() moves each step device->host.
    buffer_device = torch.device("cpu") if args_cli.buffer_on_cpu else device
    rb = AsymmetricReplayBuffer(
        record_transitions * n_env,
        obs_space,
        action_space,
        buffer_device,
        n_envs=n_env,
        n_steps=n_steps,
        gamma=gamma,
        share_policy_critic_obs=args_cli.share_policy_critic_obs,
        store_next_obs=args_cli.store_next_obs,
    )

    _elems = n_obs + n_critic_obs + n_proprio
    _streams = (1 if args_cli.share_policy_critic_obs else 2) * (2 if args_cli.store_next_obs else 1)
    print(
        f"[INFO] Replay buffer on {buffer_device} | {record_transitions * n_env:,} transitions x "
        f"~{_elems * 4 / 1024:.0f} KiB "
        f"(policy {n_obs} + critic {n_critic_obs} + proprio {n_proprio} elems, {_streams} obs stream(s))"
    )
    pbar = tqdm.tqdm(total=record_transitions, desc="Recording expert transitions")
    for _ in range(record_transitions):
        with torch.inference_mode():
            actions = policy(obs_td)

            next_obs_td, rewards, dones, extras = env.step(actions.to(env.device))
            next_obs_td = next_obs_td.to(device)
            rewards = rewards.to(device=device, dtype=torch.float)
            dones = dones.to(device=device, dtype=torch.bool)
            truncations = extras.get("time_outs", torch.zeros(n_env, dtype=torch.bool, device=device))
            truncations = truncations.to(device).bool()
            # rsl_rl reports dones = terminations | truncations; recover the pure terminal signal.
            terminations = dones & ~truncations

            next_actor_obs = torch.cat([next_obs_td[g] for g in actor_obs_keys], dim=-1)
            next_critic_obs = torch.cat([next_obs_td[g] for g in critic_obs_keys], dim=-1)
            if has_proprio:
                next_proprio_obs = torch.cat([next_obs_td[g] for g in proprio_obs_keys], dim=-1)

            obs_dict = {"policy": actor_obs, "critic": critic_obs}
            next_obs_dict = {"policy": next_actor_obs, "critic": next_critic_obs}
            if has_proprio:
                obs_dict["proprio"] = proprio_obs
                next_obs_dict["proprio"] = next_proprio_obs
            obs_for_rb = TensorDict(obs_dict, batch_size=(n_env,), device=device)
            next_obs_for_rb = TensorDict(next_obs_dict, batch_size=(n_env,), device=device)
            rb.add(obs_for_rb, next_obs_for_rb, actions.to(device), rewards, terminations, truncations, infos=[])

            obs_td = next_obs_td
            actor_obs = next_actor_obs
            critic_obs = next_critic_obs
            if has_proprio:
                proprio_obs = next_proprio_obs
        pbar.update(1)
    pbar.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Persist only the streams that exist. With share_policy_critic_obs the critic tensor IS the
    # policy tensor, and with store_next_obs=False the next_* streams are replaced by the truncation
    # side store -- writing them unconditionally would either duplicate data or crash on None.
    tensors = {
        "policy_observations": rb.policy_observations.detach().cpu(),
        "actions": rb.actions.detach().cpu(),
        "rewards": rb.rewards.detach().cpu(),
        "terminations": rb.terminations.detach().cpu(),
        "truncations": rb.truncations.detach().cpu(),
        "pos": rb.pos,
        "full": rb.full,
    }
    if not rb.share_policy_critic_obs:
        tensors["critic_observations"] = rb.critic_observations.detach().cpu()
    if rb.has_proprio:
        tensors["proprio_observations"] = rb.proprio_observations.detach().cpu()
        tensors["next_proprio_observations"] = rb.next_proprio_observations.detach().cpu()
    if rb.store_next_obs:
        tensors["next_policy_observations"] = rb.next_policy_observations.detach().cpu()
        if not rb.share_policy_critic_obs:
            tensors["next_critic_observations"] = rb.next_critic_observations.detach().cpu()
    else:
        tensors["trunc_obs_policy"] = rb.trunc_obs_policy.detach().cpu()
        if not rb.share_policy_critic_obs:
            tensors["trunc_obs_critic"] = rb.trunc_obs_critic.detach().cpu()
        tensors["trunc_slot"] = rb.trunc_slot.detach().cpu()
        tensors["trunc_owner"] = rb.trunc_owner.detach().cpu()
        tensors["trunc_ptr"] = rb.trunc_ptr
        tensors["trunc_live"] = rb.trunc_live
        # Capacity auto-grows during collection, so it must be recorded to rebuild the same layout.
        tensors["trunc_capacity"] = rb.trunc_capacity

    payload = {
        "buffer_tensors": tensors,
        "metadata": {
            "n_envs": n_env,
            "buffer_size": record_transitions,  # per-env
            "n_obs": n_obs,
            "n_critic_obs": n_critic_obs,
            "n_act": n_act,
            "n_steps": n_steps,
            "gamma": gamma,
            "actor_obs_keys": actor_obs_keys,
            "critic_obs_keys": critic_obs_keys,
            "proprio_obs_keys": proprio_obs_keys,
            "n_proprio": n_proprio,
            "share_policy_critic_obs": rb.share_policy_critic_obs,
            "store_next_obs": rb.store_next_obs,
            "task": task_name,
            "checkpoint": checkpoint_path,
            "total_transitions": n_env * record_transitions,
        },
    }
    torch.save(payload, output_path)
    n_img = (1 if rb.share_policy_critic_obs else 2) * (2 if rb.store_next_obs else 1)
    print(
        f"[INFO] Saved {n_env * record_transitions} expert transitions to: {output_path}\n"
        f"[INFO]   layout: {n_img} obs stream(s) "
        f"[share_policy_critic_obs={rb.share_policy_critic_obs}, store_next_obs={rb.store_next_obs}]"
        + ("" if rb.store_next_obs else f", truncation side store {rb.trunc_capacity} slots")
    )


def load_expert_replay_buffer(path: str, device: str = "cpu", sample_device: str | None = None) -> AsymmetricReplayBuffer:
    """Reload a saved expert buffer into a fresh AsymmetricReplayBuffer (round-trips record_expert_replay_buffer)."""
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = payload["metadata"]
    tensors = payload["buffer_tensors"]
    obs_space = spaces.Dict(
        {
            "policy": spaces.Box(low=-float("inf"), high=float("inf"), shape=(meta["n_obs"],)),
            "critic": spaces.Box(low=-float("inf"), high=float("inf"), shape=(meta["n_critic_obs"],)),
        }
    )
    if meta.get("n_proprio"):
        obs_space.spaces["proprio"] = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(meta["n_proprio"],)
        )
    action_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(meta["n_act"],))
    # Older payloads predate these keys; default to the dense layout they were written with.
    share = meta.get("share_policy_critic_obs", False)
    store_next = meta.get("store_next_obs", True)
    rb = AsymmetricReplayBuffer(
        meta["buffer_size"] * meta["n_envs"],
        obs_space,
        action_space,
        device,
        n_envs=meta["n_envs"],
        n_steps=meta["n_steps"],
        gamma=meta["gamma"],
        sample_device=sample_device,
        share_policy_critic_obs=share,
        store_next_obs=store_next,
    )
    # Sharing aliases critic onto policy, so restoring the policy tensor restores both.
    names = ["policy_observations", "actions", "rewards", "terminations", "truncations"]
    if not share:
        names.append("critic_observations")
    if meta.get("n_proprio"):
        names += ["proprio_observations", "next_proprio_observations"]
    if store_next:
        names.append("next_policy_observations")
        if not share:
            names.append("next_critic_observations")
    for name in names:
        getattr(rb, name).copy_(tensors[name].to(device))

    if not store_next:
        # Rebuild the truncation side store at the capacity it grew to during collection, otherwise
        # saved slot indices would point past the end of a default-sized store.
        cap = int(tensors["trunc_capacity"])
        if cap != rb.trunc_capacity:
            rb._grow_truncation_store(cap)
        rb.trunc_obs_policy[: cap].copy_(tensors["trunc_obs_policy"].to(device))
        if not share:
            rb.trunc_obs_critic[: cap].copy_(tensors["trunc_obs_critic"].to(device))
        rb.trunc_slot.copy_(tensors["trunc_slot"].to(device))
        rb.trunc_owner[: cap].copy_(tensors["trunc_owner"].to(device))
        rb.trunc_ptr = int(tensors["trunc_ptr"])
        rb.trunc_live = int(tensors["trunc_live"])

    rb.pos = tensors["pos"]
    rb.full = tensors["full"]
    # A shared buffer verifies policy==critic on its first add(); a restored one never adds.
    rb._shared_obs_verified = True
    return rb


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load an expert checkpoint and record an AsymmetricReplayBuffer."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve checkpoint path (same precedence as play.py)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # determine runner type and wrap env accordingly (same logic as play.py)
    is_fastsac = agent_cfg.class_name == "OnPolicyRunner" and hasattr(agent_cfg, "actor_obs_keys")
    if is_fastsac:
        # Imported here, and by path, because this directory's own vecenv_wrapper.py shadows
        # holosoma's for a plain `import vecenv_wrapper`.
        import importlib.util

        _holo = pathlib.Path(__file__).resolve().parents[2] / "holosoma" / "vecenv_wrapper.py"
        _spec = importlib.util.spec_from_file_location("holosoma_vecenv_wrapper", _holo)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        env = _mod.HolosomaVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if is_fastsac:
        runner = FastSACAgent(env, agent_cfg, log_dir=None, device=agent_cfg.device)
        runner.setup()
        runner.load(resume_path)
        policy_fn = runner.get_inference_policy(device=env.unwrapped.device)
        # Groups the EXPERT consumes. Bound to their own names: the record keys resolved below may
        # differ, and a closure over a later-reassigned name would silently feed the policy the
        # wrong observations.
        policy_actor_keys = list(agent_cfg.actor_obs_keys)
        policy_critic_keys = list(agent_cfg.critic_obs_keys)

        # FastSAC policy expects {"actor_obs": tensor}
        def policy(obs_td, _keys=policy_actor_keys):
            a_obs = torch.cat([obs_td[k] for k in _keys], dim=-1)
            return policy_fn({"actor_obs": a_obs})

    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        # rsl_rl's inference policy takes the whole obs TensorDict and selects its own groups
        # internally via runner.cfg["obs_groups"], so it is unaffected by the record keys.
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs_groups = runner.cfg["obs_groups"]
        policy_actor_keys = list(obs_groups["policy"])
        policy_critic_keys = list(obs_groups.get("critic", policy_actor_keys))
    else:
        raise ValueError(f"Unsupported runner class for expert collection: {agent_cfg.class_name}")

    # What gets STORED -- defaults to the expert's own groups, overridable to a different task's.
    actor_obs_keys, critic_obs_keys, proprio_obs_keys = resolve_record_obs_keys(
        env, policy_actor_keys, policy_critic_keys
    )

    gamma = getattr(agent_cfg.algorithm, "gamma", 0.99) if hasattr(agent_cfg, "algorithm") else 0.99
    output_path = args_cli.output or os.path.join(log_dir, "expert_replay_buffer.pt")

    record_expert_replay_buffer(
        env=env,
        policy=policy,
        actor_obs_keys=actor_obs_keys,
        critic_obs_keys=critic_obs_keys,
        proprio_obs_keys=proprio_obs_keys,
        record_transitions=args_cli.record_transitions,
        output_path=output_path,
        task_name=args_cli.task,
        checkpoint_path=resume_path,
        device=agent_cfg.device,
        gamma=gamma,
        n_steps=args_cli.num_steps,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
