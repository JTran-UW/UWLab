# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline trajectory collection + cross-seed action relabeling for an RSL-RL checkpoint.

This is a sibling to ``play.py`` that keeps the (bulky) state/action collection infra out of the
minimal play loop. It builds the env + runner exactly like ``play.py`` but adds two opt-in modes:

* ``--dump_trajectories OUT.npz`` -- roll a policy out and capture, per step, the insertive-object
  point cloud in the robot base frame plus the full observation dict and actions (a side channel that
  never touches the policy's obs groups). Pair with ``--force_sequential_reset`` so that, under a fixed
  ``--seed``, reset-state j lands in env j identically across configs and the dumps are aligned across
  policies.

* ``--relabel_actions OUT.npz --relabel_state_files <dumps...> --relabel_ckpts seed=path ...`` -- no
  rollout. Pool the first-episode observations from the dumps into ONE shared state set, then for each
  checkpoint reload that policy in full (weights + its own obs normalizer) and forward-pass the shared
  pool. The aligned ``[K_policies, M_states, A]`` action tensor answers "what would every seed's policy
  do at the states this configuration's seeds actually visit?" -- action variance across policies is
  then unbiased toward any single seed's state distribution. Consumed by
  ``analysis/action_multimodality.py``.

Examples:
    # collect paired trajectories for one (config, seed)
    python scripts/reinforcement_learning/rsl_rl/collect_actions.py \
        --task <VideoEval-task> --num_envs 512 --checkpoint <path.pt> \
        --force_sequential_reset --dump_trajectories out/s0.npz

    # relabel a config's pooled states with every seed's policy
    python scripts/reinforcement_learning/rsl_rl/collect_actions.py \
        --task <VideoEval-task> --num_envs 2 \
        --relabel_actions out/config.npz \
        --relabel_state_files out/s0.npz out/s4.npz \
        --relabel_ckpts 0=out/s0.pt 4=out/s4.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect trajectories / relabel actions for an RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record a rollout video (collection mode).")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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
parser.add_argument("--num_steps", type=int, default=None, help="Maximum number of policy steps (collection mode).")
parser.add_argument(
    "--force_sequential_reset",
    action="store_true",
    default=False,
    help="Force every MultiResetManager event term into deterministic sequential reset sampling "
    "(and disable its reset GPS curriculum). With a fixed --seed this makes the initial reset assign "
    "reset-state j to env j identically across configs, so trajectories are paired across policies.",
)
parser.add_argument(
    "--dump_trajectories",
    type=str,
    default=None,
    help="If set, capture per-step insertive-object point cloud (robot base frame) + the full obs dict "
    "+ actions and save them to this .npz path (side channel; does not alter the policy's observations).",
)
parser.add_argument(
    "--traj_num_points",
    type=int,
    default=128,
    help="Number of points sampled on the insertive-object mesh for the dumped point cloud.",
)
parser.add_argument(
    "--relabel_actions",
    type=str,
    default=None,
    help="Action-multimodality relabel mode. If set, do NOT roll out: pool the first-episode "
    "observations from the --relabel_state_files dumps and, for each checkpoint in --relabel_ckpts, "
    "forward-pass that policy over the SHARED pool to produce its action labels. Saves the aligned "
    "[K_policies, M_states, A] action tensor (+ provenance) to this .npz path and exits.",
)
parser.add_argument(
    "--relabel_state_files",
    type=str,
    nargs="+",
    default=None,
    help="Trajectory .npz dumps (from --dump_trajectories) whose first-episode obs are pooled into the "
    "shared state set that every policy is relabeled on.",
)
parser.add_argument(
    "--relabel_ckpts",
    type=str,
    nargs="+",
    default=None,
    help="Policies to relabel with, as seed=path tokens (e.g. 0=/path/s0.pt 4=/path/s4.pt). Each is "
    "loaded in full (weights + obs normalizer) and evaluated on the shared pool.",
)
parser.add_argument(
    "--relabel_max_per_seed",
    type=int,
    default=4000,
    help="Cap on first-episode states pooled per source dump (deterministic subsample).",
)
parser.add_argument(
    "--relabel_seed",
    type=int,
    default=42,
    help="RNG seed for the deterministic per-source subsample of pooled states.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# Inject UWLab distillation classes into rsl_rl's distillation_runner module so
# the runner's eval(class_name) lookup resolves them at checkpoint load time.
import rsl_rl.runners.distillation_runner as _distillation_runner_module

from uwlab_rl.rsl_rl.distillation_dagger import DistillationDAgger
from uwlab_rl.rsl_rl.distillation_dagger_weighted import DistillationDAggerWeighted
from uwlab_rl.rsl_rl.distillation_runner_split import DistillationRunnerSplit
from uwlab_rl.rsl_rl.student_teacher_mlp import StudentTeacherMLP
from uwlab_rl.rsl_rl.student_teacher_pointcloud import StudentTeacherPointCloud
from uwlab_rl.rsl_rl.student_teacher_vision import StudentTeacherVision
from uwlab_rl.rsl_rl.student_teacher_vision_recurrent import StudentTeacherVisionRecurrent

_distillation_runner_module.StudentTeacherVision = StudentTeacherVision
_distillation_runner_module.StudentTeacherVisionRecurrent = StudentTeacherVisionRecurrent
_distillation_runner_module.StudentTeacherMLP = StudentTeacherMLP
_distillation_runner_module.StudentTeacherPointCloud = StudentTeacherPointCloud
_distillation_runner_module.DistillationDAgger = DistillationDAgger
_distillation_runner_module.DistillationDAggerWeighted = DistillationDAggerWeighted
_distillation_runner_module.DistillationRunnerSplit = DistillationRunnerSplit

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

import rsl_rl.runners.on_policy_runner as _runner_module
from uwlab_rl.rsl_rl.actor_critic_encoder import ActorCriticWithEncoder
_runner_module.ActorCriticWithEncoder = ActorCriticWithEncoder
from uwlab_rl.rsl_rl.actor_critic_depth import ActorCriticDepth
from uwlab_rl.rsl_rl.bc_ppo import BCPPO
from uwlab_rl.rsl_rl.grpo import GRPO
from uwlab_rl.rsl_rl.ppo_pbrs import PPOPBRS
_runner_module.ActorCriticDepth = ActorCriticDepth
_runner_module.BCPPO = BCPPO
_runner_module.GRPO = GRPO
_runner_module.PPOPBRS = PPOPBRS
from uwlab_rl.rsl_rl.actor_critic_rma import ActorCriticRMA
from uwlab_rl.rsl_rl.ppo_rma import PPO_RMA
_runner_module.ActorCriticRMA = ActorCriticRMA
_runner_module.PPO_RMA = PPO_RMA
import rsl_rl.algorithms as _rsl_rl_algorithms
_rsl_rl_algorithms.PPO_RMA = PPO_RMA

from isaaclab_tasks.utils import get_checkpoint_path
from uwlab_tasks.utils.hydra import hydra_task_config


def _is_jit_checkpoint(path: str) -> bool:
    """Return True if *path* is a TorchScript (JIT) file, False if it is a runner checkpoint."""
    try:
        torch.jit.load(path, map_location="cpu")
        return True
    except RuntimeError:
        return False


def _build_runner(env, agent_cfg, resume_path):
    """Construct the runner for the agent's class and load *resume_path* (weights + normalizer)."""
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunnerSplit":
        runner = DistillationRunnerSplit(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "BCPPORunner":
        from uwlab_rl.rsl_rl.bc_ppo_runner import BCPPORunner
        runner = BCPPORunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "GRPOGroupedRunner":
        from uwlab_rl.rsl_rl.grpo_grouped_runner import GRPOGroupedRunner
        runner = GRPOGroupedRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerRMA":
        from uwlab_rl.rsl_rl.on_policy_runner_rma import OnPolicyRunnerRMA
        runner = OnPolicyRunnerRMA(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    try:
        runner.load(resume_path)
    except ValueError as _e:
        if "optimizer" not in str(_e).lower():
            raise
        # Optimizer param-group mismatch: checkpoint saved with a different policy architecture.
        # For inference we only need model weights, so skip the optimizer.
        print(f"[WARN] Optimizer state mismatch — loading model weights only: {_e}")
        _ckpt = torch.load(resume_path, map_location=agent_cfg.device)
        runner.alg.policy.load_state_dict(_ckpt["model_state_dict"])
    except RuntimeError as _e:
        if "size mismatch" not in str(_e):
            raise
        # Critic obs size mismatch: drop critic keys entirely and load actor only.
        print(f"[WARN] State dict size mismatch — dropping critic keys and loading actor only: {_e}")
        _ckpt = torch.load(resume_path, map_location=agent_cfg.device)
        _sd = {k: v for k, v in _ckpt["model_state_dict"].items() if not k.startswith("critic")}
        runner.alg.policy.load_state_dict(_sd, strict=False)
    return runner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect trajectories or relabel actions with an RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # make config compatible with installed rsl-rl version
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    # set the environment seed (certain randomizations occur at env init)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Optionally force deterministic sequential reset sampling on every MultiResetManager
    # event term (the State term is `reset_from_reset_states`; the ZeroG term is
    # `reset_from_states`). Sequential mode walks the reset dataset in order, so the initial
    # synchronized reset assigns reset-state j to env j -- identical across configs under a
    # fixed --seed. We also null out the reset GPS curriculum so the single reset type is
    # walked uniformly in order rather than success-rate-weighted.
    if args_cli.force_sequential_reset and hasattr(env_cfg, "events"):
        n_patched = 0
        for _name, _term in vars(env_cfg.events).items():
            _func = getattr(_term, "func", None)
            if _func is None or getattr(_func, "__name__", "") != "MultiResetManager":
                continue
            _term.params["sampling_mode"] = "sequential"
            _term.params["curriculum_target"] = None
            _term.params["use_classifier"] = False
            _term.params["use_success_critic"] = False
            n_patched += 1
            print(f"[force_sequential_reset] events.{_name} -> sampling_mode=sequential (curriculum disabled)")
        if n_patched == 0:
            print("[force_sequential_reset] WARNING: no MultiResetManager event term found on this env.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "collect"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording a rollout video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    is_jit = _is_jit_checkpoint(resume_path)

    if is_jit:
        print("[INFO]: Detected TorchScript checkpoint — loading with torch.jit.load (skipping runner).")
        policy_nn = torch.jit.load(resume_path, map_location=agent_cfg.device)
        policy_nn.eval()
        policy = policy_nn
        runner = None
    else:
        runner = _build_runner(env, agent_cfg, resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

    # ------------------------------------------------------------------
    # Action-multimodality relabel mode (no rollout): pool the first-episode
    # observations from the --relabel_state_files dumps into ONE shared state
    # set, then for each checkpoint in --relabel_ckpts reload that policy in
    # full (weights + its own obs normalizer) and forward-pass the shared pool.
    # The K aligned action-sets answer "what would every seed's policy do at the
    # states this configuration's seeds actually visit?" -- the action variance
    # across policies is then unbiased toward any single seed's state distribution.
    # ------------------------------------------------------------------
    if args_cli.relabel_actions is not None:
        import json as _json

        import numpy as _np
        from tensordict import TensorDict

        assert not is_jit, "relabel mode needs a runner checkpoint (got a JIT policy)"
        assert args_cli.relabel_state_files and args_cli.relabel_ckpts, (
            "--relabel_actions requires --relabel_state_files and --relabel_ckpts"
        )
        device = env.unwrapped.device
        rng = _np.random.default_rng(args_cli.relabel_seed)

        # --- pool first-episode states from each source dump (deterministic) ---
        pool_groups: dict = {}
        ref_action, src_seed, src_env, src_step = [], [], [], []
        for f in sorted(args_cli.relabel_state_files):
            d = _np.load(f, allow_pickle=True)
            # policy seed is the dump filename (s<seed>.npz); meta["seed"] is the fixed reset seed.
            seed = int(os.path.splitext(os.path.basename(f))[0][1:])
            groups = [k[4:] for k in d.files if k.startswith("obs_")]
            fd = d["first_done_step"]  # [N]
            T, N = d["actions"].shape[0], d["actions"].shape[1]
            # The dump is off-by-one: stored_obs[t] is the POST-step obs o_{t+1}, while
            # stored_action[t] = policy(o_t). The on-policy action AT stored_obs[t] is therefore
            # stored_action[t+1]. So we pool obs at step t and pair it with action step t+1
            # (valid t = 0..last-1, where last = first_done; this keeps states within the first episode).
            ie, it = [], []
            for n in range(N):
                last = int(fd[n]) if fd[n] >= 0 else T - 1
                if last < 1:
                    continue
                ie.extend([n] * last)
                it.extend(range(last))  # obs steps 0..last-1 -> states o_1..o_last
            ie, it = _np.asarray(ie), _np.asarray(it)
            if args_cli.relabel_max_per_seed and len(ie) > args_cli.relabel_max_per_seed:
                sel = _np.sort(rng.choice(len(ie), args_cli.relabel_max_per_seed, replace=False))
                ie, it = ie[sel], it[sel]
            for g in groups:
                pool_groups.setdefault(g, []).append(d[f"obs_{g}"][it, ie].astype(_np.float32))
            ref_action.append(d["actions"][it + 1, ie].astype(_np.float32))  # on-policy action at this obs
            src_seed.append(_np.full(len(ie), seed, dtype=_np.int64))
            src_env.append(ie.astype(_np.int64))
            src_step.append(it.astype(_np.int64))
            print(f"[relabel] pooled {len(ie)} states from s{seed} ({os.path.basename(f)})")

        pool = {g: _np.concatenate(v, 0) for g, v in pool_groups.items()}
        ref_action = _np.concatenate(ref_action, 0)
        src_seed = _np.concatenate(src_seed)
        src_env = _np.concatenate(src_env)
        src_step = _np.concatenate(src_step)
        M, A = ref_action.shape
        print(f"[relabel] shared pool: {M} states, groups {sorted(pool.keys())}")

        # --- relabel the shared pool with every policy ---
        ckpts = [(int(t.split("=", 1)[0]), t.split("=", 1)[1]) for t in args_cli.relabel_ckpts]
        actions_by_policy = _np.zeros((len(ckpts), M, A), dtype=_np.float32)
        B = 8192
        for ki, (s, p) in enumerate(ckpts):
            runner.load(p)  # restores model weights AND this seed's obs normalizer
            pol = runner.get_inference_policy(device=device)
            for start in range(0, M, B):
                end = min(M, start + B)
                td = TensorDict(
                    {g: torch.from_numpy(pool[g][start:end]).to(device) for g in pool},
                    batch_size=[end - start],
                )
                with torch.inference_mode():
                    actions_by_policy[ki, start:end] = pol(td).float().cpu().numpy()
            # self-consistency: on its OWN states a policy must reproduce the stored actions
            own = src_seed == s
            if own.any():
                err = float(_np.abs(actions_by_policy[ki, own] - ref_action[own]).max())
                print(f"[relabel] policy s{s}: relabeled {M} states  (self|Δ| max={err:.2e} on own {int(own.sum())})")
            else:
                print(f"[relabel] policy s{s}: relabeled {M} states")

        out = args_cli.relabel_actions
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        meta = {
            "task": args_cli.task,
            "policy_seeds": [s for s, _ in ckpts],
            "n_states": int(M),
            "action_dim": int(A),
            "max_per_seed": int(args_cli.relabel_max_per_seed),
            "relabel_seed": int(args_cli.relabel_seed),
            "obs_groups": sorted(pool.keys()),
        }
        _np.savez_compressed(
            out,
            actions_by_policy=actions_by_policy,  # [K, M, A]
            policy_seeds=_np.array([s for s, _ in ckpts], dtype=_np.int64),
            ref_action=ref_action,  # [M, A] stored on-policy action per pooled state
            src_seed=src_seed,  # [M] which seed's rollout the state came from
            src_env=src_env,  # [M] env index within that dump
            src_step=src_step,  # [M] timestep within that dump
            meta_json=_np.array(_json.dumps(meta)),
        )
        print(f"[relabel] saved actions_by_policy {actions_by_policy.shape} -> {out}")
        # Isaac's headless simulation_app.close() can hang on this early-return path (no rollout was
        # run). The npz is already on disk, so flush and hard-exit instead of risking a stuck process.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    dt = env.unwrapped.step_dt

    # ------------------------------------------------------------------
    # Trajectory dumping (side channel): per-step insertive-object point
    # cloud in the robot base frame, plus the full obs dict and actions.
    # Mirrors mdp.observations.MeshPointCloud (canonical points sampled once
    # in the object's local frame, transformed by its world pose each step)
    # but lives outside the obs manager so it never touches the policy's obs
    # groups (the ZeroG shared-encoder policies expect an exact group set).
    # ------------------------------------------------------------------
    traj = None
    if args_cli.dump_trajectories is not None:
        import isaaclab.utils.math as _math_utils

        from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as _omni_utils

        if args_cli.num_steps is None:
            args_cli.num_steps = int(env.unwrapped.max_episode_length)
            print(f"[dump_trajectories] --num_steps defaulted to one horizon = {args_cli.num_steps} steps")

        _base_env = env.unwrapped
        _N = _base_env.num_envs
        _P = args_cli.traj_num_points
        _insertive = _base_env.scene["insertive_object"]
        _robot = _base_env.scene["robot"]
        _prim_pat = _insertive.cfg.prim_path.replace("{ENV_REGEX_NS}", ".*")
        _canonical = _omni_utils.sample_object_point_cloud(
            num_envs=_N, num_points=_P, prim_path_pattern=_prim_pat, device=str(_base_env.device)
        )  # [N, P, 3] in object-local frame

        def _insertive_pc_base():
            """Insertive-object point cloud transformed into the robot base frame: [N, P, 3]."""
            obj_pos_w = _insertive.data.root_pos_w
            obj_quat_w = _insertive.data.root_quat_w
            pts_w = (
                _math_utils.quat_apply(obj_quat_w.unsqueeze(1).expand(-1, _P, -1), _canonical)
                + obj_pos_w.unsqueeze(1)
            )
            ref_pos_w = _robot.data.root_pos_w
            ref_quat_w = _robot.data.root_quat_w
            ref_quat_inv = _math_utils.quat_inv(ref_quat_w)
            pts_ref = _math_utils.quat_apply(
                ref_quat_inv.unsqueeze(1).expand(-1, _P, -1), pts_w - ref_pos_w.unsqueeze(1)
            )
            return pts_ref

        traj = {"insertive_pc": [], "actions": [], "rewards": [], "dones": [], "obs": {}}
        _ever_done = torch.zeros(_N, dtype=torch.bool, device=_base_env.device)
        _first_done = torch.full((_N,), -1, dtype=torch.long, device=_base_env.device)
        _succ_first = torch.zeros(_N, dtype=torch.bool, device=_base_env.device)

    # reset environment
    obs = env.get_observations()
    # capture the post-reset insertive-object PC (policy-independent initial state;
    # bit-identical across configs that share the object USD when sequential reset aligns them)
    if traj is not None:
        traj["reset_insertive_pc"] = _insertive_pc_base().half().cpu().numpy()  # [N, P, 3]
    timestep = 0
    num_episodes = 0
    num_successes = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # JIT policy takes a flat tensor directly; runner inference policy takes the obs dict.
            if is_jit:
                mean, std = policy(obs)
                actions = mean
            else:
                actions = policy(obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            # trajectory capture (side channel; CPU/float16 to keep memory bounded)
            if traj is not None:
                traj["insertive_pc"].append(_insertive_pc_base().half().cpu().numpy())
                traj["actions"].append(actions.float().cpu().numpy())
                traj["rewards"].append(rewards.float().cpu().numpy())
                traj["dones"].append(dones.bool().cpu().numpy())
                if hasattr(obs, "items"):
                    for _g, _v in obs.items():
                        traj["obs"].setdefault(_g, []).append(_v.half().cpu().numpy())
                else:
                    traj["obs"].setdefault("obs", []).append(obs.half().cpu().numpy())
                _newly = dones.bool() & (~_ever_done)
                _first_done[_newly] = timestep
                _succ_first[_newly] = (rewards > 0.1)[_newly]
                _ever_done |= dones.bool()
            if dones.any():
                num_episodes += dones.sum().item()
                num_successes += torch.logical_and(rewards > 0.1, dones).sum().item()
            # reset recurrent states for episodes that have terminated.
            if not is_jit:
                policy_nn.reset(dones)

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if args_cli.num_steps is not None and timestep >= args_cli.num_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0 and args_cli.video:
            time.sleep(sleep_time)

    print("=" * 50)
    print(f"Number of episodes: {num_episodes}")
    print(f"Number of successes: {num_successes}")
    if num_episodes:
        print(f"Success rate: {num_successes / num_episodes:.2%}")

    # save dumped trajectories (insertive-object PC + full obs + actions)
    if traj is not None:
        import json as _json

        import numpy as _np

        save = {
            "insertive_pc": _np.stack(traj["insertive_pc"], 0),  # [T, N, P, 3] float16
            "actions": _np.stack(traj["actions"], 0),  # [T, N, A] float32
            "rewards": _np.stack(traj["rewards"], 0),  # [T, N]
            "dones": _np.stack(traj["dones"], 0),  # [T, N] bool
            "first_done_step": _first_done.cpu().numpy(),  # [N] (-1 if never done)
            "success": _succ_first.cpu().numpy(),  # [N] success at first done
        }
        if "reset_insertive_pc" in traj:
            save["reset_insertive_pc"] = traj["reset_insertive_pc"]  # [N, P, 3] float16
        for _g, _lst in traj["obs"].items():
            save[f"obs_{_g}"] = _np.stack(_lst, 0)  # [T, N, D_g] float16
        meta = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "num_envs": int(_N),
            "num_points": int(_P),
            "num_steps": int(args_cli.num_steps or 0),
            "seed": args_cli.seed,
            "reset_type": "ObjectAnywhereEEAnywhere",
            "sampling_mode": "sequential" if args_cli.force_sequential_reset else "default",
            "ref_frame": "robot_base",
            "obs_groups": sorted(traj["obs"].keys()),
        }
        save["meta_json"] = _np.array(_json.dumps(meta))
        out_path = args_cli.dump_trajectories
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        _np.savez_compressed(out_path, **save)
        n_done = int((_first_done >= 0).sum().item())
        print(
            f"[dump_trajectories] saved insertive_pc {save['insertive_pc'].shape} + "
            f"{len(traj['obs'])} obs groups + actions {save['actions'].shape} -> {out_path}  "
            f"({n_done}/{_N} envs finished a first episode, "
            f"success={float(_succ_first.float().mean()):.1%})"
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
