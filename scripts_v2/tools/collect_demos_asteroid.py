"""BAMDP-aware ASTEROID demo collection (port of UWLab-ICL/scripts_v2/tools/collect_demos_asteroid.py).

Differences from the UWLab-ICL version:

- **No single global expert.** The K experts are owned by the env-side
  :class:`BAMDPLatentSampler`; the per-env rescue expert is selected at reset
  from the BAMDP latent θ. This script asks the env's bamdp state for the
  rescue action via :func:`task_mdp.compute_rescue_action`.

- **Supervision label = rescue action.** The recorded action is what the
  rescue expert would have done at every step — counterfactual labels. The
  env's patched ``process_action`` writes the current step's rescue action
  into ``obs_buf["data_collection"]["expert_action_mean"]`` so the recorder
  picks it up on the same step (no one-step lag).

- **Forced-failure injection is env-side** (no script-side wrapper). When the
  student drives during exploration horizon, the env may stall+rescue
  transparently. The script doesn't need to know about it.

- **Inverse-action and discretization removed.** The two experts here share
  the env's action scale, and discretization is a separable concern handled
  by the trainer rather than collection.
"""

# NOTE: Pre-import numpy and numba BEFORE isaaclab / AppLauncher (Kit pip_prebundle
# mutates sys.path and would otherwise shadow our conda env's pinned numba 0.64).
import numpy  # noqa: F401
import numba  # noqa: F401

import argparse
import contextlib
import os
from types import MethodType
from typing import Sequence

import gymnasium as gym
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect BAMDP ASTEROID demonstrations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-BAMDPFailures-Train-v0",
    help="BAMDP-failures task id.",
)
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.zarr",
                    help="Output dataset path (.zarr).")
parser.add_argument("--num_demos", type=int, default=10, help="Number of successful demos to record.")
parser.add_argument(
    "--exploration_checkpoint",
    type=str,
    default=None,
    help=(
        "Optional diffusion-policy exploration checkpoint. If unset, the rescue expert "
        "drives every step (iter-0 collection)."
    ),
)
parser.add_argument(
    "--min_exploration_horizon",
    type=float,
    default=0.0,
    help="Min fraction of episode driven by the exploration policy (per env). Default 0.0.",
)
parser.add_argument(
    "--max_exploration_horizon",
    type=float,
    default=0.0,
    help=(
        "Max fraction of episode driven by the exploration policy. "
        "0.0 means: rescue expert drives every step (iter-0). For iter-N>0, "
        "pass e.g. 0.5 to mix in student exploration."
    ),
)
parser.add_argument("--episode_length_s", type=float, default=16.0, help="Per-episode time horizon.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--disable_exploration_ratio_filter",
    action="store_true",
    default=False,
    help="If set, do NOT reject demos where the learner drove >95%% of the episode.",
)
parser.add_argument(
    "--disable_task_success_filter",
    action="store_true",
    default=False,
    help=(
        "If set, admit every completed episode regardless of the `success` termination. "
        "Requires the exploration-ratio filter to stay ON, otherwise nothing is gated."
    ),
)
parser.add_argument(
    "--transformer_mini_batch_size", type=int, default=64,
    help="Mini-batch size for the diffusion-policy exploration wrapper.",
)
parser.add_argument("--no_kv_cache", action="store_true", default=False)
parser.add_argument("--kv_cache_max_seq_len", type=int, default=None)
parser.add_argument(
    "--insertive_object", type=str, default="peg",
    help="Hydra variant for insertive object. Spliced into hydra_args.",
)
parser.add_argument(
    "--receptive_object", type=str, default="peghole",
    help="Hydra variant for receptive object. Spliced into hydra_args.",
)
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--n_avg", type=float, default=25.0,
                    help="Mean episode length for the BAMDP hazard budget. Overrides env cfg default.")
parser.add_argument("--temperature", type=float, default=0.3,
                    help="Discriminator softmax temperature. T<1 sharpens. Default 0.3.")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="Argmax discriminator output (T → 0 limit). Overrides --temperature.")
parser.add_argument(
    "--bamdp_disabled", action="store_true", default=False,
    help=(
        "Disable the env-side BAMDP layer entirely (no discriminator, no hazard mixing, "
        "no stall, no rescue takeover). Pair with --multi_expert for iter-0 multi-modal "
        "demo collection. Default False (BAMDP active)."
    ),
)
parser.add_argument(
    "--multi_expert", action="store_true", default=False,
    help=(
        "Iter-0 mode: drive each env with a random expert from the ensemble. Expert "
        "assignment is resampled at every env reset. The recorded action label is "
        "the assigned expert's action. Only makes sense with --bamdp_disabled."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Splice the object variants into hydra so they pass through the registered task's variant lookup.
hydra_args = list(hydra_args) + [
    f"env.scene.insertive_object={args_cli.insertive_object}",
    f"env.scene.receptive_object={args_cli.receptive_object}",
]
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# isaaclab imports come AFTER AppLauncher up.
import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers.recorder_manager import DatasetExportMode  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

from uwlab.utils.datasets import ZarrDatasetFileHandler  # noqa: E402

import uwlab_tasks  # noqa: F401, E402
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.bamdp_failures import compute_rescue_action  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders_cfg import (  # noqa: E402
    ActionStateRecorderManagerTransformedActionCfg,
)
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


# ---------------------------------------------------------------------------
# Recorder gating: only admit demos that pass success + exploration-ratio filters.
# ---------------------------------------------------------------------------


def record_pre_reset(self, env_ids: Sequence[int] | None, force_export_or_skip=None) -> None:
    """Patched recorder_manager.record_pre_reset that gates by success + exploration ratio."""
    if len(self.active_terms) == 0:
        return

    if env_ids is None:
        env_ids = list(range(self._env.num_envs))
    if isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.tolist()

    for term in self._terms.values():
        key, value = term.record_pre_reset(env_ids)
        self.add_to_episodes(key, value, env_ids)

    device = self._env.device
    n = len(env_ids)

    task_success = torch.zeros(n, dtype=bool, device=device)
    if hasattr(self._env, "termination_manager") and "success" in self._env.termination_manager.active_terms:
        task_success |= self._env.termination_manager.get_term("success")[env_ids]

    episode_lengths = self._env.episode_length_buf[env_ids]
    ratio_pass = torch.ones(n, dtype=bool, device=device)
    if hasattr(self, "exploration_lengths"):
        exploration_lengths = self.exploration_lengths[env_ids]
        exploration_ratios = exploration_lengths / torch.clamp(episode_lengths, min=1)
        ratio_pass = exploration_ratios < 0.95

    task_gate = (
        torch.ones(n, dtype=bool, device=device)
        if getattr(self, "disable_task_success_filter", False)
        else task_success
    )
    ratio_gate = (
        ratio_pass
        if getattr(self, "apply_exploration_ratio_filter", False)
        else torch.ones(n, dtype=bool, device=device)
    )
    success_results = task_gate & ratio_gate

    self.set_success_to_episodes(env_ids, success_results)
    if force_export_or_skip or (force_export_or_skip is None and self.cfg.export_in_record_pre_reset):
        self.export_episodes(env_ids)


# ---------------------------------------------------------------------------
# Exploration policy loader (shared with UWLab-ICL's diffusion-policy wrapper).
# ---------------------------------------------------------------------------


def load_exploration_policy(
    checkpoint_path: str,
    device: torch.device,
    num_envs: int,
    mini_batch_size: int = 64,
    use_kv_cache: bool = True,
    kv_cache_max_seq_len: int | None = None,
):
    """Defer import until we know we need it (diffusion_policy + dill are heavy)."""
    import dill
    import hydra
    from diffusion_policy.policy.base_image_policy import BaseImagePolicy
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    from uwlab_rl.wrappers.diffusion import DiffusionPolicyWrapper

    with open(checkpoint_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy: BaseImagePolicy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy = policy.eval().to(device)
    return DiffusionPolicyWrapper(
        policy,
        device,
        n_obs_steps=policy.n_obs_steps,
        num_envs=num_envs,
        mini_batch_size=mini_batch_size,
        use_kv_cache=use_kv_cache,
        kv_cache_max_seq_len=kv_cache_max_seq_len,
        profile_name="exploration",
    )


def sample_exploration_horizons(num_envs: int, min_h: int, max_h: int, device: torch.device) -> torch.Tensor:
    if max_h <= 0:
        return torch.zeros((num_envs,), device=device, dtype=torch.int32)
    min_h = min(max(min_h, 0), max_h)
    return torch.randint(min_h, max_h + 1, (num_envs,), device=device)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)
    os.makedirs(output_dir, exist_ok=True)

    # Recorder cfg.
    env_cfg.recorders = ActionStateRecorderManagerTransformedActionCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = ZarrDatasetFileHandler

    # Env knobs.
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.episode_length_s
    # Note: BAMDP env's ObsCfg exposes flat top-level groups (proprio, pointcloud, ...),
    # not a nested `policy` group, so we don't touch policy.concatenate_terms here.

    # BAMDP knobs forwarded into the latent sampler.
    env_cfg.events.bamdp_latent_sampler.params["n_avg"] = float(args_cli.n_avg)
    env_cfg.events.bamdp_latent_sampler.params["discriminator_temperature"] = float(args_cli.temperature)
    env_cfg.events.bamdp_latent_sampler.params["discriminator_argmax"] = bool(args_cli.argmax)
    env_cfg.events.bamdp_latent_sampler.params["bamdp_disabled"] = bool(args_cli.bamdp_disabled)

    print(f"[collect] building env {args_cli.task!r} with {env_cfg.scene.num_envs} envs...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env)
    bamdp = env.unwrapped.bamdp
    num_envs = env.num_envs
    action_dim = env.action_space.shape[-1]
    device = torch.device(env_cfg.sim.device if isinstance(env_cfg.sim.device, str) else "cuda:0")
    print(f"[collect] env up: num_envs={num_envs} action_dim={action_dim} K={bamdp.K}", flush=True)

    # Recorder gating setup (matches UWLab-ICL collect_demos_asteroid).
    recorder_manager = env.unwrapped.recorder_manager
    expert_mask_recorder = recorder_manager._terms.get("record_pre_step_expert_mask")
    if expert_mask_recorder is None:
        raise RuntimeError("record_pre_step_expert_mask recorder term is not configured.")
    recorder_manager.record_pre_reset = MethodType(record_pre_reset, recorder_manager)
    recorder_manager.apply_exploration_ratio_filter = not args_cli.disable_exploration_ratio_filter
    recorder_manager.disable_task_success_filter = args_cli.disable_task_success_filter
    if args_cli.disable_task_success_filter:
        assert not args_cli.disable_exploration_ratio_filter, (
            "--disable_task_success_filter requires the exploration-ratio filter to stay ON."
        )
        print("[collect] task-success filter DISABLED; only exploration-ratio gate applies.", flush=True)
    if args_cli.disable_exploration_ratio_filter:
        print("[collect] exploration-ratio<0.95 filter DISABLED.", flush=True)

    # Episode-length / horizon bookkeeping (in env steps).
    step_dt = env.unwrapped.step_dt
    episode_length_steps = int(env_cfg.episode_length_s / step_dt)
    max_exploration_horizon_steps = int(args_cli.max_exploration_horizon * episode_length_steps)
    min_exploration_horizon_steps = int(args_cli.min_exploration_horizon * episode_length_steps)
    print(
        f"[collect] step_dt={step_dt:.4f}s episode_length_steps={episode_length_steps}"
        f"  exploration_horizon_steps=[{min_exploration_horizon_steps}, {max_exploration_horizon_steps}]",
        flush=True,
    )

    # Exploration policy (optional).
    exploration_policy = None
    if args_cli.exploration_checkpoint:
        exploration_policy = load_exploration_policy(
            args_cli.exploration_checkpoint,
            device,
            num_envs,
            mini_batch_size=args_cli.transformer_mini_batch_size,
            use_kv_cache=not args_cli.no_kv_cache,
            kv_cache_max_seq_len=args_cli.kv_cache_max_seq_len,
        )
        exploration_policy.reset(torch.arange(num_envs, device=device))
        print(f"[collect] exploration policy loaded: {args_cli.exploration_checkpoint}", flush=True)
    elif args_cli.multi_expert:
        print("[collect] no exploration policy — multi_expert mode drives each env with a random expert (iter-0).", flush=True)
    else:
        print("[collect] no exploration policy — rescue expert drives every step (iter-0 single-expert).", flush=True)

    exploration_horizons = sample_exploration_horizons(
        num_envs, min_exploration_horizon_steps, max_exploration_horizon_steps, device
    )
    exploration_lengths = torch.zeros((num_envs,), device=device, dtype=torch.int32)
    recorder_manager.exploration_lengths = exploration_lengths

    # Multi-expert iter-0 mode: pick a random expert per env at the start, and
    # resample on every reset so each new episode gets an independent expert
    # draw. Demos are then a near-uniform mix across the K experts — the
    # student's BC objective becomes "represent the multimodal expert dist".
    multi_expert = bool(args_cli.multi_expert)
    K = bamdp.K
    if multi_expert:
        assigned_expert = torch.randint(0, K, (num_envs,), device=device)
        if not args_cli.bamdp_disabled:
            print(
                "[collect] WARNING: --multi_expert without --bamdp_disabled means triggers"
                " can fire on episodes driven by the non-rescue expert. Demos will be"
                " contaminated by stall + rescue takeovers.",
                flush=True,
            )
        print(f"[collect] multi_expert ENABLED — initial assignment: {assigned_expert.cpu().tolist()[:8]}…", flush=True)
    else:
        assigned_expert = None

    current_recorded_demo_count = 0
    num_episodes = 0
    num_successes = 0
    num_trigger_episodes = 0

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        pbar = tqdm(total=args_cli.num_demos, desc="BAMDP demos (success: 0.00%)", unit="demo")

        while True:
            # ── (1) Decide who drives each env this step. ──
            episode_steps = env.unwrapped.episode_length_buf
            use_exploration = (episode_steps < exploration_horizons) & (exploration_policy is not None)
            use_expert = ~use_exploration
            exploration_lengths += use_exploration.int()
            recorder_manager.exploration_lengths = exploration_lengths

            obs_buf = env.unwrapped.obs_buf
            proprio = obs_buf["proprio"].to(device)
            pc = obs_buf["pointcloud"].to(device)

            actions = torch.zeros((num_envs, action_dim), device=device)
            if multi_expert:
                # Iter-0 multi-modal demo collection: each env runs its assigned expert.
                with torch.no_grad():
                    all_actions = torch.stack(
                        [exp(proprio, pc) for exp in bamdp.experts], dim=1
                    )  # (B, K, A)
                row_idx = torch.arange(num_envs, device=device)
                expert_action_for_env = all_actions[row_idx, assigned_expert]
                # In multi_expert mode the "expert" label is the assigned expert
                # rather than the BAMDP latent's rescue idx. Drive every env
                # with this action regardless of use_exploration (iter-0 has no
                # exploration_horizon).
                actions[:] = expert_action_for_env
            else:
                # ── (2) Compute rescue expert action for ALL envs (label + driver for use_expert). ──
                rescue_action = compute_rescue_action(bamdp, proprio, pc)
                if use_expert.any():
                    actions[use_expert] = rescue_action[use_expert]
                if use_exploration.any() and exploration_policy is not None:
                    exploration_env_ids = use_exploration.nonzero(as_tuple=False).reshape(-1)
                    policy_obs = obs_buf.get("policy", obs_buf) if isinstance(obs_buf, dict) else obs_buf
                    if not isinstance(policy_obs, dict):
                        policy_obs = {"_": policy_obs}
                    exploration_obs = {k: v[use_exploration] for k, v in policy_obs.items()}
                    exploration_actions = exploration_policy.predict_action(exploration_obs, exploration_env_ids)
                    actions[use_exploration] = exploration_actions.to(device)

            # NOTE: UWLab-ICL's collect_demos_asteroid.py masks the first step to
            # (zero arm, gripper closed) because its EEGrasped reset types start
            # with the gripper already closed on the object. The gravity env's
            # reset types (ZeroGAnywhere / ZeroGPartialAssembly) do NOT start
            # with the gripper closed — forcing close on step 1 grabs empty air
            # and immediately puts the rescue out of distribution. So we let the
            # rescue drive from step 0.

            # ── (3) Stash who-drove-this-step for the recorder. ──
            expert_mask_recorder.set_mask(use_expert.unsqueeze(-1))

            # ── (4) Step the env. The env-side BAMDP layer handles failure injection. ──
            _obs, rewards, dones, _infos = env.step(actions)

            if dones.any():
                done_long = dones.bool()
                num_episodes += int(done_long.sum().item())
                # Count successes via the termination manager.
                tm = env.unwrapped.termination_manager
                if "success" in tm.active_terms:
                    num_successes += int(tm.get_term("success")[done_long].sum().item())
                # Track how many done envs had at least one trigger this episode.
                # (BAMDPLatentSampler resets `failure_count` per-env on reset, so we
                # look at the failure_count before reset via dones mask.)
                if hasattr(bamdp, "failure_count"):
                    num_trigger_episodes += int((bamdp.failure_count[done_long] > 0).sum().item())

                # Resample exploration horizons + clear exploration_lengths for reset envs.
                reset_ids = done_long.nonzero(as_tuple=False).reshape(-1)
                exploration_horizons[reset_ids] = sample_exploration_horizons(
                    len(reset_ids), min_exploration_horizon_steps, max_exploration_horizon_steps, device
                )
                exploration_lengths[reset_ids] = 0
                if exploration_policy is not None:
                    exploration_policy.reset(reset_ids)
                # Multi-expert: resample assignment for newly-reset envs so each
                # new episode is an independent expert draw.
                if multi_expert:
                    assigned_expert[reset_ids] = torch.randint(0, K, (len(reset_ids),), device=device)

            # Progress bar update.
            new_count = recorder_manager.exported_successful_episode_count
            if new_count > current_recorded_demo_count:
                inc = new_count - current_recorded_demo_count
                current_recorded_demo_count = new_count
                pbar.update(inc)
                rate = (num_successes / num_episodes * 100) if num_episodes > 0 else 0.0
                trig_rate = (num_trigger_episodes / num_episodes * 100) if num_episodes > 0 else 0.0
                pbar.set_description(
                    f"BAMDP demos (success: {rate:.1f}%, triggered: {trig_rate:.1f}%)"
                )

            if args_cli.num_demos > 0 and new_count >= args_cli.num_demos:
                print(f"\n[collect] {args_cli.num_demos} demos recorded — exiting.", flush=True)
                break

            if env.unwrapped.sim.is_stopped():
                break

            if args_cli.render:
                env.render()

        pbar.close()

    print(f"[collect] episodes={num_episodes}  successes={num_successes}  trigger_episodes={num_trigger_episodes}",
          flush=True)
    print(f"[collect] dataset: {args_cli.dataset_file}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()  # type: ignore[misc]
    finally:
        simulation_app.close()
