"""Thin sequential orchestrator for BAMDP-ASTEROID smoke runs.

Loops one iteration of:
  1. collect demos with `scripts_v2/tools/collect_demos_asteroid.py` (in patlab env)
  2. train a diffusion student via `diffusion_policy/train.py` (from UWLab-ICL,
     pip install -e'd into the patlab env in this session)
  3. eval the trained checkpoint via `scripts_v2/tools/eval_distilled_policy_bamdp.py`

No GPU pipelining, no long-lived data worker — every step boots Isaac Sim
fresh (~60s overhead) and runs to completion. The hard "production" version
would mirror UWLab-ICL's `run_incontext_exploration_parallel.py`: long-lived
worker on the data GPU, training subprocess on a separate GPU, eval pipelined
with the next iteration's collection. That can land later once the loop is
known-good.

Defaults are smoke-sized (32 envs, 40 demos, 2k train steps, 32 eval episodes).
Iteration 0 is rescue-only; iteration ≥ 1 uses the previous iter's checkpoint
as the exploration policy (so the env-side BAMDP injector fires forced
failures during the student-driven prefix of each episode).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


ROOT = "/mnt/storage/lti/UWLab-patrick-private"
UWLAB_ICL = "/mnt/storage/lti/UWLab-ICL"

DIFFUSION_TRAIN = os.path.join(UWLAB_ICL, "diffusion_policy/train.py")
DIFFUSION_CONFIG_DIR = os.path.join(UWLAB_ICL, "diffusion_policy/diffusion_policy/config")
COLLECT_SCRIPT = os.path.join(ROOT, "scripts_v2/tools/collect_demos_asteroid.py")
EVAL_SCRIPT = os.path.join(ROOT, "scripts_v2/tools/eval_distilled_policy_bamdp.py")


def _run(cmd: list[str], cwd: str | None = None, env_extra: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if env_extra:
        env.update(env_extra)
    print(f"\n[orchestrator] $ {' '.join(cmd)}\n  (cwd={cwd or os.getcwd()})", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=cwd or os.getcwd(), env=env).returncode
    dt = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"step failed (rc={rc}, {dt:.1f}s): {' '.join(cmd)}")
    print(f"[orchestrator] step done in {dt:.1f}s", flush=True)


def _collect(iteration: int, output_dir: str, args) -> str:
    """Run a collection job; returns the path to the written zarr."""
    dataset_dir = os.path.join(output_dir, f"dataset-iteration-{iteration}")
    dataset_zarr = os.path.join(dataset_dir, "data.zarr")
    exploration_ckpt = args.exploration_checkpoint if iteration > 0 else None
    cmd = [
        sys.executable,
        COLLECT_SCRIPT,
        "--num_envs", str(args.num_envs),
        "--num_demos", str(args.num_demos),
        "--episode_length_s", str(args.episode_length_s),
        "--dataset_file", dataset_zarr,
        "--seed", str(args.seed),
        "--n_avg", str(args.n_avg),
        "--temperature", str(args.discriminator_temperature),
        "--insertive_object", args.insertive_object,
        "--receptive_object", args.receptive_object,
    ]
    if iteration == 0:
        # Iter-0 ASTEROID convention: BAMDP off + uniform multi-expert demos.
        # No synthetic failures, no rescue takeover — just clean expert
        # demonstrations spread across the K experts. Trains the BC student to
        # represent the multimodal expert distribution.
        cmd += ["--bamdp_disabled", "--multi_expert"]
    if iteration > 0 and exploration_ckpt:
        cmd += ["--exploration_checkpoint", exploration_ckpt,
                "--max_exploration_horizon", str(args.max_exploration_horizon)]
    _run(cmd, cwd=ROOT)
    return dataset_zarr


def _train(iteration: int, dataset_zarr: str, output_dir: str, args) -> str:
    """Run training; returns the path to the resulting checkpoint dir."""
    iter_dir = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(os.path.join(iter_dir, "wandb"), exist_ok=True)  # train.py writes wandb runs here
    dataset_dir = os.path.dirname(dataset_zarr)  # train.py wants the parent dir
    dataset_str = f"task.dataset.dataset_config=[{{dataset_dir: {dataset_dir}, sampling_ratio: 1.0}}]"
    # The training workspace runs `for _ in range(num_epochs):` and only
    # increments global_step on non-last batches. So total gradient steps are
    # roughly bounded by num_epochs * (batches_per_epoch - 1). For our small
    # smoke datasets we need to crank num_epochs up so global_step has a chance
    # to climb past `checkpoint_every` and trigger a save. We also cap
    # checkpoint_every so we get at least one save before train_steps exhausts.
    ckpt_every = max(10, min(args.train_steps // 2, 1000))
    # Heuristic: bias num_epochs high enough that max_gradient_steps is what
    # actually terminates training, not num_epochs running out.
    num_epochs = max(args.train_steps, 100)
    cmd = [
        sys.executable,
        DIFFUSION_TRAIN,
        "--config-name", args.config_name,
        "--config-dir", DIFFUSION_CONFIG_DIR,
        f"task={args.task_config}",
        f"output_dir={iter_dir}",
        dataset_str,
        f"name={args.exp_name}",
        f"exp_name={args.exp_name}",
        f"logging.project={args.wandb_project}",
        f"logging.group={args.exp_name}",
        f"optimizer.lr={args.lr}",
        f"seed={args.seed}",
        f"iteration={iteration}",
        f"training.max_gradient_steps={args.train_steps}",
        f"training.checkpoint_every={ckpt_every}",
        f"training.num_epochs={num_epochs}",
    ]
    _run(cmd, cwd=ROOT)
    # The diffusion-policy trainer writes checkpoints to <iter_dir>/checkpoints/.
    return os.path.join(iter_dir, "checkpoints")


def _eval(iteration: int, ckpt_dir: str, output_dir: str, args) -> str:
    """Eval the latest checkpoint in ckpt_dir; returns the stats JSON path."""
    ckpts = sorted(
        [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    )
    if not ckpts:
        raise RuntimeError(f"no checkpoints found under {ckpt_dir}")
    # Prefer latest.ckpt if it exists, else highest step_*.ckpt.
    pref = [c for c in ckpts if c.endswith("/latest.ckpt")]
    ckpt = pref[0] if pref else ckpts[-1]

    stats_path = os.path.join(output_dir, f"iteration_{iteration}", "eval_stats.json")
    cmd = [
        sys.executable,
        EVAL_SCRIPT,
        "--task", args.eval_task,
        "--checkpoint", ckpt,
        "--num_envs", str(args.num_eval_envs),
        "--num_trajectories", str(args.num_eval_episodes),
        "--headless",
        "--stats_output_path", stats_path,
        f"env.scene.insertive_object={args.insertive_object}",
        f"env.scene.receptive_object={args.receptive_object}",
        f"env.episode_length_s={args.episode_length_s}",
    ]
    _run(cmd, cwd=ROOT)
    return stats_path


def main() -> None:
    parser = argparse.ArgumentParser("BAMDP-ASTEROID sequential smoke orchestrator.")
    parser.add_argument("--output_dir", type=str, default="logs/bamdp_asteroid",
                        help="Per-run output dir under patlab repo root.")
    parser.add_argument("--exp_name", type=str, default="bamdp_asteroid_smoke")
    parser.add_argument("--wandb_project", type=str, default="bamdp_asteroid")
    parser.add_argument("--max_iterations", type=int, default=1)

    # Collection knobs.
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--num_demos", type=int, default=40)
    parser.add_argument("--episode_length_s", type=float, default=12.0)
    parser.add_argument("--n_avg", type=float, default=25.0)
    parser.add_argument("--discriminator_temperature", type=float, default=0.3)
    parser.add_argument("--max_exploration_horizon", type=float, default=0.5,
                        help="iter > 0 only — fraction of episode driven by the exploration policy.")
    parser.add_argument("--exploration_checkpoint", type=str, default=None,
                        help="Override the exploration policy for iter > 0 (defaults to the previous iter's ckpt).")

    # Training knobs.
    parser.add_argument("--config_name", type=str, default="in_context_adaptation_interleave.yaml",
                        help="Hydra config name under diffusion_policy/config/.")
    parser.add_argument("--task_config", type=str, default="bamdp_scene_pc",
                        help="Hydra task config name (under diffusion_policy/config/task/).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    # Eval knobs.
    parser.add_argument(
        "--eval_task", type=str,
        default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-ZeroG-ScenePC-BAMDPFailures-StudentEval-v0",
    )
    parser.add_argument("--num_eval_envs", type=int, default=32)
    parser.add_argument("--num_eval_episodes", type=int, default=64)

    # Task variants.
    parser.add_argument("--insertive_object", type=str, default="peg")
    parser.add_argument("--receptive_object", type=str, default="peghole")

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[orchestrator] output_dir = {output_dir}", flush=True)

    exploration_ckpt = args.exploration_checkpoint

    for iteration in range(args.max_iterations):
        print(f"\n========================================================")
        print(f"  BAMDP-ASTEROID iteration {iteration}")
        print(f"========================================================")

        if iteration > 0:
            args.exploration_checkpoint = exploration_ckpt

        zarr_path = _collect(iteration, output_dir, args)
        print(f"[orchestrator] iter {iteration} dataset → {zarr_path}", flush=True)

        ckpt_dir = _train(iteration, zarr_path, output_dir, args)
        print(f"[orchestrator] iter {iteration} checkpoints → {ckpt_dir}", flush=True)

        stats = _eval(iteration, ckpt_dir, output_dir, args)
        print(f"[orchestrator] iter {iteration} eval stats → {stats}", flush=True)

        # Use this iter's latest checkpoint as the next iter's exploration policy.
        latest = os.path.join(ckpt_dir, "latest.ckpt")
        if os.path.exists(latest):
            exploration_ckpt = latest
            print(f"[orchestrator] next iter exploration ckpt: {exploration_ckpt}", flush=True)

    print(f"\n[orchestrator] DONE.", flush=True)


if __name__ == "__main__":
    main()
