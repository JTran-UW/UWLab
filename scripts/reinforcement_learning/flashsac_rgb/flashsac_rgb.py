# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FlashSAC-RGB on the OmniReset vision task: standalone port of ManiSkill-opt-all flashsac_rgb.py.

Defaults are the PegInsertionSide recipe (n-step 3, gamma 0.97, tau 0.01, UTD 1/64, image embed 256, encoder EMA 0.05,
RLPD expert ratio 0.2, bf16 + compile). Differences from the ManiSkill script: Isaac Lab env (via UWLabVisionVecEnv,
which owns a fixed horizon and resets all envs together), evaluation runs one deterministic episode on the training
envs (no separate eval env in one Isaac Lab process), and policy actions in [-1, 1] are scaled by --action_scale before env.step.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FlashSAC-RGB (UWLab port).")
parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Vision-v0")
parser.add_argument("--exp_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--total_timesteps", type=int, default=15_000_000)
parser.add_argument("--horizon", type=int, default=160)
parser.add_argument("--action_scale", type=str, default="10.0", help="Policy actions in [-1, 1] are scaled by this before env.step; "
                    "a comma-separated list gives one scale per action dim.")
parser.add_argument("--no_tanh", action="store_true", help="Unsquashed Gaussian actor (actions not bounded to [-1, 1]); pair with an expert buffer collected with --no_clip.")
parser.add_argument("--log_std_min", type=float, default=-10.0)
parser.add_argument("--log_std_max", type=float, default=2.0)
parser.add_argument("--state_group", type=str, default="vision_state", help="Low-dim state group (vision_state = non-privileged 25-D, policy = privileged 43-D, asym = both: critic sees all, actor the first --actor_state_dim; privstore = both stored, symmetric networks see the first --model_state_dim).")
parser.add_argument("--actor_state_dim", type=int, default=25, help="With --state_group asym: leading state dims the actor sees.")
parser.add_argument("--model_state_dim", type=int, default=None, help="Leading stored-state dims the encoders see (default all); "
                    "with --state_group privstore, 25 hides the privileged slice from the networks.")
parser.add_argument("--bc_expert", type=str, default=None, help="RSL-RL state expert checkpoint; adds --bc_coef * ||mean action - expert "
                    "action||^2 to the actor loss, the expert queried on the stored privileged slice (needs --state_group privstore).")
parser.add_argument("--bc_coef", type=float, default=0.0)
parser.add_argument("--eval_freq", type=int, default=1_000_000, help="Env steps between deterministic eval episodes (0 = off).")
parser.add_argument("--save_freq", type=int, default=1_000_000)
parser.add_argument("--log_freq", type=int, default=16_384)
parser.add_argument("--target_success", type=float, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Load actor/encoder/critics from a flashsac_rgb checkpoint.")
parser.add_argument("--resume", action="store_true", help="With --checkpoint: also restore alpha, SGS state and the step count "
                    "(LR schedule continues); optimizers and replay buffer start fresh.")
parser.add_argument("--warmup_episodes", type=int, default=0, help="After --resume: collect this many policy episodes before updating.")
parser.add_argument("--track", action="store_true", help="Log to wandb.")
parser.add_argument("--wandb_project", type=str, default="uwlab_vision_rl")
parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
# algorithm (names follow the ManiSkill script)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--buffer_size", type=int, default=1_024_000, help="Rounded down to a multiple of horizon * num_envs.")
parser.add_argument("--gamma", type=float, default=0.97)
parser.add_argument("--tau", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--learning_starts", type=int, default=10_240)
parser.add_argument("--policy_lr", type=float, default=3e-4)
parser.add_argument("--q_lr", type=float, default=3e-4)
parser.add_argument("--alpha_lr", type=float, default=3e-4)
parser.add_argument("--policy_frequency", type=int, default=2)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--no_autotune", action="store_true")
parser.add_argument("--training_freq", type=int, default=None, help="Env steps between update bursts (default num_envs).")
parser.add_argument("--utd", type=float, default=0.015625)
parser.add_argument("--n_step_returns", type=int, default=3)
parser.add_argument("--v_min", type=float, default=-2.0, help="OmniReset per-step reward is ~[-0.05, 0.12]; discounted values stay within ~[-2, 6].")
parser.add_argument("--v_max", type=float, default=6.0)
parser.add_argument("--reward_scale", type=float, default=1.0, help="Multiplies sampled (n-step) rewards; widen --v_min/--v_max to match.")
parser.add_argument("--target_entropy_sigma", type=float, default=0.15)
parser.add_argument("--noise_repeat_zeta_mu", type=float, default=2.0)
parser.add_argument("--noise_repeat_zeta_max", type=int, default=16)
parser.add_argument("--lr_end_ratio", type=float, default=0.5)
parser.add_argument("--image_embed_dim", type=int, default=256)
parser.add_argument("--state_embed_dim", type=int, default=64)
parser.add_argument("--encoder_ema", type=float, default=0.05)
parser.add_argument("--freeze_encoder", action="store_true")
parser.add_argument("--expert_buffer", type=str, default=None)
parser.add_argument("--expert_ratio", type=float, default=0.2)
parser.add_argument("--aug_pad", type=int, default=0)
parser.add_argument("--blank_images", action="store_true", help="Debug: zero all images (online and expert).")
parser.add_argument("--sgs", action="store_true", help="Success Guided Sampling over all reset states (arXiv 2610.12465).")
parser.add_argument("--sgs_target", type=float, default=0.5)
parser.add_argument("--sgs_kappa", type=float, default=1.0)
parser.add_argument("--sgs_temperature", type=float, default=2.0)
parser.add_argument("--sgs_eps", type=float, default=1e-4)
parser.add_argument("--sgs_history", type=int, default=100)
parser.add_argument("--sgs_empty_rate", type=float, default=0.5, help="Success estimate of a state with no outcomes yet.")
parser.add_argument("--no_bf16", action="store_true")
parser.add_argument("--no_compile", action="store_true")
parser.add_argument("--no_channels_last", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, remaining_args = parser.parse_known_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import time
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
from buffers import ImageReplayBuffer, cat_image_samples  # noqa: E402
from networks.flashsac_actor import FlashSACActor  # noqa: E402
from networks.flashsac_critic import FlashSACQNetwork  # noqa: E402
from networks.flashsac_utils import FlashSACEncoder  # noqa: E402
from push_t_vision_env import PushTVisionVecEnv, extend_episode_length  # noqa: E402
from vision_env import IMAGE_KEYS, UWLabVisionVecEnv  # noqa: E402


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ random shift on (N, k, H, W, C) images (copied from ManiSkill-opt-all utils/augmentations.py)."""
    if pad <= 0:
        return x
    N, k, H, W, C = x.shape
    shift = torch.randint(-pad, pad + 1, (N, 2), device=x.device)
    rows = (torch.arange(H, device=x.device) + shift[:, :1]).clamp_(0, H - 1)
    cols = (torch.arange(W, device=x.device) + shift[:, 1:]).clamp_(0, W - 1)
    n = torch.arange(N, device=x.device)[:, None, None]
    return x[n, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2, 4)


def _build_truncated_zeta_cdf(mu: float, max_n: int, device) -> torch.Tensor:
    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    return torch.cumsum(pmf, dim=0)


def _sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)


def bf16_forward(module):
    """Run module.forward under bf16 autocast and return fp32, without changing the module or its state_dict."""
    fwd = module.forward

    def wrapped(*a, **kw):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return fwd(*a, **kw).float()

    module.forward = wrapped


def _sample_rollout_action(x_embed, actor, noise, cur_count, cur_n, zeta_cdf, training: bool):
    mean, log_std = actor(x_embed, training=training)
    std = log_std.exp()
    reinit = (cur_count == 0) | (cur_count >= cur_n)
    new_noise = torch.randn_like(mean)
    new_n = _sample_integer_from_cdf(zeta_cdf)
    noise = torch.where(reinit, new_noise, noise)
    cur_n = torch.where(reinit, new_n, cur_n)
    cur_count = torch.where(reinit, torch.zeros_like(cur_count), cur_count)
    x_t = mean + std * noise
    if actor.use_tanh:
        action = torch.tanh(x_t) * actor.action_scale + actor.action_bias
    else:
        action = x_t
    return action, noise, cur_count + 1, cur_n


class Logger:
    def __init__(self, writer: SummaryWriter, log_wandb: bool):
        self.writer, self.log_wandb = writer, log_wandb
        self.counters = {"per_env_step": 0, "train_episodes": 0}

    def add_scalar(self, tag, value, step):
        value = float(value)
        if self.log_wandb:
            wandb.log({tag: value, **self.counters}, step=step)
        self.writer.add_scalar(tag, value, step)


@hydra_task_compose(args.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    global wandb
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    push_t = args.task.startswith("PushT-")
    if push_t:
        extend_episode_length(env_cfg, args.horizon)
    if args.sgs:
        env_cfg.events.reset_from_reset_states.params["sgs"] = dict(
            history=args.sgs_history, target=args.sgs_target, kappa=args.sgs_kappa,
            temperature=args.sgs_temperature, eps=args.sgs_eps, empty_rate=args.sgs_empty_rate,
        )
    args.action_scale = [float(v) for v in args.action_scale.split(",")] if "," in args.action_scale else float(args.action_scale)
    gym_env = gym.make(args.task, cfg=env_cfg)
    bc_policy = None
    if args.bc_expert is not None:
        assert args.state_group == "privstore", "--bc_expert needs the privileged slice in the replay buffer (--state_group privstore)"
        from isaaclab.utils.assets import retrieve_file_path
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner

        import cli_args

        rl_cfg = cli_args.sanitize_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"))
        bc_runner = OnPolicyRunner(RslRlVecEnvWrapper(gym_env), rl_cfg.to_dict(), log_dir=None, device=rl_cfg.device)
        bc_runner.load(retrieve_file_path(args.bc_expert))
        bc_policy = bc_runner.get_inference_policy(device=gym_env.unwrapped.device)
        print(f"[bc] expert {args.bc_expert}, coef {args.bc_coef}", flush=True)
    env_cls = PushTVisionVecEnv if push_t else UWLabVisionVecEnv
    envs = env_cls(gym_env, args.horizon, args.action_scale, args.state_group)
    envs.blank_images = args.blank_images
    device = envs.device
    full_state_dim = envs.single_observation_space["state"].shape[-1]
    model_state_dim = args.model_state_dim or full_state_dim
    net_space = type("NetObs", (), {"single_observation_space": gym.spaces.Dict({
        "rgb": envs.single_observation_space["rgb"],
        "state": gym.spaces.Box(-np.inf, np.inf, (1, model_state_dim), np.float32)})})()
    priv_slice = slice(args.actor_state_dim, full_state_dim)

    def enc_in(o):
        return o if model_state_dim == full_state_dim else {"rgb": o["rgb"], "state": o["state"][..., :model_state_dim]}
    bf16, compile_nets, autotune = not args.no_bf16, not args.no_compile, not args.no_autotune
    channels_last = bf16 and not args.no_channels_last
    training_freq = args.training_freq or args.num_envs
    grad_steps_per_iteration = int(training_freq * args.utd)
    steps_per_env = training_freq // args.num_envs
    embed_dim = args.image_embed_dim + args.state_embed_dim
    run_name = args.exp_name or f"flashsac_rgb_s{args.seed}_{int(time.time())}"
    log_dir = os.path.join("logs", "flashsac_rgb", run_name)
    os.makedirs(log_dir, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}

    writer = SummaryWriter(log_dir)
    if args.track:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=config, group="FlashSAC-RGB")
        wandb.define_metric("per_env_step")
        wandb.define_metric("*", step_metric="per_env_step")
    logger = Logger(writer, args.track)

    act_shape = envs.single_action_space.shape
    actor = FlashSACActor(
        envs, input_dim=embed_dim, action_scale=torch.ones(act_shape),
        action_bias=torch.zeros(act_shape), use_tanh=not args.no_tanh, log_std_min=args.log_std_min, log_std_max=args.log_std_max,
    ).to(device)
    act_scale_t = torch.as_tensor(args.action_scale, dtype=torch.float32, device=device)
    zeta_cdf = _build_truncated_zeta_cdf(args.noise_repeat_zeta_mu, args.noise_repeat_zeta_max, device)
    noise_repeat_n = torch.ones((), dtype=torch.int32, device=device)
    noise_repeat_count = torch.zeros((), dtype=torch.int32, device=device)
    cached_noise = torch.randn((args.num_envs,) + act_shape, device=device)
    qf1, qf2, qf1_target, qf2_target = (
        FlashSACQNetwork(envs, input_dim=embed_dim, v_min=args.v_min, v_max=args.v_max, device=device).to(device)
        for _ in range(4)
    )
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    encoder = FlashSACEncoder(net_space, image_embed_dim=args.image_embed_dim, state_embed_dim=args.state_embed_dim, channels_last=channels_last).to(device)
    target_encoder = FlashSACEncoder(net_space, image_embed_dim=args.image_embed_dim, state_embed_dim=args.state_embed_dim, channels_last=channels_last).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    if bf16:
        torch.backends.cudnn.benchmark = True
        if channels_last:
            for m in (encoder, target_encoder):
                m.to(memory_format=torch.channels_last)
        for m in (encoder, target_encoder, qf1, qf2, qf1_target, qf2_target):
            bf16_forward(m)
    if args.freeze_encoder:
        encoder.requires_grad_(False)
        target_encoder.requires_grad_(False)
    encoder_params = [] if args.freeze_encoder else list(encoder.parameters())
    q_optimizer = optim.Adam(encoder_params + list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, fused=True)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, fused=True)
    ema_pairs = [(list(qf1_target.parameters()) + list(qf2_target.parameters()), list(qf1.parameters()) + list(qf2.parameters()), args.tau)]
    if not args.freeze_encoder:
        ema_pairs.insert(0, (list(target_encoder.parameters()), list(encoder.parameters()), args.encoder_ema))

    asym = args.state_group == "asym"
    actor_encoder = None
    if asym:
        # asymmetric actor-critic: the critic (privileged state) can ignore the images, so its encoder learns no
        # visual features; the actor gets its own image + non-privileged-state encoder, trained by the actor loss
        rgb_space = envs.single_observation_space["rgb"]
        actor_obs_space = gym.spaces.Dict({"rgb": rgb_space, "state": gym.spaces.Box(-np.inf, np.inf, (1, args.actor_state_dim), np.float32)})
        actor_encoder = FlashSACEncoder(
            type("ActorObs", (), {"single_observation_space": actor_obs_space})(),
            image_embed_dim=args.image_embed_dim, state_embed_dim=args.state_embed_dim, channels_last=channels_last,
        ).to(device)
        if channels_last:
            actor_encoder.to(memory_format=torch.channels_last)
            bf16_forward(actor_encoder)
        actor_optimizer.add_param_group({"params": list(actor_encoder.parameters())})

    def actor_input(embed, obs):
        if not asym:
            return embed.detach()
        s = obs["state"]
        return actor_encoder({"rgb": obs["rgb"], "state": s[..., : args.actor_state_dim]})

    def normalize_actor():
        actor.normalize_parameters()
        if asym:
            actor_encoder.normalize_parameters()

    for m in (qf1, qf2, qf1_target, qf2_target, encoder):
        m.normalize_parameters()
    normalize_actor()
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        encoder.load_state_dict(ckpt["encoder"])
        target_encoder.load_state_dict(ckpt.get("target_encoder", ckpt["encoder"]))
        for online, target, key in ((qf1, qf1_target, "qf1"), (qf2, qf2_target, "qf2")):
            online.load_state_dict(ckpt[key])
            target.load_state_dict(ckpt[key])
        print(f"[checkpoint] loaded actor/encoder/critics from {args.checkpoint} (step {ckpt.get('global_step')})")
    resume_ckpt = ckpt if (args.checkpoint is not None and args.resume) else None

    def renorm_and_ema():
        if not args.freeze_encoder:
            encoder.normalize_parameters()
        qf1.normalize_parameters()
        qf2.normalize_parameters()
        with torch.no_grad():
            for targets, online, w in ema_pairs:
                torch._foreach_lerp_(targets, online, w)

    if compile_nets:
        for m in (encoder, target_encoder, qf1, qf2, qf1_target, qf2_target, actor):
            m.compile()
        actor.get_action = torch.compile(actor.get_action)
        for m in (qf1_target, qf2_target):
            m.projection = torch.compile(m.projection)
        renorm_and_ema = torch.compile(renorm_and_ema)

    if autotune:
        target_entropy = 0.5 * float(np.prod(act_shape)) * math.log(2 * math.pi * math.e * args.target_entropy_sigma**2)
        log_alpha = torch.tensor([math.log(args.alpha)], requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr, fused=True)
    else:
        alpha = args.alpha

    total_updates = max(1, (args.total_timesteps - args.learning_starts) // (args.num_envs * steps_per_env) * grad_steps_per_iteration)

    def cosine_lr(step):
        progress = min(step / total_updates, 1.0)
        return args.lr_end_ratio + (1 - args.lr_end_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    q_scheduler = optim.lr_scheduler.LambdaLR(q_optimizer, cosine_lr)
    actor_scheduler = optim.lr_scheduler.LambdaLR(actor_optimizer, cosine_lr)
    if autotune:
        a_scheduler = optim.lr_scheduler.LambdaLR(a_optimizer, cosine_lr)
    resume_step = resume_update = 0
    if resume_ckpt is not None:
        resume_step = int(resume_ckpt["global_step"])
        resume_update = max(0, (resume_step - args.learning_starts) // (args.num_envs * steps_per_env) * grad_steps_per_iteration)
        for _ in range(resume_update):
            q_scheduler.step()
        for _ in range(resume_update // args.policy_frequency):
            actor_scheduler.step()
            if autotune:
                a_scheduler.step()
        if autotune and "log_alpha" in resume_ckpt:
            log_alpha.data.copy_(resume_ckpt["log_alpha"].to(device))
            alpha = log_alpha.detach().exp()
        if "sgs" in resume_ckpt and envs._reset_term.sgs is not None:
            envs._reset_term.sgs.load_state_dict({k: v.to(device) for k, v in resume_ckpt["sgs"].items()})
        print(f"[resume] step {resume_step:,} update {resume_update:,} alpha {float(alpha):.5f} q_lr {q_optimizer.param_groups[0]['lr']:.2e} "
              f"sgs {'restored' if 'sgs' in resume_ckpt and envs._reset_term.sgs is not None else 'n/a'}; "
              f"fresh optimizers + replay buffer, {args.warmup_episodes} warmup episodes without updates", flush=True)

    chunk = args.horizon * args.num_envs
    buffer_size = max(chunk, args.buffer_size // chunk * chunk)
    rgb_shape = envs.single_observation_space["rgb"].shape
    rb = ImageReplayBuffer(
        pixels_obs_space=rgb_shape, state_obs_dim=envs.single_observation_space["state"].shape[1], act_dim=act_shape[0],
        size=buffer_size, horizon=args.horizon, num_envs=args.num_envs, n_frame_stack=1, n_steps=args.n_step_returns,
        gamma=args.gamma, device=device,
    )
    expert_rb, n_expert = None, 0
    if args.expert_buffer is not None:
        d = torch.load(args.expert_buffer, weights_only=False)
        meta = d["metadata"]
        expected = dict(
            task=args.task, image_keys=list(envs.image_keys), rgb_shape=tuple(rgb_shape), state_dim=rb.state_obs_buf.shape[-1],
            horizon=args.horizon, action_scale=args.action_scale, state_group=args.state_group,
        )
        mismatch = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
        if meta.get("expert_clip", True) == args.no_tanh:
            mismatch["expert_clip"] = (meta.get("expert_clip", True), not args.no_tanh)
        assert not mismatch, f"expert buffer does not match this run (buffer, run): {mismatch}"
        expert_rb = ImageReplayBuffer.from_state_dict(d, n_steps=args.n_step_returns, gamma=args.gamma, device=device)
        del d
        n_expert = int(round(args.batch_size * args.expert_ratio))
        print(f"[rlpd] expert buffer {args.expert_buffer}: {len(expert_rb):,} transitions, expert success "
              f"{meta.get('success_once', float('nan')):.3f}, {n_expert}/{args.batch_size} per batch")
    print(f"[flashsac_rgb] {args.num_envs} envs, horizon {args.horizon}, buffer {buffer_size:,}, "
          f"{grad_steps_per_iteration} grad steps per {training_freq} env steps, obs rgb {tuple(rgb_shape)} state "
          f"{tuple(envs.single_observation_space['state'].shape)} ({args.state_group}), action scale {args.action_scale}", flush=True)

    start_time = time.time()
    obs, _ = envs.reset()
    global_step, global_update = resume_step, resume_update
    warmup_until = resume_step + args.warmup_episodes * args.horizon * args.num_envs
    actor_loss = policy_entropy = alpha_loss = encoder_grad_norm = bc_loss = torch.zeros((), device=device)
    qf1_target_values = qf2_target_values = qf1_loss = qf2_loss = qf_loss = embed_all = None
    recent_success = deque(maxlen=5)
    cumulative_times = defaultdict(float)
    next_eval = resume_step + args.eval_freq if args.eval_freq > 0 else float("inf")
    next_save = resume_step + args.save_freq if args.save_freq > 0 else float("inf")
    next_log = resume_step + args.log_freq
    best_eval = -1.0

    def save(path):
        torch.save({
            "actor": actor.state_dict(), "encoder": encoder.state_dict(), "target_encoder": target_encoder.state_dict(),
            "qf1": qf1_target.state_dict(), "qf2": qf2_target.state_dict(),
            **({"actor_encoder": actor_encoder.state_dict()} if asym else {}),
            **({"log_alpha": log_alpha} if autotune else {}),
            **({"sgs": envs._reset_term.sgs.state_dict()} if envs._reset_term.sgs is not None else {}),
            "config": config, "global_step": global_step,
        }, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"[flashsac_rgb] saved {path}", flush=True)

    reset_types = [] if push_t else env_cfg.events.reset_from_reset_states.params["reset_types"]

    def per_type_success(final_info):
        s, tid = final_info["episode"]["success_once"].float(), final_info["task_id"]
        return {f"success_once_{rt}": s[tid == k].mean().item() for k, rt in enumerate(reset_types) if (tid == k).any()}

    def evaluate():
        nonlocal obs
        stime = time.perf_counter()
        actor.eval()
        metrics = None
        obs, _ = envs.reset_for_eval()
        for _ in range(args.horizon):
            with torch.no_grad():
                actions = actor.get_action(actor_input(encoder(enc_in(obs)), obs), training=False)[2]
            obs, _, _, _, infos = envs.step(actions)
            if "final_info" in infos:
                metrics = {k: v.float().mean().item() for k, v in infos["final_info"]["episode"].items()}
                metrics.update(per_type_success(infos["final_info"]))
        actor.train()
        for k, v in metrics.items():
            logger.add_scalar(f"eval/{k}", v, global_step)
        logger.add_scalar("time/eval_time", time.perf_counter() - stime, global_step)
        print(f"[eval] step {global_step:,}: " + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items()), flush=True)
        return metrics

    while global_step < args.total_timesteps:
        if global_step >= next_eval and envs.t == 0:
            next_eval += args.eval_freq
            metrics = evaluate()
            if metrics["success_once"] > best_eval:
                best_eval = metrics["success_once"]
                save(os.path.join(log_dir, "best_ckpt.pt"))
        if global_step >= next_save:
            next_save += args.save_freq
            save(os.path.join(log_dir, f"ckpt_{global_step}.pt"))

        rollout_time = time.perf_counter()
        for _ in range(steps_per_env):
            global_step += args.num_envs
            logger.counters["per_env_step"] += 1
            if global_step < args.learning_starts:
                actions = torch.rand((args.num_envs,) + act_shape, device=device) * 2 - 1
            else:
                with torch.no_grad():
                    actions, cached_noise, noise_repeat_count, noise_repeat_n = _sample_rollout_action(
                        actor_input(encoder(enc_in(obs)), obs), actor, cached_noise, noise_repeat_count, noise_repeat_n, zeta_cdf, training=False
                    )
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = {k: v.clone() for k, v in next_obs.items()}
            if "final_info" in infos:
                for k in real_next_obs:
                    real_next_obs[k][truncations] = infos["final_observation"][k][truncations]
                logger.counters["train_episodes"] += args.num_envs
                episode = infos["final_info"]["episode"]
                for k, v in episode.items():
                    logger.add_scalar(f"train/{k}", v.float().mean(), global_step)
                for k, v in per_type_success(infos["final_info"]).items():
                    logger.add_scalar(f"train/{k}", v, global_step)
                for k, v in envs.sgs_stats().items():
                    logger.add_scalar(k, v, global_step)
                recent_success.append(episode["success_once"].float().mean().item())
            rb.add(obs, real_next_obs, actions, rewards, terminations)
            obs = next_obs
        cumulative_times["rollout_time"] += time.perf_counter() - rollout_time

        if args.target_success is not None and len(recent_success) == recent_success.maxlen and np.mean(recent_success) >= args.target_success:
            print(f"[flashsac_rgb] train success_once {np.mean(recent_success):.3f} >= {args.target_success}; stopping at {global_step}")
            break
        if global_step < args.learning_starts or global_step < warmup_until:
            continue

        update_time = time.perf_counter()
        for _ in range(grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size - n_expert)
            if n_expert:
                data = cat_image_samples(data, expert_rb.sample(n_expert))
            obs_all = {k: torch.cat([data.observations[k], data.next_observations[k]], dim=0) for k in data.observations}
            obs_all["rgb"] = random_shift(obs_all["rgb"], args.aug_pad)
            if args.blank_images:
                obs_all["rgb"] = torch.zeros_like(obs_all["rgb"])
            embed_all = encoder(enc_in(obs_all))
            data_embed = embed_all[: args.batch_size].detach()
            data_next_embed = embed_all[args.batch_size :].detach()
            actor_in = actor_input(embed_all[: args.batch_size], {k: v[: args.batch_size] for k, v in obs_all.items()})
            with torch.no_grad():
                actor_next_in = actor_input(embed_all[args.batch_size :], {k: v[args.batch_size :] for k, v in obs_all.items()})
            bootstrap = (~data.dones.bool()).float()
            rewards = data.rewards * args.reward_scale

            with torch.no_grad():
                next_state_actions, next_state_log_probs, _ = actor.get_action(actor_next_in, training=False)
                actions_all = torch.cat([data.actions, next_state_actions], dim=0)
                discount = args.gamma**data.effective_n_steps
                entropy_coeff = log_alpha.exp() if autotune else alpha
                target_embed_all = target_encoder(enc_in(obs_all))
                qf1_next = F.softmax(qf1_target(target_embed_all, actions_all, training=True), dim=-1).chunk(2, dim=0)[1]
                qf2_next = F.softmax(qf2_target(target_embed_all, actions_all, training=True), dim=-1).chunk(2, dim=0)[1]
                target_rewards = rewards.squeeze(-1) - discount * bootstrap.squeeze(-1) * entropy_coeff * next_state_log_probs.squeeze(-1)
                qf1_target_dist = qf1_target.projection(qf1_next, next_state_actions, target_rewards, bootstrap.squeeze(-1), discount, training=True)
                qf2_target_dist = qf2_target.projection(qf2_next, next_state_actions, target_rewards, bootstrap.squeeze(-1), discount, training=True)
                qf1_target_values = qf1_target.get_value(qf1_target_dist)
                qf2_target_values = qf2_target.get_value(qf2_target_dist)
                use_qf1 = (qf1_target_values <= qf2_target_values).unsqueeze(-1)
                qf_target_dist = torch.where(use_qf1, qf1_target_dist, qf2_target_dist)

            qf1_outputs = qf1(embed_all, actions_all, training=True).chunk(2, dim=0)[0]
            qf2_outputs = qf2(embed_all, actions_all, training=True).chunk(2, dim=0)[0]
            qf1_loss = -torch.sum(qf_target_dist * F.log_softmax(qf1_outputs, dim=-1), dim=-1)
            qf2_loss = -torch.sum(qf_target_dist * F.log_softmax(qf2_outputs, dim=-1), dim=-1)
            qf_loss = torch.stack([qf1_loss, qf2_loss]).mean(dim=1).sum(dim=0)
            q_optimizer.zero_grad()
            qf_loss.backward()
            if not args.freeze_encoder:
                encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), float("inf"))
            q_optimizer.step()
            q_scheduler.step()

            if global_update % args.policy_frequency == 0:
                pi, log_pi, mean_pi = actor.get_action(actor_in, training=True)
                with torch.no_grad():
                    policy_entropy = -log_pi.mean()
                qf1_value = qf1.get_value(F.softmax(qf1(data_embed, pi, training=False), dim=-1))
                qf2_value = qf2.get_value(F.softmax(qf2(data_embed, pi, training=False), dim=-1))
                actor_loss = ((alpha * log_pi) - torch.min(qf1_value, qf2_value)).mean()
                if bc_policy is not None and args.bc_coef > 0:
                    with torch.no_grad():
                        priv = data.observations["state"][..., priv_slice].reshape(args.batch_size, -1).float().clone()
                        priv[:, : act_shape[0]] *= act_scale_t
                        bc_target = bc_policy(TensorDict({"policy": priv}, batch_size=[args.batch_size])) / act_scale_t
                        if not args.no_tanh:
                            bc_target = bc_target.clamp(-1.0, 1.0)
                    bc_loss = ((mean_pi - bc_target) ** 2).mean()
                    actor_loss = actor_loss + args.bc_coef * bc_loss
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                actor_scheduler.step()
                if autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(actor_in.detach(), training=False)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    a_scheduler.step()
                    alpha = log_alpha.detach().exp()
                normalize_actor()
            renorm_and_ema()
        cumulative_times["update_time"] += time.perf_counter() - update_time

        if global_step >= next_log:
            next_log += args.log_freq
            sps = int((global_step - resume_step) / (time.time() - start_time))
            logger.add_scalar("losses/qf1_values", qf1_target_values.mean(), global_step)
            logger.add_scalar("losses/qf2_values", qf2_target_values.mean(), global_step)
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/bc_loss", bc_loss.item(), global_step)
            logger.add_scalar("losses/alpha", float(alpha), global_step)
            if autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            logger.add_scalar("metrics/policy_entropy", policy_entropy.item(), global_step)
            logger.add_scalar("charts/q_lr", q_optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("charts/expert_frac", n_expert / args.batch_size, global_step)
            logger.add_scalar("time/wall", time.time() - start_time, global_step)
            logger.add_scalar("time/rollout_time", cumulative_times["rollout_time"], global_step)
            logger.add_scalar("time/update_time", cumulative_times["update_time"], global_step)
            logger.add_scalar("encoder/grad_norm", encoder_grad_norm.item(), global_step)
            logger.add_scalar("encoder/embed_std", embed_all.detach().std(0).mean(), global_step)
            print(
                f"[train] step {global_step:,} sps {sps} qf_loss {qf_loss.item() / 2:.3f} q {qf1_target_values.mean().item():.2f} "
                f"actor {actor_loss.item():.3f} alpha {float(alpha):.4f} entropy {policy_entropy.item():.2f} "
                f"train success_once {recent_success[-1] if recent_success else float('nan'):.3f} "
                f"(rollout {cumulative_times['rollout_time']:.0f}s update {cumulative_times['update_time']:.0f}s)",
                flush=True,
            )

    save(os.path.join(log_dir, "final_ckpt.pt"))
    writer.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
