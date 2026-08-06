# Squint on the asymmetric vision peg-insertion task.
#
# A deliberately stripped-down alternative to mrq_flashsac_continuous_action_vision_asymmetric.py,
# which this file is otherwise a copy of. Three things are gone:
#
#   * no FlashSAC residual blocks    -- plain Linear + LayerNorm + ReLU stacks
#   * no unit-norm weight constraint -- no UnitLinear/UnitBatchNorm/UnitRMSNorm, and with them the
#     per-update normalize_parameters() pass disappears entirely
#   * no MR.Q model-based auxiliary loss -- no zsa, no MDP predictor, no encoder target
#
# What remains is a plain SAC:
#
#   actor  : CNNEncoder(images) -> Projection(rgb, proprio) -> 3x(Linear,LayerNorm,ReLU) -> mean/log_std,
#            tanh-squashed with the matching log-prob correction
#   critic : Projection(state) -> num_q x 3x(Linear,LayerNorm,ReLU) -> C51 logits, the ensemble held
#            as one stacked parameter set and evaluated with a single vmap
#
# Still asymmetric: the actor sees pixels + proprioception, the critic sees the full state vector and
# never touches the encoder. Removing the auxiliary loss therefore leaves the ACTOR LOSS as the only
# gradient reaching the CNN -- the encoder is optimized by --encoder_lr on the policy gradient alone.
# That is the intended design here, but it is a weaker signal than MR.Q's auxiliary objective, so
# encoder collapse is the first thing to suspect if representations stop improving.
#
# Pairs with the same Grayscale-Asymmetric tasks in rl_state_cfg.py, subject to one new constraint:
# CNNEncoder only accepts square images of side 64, 32 or 16.
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
import math
import os
import random
import statistics
import time
from dataclasses import dataclass

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call, stack_module_state
import tyro
from torch.utils.tensorboard import SummaryWriter

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cleanrl_utils.buffers import AsymmetricReplayBuffer, cat_samples, load_expert_replay_buffer
from cleanrl_utils.utils import EmpiricalNormalization

from vecenv_wrapper import IsaacLabVectorEnv

_NULL_CTX = nullcontext()


def _fmt_bytes(n: float) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:7.2f} {unit}"
        n /= 1024.0
    return f"{n:7.2f} PiB"


def cuda_used_bytes() -> int:
    """Bytes in use on the current CUDA device, as the *driver* sees it.

    Deliberately not torch.cuda.memory_allocated(): Isaac Sim's physics and rendering allocate
    outside PyTorch's caching allocator, so the torch-level counters are blind to the single largest
    consumer in this script. mem_get_info reflects the whole device, so before/after deltas around
    env construction actually capture it.
    """
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info()
    return total - free


def _host_rss_bytes() -> int:
    """Resident set size of this process (host RAM), or 0 if psutil is unavailable."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return 0


def tensor_bytes_by_device(tensors) -> dict[str, int]:
    """Sum tensor storage per device, de-duplicated by identity (modules may share parameters)."""
    seen, out = set(), defaultdict(int)
    for t in tensors:
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        out[str(t.device)] += t.numel() * t.element_size()
    return dict(out)


def module_bytes_by_device(*modules) -> dict[str, int]:
    """Parameter + buffer bytes per device across modules, de-duplicated (e.g. a shared encoder)."""
    tensors = []
    for mod in modules:
        tensors.extend(mod.parameters())
        tensors.extend(mod.buffers())
    return tensor_bytes_by_device(tensors)


def optimizer_bytes_by_device(*optimizers) -> dict[str, int]:
    """Currently-allocated optimizer state bytes per device.

    Note this reads 0 before the first step(): torch allocates exp_avg/exp_avg_sq lazily, per
    parameter, on the step that first sees a gradient. Use optimizer_projected_bytes_by_device for
    a setup-time estimate.
    """
    tensors = []
    for opt in optimizers:
        for state in opt.state.values():
            tensors.extend(v for v in state.values() if torch.is_tensor(v))
    return tensor_bytes_by_device(tensors)


def optimizer_projected_bytes_by_device(*optimizers, per_param: int = 2) -> dict[str, int]:
    """Steady-state optimizer state, as `per_param` copies of every owned parameter.

    AdamW keeps exp_avg + exp_avg_sq, hence per_param=2 -- so optimizer state ends up roughly twice
    the size of the networks themselves, which is easy to forget when sizing a GPU. Projected rather
    than measured because the report runs before any step(), when the real state is still empty.
    """
    params = []
    for opt in optimizers:
        for group in opt.param_groups:
            params.extend(group["params"])
    return {dev: n * per_param for dev, n in tensor_bytes_by_device(params).items()}


def replay_buffer_bytes_by_device(rb) -> dict[str, int]:
    """Storage bytes per device for every tensor field the replay buffer owns."""
    fields = [
        "policy_observations", "critic_observations",
        "next_policy_observations", "next_critic_observations",
        "proprio_observations", "next_proprio_observations",
        "actions", "rewards", "terminations", "truncations",
        # terminal-observation side store used when store_next_obs=False; it is not free and was
        # missing here originally, which made the reported buffer size smaller than reality
        "trunc_obs_policy", "trunc_obs_critic", "trunc_slot", "trunc_owner",
    ]
    return tensor_bytes_by_device(getattr(rb, f, None) for f in fields)


def _merge_by_device(*dicts) -> dict[str, int]:
    out = defaultdict(int)
    for d in dicts:
        for dev, n in d.items():
            out[dev] += n
    return dict(out)


class BlockTimer:
    """Accumulates wall-clock time per named block, for finding what actually dominates a step.

    CUDA kernels launch asynchronously, so wrapping GPU work in bare `time.perf_counter()` measures
    launch time, not execution time. This synchronizes the device at both ends of every block, which
    makes the numbers meaningful but perturbs the very thing being measured (it serializes work that
    would otherwise overlap). That is why profiling is opt-in via --profile_timing and off by
    default: an instrumented run is slower than a production run, and its absolute SPS should not be
    compared against one.

    Nested blocks are fine -- each name accumulates independently -- but note that a parent's total
    then includes its children's, so percentages sum past 100%. Names prefixed "total/" are treated
    as such roll-ups and excluded from the percentage base.
    """

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = enabled
        self._sync = enabled and device.type == "cuda"
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._open: dict[str, float] = {}

    def _now(self) -> float:
        if self._sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    def start(self, name: str) -> None:
        if self.enabled:
            self._open[name] = self._now()

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        t0 = self._open.pop(name, None)
        if t0 is None:  # stop() without a matching start(), e.g. a branch that never ran
            return
        self.totals[name] += self._now() - t0
        self.counts[name] += 1

    def block(self, name: str):
        """Context-manager form, for blocks small enough to indent."""
        if not self.enabled:
            return _NULL_CTX
        return self._block(name)

    @contextmanager
    def _block(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def rows(self) -> list[tuple[str, float, int, float, float]]:
        """(name, total_s, calls, mean_ms, pct_of_leaf_time), slowest first."""
        base = sum(v for k, v in self.totals.items() if not k.startswith("total/"))
        out = []
        for name, tot in self.totals.items():
            n = max(self.counts[name], 1)
            pct = 100.0 * tot / base if base > 0 and not name.startswith("total/") else float("nan")
            out.append((name, tot, self.counts[name], 1e3 * tot / n, pct))
        return sorted(out, key=lambda r: -r[1])

    def reset(self) -> None:
        """Clear accumulated totals, starting a new reporting window.

        Deliberately does NOT clear `_open`: a block may be mid-flight when the window rolls over
        (total/iteration always is, since reporting happens inside the loop). Dropping its start
        would silently lose that sample; keeping it just attributes the span to the next window.
        """
        self.totals.clear()
        self.counts.clear()


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    run_name: str = ""
    """if set, overrides the auto-generated run name (env_id__exp_name__seed__timestamp)"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    num_learning_iterations: int = 25000
    """total iterations of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 3
    """n-step returns"""
    action_repeat: int = 1
    """number of env sim-steps each sampled action is repeated for"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    expert_rb_path: str = ""
    """path to an expert replay buffer written by collect_expert_replay_buffer.py. When set, each
    update samples a mixture of expert and online transitions (see --expert_ratio). Empty = off."""
    expert_ratio: float = 0.5
    """fraction of every training batch drawn from the expert buffer; the rest comes from online
    experience. 0 = online only, 1 = expert only. Held constant (no annealing)."""
    expert_buffer_on_cpu: bool = True
    """store the EXPERT buffer on cpu, independently of the online buffer.

    Kept separate because the two have different economics: the expert buffer is sized by whoever
    collected it rather than by --buffer_size, and load_expert_replay_buffer transiently holds two
    copies (th.load payload + the rebuilt buffer), so a 32 GiB file peaks near 64 GiB -- survivable
    on host RAM, fatal on VRAM. It also supplies only expert_ratio of each batch (~5% by default),
    so it has far less to gain from GPU residency than the online buffer does."""
    buffer_on_cpu: bool = True
    """store the replay buffer on cpu instead of the compute device.

    Set False to keep it on the GPU: that removes the per-update gather + host->device copy
    (measured 13.4 ms -> 0.15 ms for the image stream at a 63504-element obs) and matters most at
    high num_updates, since the sample path runs once per update. It costs roughly
    buffer_size * (policy + critic + proprio + action + 3) * 4 B of VRAM on top of ~14 GiB of Isaac
    env -- ~0.9 GiB at the ablated 2048-element obs, but ~24 GiB at 63504, which will not fit
    alongside the env on a 24 GiB card."""
    share_policy_critic_obs: bool = False
    """MUST stay False in this script. Sharing one allocation between the policy and critic streams
    requires them to be bitwise identical, which is true for the symmetric vision tasks but never
    here: the actor sees an image and the critic a state vector, so the dims cannot even match. The
    buffer rejects it at construction. Kept as a flag only for parity with the symmetric script."""
    store_next_obs: bool = False
    """if False (default), do not store next_* image streams; reconstruct them as observations[t+1]
    at sample time (~2x less replay memory again). Terminal observations for truncated steps are
    kept in a small auto-growing side store, so targets stay bit-exact -- see AsymmetricReplayBuffer."""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 100
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    num_atoms: int = 101
    """number of atoms in the C51 critic support (every distribution-space op scales with this)"""
    alpha_lr: float = 3e-4
    """the learning rate of alpha"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.001
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    save_model: bool = True
    """if toggled, periodically save model checkpoints to runs/{run_name}/ (and upload to wandb if --track)"""
    save_interval: int = 1_000
    """checkpoint save interval, in global_step units (learning iterations, not transitions), so a
    default 25k-iteration run yields 25 checkpoints"""
    ckpt_dir: str = "logs/runs"
    """parent dir for checkpoints; the run writes to {ckpt_dir}/{run_name}/. Defaults under logs/
    because on the cluster only /workspace/uwlab/logs is bound to persistent GPFS -- the rest of
    /workspace/uwlab is node-local scratch that run_singularity.sh rm -rf's when the job ends, so a
    bare runs/ dir survives only via the wandb upload."""
    obs_normalization: bool = True
    """use obs normalization"""
    # ── Squint architecture ──
    encoder_lr: float = 3e-4
    """learning rate for the CNN encoder. With no auxiliary loss the encoder is trained ONLY by the
    actor loss (the critic is state-only), so this is the sole knob on representation learning."""
    actor_hidden_dim: int = 256
    """width of the actor's 3-layer Linear/LayerNorm/ReLU trunk"""
    critic_hidden_dim: int = 512
    """width of each Q-network's 3-layer Linear/LayerNorm/ReLU trunk"""
    proj_rgb_dim: int = 50
    """width of the projection's image head (Linear -> LayerNorm -> Tanh)"""
    proj_state_dim: int = 256
    """width of the projection's state head (Linear -> LayerNorm -> ReLU)"""
    num_q: int = 2
    """size of the C51 critic ensemble. Members are stacked into one parameter set and evaluated
    with a single vmap, so raising this is far cheaper than adding separate critics: batched matmul
    only beats separate GEMMs from about 4 members up."""
    v_min: float = -20.0
    """lower edge of the C51 atom support"""
    v_max: float = 20.0
    """upper edge of the C51 atom support"""
    fused_optimizer: bool = True
    """use the fused AdamW kernel instead of the default multi-tensor (foreach) path. One kernel per
    optimizer step instead of several, which matters at high num_updates where the step runs
    num_updates times per env step. Requires fp32 CUDA params (true here). Set False to fall back if
    a build lacks the fused kernel."""
    tf32_matmul: bool = True
    """allow TF32 for fp32 matmuls / convolutions (Ampere and newer: a100, l40s, h200). Roughly free
    throughput on the MLP-heavy critic and actor at the cost of ~10 bits of mantissa on matmul
    accumulation. Set False if you suspect numerical trouble in the C51 head."""
    num_updates: float = 0.5
    """Update-to-data ratio: gradient updates per env step. Values below 1 (e.g. 0.5) run an
    update only once every 1/num_updates env steps, via a fractional accumulator."""
    temp_inital_value: float = 0.01
    temp_target_sigma: float = 0.15
    noise_repeat_zeta_mu: float = 2.0
    """zeta-distribution shape parameter for noise-repetition exploration"""
    noise_repeat_zeta_max: int = 16
    """max repeat length (in env steps) for noise-repetition exploration"""
    metrics_window: int = 10
    """rolling window size (in completed episodes) for reward/length/termination-rate logging"""
    save_depth_video: bool = False
    """save env-0's depth camera feed as a per-episode mp4"""
    depth_video_max_m: float = 1.25
    """depth value (in meters) mapped to white when saving the per-episode depth video"""
    depth_video_fps: int = 10
    """frames-per-second for the saved per-episode depth video"""
    weight_decay: float = 0.001
    """the weight decay of the optimizer"""
    logging_interval: int = 100
    """the interval to log the metrics"""
    enable_kl_diag: bool = False
    """measure KL(new||old) of the policy across each update epoch. Off by default: it costs an
    extra replay-buffer sample plus two actor forward passes per epoch -- the largest non-env_step
    cost in profiling -- and is purely diagnostic. metrics/kl_divergence logs NaN when off."""
    profile_memory: bool = True
    """print a one-time memory breakdown after setup (envs / networks / optimizers / replay buffer,
    with the device each lives on) and log mem/* scalars every logging_interval. Cheap -- the
    breakdown is a few driver queries and a walk over already-allocated tensors."""
    profile_timing: bool = False
    """if toggled, time each major block (rollout, env step, buffer add/sample incl. host<->device
    transfer, encoder+model loss, C51 target, critic/actor/alpha updates, target EMA, KL, Hessian,
    logging) and print a breakdown every logging_interval. Adds torch.cuda.synchronize() around
    every block, which slows the run -- do not compare an instrumented run's SPS to a normal one."""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def weight_init(m):
    """Orthogonal init, ReLU gain on convs, zero bias -- Squint's initializer."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight.data, nn.init.calculate_gain("relu"))
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class CNNEncoder(nn.Module):
    """Squint's plain conv trunk over the stacked camera images.

    Two deliberate departures from the reference implementation, both forced by this codebase's
    observation contract:

      * No NHWC permute. Observations arrive as (history, C, H, W) and are stored flat; history and
        camera channels fold together into the conv input channels, exactly as the MR.Q encoder did.
      * No `/255 - 0.5` rescale. These are normalized grayscale or metric-depth images, not uint8
        frames, which is the same reason the MR.Q port skipped it.

    The three published strides assume a square image of side 64, 32 or 16; each lands on a 4x4
    map with 64 channels (1024 features). Anything else is rejected rather than silently reshaped.
    """

    def __init__(self, obs_shape, device=None):
        super().__init__()
        h, w = int(obs_shape[-2]), int(obs_shape[-1])
        if h != w:
            raise ValueError(f"CNNEncoder needs a square image, got {h}x{w}")
        self.image_size = h
        self.num_channels = int(np.prod(obs_shape[:-2]))  # history * cameras
        self.obs_hw = (h, w)

        c = self.num_channels
        if self.image_size == 64:
            layers = [nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
                      nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                      nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()]
        elif self.image_size == 32:
            layers = [nn.Conv2d(c, 32, 4, stride=2), nn.ReLU(),
                      nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                      nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()]
        elif self.image_size == 16:
            layers = [nn.Conv2d(c, 32, 4, stride=2), nn.ReLU(),
                      nn.Conv2d(32, 64, 4, stride=1), nn.ReLU()]
        else:
            raise ValueError(
                f"No Squint CNN encoder for image size {self.image_size}; supported: 64, 32, 16. "
                "Point --env_id at a task whose cameras render one of those."
            )
        self.conv = nn.Sequential(*layers, nn.Flatten())

        self.apply(weight_init)
        self.conv = self.conv.to(memory_format=torch.channels_last)
        if device is not None:
            self.to(device)

        with torch.no_grad():
            probe = torch.zeros(1, c, h, w, device=device)
            self.repr_dim = int(self.conv(probe.contiguous(memory_format=torch.channels_last)).shape[1])
        print(f"[encoder] {c}x{h}x{w} -> {self.repr_dim} features ({self.image_size}px stride stack)")

    def forward(self, obs_flat):
        """`obs_flat` is (B, prod(obs_shape)); reshaped to (B, history*C, H, W)."""
        x = obs_flat.reshape(obs_flat.shape[0], self.num_channels, *self.obs_hw)
        x = x.contiguous(memory_format=torch.channels_last)
        return self.conv(x)


class Projection(nn.Module):
    """Squint's two-stream input projection: a narrow tanh head on image features, a wider ReLU
    head on the state vector, concatenated.

    Both streams are optional so one class covers all three uses here: the actor projects
    (image features, proprio); the asymmetric critic projects the state vector alone; a task
    without a proprio group leaves the actor with the image stream only.
    """

    def __init__(self, n_obs=None, n_state=None, rgb_dim=50, state_dim=256, device=None):
        super().__init__()
        self.rgb_proj = None
        self.state_proj = None
        self.repr_dim = 0
        if n_obs:
            self.rgb_proj = nn.Sequential(
                nn.Linear(n_obs, rgb_dim, device=device), nn.LayerNorm(rgb_dim, device=device), nn.Tanh(),
            )
            self.repr_dim += rgb_dim
        if n_state:
            self.state_proj = nn.Sequential(
                nn.Linear(n_state, state_dim, device=device), nn.LayerNorm(state_dim, device=device), nn.ReLU(),
            )
            self.repr_dim += state_dim
        if self.repr_dim == 0:
            raise ValueError("Projection needs at least one of n_obs / n_state")

    def forward(self, rgb=None, state=None):
        parts = []
        if self.rgb_proj is not None:
            parts.append(self.rgb_proj(rgb))
        if self.state_proj is not None:
            parts.append(self.state_proj(state))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


class Actor(nn.Module):
    """Squint actor: projection -> 3x (Linear, LayerNorm, ReLU) -> mean / log_std heads.

    Always tanh-squashed, unlike the FlashSAC actor this replaces (which had a --use_tanh switch and
    defaulted to raw Gaussian actions). The log-prob carries the matching tanh Jacobian correction.
    """

    def __init__(self, env, n_obs, n_state, hidden_dim=256, device=None):
        super().__init__()
        n_act = int(np.prod(env.single_action_space.shape))
        activ = nn.ReLU

        self.proj = Projection(n_obs, n_state, device=device)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
        )
        self.fc_mean = nn.Linear(hidden_dim, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim, n_act, device=device)

        # The UWLab manipulation tasks expose an UNBOUNDED action space. Squint's reference
        # implementation assumes a finite Box, and (inf - -inf)/2 = inf with (inf + -inf)/2 = nan,
        # which makes every action NaN from the first rollout step -- the NaN then reaches PhysX and
        # brings the sim down with an illegal memory access rather than anything legible. Fall back
        # to the identity rescale, matching FlashSACActor's guard.
        action_space = env.single_action_space
        if np.isinf(action_space.high).any() or np.isinf(action_space.low).any():
            action_scale, action_bias = 1.0, 0.0
        else:
            action_scale = (action_space.high - action_space.low) / 2.0
            action_bias = (action_space.high + action_space.low) / 2.0
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32, device=device))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32, device=device))

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.apply(weight_init)

    def forward(self, rgb, state, get_log_std=False):
        x = self.fc(self.proj(rgb, state))
        mean = self.fc_mean(x)
        if get_log_std:
            log_std = torch.tanh(self.fc_logstd(x))
            log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
            return mean, log_std
        return mean

    def get_eval_action(self, rgb, state):
        return torch.tanh(self.forward(rgb, state)) * self.action_scale + self.action_bias

    def get_action(self, rgb, state):
        mean, log_std = self.forward(rgb, state, get_log_std=True)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Critic(nn.Module):
    """Squint distributional (C51) Q-ensemble over the STATE observation.

    Asymmetric by construction: no image stream. The actor sees pixels, the critic sees the full
    state vector, so `Projection` here carries only its state head and the encoder never appears on
    this path -- which is why the actor loss is the sole gradient source for the CNN.

    The `num_q` members are held as one stack of parameters and evaluated with a single vmap over a
    meta-device template, rather than as `num_q` separate modules. The reference implementation uses
    `tensordict.nn.from_modules`; this environment's tensordict predates it, so the equivalent
    `torch.func.stack_module_state` + `functional_call` is used instead.
    """

    def __init__(self, n_state, n_act, num_atoms, v_min, v_max, num_q=2, hidden_dim=512,
                 state_dim=256, device=None):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_q = num_q
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_dim = hidden_dim
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

        self.proj = Projection(None, n_state, state_dim=state_dim, device=device)
        self.proj.apply(weight_init)

        q_input_dim = self.proj.repr_dim + n_act
        q_nets = [self._build_q_network(q_input_dim, num_atoms, device=device) for _ in range(num_q)]
        for qn in q_nets:
            qn.apply(weight_init)

        # Stacked parameters: one tensor per layer with a leading ensemble dim. Registered through a
        # ParameterDict so the optimizer, state_dict and the Polyak update all see them normally.
        # ParameterDict keys cannot contain '.', hence the sanitized names + the reverse map.
        stacked, _ = stack_module_state(q_nets)
        self._q_key_map = {k.replace(".", "__"): k for k in stacked}
        self.q_params = nn.ParameterDict({sk: nn.Parameter(stacked[k]) for sk, k in self._q_key_map.items()})

        # Shape-only template for functional_call. object.__setattr__ keeps it out of
        # parameters()/state_dict(): meta tensors have no storage and would break both.
        object.__setattr__(self, "_q_meta", self._build_q_network(q_input_dim, num_atoms, device="meta"))
        object.__setattr__(self, "_q_repr", repr(q_nets[0]))

    def __repr__(self):
        lines = [f"{self.__class__.__name__}(", f"  (proj): {self.proj}"]
        lines += [f"  (q{i}): {self._q_repr}" for i in range(self.num_q)]
        return "\n".join(lines + [")"])

    def _build_q_network(self, input_dim, num_atoms, device=None):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, device=device), nn.LayerNorm(self.hidden_dim, device=device), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, device=device), nn.LayerNorm(self.hidden_dim, device=device), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, device=device), nn.LayerNorm(self.hidden_dim, device=device), nn.ReLU(),
            nn.Linear(self.hidden_dim, num_atoms, device=device),
        )

    def _stacked(self, detach=False):
        """The stacked parameters keyed by their original module paths, for functional_call."""
        return {k: (self.q_params[sk].detach() if detach else self.q_params[sk])
                for sk, k in self._q_key_map.items()}

    def _vmap_q(self, params, x):
        return functional_call(self._q_meta, params, (x,))

    def forward(self, state, actions):
        """Logits from every ensemble member: (num_q, batch, num_atoms)."""
        x = torch.cat([self.proj(state=state), actions], dim=-1)
        return torch.vmap(self._vmap_q, (0, None))(self._stacked(), x)

    def get_q_values(self, state, actions, detach_critic=False):
        """Expected Q per member: (num_q, batch).

        With `detach_critic` the critic's own weights are frozen while gradients still flow through
        `actions` -- the actor's policy gradient, which here also carries all the way back into the
        CNN encoder that produced the actor's input.
        """
        if detach_critic:
            with torch.no_grad():
                proj = self.proj(state=state)
            x = torch.cat([proj, actions], dim=-1)
            logits = torch.vmap(self._vmap_q, (0, None))(self._stacked(detach=True), x)
        else:
            logits = self.forward(state, actions)
        return torch.sum(F.softmax(logits, dim=-1) * self.q_support, dim=-1)

    def categorical(self, state, actions, rewards, bootstrap, discount):
        """C51 projection of the TD target onto the atom support: (num_q, batch, num_atoms).

        The atom indices depend only on the target, not on any member's output, so they are computed
        once and broadcast across the ensemble, and both scatters run over the flattened
        (num_q * batch) rows in a single kernel each.
        """
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        dev = rewards.device

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * self.q_support
        target_z = target_z.clamp(self.v_min, self.v_max)

        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and(lower > 0, is_integer)
        upper_mask = torch.logical_and(lower == 0, is_integer)
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dists = F.softmax(self.forward(state, actions), dim=-1)

        total_batch = self.num_q * batch_size
        next_dists_flat = next_dists.reshape(total_batch, self.num_atoms)
        offset = torch.arange(total_batch, device=dev).unsqueeze(1) * self.num_atoms

        exp = lambda t: t.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)
        lower_exp, upper_exp, b_exp = exp(lower), exp(upper), exp(b)

        max_index = total_batch * self.num_atoms - 1
        lower_indices = torch.clamp((lower_exp + offset).view(-1), 0, max_index)
        upper_indices = torch.clamp((upper_exp + offset).view(-1), 0, max_index)

        proj_flat = torch.zeros_like(next_dists_flat)
        proj_flat.view(-1).index_add_(0, lower_indices, (next_dists_flat * (upper_exp.float() - b_exp)).view(-1))
        proj_flat.view(-1).index_add_(0, upper_indices, (next_dists_flat * (b_exp - lower_exp.float())).view(-1))
        return proj_flat.reshape(self.num_q, batch_size, self.num_atoms)


def _build_truncated_zeta_cdf(mu: float, max_n: int, device) -> torch.Tensor:
    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    return torch.cumsum(pmf, dim=0)


def _sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)


def _sample_rollout_action(actor, rgb, state, noise, cur_count, cur_n, zeta_cdf):
    """Noise-repetition exploration around the actor's mean, tanh-squashed into the action box.

    Squint's actor is always squashed (there is no --use_tanh switch as in the FlashSAC port), so
    the rollout squashes here too and the stored action stays inside the env's bounds.
    """
    mean, log_std = actor(rgb, state, get_log_std=True)
    std = log_std.exp()

    reinit = (cur_count == 0) | (cur_count >= cur_n)
    new_noise = torch.randn_like(mean)
    new_n = _sample_integer_from_cdf(zeta_cdf)

    noise = torch.where(reinit, new_noise, noise)
    cur_n = torch.where(reinit, new_n, cur_n)
    cur_count = torch.where(reinit, torch.zeros_like(cur_count), cur_count)

    x_t = mean + std * noise
    action = torch.tanh(x_t) * actor.action_scale + actor.action_bias

    return action, noise, cur_count + 1, cur_n


def _depth_frame_to_uint8(frame: torch.Tensor, max_m: float) -> np.ndarray:
    """Convert a single (H, W) depth frame (meters) to a grayscale uint8 image for video export."""
    frame = (frame.clamp(0.0, max_m) / max_m * 255.0).to(torch.uint8)
    return frame.cpu().numpy()


if __name__ == "__main__":

    args, launcher_args = tyro.cli(Args, return_unknown_args=True)
    run_name = args.run_name if args.run_name else f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    if args.save_depth_video:
        depth_video_dir = f"runs/{run_name}/depth_videos"
        os.makedirs(depth_video_dir, exist_ok=True)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # TF32 for fp32 matmul/conv on Ampere+ (a100 / l40s / h200). The critic and actor are MLP-heavy
    # and run num_updates times per env step, so this is one of the cheapest throughput knobs
    # available. It costs mantissa bits on matmul accumulation only -- master weights stay fp32.
    if args.tf32_matmul and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[perf] TF32 enabled for fp32 matmul/conv")
    print(f"[perf] fused AdamW: {args.fused_optimizer}")

    # ── memory accounting: baseline before Isaac Sim starts ──
    # Taken here so the env delta below includes Isaac's PhysX/rendering allocations, which live
    # outside PyTorch's allocator. The baseline itself is mostly this process's CUDA context.
    mem_baseline_gpu = cuda_used_bytes()
    mem_baseline_host = _host_rss_bytes()

    # env setup
    envs = IsaacLabVectorEnv(args.env_id, args.num_envs, launcher_args=launcher_args)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    mem_after_env_gpu = cuda_used_bytes()
    mem_after_env_host = _host_rss_bytes()

    policy_obs_shape = envs.single_observation_space["policy"].shape   # (history, C, H, W) image
    critic_obs_shape = envs.single_observation_space["critic"].shape   # flat state vector
    critic_obs_dim = int(np.prod(critic_obs_shape))
    # No shape assertion here: unlike the symmetric script there is no shared encoder, so the actor
    # (image) and critic (state) streams are expected to differ.
    print(f"[obs] asymmetric: actor {tuple(policy_obs_shape)} image | critic {critic_obs_dim}-d state")

    action_dim = int(np.prod(envs.single_action_space.shape))

    # Optional proprioception group (e.g. the grayscale reaching task). Tasks that only define
    # policy/critic leave proprio_dim at 0, which reproduces the vision-only encoder exactly.
    has_proprio = "proprio" in envs.single_observation_space.spaces
    proprio_dim = int(np.prod(envs.single_observation_space["proprio"].shape)) if has_proprio else 0
    print(f"[obs] proprio group: {'present' if has_proprio else 'absent'} (dim={proprio_dim})")

    if args.obs_normalization:
        policy_obs_dim = int(np.prod(envs.single_observation_space["policy"].shape))
        critic_obs_dim = int(np.prod(envs.single_observation_space["critic"].shape))
        actor_obs_normalizer = EmpiricalNormalization(shape=(policy_obs_dim,), device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=(critic_obs_dim,), device=device)
        if has_proprio:
            proprio_obs_normalizer = EmpiricalNormalization(shape=(proprio_dim,), device=device)

    mem_before_nets_gpu = cuda_used_bytes()

    # Squint CNN encoder over the image stream. There is no encoder target: the C51 target action
    # comes from the CURRENT policy (standard SAC), and with the auxiliary loss gone there is no
    # dynamics target that would need one.
    encoder = CNNEncoder(policy_obs_shape, device=device)

    actor = Actor(
        envs, encoder.repr_dim, proprio_dim,
        hidden_dim=args.actor_hidden_dim, device=device,
    ).to(device)

    def make_critic():
        return Critic(
            critic_obs_dim, action_dim,
            num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max,
            num_q=args.num_q, hidden_dim=args.critic_hidden_dim,
            state_dim=args.proj_state_dim, device=device,
        ).to(device)

    critic = make_critic()
    critic_target = make_critic()
    critic_target.load_state_dict(critic.state_dict())

    zeta_cdf = _build_truncated_zeta_cdf(args.noise_repeat_zeta_mu, args.noise_repeat_zeta_max, device)
    noise_repeat_n = torch.ones((), dtype=torch.int32, device=device)
    noise_repeat_count = torch.zeros((), dtype=torch.int32, device=device)
    cached_noise = torch.randn((args.num_envs,) + envs.single_action_space.shape, device=device)

    # Flat (online, target) lists for the Polyak update, built once so the per-update EMA is a
    # couple of _foreach_ kernels instead of ~3 launches per parameter tensor. Only the critic has a
    # target; parameters() ordering is identical across the two identically-built instances.
    ema_online_params = list(critic.parameters())
    ema_target_params = list(critic_target.parameters())
    assert len(ema_online_params) == len(ema_target_params)

    # The encoder's only gradient source is the actor loss (see the module docstring), but it keeps
    # its own optimizer so the representation can be tuned independently of the policy head.
    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=args.encoder_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=args.fused_optimizer)
    q_optimizer = optim.AdamW(critic.parameters(), lr=args.q_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=args.fused_optimizer)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.policy_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=args.fused_optimizer)

    # Automatic entropy tuning. Kept on the FlashSAC port's "match this sigma" target so runs stay
    # comparable; note Squint's actor is tanh-squashed, for which the classic SAC target is -|A|.
    if args.autotune:
        target_entropy = 0.5 * torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item() * torch.log(torch.tensor(2 * torch.pi * torch.e * args.temp_target_sigma ** 2)).item()
        # log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # alpha = log_alpha.exp().item()
        alpha = torch.tensor(args.alpha, device=device)
        log_alpha = torch.tensor([torch.log(alpha)], requires_grad=True, device=device)
        a_optimizer = optim.AdamW([log_alpha], lr=args.alpha_lr, betas=(0.9, 0.95), fused=args.fused_optimizer)
    else:
        alpha = args.alpha

    mem_after_nets_gpu = cuda_used_bytes()

    envs.single_observation_space.dtype = np.float32
    buffer_device = torch.device("cpu") if args.buffer_on_cpu else device
    mem_before_rb_gpu, mem_before_rb_host = cuda_used_bytes(), _host_rss_bytes()
    rb = AsymmetricReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        buffer_device,
        n_envs=args.num_envs,
        n_steps=args.num_steps,
        sample_device=device,
        share_policy_critic_obs=args.share_policy_critic_obs,
        store_next_obs=args.store_next_obs,
    )
    mem_after_rb_gpu, mem_after_rb_host = cuda_used_bytes(), _host_rss_bytes()

    # ── expert replay buffer (optional) ──
    expert_rb = None
    if args.expert_rb_path:
        # n_steps/gamma are overridden to this run's values: the expert buffer stores single-step
        # transitions in temporal order, so its n-step returns are built at sample time and should
        # use the consumer's horizon, not whatever the collector was tagged with.
        # Deliberately its own device: the expert buffer is large, fixed-size and loaded with a
        # transient 2x peak, while contributing only expert_ratio of each batch.
        expert_buffer_device = torch.device("cpu") if args.expert_buffer_on_cpu else device
        expert_rb = load_expert_replay_buffer(
            args.expert_rb_path, device=expert_buffer_device, sample_device=device,
            n_steps=args.num_steps, gamma=args.gamma,
        )
        # Fail loudly on layout mismatch rather than training on silently wrong observations.
        for name, got, want in (
            ("policy obs", expert_rb.policy_observations.shape[-1], rb.policy_observations.shape[-1]),
            ("critic obs", expert_rb.critic_observations.shape[-1], rb.critic_observations.shape[-1]),
            ("action", expert_rb.actions.shape[-1], rb.actions.shape[-1]),
        ):
            if got != want:
                raise ValueError(
                    f"Expert buffer {name} dim {got} != this task's {want}. The expert buffer must be "
                    f"collected on a task exposing the same observation groups."
                )
        if expert_rb.has_proprio != rb.has_proprio:
            raise ValueError(
                f"Expert buffer has_proprio={expert_rb.has_proprio} but this task has "
                f"has_proprio={rb.has_proprio}; re-collect with --record_proprio_obs_keys."
            )
        if expert_rb.has_proprio and (
            expert_rb.proprio_observations.shape[-1] != rb.proprio_observations.shape[-1]
        ):
            raise ValueError("Expert buffer proprio dim does not match this task's.")
        n_expert_tr = expert_rb.buffer_size * expert_rb.n_envs
        print(
            f"[expert] loaded {args.expert_rb_path}: {n_expert_tr:,} transitions "
            f"({expert_rb.buffer_size} steps x {expert_rb.n_envs} envs), "
            f"ratio {args.expert_ratio} -> {round(args.batch_size * args.expert_ratio)}/{args.batch_size} "
            f"of each batch"
        )

    # ── memory report ───────────────────────────────────────────────────────────────────────────
    # Two independent views, because neither alone is trustworthy:
    #   * "exact"    -- summed tensor storage, attributed to the device each tensor lives on. Precise
    #                   for the buffer/networks/optimizers, but blind to anything not a torch tensor.
    #   * "observed" -- driver-level before/after deltas (mem_get_info). Catches Isaac's non-torch
    #                   allocations, but is noisy: allocator caching and rounding inflate it, and a
    #                   lazily-initialized subsystem can land in whichever window happens to touch it.
    if args.profile_memory:
        nets = {
            "encoder (CNN)": module_bytes_by_device(encoder),
            f"critic (x{args.num_q} C51)": module_bytes_by_device(critic),
            "critic_target": module_bytes_by_device(critic_target),
            "actor": module_bytes_by_device(actor),
        }
        if args.obs_normalization:
            norms = [actor_obs_normalizer, critic_obs_normalizer] + ([proprio_obs_normalizer] if has_proprio else [])
            nets["obs normalizers"] = module_bytes_by_device(*norms)
        # Projected, not measured: this runs pre-step(), when AdamW's state is still unallocated.
        opt_bytes = optimizer_projected_bytes_by_device(encoder_optimizer, q_optimizer, actor_optimizer)
        rb_bytes = replay_buffer_bytes_by_device(rb)
        nets_total = _merge_by_device(*nets.values())

        def _row(label, by_dev):
            total = sum(by_dev.values())
            where = ", ".join(f"{dev}: {_fmt_bytes(n)}" for dev, n in sorted(by_dev.items())) or "-"
            return f"  {label:<28}{_fmt_bytes(total)}   [{where}]"

        bar = "─" * 78
        obs_el = int(np.prod(critic_obs_shape))
        lines = [
            f"\n{bar}",
            f"{' memory breakdown ':─^78}",
            "  exact (summed tensor storage, by device)",
            *[_row(name, b) for name, b in nets.items()],
            _row("networks TOTAL", nets_total),
            _row("optimizer state (AdamW, ~2x)", opt_bytes),
            _row("replay buffer", rb_bytes),
            # Report the streams actually allocated rather than assuming 4: with
            # share_policy_critic_obs / store_next_obs the count drops to 2 or 1.
            f"    buffer detail: {rb.buffer_size} steps x {rb.n_envs} envs x {obs_el} obs-elem x 4 B"
            f" x {(1 if rb.share_policy_critic_obs else 2) * (2 if rb.store_next_obs else 1)}"
            f" image stream(s)"
            f"  [share_policy_critic_obs={rb.share_policy_critic_obs},"
            f" store_next_obs={rb.store_next_obs}]"
            + (f"\n    terminal side store: {rb.trunc_capacity} slots"
               f" ({_fmt_bytes(sum(tensor_bytes_by_device([rb.trunc_obs_policy, rb.trunc_obs_critic]).values())).strip()})"
               if not rb.store_next_obs else "")
            + (f"\n    proprio streams: {'2' if rb.store_next_obs else '2'} x {proprio_dim} elem"
               if rb.has_proprio else ""),
            "",
            "  observed (driver deltas; includes non-torch allocations)",
            f"  {'CUDA context baseline':<28}{_fmt_bytes(mem_baseline_gpu)}",
            f"  {'envs (Isaac sim+render)':<28}{_fmt_bytes(mem_after_env_gpu - mem_baseline_gpu)}   "
            f"[{device}]  <- physics/rendering, invisible to torch counters",
            f"  {'networks+optimizers':<28}{_fmt_bytes(mem_after_nets_gpu - mem_before_nets_gpu)}   [{device}]",
            f"  {'replay buffer':<28}{_fmt_bytes(mem_after_rb_gpu - mem_before_rb_gpu)}   [{device}]"
            f"  (online buffer on {'cpu' if args.buffer_on_cpu else device}"
            + (f", expert on {'cpu' if args.expert_buffer_on_cpu else device}" if args.expert_rb_path else "")
            + ")",
            f"  {'GPU in use now':<28}{_fmt_bytes(cuda_used_bytes())}",
        ]
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            lines.append(f"  {'GPU capacity':<28}{_fmt_bytes(total)}  (free {_fmt_bytes(free)})")
            lines.append(f"  {'torch allocated / reserved':<28}"
                         f"{_fmt_bytes(torch.cuda.memory_allocated())} / {_fmt_bytes(torch.cuda.memory_reserved())}")
        host_rss = _host_rss_bytes()
        if host_rss:
            lines.append(f"  {'host RSS now':<28}{_fmt_bytes(host_rss)}   "
                         f"(env +{_fmt_bytes(mem_after_env_host - mem_baseline_host)}, "
                         f"buffer +{_fmt_bytes(mem_after_rb_host - mem_before_rb_host)})")
        lines.append(bar)
        print("\n".join(lines), flush=True)

    start_time = time.time()

    if args.obs_normalization:
        normalize_obs = actor_obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward

    n_expert_per_batch = int(min(max(round(args.batch_size * args.expert_ratio), 0), args.batch_size))

    def sample_batch(batch_size: int):
        """Online sample, or an expert/online mixture when an expert buffer is loaded."""
        if expert_rb is None or n_expert_per_batch == 0:
            return rb.sample(batch_size)
        n_expert = int(round(batch_size * args.expert_ratio))
        n_online = batch_size - n_expert
        if n_online <= 0:
            return expert_rb.sample(batch_size)
        return cat_samples(rb.sample(n_online), expert_rb.sample(n_expert))

    def get_proprio(obs_dict, update: bool = False):
        """Flattened (and optionally normalized) proprio from an env obs dict; None if absent."""
        if not has_proprio:
            return None
        p = obs_dict["proprio"].reshape(obs_dict["proprio"].shape[0], -1)
        return proprio_obs_normalizer(p, update=update) if args.obs_normalization else p

    def norm_proprio(p, update: bool = False):
        """Normalize an already-flattened proprio batch (e.g. straight out of the replay buffer)."""
        if p is None:
            return None
        return proprio_obs_normalizer(p, update=update) if args.obs_normalization else p

    # Logging stuff
    rewbuffer = deque(maxlen=args.metrics_window)
    lenbuffer = deque(maxlen=args.metrics_window)
    total_episodes = 0
    num_success_episodes_log = 0
    num_episodes_log = 0
    termination_buffers: dict[str, deque] = {}
    # NaN unless --enable_kl_diag: the logging block below reads it unconditionally.
    kl_old_policy = torch.tensor(float('nan'))
    updates_pending = 0.0
    update_step = 0
    depth_video_frames: list[np.ndarray] = []
    depth_video_episode = 0

    timer = BlockTimer(args.profile_timing, device)
    if args.profile_timing:
        print("[profile] --profile_timing on: CUDA syncs added around every block; SPS is not "
              "comparable to a normal run.")

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    while global_step < args.num_learning_iterations:
        timer.start("total/iteration")
        with timer.block("rollout/act"):
            with torch.no_grad():
                policy_obs = obs["policy"].reshape(obs["policy"].shape[0], -1)
                if args.obs_normalization:
                    policy_obs = normalize_obs(policy_obs, update=False)

            with torch.no_grad():
                policy_rgb = encoder(policy_obs)
                actions, cached_noise, noise_repeat_count, noise_repeat_n = _sample_rollout_action(
                    actor, policy_rgb, get_proprio(obs, update=False),
                    cached_noise, noise_repeat_count, noise_repeat_n, zeta_cdf,
                )

        # TRY NOT TO MODIFY: execute the game and log data.
        # Sim step (incl. rendering the cameras, usually the single largest cost for image tasks).
        timer.start("rollout/env_step")
        rewards = None
        for _ in range(args.action_repeat):
            next_obs, step_rewards, terminations, truncations, infos = envs.step(actions)
            rewards = step_rewards if rewards is None else rewards + step_rewards
        dones = terminations | truncations
        timer.stop("rollout/env_step")

        # env-0 depth video: capture the most-recent frame in the history stack, flush to disk
        # as an mp4 whenever env 0's episode ends.
        if args.save_depth_video:
            with timer.block("io/depth_video"):
                depth_video_frames.append(_depth_frame_to_uint8(next_obs["policy"][0, -1, 0], args.depth_video_max_m))
                if dones[0]:
                    video_path = os.path.join(depth_video_dir, f"episode_{depth_video_episode:06d}.mp4")
                    imageio.mimsave(video_path, depth_video_frames, fps=args.depth_video_fps)
                    depth_video_episode += 1
                    depth_video_frames = []

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # Note: the .cpu()/.any() calls here force device->host syncs every iteration.
        with timer.block("rollout/episode_stats"):
            rewbuffer.extend(infos["ep_reward"].cpu().tolist())
            lenbuffer.extend(infos["ep_length"].cpu().tolist())
            num_episodes_log += dones.int().sum()
            total_episodes += dones.int().sum()
            num_success_episodes_log += infos["ep_success"].to(device).float().sum()
            if dones.any():
                for key, val in infos.get("log", {}).items():
                    if key.startswith("Episode_Termination/"):
                        term_name = key[len("Episode_Termination/"):]
                        termination_buffers.setdefault(term_name, deque(maxlen=args.metrics_window)).append(val)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # With --buffer_on_cpu (default) rb.add() copies the image observations device->host, so
        # this block is dominated by PCIe transfer rather than compute.
        with timer.block("buffer/add_D2H"):
            real_next_obs = next_obs.copy()
            if truncations.any():
                real_next_obs[truncations.bool()] = infos["final_obs"]
            rb.add(obs, real_next_obs, actions, rewards, terminations, truncations, infos)
        # import pdb; pdb.set_trace()

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            updates_pending += args.num_updates
            n_updates_this_step = int(updates_pending)
            updates_pending -= n_updates_this_step

        if global_step > args.learning_starts and n_updates_this_step > 0:
            # Snapshot the policy on a reference batch before this epoch's updates, to
            # measure how far the policy moves (KL) over the course of the num_updates
            # gradient steps below.
            if args.enable_kl_diag:
                with timer.block("diag/kl_snapshot"), torch.no_grad():
                    kl_ref_sample = rb.sample(args.batch_size)
                    kl_ref_obs = kl_ref_sample.policy_observations
                    if args.obs_normalization:
                        kl_ref_obs = normalize_obs(kl_ref_obs, update=False)
                    kl_ref_proprio = norm_proprio(kl_ref_sample.proprio_observations, update=False)
                    old_mean, old_log_std = actor(
                        encoder(kl_ref_obs), kl_ref_proprio, get_log_std=True
                    )
                    old_mean = old_mean.detach()
                    old_log_std = old_log_std.detach()

            for upd_i in range(n_updates_this_step):
                # rb.sample() gathers on the buffer device then moves the batch host->device when
                # --buffer_on_cpu; for image observations that copy usually dwarfs the indexing.
                with timer.block("buffer/sample_H2D"):
                    data = sample_batch(args.batch_size)

                with timer.block("update/normalize_obs"):
                    policy_obs = data.policy_observations
                    next_policy_obs = data.next_policy_observations
                    critic_obs = data.critic_observations
                    next_critic_obs = data.next_critic_observations

                    if args.obs_normalization:
                        policy_obs = normalize_obs(data.policy_observations)
                        next_policy_obs = normalize_obs(data.next_policy_observations)
                        critic_obs = normalize_critic_obs(data.critic_observations)
                        next_critic_obs = normalize_critic_obs(data.next_critic_observations)

                    # Proprio is the actor Projection's state stream, concatenated with the
                    # projected image features; the critic never sees it.
                    proprio = norm_proprio(data.proprio_observations, update=True)
                    next_proprio = norm_proprio(data.next_proprio_observations, update=True)

                bootstrap = (~data.dones.bool()).float()

                with torch.no_grad():
                    discount = args.gamma ** data.effective_n_steps

                # ── MR.Q on the ACTOR/vision stream: f/g/model, trained by the auxiliary loss only ──
                # ── C51 target: action from the CURRENT policy on the next observation ──
                # No encoder target (nothing here needs one) and no auxiliary loss, so this is the
                # only encoder forward on non-actor-update steps.
                with timer.block("critic/c51_target"), torch.no_grad():
                    next_rgb = encoder(next_policy_obs)
                    next_state_actions, next_state_log_probs, _ = actor.get_action(next_rgb, next_proprio)
                    reward_term = (
                        data.rewards.squeeze(-1)
                        - discount * bootstrap.squeeze(-1) * log_alpha.exp() * next_state_log_probs.squeeze(-1)
                    )
                    target_dists = critic_target.categorical(
                        next_critic_obs, next_state_actions,
                        reward_term, bootstrap.squeeze(-1), discount,
                    )                                                    # (num_q, B, atoms)
                    target_values = torch.sum(target_dists * critic_target.q_support, dim=-1)  # (num_q, B)
                    # Clipped double-Q generalized to num_q: take the distribution of whichever
                    # member is most pessimistic per sample, not the elementwise min of the two.
                    least = target_values.argmin(dim=0)
                    qf_target_dist = target_dists[least, torch.arange(least.shape[0], device=device)]

                # Critic loss straight on the state observation -- nothing to detach, the critic owns
                # every parameter on this path.
                with timer.block("critic/loss_bwd"):
                    qf_log_probs = F.log_softmax(critic(critic_obs, data.actions), dim=-1)  # (num_q, B, atoms)
                    qf_losses = -torch.sum(qf_target_dist.unsqueeze(0) * qf_log_probs, dim=-1)  # (num_q, B)
                    qf_loss = qf_losses.mean(dim=1).sum(dim=0)

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                if update_step % args.policy_frequency == 0:
                    # UNLIKE the MR.Q port, the encoder forward here is NOT detached: with a
                    # state-only critic and no auxiliary loss, the actor loss is the only gradient
                    # that ever reaches the CNN, so both optimizers step on it.
                    with timer.block("actor/update"):
                        rgb = encoder(policy_obs)
                        pi, log_pi, _ = actor.get_action(rgb, proprio)
                        with torch.no_grad():
                            policy_entropy = -log_pi.mean()
                        # detach_critic freezes the critic's own weights while leaving the path
                        # through `pi` -- and hence back into the encoder -- intact.
                        q_pi = critic.get_q_values(critic_obs, pi, detach_critic=True)   # (num_q, B)
                        min_qf_pi = q_pi.min(dim=0).values.unsqueeze(-1)                 # (B, 1) like log_pi
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        encoder_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()
                        encoder_optimizer.step()

                    if args.autotune:
                        with timer.block("actor/alpha_update"):
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(rgb.detach(), proprio)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            # Kept as a detached 0-dim DEVICE tensor rather than .item(): .item() is a
                            # device->host sync, and at high UTD it runs once per policy update
                            # (num_updates/policy_frequency times per env step -- 128x at UTD 256),
                            # stalling the CPU until the GPU drains each time. detach() preserves the
                            # old semantics, since the actor loss must not backprop into log_alpha.
                            alpha = log_alpha.detach().exp().squeeze()

                update_step += 1

                # Normalize the params
                # update the target network (critic only -- the encoder has no target)
                if global_step % args.target_network_frequency == 0:
                    with timer.block("update/target_ema"), torch.no_grad():
                        # lerp_(online, tau) == (1 - tau) * target + tau * online, i.e. exactly the
                        # Polyak update this replaces -- but batched over every parameter tensor.
                        torch._foreach_lerp_(ema_target_params, ema_online_params, args.tau)
            
        
            # KL(new policy || old policy) on the reference batch, measured over this epoch's
            # num_updates gradient steps. Closed-form KL between diagonal Gaussians, computed in
            # pre-tanh space (KL is invariant under the shared deterministic tanh bijection).
            if args.enable_kl_diag:
                with timer.block("diag/kl"), torch.no_grad():
                    new_mean, new_log_std = actor(
                        encoder(kl_ref_obs), kl_ref_proprio, get_log_std=True
                    )
                    old_std = old_log_std.exp()
                    new_std = new_log_std.exp()
                    kl_old_policy = (
                        old_log_std - new_log_std
                        + (new_std.pow(2) + (new_mean - old_mean).pow(2)) / (2.0 * old_std.pow(2))
                        - 0.5
                    ).sum(dim=-1).mean()

            if global_step % args.logging_interval == 0:
                # Every .item() below is a device->host sync; with many scalars this is not free.
                timer.start("io/logging")
                for qi in range(args.num_q):
                    writer.add_scalar(f"losses/qf{qi + 1}_values", target_values[qi].mean().item(), global_step)
                    writer.add_scalar(f"losses/qf{qi + 1}_loss", qf_losses[qi].mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / args.num_q, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", float(alpha), global_step)
                writer.add_scalar("metrics/kl_divergence", kl_old_policy.item(), global_step)
                writer.add_scalar("metrics/policy_entropy", policy_entropy.item(), global_step)
                sps = global_step / (time.time() - start_time)
                writer.add_scalar("charts/SPS", sps, global_step)
                samples_per_sec = sps * args.num_envs * args.num_updates * args.batch_size
                writer.add_scalar("charts/samples_per_sec", samples_per_sec, global_step)

                if len(rewbuffer) > 0:
                    writer.add_scalar("charts/episodic_return", statistics.mean(rewbuffer), global_step)
                    writer.add_scalar("charts/episodic_length", statistics.mean(lenbuffer), global_step)
                
                writer.add_scalar("charts/success_rate", num_success_episodes_log / num_episodes_log, global_step)
                num_success_episodes_log = 0
                num_episodes_log = 0

                writer.add_scalar("charts/num_episodes", total_episodes, global_step)

                for term_name, buf in termination_buffers.items():
                    if len(buf) > 0:
                        writer.add_scalar(f"charts/termination_{term_name}", statistics.mean(buf), global_step)

                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                # ─── pretty console log ───
                elapsed = time.time() - start_time
                progress = global_step / args.num_learning_iterations
                eta_sec = (elapsed / max(progress, 1e-9)) * (1 - progress)
                eta_str = f"{int(eta_sec // 3600):d}h{int((eta_sec % 3600) // 60):02d}m"

                ep_ret = statistics.mean(rewbuffer) if len(rewbuffer) > 0 else float("nan")
                ep_len = statistics.mean(lenbuffer) if len(lenbuffer) > 0 else float("nan")
                pad = 22
                bar = "─" * 60
                header = f" step={global_step:,}  ({progress*100:5.1f}%)  ETA={eta_str} "
                lines = [
                    f"\n{bar}",
                    f"{header.center(60, '─')}",
                    # sps is a float (true division); ',d' would raise ValueError every logging pass.
                    f"{'SPS:':>{pad}} {sps:>10.2f}",
                    f"{'episodes done:':>{pad}} {int(total_episodes):,d}",
                    f"{'mean episodic return:':>{pad}} {ep_ret:>10.3f}",
                    f"{'mean episode length:':>{pad}} {ep_len:>10.2f}",
                    f"{'q values (per member):':>{pad}} "
                    + "  /  ".join(f"{target_values[qi].mean().item():.3f}" for qi in range(args.num_q)),
                    f"{'qf_loss:':>{pad}} {(qf_loss.item() / args.num_q):>10.4f}",
                    f"{'actor_loss:':>{pad}} {actor_loss.item():>10.4f}",
                    f"{'log_pi (mean):':>{pad}} {log_pi.mean().item():>10.3f}",
                    f"{'kl_divergence:':>{pad}} {kl_old_policy.item():>10.5f}",
                    f"{'policy_entropy:':>{pad}} {policy_entropy.item():>10.4f}",
                    f"{'alpha:':>{pad}} {float(alpha):>10.4f}"
                    + (f"   alpha_loss={alpha_loss.item():.4f}" if args.autotune else ""),
                    bar,
                ]

                # ─── memory tracking (--profile_memory) ───
                # Peak reserved is the number that decides whether a run OOMs; it only ever grows
                # within a window, so reset_peak_memory_stats makes each report window-local.
                if args.profile_memory and torch.cuda.is_available():
                    gpu_used = cuda_used_bytes()
                    writer.add_scalar("mem/gpu_used_total_gb", gpu_used / 2**30, global_step)
                    writer.add_scalar("mem/torch_allocated_gb", torch.cuda.memory_allocated() / 2**30, global_step)
                    writer.add_scalar("mem/torch_reserved_gb", torch.cuda.memory_reserved() / 2**30, global_step)
                    writer.add_scalar("mem/torch_peak_reserved_gb", torch.cuda.max_memory_reserved() / 2**30, global_step)
                    host_rss = _host_rss_bytes()
                    if host_rss:
                        writer.add_scalar("mem/host_rss_gb", host_rss / 2**30, global_step)
                    lines.append(
                        f"{'GPU mem:':>{pad}} {_fmt_bytes(gpu_used)} used"
                        f"   (torch reserved {_fmt_bytes(torch.cuda.memory_reserved())},"
                        f" peak {_fmt_bytes(torch.cuda.max_memory_reserved())})"
                    )
                    if host_rss:
                        lines.append(f"{'host RSS:':>{pad}} {_fmt_bytes(host_rss)}")
                    torch.cuda.reset_peak_memory_stats()

                # ─── timing breakdown (--profile_timing) ───
                # Totals accumulated since the last report, so percentages describe this window.
                # Close io/logging before reporting: it must land in this window's totals, and
                # timer.reset() below would otherwise discard the still-open start.
                timer.stop("io/logging")
                if args.profile_timing:
                    rows = timer.rows()
                    window = sum(t for n, t, *_ in rows if not n.startswith("total/"))
                    lines.append(f"{'timing (last %d steps)' % args.logging_interval:^60}")
                    lines.append(f"  {'block':<28}{'total s':>9}{'calls':>7}{'ms/call':>10}{'%':>6}")
                    for name, tot, calls, ms, pct in rows:
                        pct_s = "  --  " if math.isnan(pct) else f"{pct:5.1f}%"
                        lines.append(f"  {name:<28}{tot:9.3f}{calls:7d}{ms:10.3f}{pct_s:>6}")
                    lines.append(f"  {'(measured total)':<28}{window:9.3f}")
                    lines.append(bar)
                    for name, tot, calls, ms, _ in rows:
                        writer.add_scalar(f"time/{name}_ms_per_call", ms, global_step)
                        writer.add_scalar(f"time/{name}_total_s", tot, global_step)
                    timer.reset()

                print("\n".join(lines), flush=True)

        # Periodic checkpoint save (after training starts so model isn't all-zeros)
        if args.save_model and global_step > args.learning_starts:
            # global_step counts learning iterations (+=1 per iter), so an exact modulo fires once
            # per boundary. A `% interval < num_envs` guard would assume transition-counting and
            # dump num_envs consecutive checkpoints per boundary.
            if global_step % args.save_interval == 0:
                ckpt_dir = f"{args.ckpt_dir}/{run_name}"
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = f"{ckpt_dir}/model_{global_step:010d}.pt"
                ckpt = {
                    "actor": actor.state_dict(),
                    "encoder": encoder.state_dict(),
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                    "actor_obs_normalizer": actor_obs_normalizer.state_dict() if args.obs_normalization else None,
                    "critic_obs_normalizer": critic_obs_normalizer.state_dict() if args.obs_normalization else None,
                    "proprio_obs_normalizer": (
                        proprio_obs_normalizer.state_dict() if (args.obs_normalization and has_proprio) else None
                    ),
                    "encoder_optimizer": encoder_optimizer.state_dict(),
                    "q_optimizer": q_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "global_step": global_step,
                    "args": vars(args),
                }
                if args.autotune:
                    ckpt["log_alpha"] = log_alpha.detach()
                    ckpt["a_optimizer"] = a_optimizer.state_dict()
                with timer.block("io/checkpoint"):
                    torch.save(ckpt, ckpt_path)
                    print(f"[ckpt] saved {ckpt_path}")
                    if args.track:
                        wandb.save(ckpt_path, base_path=ckpt_dir, policy="now")

        timer.stop("total/iteration")
        global_step += 1

    # Final checkpoint at end of training
    if args.save_model:
        ckpt_dir = f"{args.ckpt_dir}/{run_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f"{ckpt_dir}/model_final.pt"
        ckpt = {
            "actor": actor.state_dict(),
            "encoder": encoder.state_dict(),
            "critic": critic.state_dict(),
            "critic_target": critic_target.state_dict(),
            "actor_obs_normalizer": actor_obs_normalizer.state_dict() if args.obs_normalization else None,
            "critic_obs_normalizer": critic_obs_normalizer.state_dict() if args.obs_normalization else None,
            "proprio_obs_normalizer": (
                proprio_obs_normalizer.state_dict() if (args.obs_normalization and has_proprio) else None
            ),
            "encoder_optimizer": encoder_optimizer.state_dict(),
            "q_optimizer": q_optimizer.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "global_step": global_step,
            "args": vars(args),
        }
        if args.autotune:
            ckpt["log_alpha"] = log_alpha.detach()
            ckpt["a_optimizer"] = a_optimizer.state_dict()
        torch.save(ckpt, ckpt_path)
        print(f"[ckpt] saved final {ckpt_path}")
        if args.track:
            wandb.save(ckpt_path, base_path=ckpt_dir, policy="now")

    envs.close()
    writer.close()
