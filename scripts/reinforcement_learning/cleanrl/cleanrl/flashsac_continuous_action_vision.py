# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
from collections import deque
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
import tyro
from torch.utils.tensorboard import SummaryWriter

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cleanrl_utils.buffers import AsymmetricReplayBuffer
from cleanrl_utils.utils import EmpiricalNormalization

from vecenv_wrapper import IsaacLabVectorEnv

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
    action_repeat: int = 2
    """number of env sim-steps each sampled action is repeated for"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    buffer_on_cpu: bool = True
    """store the replay buffer on cpu instead of the compute device"""
    share_policy_critic_obs: bool = False
    """store ONE image stream instead of separate policy/critic copies (~2x less replay memory).
    Only valid when the two obs groups are bitwise identical -- true for the reaching tasks, where
    they wrap the same terms. Verified on the first add(), which raises if they ever differ."""
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
    save_model: bool = False
    """if toggled, periodically save model checkpoints to runs/{run_name}/ (and upload to wandb if --track)"""
    save_interval: int = 1_000_000
    """checkpoint save interval, in global_step units (transitions)"""
    use_tanh: bool = False
    """use tanh layer after policy"""
    share_encoder: bool = True
    """tie actor/qf1/qf2 depth-encoder weights into one shared encoder (DrQ-v2 style); actor
    gradients are detached from it so only the critic updates the shared encoder"""
    obs_normalization: bool = True
    """use obs normalization"""
    drq_aug: bool = False
    """enable DrQ-style random-shift image augmentation on replay-buffer samples only, not on
    action-collection (arxiv.org/abs/2004.13649). Applied after obs normalization, at the encoder input."""
    drq_pad: int = 4
    """DrQ random-shift pad (pixels): replicate-pad each spatial side by this, then take a random
    H x W crop, yielding shifts of +/- drq_pad"""
    drq_k: int = 2
    """DrQ number of target augmentations K: K augmented next-obs target distributions, averaged
    into the target distribution used for the critic update"""
    drq_m: int = 2
    """DrQ number of critic-update augmentations M: M augmented current-obs critic losses, averaged
    into the critic loss"""
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
    enable_hessian_diag: bool = False
    """compute the qf1 critic-loss Hessian diagnostic (arxiv.org/abs/2509.25174). Off by default:
    it costs an extra replay-buffer sample plus hessian_lanczos_iters double-backward passes, and is
    a diagnostic, not part of training. Turn on only when specifically investigating conditioning."""
    hessian_interval: int = 100
    """interval (in global_step) at which to estimate the qf1 critic loss Hessian's top/bottom
    eigenvalues and condition number, per arxiv.org/abs/2509.25174. Expensive (double-backward
    power/Lanczos iterations), so this runs far less often than logging_interval."""
    hessian_lanczos_iters: int = 20
    """number of Lanczos iterations used to estimate the extremal Hessian eigenvalues"""


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


class UnitLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        nn.init.orthogonal_(self.linear.weight, gain=1)
    
    def forward(self, x):
        return self.linear(x)

    def normalize_parameters(self):
        self.linear.weight.copy_(F.normalize(self.linear.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    def __init__(self, input_dim, momentum=0.01, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x, training: bool):
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def normalize_parameters(self):
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.RMSNorm(input_dim)
    
    def forward(self, x):
        return self.norm(x)
    
    def normalize_parameters(self):
        scale = self.norm.weight
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.norm.weight.copy_(scale * norm_factor)


class DepthCNNEncoder(nn.Module):
    """DrQ-v2-style encoder: num_layers conv layers (first stride 2, rest stride 1) then a linear
    bottleneck to feature_dim, followed by LayerNorm + Tanh (matches FlashSAC paper Table 11)."""

    def __init__(self, in_channels, input_hw, feature_dim=50, num_layers=3, num_channels=32):
        super().__init__()
        conv_layers = []
        c_in = in_channels
        for i in range(num_layers):
            stride = 2 if i == 0 else 1
            conv_layers += [nn.Conv2d(c_in, num_channels, kernel_size=3, stride=stride), nn.ReLU()]
            c_in = num_channels
        self.convnet = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_hw)
            flatten_dim = self.convnet(dummy).flatten(1).shape[1]

        self.trunk = nn.Sequential(nn.Linear(flatten_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

    def forward(self, x):
        h = self.convnet(x)
        h = h.flatten(1)
        return self.trunk(h)


def random_shift_aug(obs_flat, obs_shape, pad):
    """DrQ random-shift augmentation (arxiv.org/abs/2004.13649).

    Takes flattened obs `(B, prod(obs_shape))`, reshapes to an image stack `(B, channels, H, W)`
    (all leading obs dims -- history and channels -- fold into `channels`), replicate-pads each
    spatial side by `pad`, then takes an independent random H x W crop per batch element (all
    channels of a sample share one shift). Returns flattened obs of the original shape.
    """
    B = obs_flat.shape[0]
    H, W = int(obs_shape[-2]), int(obs_shape[-1])
    channels = int(np.prod(obs_shape[:-2]))
    x = obs_flat.reshape(B, channels, H, W)
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    h0 = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    w0 = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = h0[:, None] + torch.arange(H, device=x.device)[None, :]  # (B, H)
    cols = w0[:, None] + torch.arange(W, device=x.device)[None, :]  # (B, W)
    b_idx = torch.arange(B, device=x.device)[:, None, None, None]
    c_idx = torch.arange(channels, device=x.device)[None, :, None, None]
    x = x[b_idx, c_idx, rows[:, None, :, None], cols[:, None, None, :]]  # (B, channels, H, W)
    return x.reshape(B, -1)


class FlashSACBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = 4 * input_dim
        self.linear1 = UnitLinear(input_dim, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim)
        self.linear2 = UnitLinear(hidden_dim, input_dim)
        self.norm2 = UnitBatchNorm(input_dim)
    
    def forward(self, x, training: bool):
        x = F.relu(self.norm1(self.linear1(x), training=training))
        x = F.relu(self.norm2(self.linear2(x), training=training))
        return x
    
    def normalize_parameters(self):
        self.linear1.normalize_parameters()
        self.linear2.normalize_parameters()
        self.norm1.normalize_parameters()
        self.norm2.normalize_parameters()


# ALGO LOGIC: initialize agent here:
class FlashSACQNetwork(nn.Module):
    def __init__(
        self,
        env,
        hidden_dim=256,
        num_atoms=101,
        v_min=-20.0,
        v_max=20.0,
        num_blocks=2,
        encoder=None,
        proprio_dim=0,
    ):
        super().__init__()

        obs_shape = env.single_observation_space["critic"].shape  # (history, C, H, W)
        img_channels = obs_shape[0] * obs_shape[1]
        self.obs_hw = obs_shape[2:]
        action_dim = np.prod(env.single_action_space.shape)
        self.proprio_dim = proprio_dim

        self.encoder = encoder if encoder is not None else DepthCNNEncoder(
            in_channels=img_channels, input_hw=obs_shape[2:], feature_dim=50
        )
        # Proprioception is concatenated onto the encoder's feature vector (it bypasses the CNN),
        # so the trunk widens by proprio_dim. proprio_dim=0 reproduces the vision-only network.
        trunk_in = 50 + proprio_dim + action_dim
        self.norm1 = UnitBatchNorm(trunk_in)
        self.linear1 = UnitLinear(trunk_in, hidden_dim)
        self.blocks = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.norm2 = UnitRMSNorm(hidden_dim)
        self.linear2 = UnitLinear(hidden_dim, num_atoms, bias=True)

        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.q_support = torch.linspace(v_min, v_max, num_atoms, device="cuda")

    def forward(self, x, a, training: bool, proprio: torch.Tensor = None):
        x = x.reshape(x.shape[0], -1, *self.obs_hw)
        x = self.encoder(x)
        if self.proprio_dim > 0:
            x = torch.cat([x, proprio], 1)
        x = torch.cat([x, a], 1)
        x = self.linear1(self.norm1(x, training=training))
        for block in self.blocks:
            x = block(x, training=training) + x
        x = self.linear2(self.norm2(x))
        return x

    def projection(
        self,
        obs,
        actions,
        rewards,
        bootstrap,
        discount,
        training: bool,
        next_dist: torch.Tensor = None,
        proprio: torch.Tensor = None,
    ):
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        # import pdb; pdb.set_trace()

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * self.q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)

        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        if next_dist is None:
            next_dist = F.softmax(self(obs, actions, training=training, proprio=proprio), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        # Additional safety check for indices
        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        max_index = proj_dist.numel() - 1

        lower_indices = torch.clamp(lower_indices, 0, max_index)
        upper_indices = torch.clamp(upper_indices, 0, max_index)

        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))
        return proj_dist
    
    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=-1)

    @torch.no_grad()
    def normalize_parameters(self):
        self.norm1.normalize_parameters()
        self.linear1.normalize_parameters()
        for block in self.blocks:
            block.normalize_parameters()
        self.norm2.normalize_parameters()
        self.linear2.normalize_parameters()


LOG_STD_MAX = 0
LOG_STD_MIN = -5


class FlashSACActor(nn.Module):
    def __init__(self,
        env,
        action_scale=None,
        action_bias=None,
        hidden_dim=128,
        num_blocks=2,
        use_tanh=False,
        encoder=None,
        detach_encoder=False,
        proprio_dim=0,
    ):
        super().__init__()
        obs_shape = env.single_observation_space["policy"].shape  # (history, C, H, W)
        img_channels = obs_shape[0] * obs_shape[1]
        self.obs_hw = obs_shape[2:]
        action_dim = np.prod(env.single_action_space.shape)
        self.proprio_dim = proprio_dim

        self.encoder = encoder if encoder is not None else DepthCNNEncoder(
            in_channels=img_channels, input_hw=obs_shape[2:], feature_dim=50
        )
        self.detach_encoder = detach_encoder
        # Proprioception is concatenated onto the encoder features (bypassing the CNN); the actor
        # trunk widens accordingly. proprio_dim=0 reproduces the vision-only network.
        trunk_in = 50 + proprio_dim
        self.norm1 = UnitBatchNorm(trunk_in)
        self.linear1 = UnitLinear(trunk_in, hidden_dim)
        self.blocks = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.norm2 = UnitRMSNorm(hidden_dim)

        self.fc_mu = UnitLinear(hidden_dim, action_dim, bias=True)
        self.fc_logstd = UnitLinear(hidden_dim, action_dim, bias=True)
        self.use_tanh = use_tanh

        # action rescaling
        if action_scale != None and action_bias != None:
            action_scale = action_scale
            action_bias = action_bias
        elif env.single_action_space.high[0] == torch.inf:
            action_scale = 1.0
            action_bias = 0.0
        else:
            action_scale = (env.single_action_space.high - env.single_action_space.low) / 2.0
            action_bias = (env.single_action_space.high + env.single_action_space.low) / 2.0

        self.register_buffer(
            "action_scale",
            torch.tensor(
                action_scale,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                action_bias,
                dtype=torch.float32,
            ),
        )

    def forward(self, x, training: bool, proprio: torch.Tensor = None):
        x = x.reshape(x.shape[0], -1, *self.obs_hw)
        x = self.encoder(x)
        if self.detach_encoder:
            x = x.detach()
        # Concatenated after the detach: proprio is a direct input, not a shared-encoder feature, so
        # it must stay on the actor's gradient path even when the encoder is critic-owned.
        if self.proprio_dim > 0:
            x = torch.cat([x, proprio], 1)
        x = self.linear1(self.norm1(x, training=training))
        for block in self.blocks:
            x = block(x, training=training) + x
        x = self.norm2(x)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, training: bool, proprio: torch.Tensor = None):
        mean, log_std = self(x, training=training, proprio=proprio)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        if self.use_tanh:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = x_t
            log_prob = normal.log_prob(x_t)
            log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean
    
    @torch.no_grad()
    def normalize_parameters(self):
        self.norm1.normalize_parameters()
        self.linear1.normalize_parameters()
        for block in self.blocks:
            block.normalize_parameters()
        self.norm2.normalize_parameters()
        self.fc_mu.normalize_parameters()
        self.fc_logstd.normalize_parameters()


def _build_truncated_zeta_cdf(mu: float, max_n: int, device) -> torch.Tensor:
    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    return torch.cumsum(pmf, dim=0)


def _sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)


def _sample_rollout_action(actor, x, noise, cur_count, cur_n, zeta_cdf, training: bool, proprio=None):
    mean, log_std = actor(x, training=training, proprio=proprio)
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


def _depth_frame_to_uint8(frame: torch.Tensor, max_m: float) -> np.ndarray:
    """Convert a single (H, W) depth frame (meters) to a grayscale uint8 image for video export."""
    frame = (frame.clamp(0.0, max_m) / max_m * 255.0).to(torch.uint8)
    return frame.cpu().numpy()


def unique_params(*modules):
    """Parameters across modules, de-duplicated by identity (needed when modules share a submodule)."""
    seen = set()
    params = []
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def unique_param_pairs(*module_pairs):
    """(online, target) parameter pairs across (online, target) module pairs, de-duplicated by
    the online param's identity (needed when e.g. qf1/qf2 share an encoder)."""
    seen = set()
    pairs = []
    for online, target in module_pairs:
        for p, tp in zip(online.parameters(), target.parameters()):
            if id(p) not in seen:
                seen.add(id(p))
                pairs.append((p, tp))
    return pairs


def _make_hvp_fn(loss: torch.Tensor, params: list[torch.Tensor]):
    """Build a Hessian-vector-product closure for `loss`'s Hessian w.r.t. `params`.

    Computes the first-order gradient graph once (with create_graph=True); each call to the
    returned closure does one additional (cheap, retained-graph) backward pass to get H @ vec,
    rather than rebuilding the forward/first-backward graph per Hessian-vector product.
    """
    flat_grad = torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, params, create_graph=True)])

    def hvp(vec: torch.Tensor) -> torch.Tensor:
        grad_dot_v = torch.dot(flat_grad, vec)
        hvp_grads = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
        return torch.cat([h.reshape(-1) for h in hvp_grads]).detach()

    return hvp


def lanczos_extreme_eigenvalues(loss: torch.Tensor, params: list[torch.Tensor], num_iters: int, device) -> tuple[float, float]:
    """Estimate the largest and smallest eigenvalues of the Hessian of `loss` w.r.t. `params`.

    Uses the Lanczos algorithm (Golub & Welsch 1969), the same core primitive as the stochastic
    Lanczos quadrature method in arxiv.org/abs/2509.25174, restricted here to just the extremal
    Ritz values (top/bottom eigenvalues of the small resulting tridiagonal matrix) rather than
    the full spectral density. Uses only Hessian-vector products (no explicit Hessian matrix).
    Full reorthogonalization against all prior Lanczos vectors, cheap since num_iters is small.
    """
    hvp = _make_hvp_fn(loss, params)
    n_params = sum(p.numel() for p in params)

    v = torch.randn(n_params, device=device)
    v = v / v.norm()
    vs = [v]
    alphas: list[float] = []
    betas: list[float] = []
    v_prev = torch.zeros_like(v)
    beta_prev = 0.0

    for i in range(num_iters):
        w = hvp(vs[-1])
        alpha = torch.dot(w, vs[-1])
        w = w - alpha * vs[-1] - beta_prev * v_prev
        for vj in vs:  # full reorthogonalization
            w = w - torch.dot(w, vj) * vj
        beta = w.norm()
        alphas.append(alpha.item())
        if beta.item() < 1e-8 or i == num_iters - 1:
            betas.append(0.0)
            break
        betas.append(beta.item())
        v_prev = vs[-1]
        beta_prev = beta
        vs.append(w / beta)

    m = len(alphas)
    T = torch.zeros(m, m, device=device)
    for i in range(m):
        T[i, i] = alphas[i]
        if i < m - 1 and betas[i] != 0.0:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    ritz = torch.linalg.eigvalsh(T)
    return ritz[-1].item(), ritz[0].item()


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

    # env setup
    envs = IsaacLabVectorEnv(args.env_id, args.num_envs, launcher_args=launcher_args)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    policy_obs_shape = envs.single_observation_space["policy"].shape  # (history, C, H, W)
    critic_obs_shape = envs.single_observation_space["critic"].shape  # (history, C, H, W)

    # Optional proprioception group (e.g. the grayscale reaching task). Tasks that only define
    # policy/critic leave proprio_dim at 0, which reproduces the vision-only networks exactly.
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

    qf1 = FlashSACQNetwork(envs, proprio_dim=proprio_dim).to(device)
    qf2 = FlashSACQNetwork(
        envs, encoder=(qf1.encoder if args.share_encoder else None), proprio_dim=proprio_dim
    ).to(device)
    actor = FlashSACActor(
        envs, use_tanh=args.use_tanh, encoder=(qf1.encoder if args.share_encoder else None),
        detach_encoder=args.share_encoder, proprio_dim=proprio_dim,
    ).to(device)
    zeta_cdf = _build_truncated_zeta_cdf(args.noise_repeat_zeta_mu, args.noise_repeat_zeta_max, device)
    noise_repeat_n = torch.ones((), dtype=torch.int32, device=device)
    noise_repeat_count = torch.zeros((), dtype=torch.int32, device=device)
    cached_noise = torch.randn((args.num_envs,) + envs.single_action_space.shape, device=device)
    qf1_target = FlashSACQNetwork(envs, proprio_dim=proprio_dim).to(device)
    qf2_target = FlashSACQNetwork(
        envs, encoder=(qf1_target.encoder if args.share_encoder else None), proprio_dim=proprio_dim
    ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.AdamW(unique_params(qf1, qf2), lr=args.q_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if args.share_encoder:
        actor_params = [p for n, p in actor.named_parameters() if not n.startswith("encoder.")]
    else:
        actor_params = list(actor.parameters())
    actor_optimizer = optim.AdamW(actor_params, lr=args.policy_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # Normalize after init
    actor.normalize_parameters()
    qf1.normalize_parameters()
    qf2.normalize_parameters()
    qf1_target.normalize_parameters()
    qf2_target.normalize_parameters()

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = 0.5 * torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item() * torch.log(torch.tensor(2 * torch.pi * torch.e * args.temp_target_sigma ** 2)).item()
        # log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # alpha = log_alpha.exp().item()
        alpha = torch.tensor(args.alpha)
        log_alpha = torch.tensor([torch.log(alpha)], requires_grad=True, device=device)
        a_optimizer = optim.AdamW([log_alpha], lr=args.alpha_lr, betas=(0.9, 0.95))
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    buffer_device = torch.device("cpu") if args.buffer_on_cpu else device
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
    start_time = time.time()

    if args.obs_normalization:
        normalize_obs = actor_obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward

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

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    while global_step < args.num_learning_iterations:
        with torch.no_grad():
            policy_obs = obs["policy"].reshape(obs["policy"].shape[0], -1)
            if args.obs_normalization:
                policy_obs = normalize_obs(policy_obs, update=False)

        with torch.no_grad():
            rollout_proprio = get_proprio(obs, update=False)
            actions, cached_noise, noise_repeat_count, noise_repeat_n = _sample_rollout_action(
                actor, policy_obs, cached_noise, noise_repeat_count, noise_repeat_n, zeta_cdf,
                training=False, proprio=rollout_proprio,
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        rewards = None
        for _ in range(args.action_repeat):
            next_obs, step_rewards, terminations, truncations, infos = envs.step(actions)
            rewards = step_rewards if rewards is None else rewards + step_rewards
        dones = terminations | truncations

        # env-0 depth video: capture the most-recent frame in the history stack, flush to disk
        # as an mp4 whenever env 0's episode ends.
        if args.save_depth_video:
            depth_video_frames.append(_depth_frame_to_uint8(next_obs["policy"][0, -1, 0], args.depth_video_max_m))
            if dones[0]:
                video_path = os.path.join(depth_video_dir, f"episode_{depth_video_episode:06d}.mp4")
                imageio.mimsave(video_path, depth_video_frames, fps=args.depth_video_fps)
                depth_video_episode += 1
                depth_video_frames = []

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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
                with torch.no_grad():
                    kl_ref_sample = rb.sample(args.batch_size)
                    kl_ref_obs = kl_ref_sample.policy_observations
                    if args.obs_normalization:
                        kl_ref_obs = normalize_obs(kl_ref_obs, update=False)
                    kl_ref_proprio = norm_proprio(kl_ref_sample.proprio_observations, update=False)
                    old_mean, old_log_std = actor(kl_ref_obs, training=False, proprio=kl_ref_proprio)
                    old_mean = old_mean.detach()
                    old_log_std = old_log_std.detach()

            for upd_i in range(n_updates_this_step):
                data = rb.sample(args.batch_size)

                policy_obs = data.policy_observations
                next_policy_obs = data.next_policy_observations
                critic_obs = data.critic_observations
                next_critic_obs = data.next_critic_observations

                if args.obs_normalization:
                    policy_obs = normalize_obs(data.policy_observations)
                    next_policy_obs = normalize_obs(data.next_policy_observations)
                    critic_obs = normalize_critic_obs(data.critic_observations)
                    next_critic_obs = normalize_critic_obs(data.next_critic_observations)

                # Proprio is never image-augmented -- DrQ's random shift applies to pixels only.
                proprio = norm_proprio(data.proprio_observations, update=True)
                next_proprio = norm_proprio(data.next_proprio_observations, update=True)

                bootstrap = (~data.dones.bool()).float()

                with torch.no_grad():
                    discount = args.gamma ** data.effective_n_steps

                if args.drq_aug:
                    # DrQ: build the target distribution by averaging over K augmentations of the
                    # next obs. Each k draws its own random shift for next_policy_obs (used to sample
                    # the target action) and next_critic_obs (fed to the target critics), then does the
                    # usual clipped-double-Q min-selection at the distribution level. The K target
                    # distributions are averaged into a single categorical target.
                    with torch.no_grad():
                        qf_target_dist = torch.zeros(args.batch_size, qf1_target.num_atoms, device=device)
                        for _ in range(args.drq_k):
                            next_policy_aug = random_shift_aug(next_policy_obs, policy_obs_shape, args.drq_pad)
                            next_critic_aug = random_shift_aug(next_critic_obs, critic_obs_shape, args.drq_pad)
                            next_state_actions, next_state_log_probs, _ = actor.get_action(
                                next_policy_aug, training=False, proprio=next_proprio
                            )
                            reward_term = (
                                data.rewards.squeeze(-1)
                                - discount * bootstrap.squeeze(-1) * log_alpha.exp() * next_state_log_probs.squeeze(-1)
                            )
                            qf1_next_dist = F.softmax(
                                qf1_target(next_critic_aug, next_state_actions, training=True, proprio=next_proprio), dim=-1
                            )
                            qf2_next_dist = F.softmax(
                                qf2_target(next_critic_aug, next_state_actions, training=True, proprio=next_proprio), dim=-1
                            )
                            qf1_target_dist = qf1_target.projection(
                                next_critic_aug, next_state_actions, reward_term,
                                bootstrap.squeeze(-1), discount, training=True, next_dist=qf1_next_dist,
                                proprio=next_proprio,
                            )
                            qf2_target_dist = qf2_target.projection(
                                next_critic_aug, next_state_actions, reward_term,
                                bootstrap.squeeze(-1), discount, training=True, next_dist=qf2_next_dist,
                                proprio=next_proprio,
                            )
                            qf1_target_values = qf1_target.get_value(qf1_target_dist)
                            qf2_target_values = qf2_target.get_value(qf2_target_dist)
                            use_qf1 = (qf1_target_values <= qf2_target_values).unsqueeze(-1)
                            qf_target_dist = qf_target_dist + torch.where(use_qf1, qf1_target_dist, qf2_target_dist)
                        qf_target_dist = qf_target_dist / args.drq_k

                    # DrQ: average the critic loss over M augmentations of the current critic obs,
                    # each scored against the same K-averaged target distribution.
                    qf1_loss = torch.zeros(args.batch_size, device=device)
                    qf2_loss = torch.zeros(args.batch_size, device=device)
                    for _ in range(args.drq_m):
                        critic_aug = random_shift_aug(critic_obs, critic_obs_shape, args.drq_pad)
                        qf1_log_probs = F.log_softmax(
                            qf1(critic_aug, data.actions, training=True, proprio=proprio), dim=-1
                        )
                        qf2_log_probs = F.log_softmax(
                            qf2(critic_aug, data.actions, training=True, proprio=proprio), dim=-1
                        )
                        qf1_loss = qf1_loss - torch.sum(qf_target_dist * qf1_log_probs, dim=-1)
                        qf2_loss = qf2_loss - torch.sum(qf_target_dist * qf2_log_probs, dim=-1)
                    qf1_loss = qf1_loss / args.drq_m
                    qf2_loss = qf2_loss / args.drq_m
                    qf_loss = torch.stack([qf1_loss, qf2_loss]).mean(dim=1).sum(dim=0)
                else:
                    with torch.no_grad():
                        next_state_actions, next_state_log_probs, _ = actor.get_action(
                            next_policy_obs, training=False, proprio=next_proprio
                        )

                    critic_obs_all = torch.cat([critic_obs, next_critic_obs], dim=0)
                    actions_all = torch.cat([data.actions, next_state_actions], dim=0)
                    # Mirror the current/next stacking used for obs+actions so the batched forward
                    # keeps proprio aligned with its own half.
                    proprio_all = torch.cat([proprio, next_proprio], dim=0) if has_proprio else None

                    with torch.no_grad():
                        qf1_target_outputs_all = qf1_target(critic_obs_all, actions_all, training=True, proprio=proprio_all)
                        qf2_target_outputs_all = qf2_target(critic_obs_all, actions_all, training=True, proprio=proprio_all)
                        qf1_target_next_dist = F.softmax(qf1_target_outputs_all, dim=-1).chunk(2, dim=0)[1]
                        qf2_target_next_dist = F.softmax(qf2_target_outputs_all, dim=-1).chunk(2, dim=0)[1]

                        qf1_target_dist = qf1_target.projection(
                            next_critic_obs,
                            next_state_actions,
                            data.rewards.squeeze(-1) - discount * bootstrap.squeeze(-1) * log_alpha.exp() * next_state_log_probs.squeeze(-1),
                            bootstrap.squeeze(-1),
                            discount,
                            training=True,
                            next_dist=qf1_target_next_dist,
                            proprio=next_proprio,
                        )
                        qf2_target_dist = qf2_target.projection(
                            next_critic_obs,
                            next_state_actions,
                            data.rewards.squeeze(-1) - discount * bootstrap.squeeze(-1) * log_alpha.exp() * next_state_log_probs.squeeze(-1),
                            bootstrap.squeeze(-1),
                            discount,
                            training=True,
                            next_dist=qf2_target_next_dist,
                            proprio=next_proprio,
                        )
                        qf1_target_values = qf1_target.get_value(qf1_target_dist)
                        qf2_target_values = qf2_target.get_value(qf2_target_dist)
                        use_qf1 = (qf1_target_values <= qf2_target_values).unsqueeze(-1)
                        qf_target_dist = torch.where(use_qf1, qf1_target_dist, qf2_target_dist)

                    qf1_outputs_all = qf1(critic_obs_all, actions_all, training=True, proprio=proprio_all)
                    qf2_outputs_all = qf2(critic_obs_all, actions_all, training=True, proprio=proprio_all)
                    qf1_outputs = qf1_outputs_all.chunk(2, dim=0)[0]
                    qf2_outputs = qf2_outputs_all.chunk(2, dim=0)[0]
                    qf1_log_probs = F.log_softmax(qf1_outputs, dim=-1)
                    qf2_log_probs = F.log_softmax(qf2_outputs, dim=-1)
                    qf1_loss = -torch.sum(qf_target_dist * qf1_log_probs, dim=-1)
                    qf2_loss = -torch.sum(qf_target_dist * qf2_log_probs, dim=-1)
                    qf_loss = torch.stack([qf1_loss, qf2_loss]).mean(dim=1).sum(dim=0)

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if update_step % args.policy_frequency == 0:
                    if args.drq_aug:
                        # DrQ augments the policy update too: single random shift of the obs the
                        # actor and the Q-value-for-actor see.
                        actor_policy_obs = random_shift_aug(policy_obs, policy_obs_shape, args.drq_pad)
                        actor_critic_obs = random_shift_aug(critic_obs, critic_obs_shape, args.drq_pad)
                    else:
                        actor_policy_obs = policy_obs
                        actor_critic_obs = critic_obs
                    pi, log_pi, _ = actor.get_action(actor_policy_obs, training=True, proprio=proprio)
                    with torch.no_grad():
                        policy_entropy = -log_pi.mean()
                    qf1_pi = qf1(actor_critic_obs, pi, training=False, proprio=proprio)
                    qf2_pi = qf2(actor_critic_obs, pi, training=False, proprio=proprio)
                    qf1_probs = F.softmax(qf1_pi, dim=-1)
                    qf2_probs = F.softmax(qf2_pi, dim=-1)
                    qf1_value = qf1.get_value(qf1_probs)
                    qf2_value = qf2.get_value(qf2_probs)
                    min_qf_pi = torch.min(qf1_value, qf2_value)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(actor_policy_obs, training=False, proprio=proprio)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                update_step += 1

                # Normalize the params
                actor.normalize_parameters()
                qf1.normalize_parameters()
                qf2.normalize_parameters()

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in unique_param_pairs((qf1, qf1_target), (qf2, qf2_target)):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
        
            # KL(new policy || old policy) on the reference batch, measured over this epoch's
            # num_updates gradient steps. Closed-form KL between diagonal Gaussians, computed in
            # pre-tanh space (valid regardless of use_tanh, since KL is invariant under a shared
            # deterministic bijection).
            if args.enable_kl_diag:
                with torch.no_grad():
                    new_mean, new_log_std = actor(kl_ref_obs, training=False, proprio=kl_ref_proprio)
                    old_std = old_log_std.exp()
                    new_std = new_log_std.exp()
                    kl_old_policy = (
                        old_log_std - new_log_std
                        + (new_std.pow(2) + (new_mean - old_mean).pow(2)) / (2.0 * old_std.pow(2))
                        - 0.5
                    ).sum(dim=-1).mean()

            # qf1 critic-loss Hessian top/bottom eigenvalues + condition number (arxiv.org/abs/2509.25174).
            # Expensive (double-backward Lanczos), so this runs on its own coarse interval, on a
            # fresh replay-buffer batch, and is NOT part of the actual training update.
            if args.enable_hessian_diag and global_step % args.hessian_interval == 0:
                h_data = rb.sample(args.batch_size)
                h_policy_obs = h_data.policy_observations
                h_next_policy_obs = h_data.next_policy_observations
                h_critic_obs = h_data.critic_observations
                h_next_critic_obs = h_data.next_critic_observations
                if args.obs_normalization:
                    h_policy_obs = normalize_obs(h_policy_obs, update=False)
                    h_next_policy_obs = normalize_obs(h_next_policy_obs, update=False)
                    h_critic_obs = normalize_critic_obs(h_critic_obs, update=False)
                    h_next_critic_obs = normalize_critic_obs(h_next_critic_obs, update=False)

                h_proprio = norm_proprio(h_data.proprio_observations, update=False)
                h_next_proprio = norm_proprio(h_data.next_proprio_observations, update=False)

                h_bootstrap = (~h_data.dones.bool()).float()
                with torch.no_grad():
                    h_next_actions, h_next_log_probs, _ = actor.get_action(
                        h_next_policy_obs, training=False, proprio=h_next_proprio
                    )
                    h_discount = args.gamma ** h_data.effective_n_steps
                    qf1_target_dist_h = qf1_target.projection(
                        h_next_critic_obs,
                        h_next_actions,
                        h_data.rewards.squeeze(-1) - h_discount * h_bootstrap.squeeze(-1) * log_alpha.exp() * h_next_log_probs.squeeze(-1),
                        h_bootstrap.squeeze(-1),
                        h_discount,
                        training=True,
                        proprio=h_next_proprio,
                    )

                qf1_outputs_h = qf1(h_critic_obs, h_data.actions, training=False, proprio=h_proprio)
                qf1_log_probs_h = F.log_softmax(qf1_outputs_h, dim=-1)
                qf1_loss_h = (-torch.sum(qf1_target_dist_h * qf1_log_probs_h, dim=-1)).mean()

                qf1_params = list(qf1.parameters())
                hess_lambda_max, hess_lambda_min = lanczos_extreme_eigenvalues(
                    qf1_loss_h, qf1_params, num_iters=args.hessian_lanczos_iters, device=device
                )
                hess_cond_number = abs(hess_lambda_max) / max(abs(hess_lambda_min), 1e-12)

                # log10 versions for log-scale plotting: these metrics (per arxiv.org/abs/2509.25174)
                # commonly span several orders of magnitude, so a linear-axis chart of the raw value
                # is hard to read. top_eig can in principle be <= 0 (non-convex loss), so we take
                # log10 of its magnitude and clamp both away from 0 to avoid log(0).
                writer.add_scalar("metrics/critic_hessian_top_eig_log10", math.log10(max(abs(hess_lambda_max), 1e-12)), global_step)
                writer.add_scalar("metrics/critic_hessian_cond_number_log10", math.log10(max(hess_cond_number, 1e-12)), global_step)

            if global_step % args.logging_interval == 0:
                writer.add_scalar("losses/qf1_values", qf1_target_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_target_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.mean().item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("metrics/kl_divergence", kl_old_policy.item(), global_step)
                writer.add_scalar("metrics/policy_entropy", policy_entropy.item(), global_step)
                sps = int(global_step / (time.time() - start_time))
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
                    f"{'SPS:':>{pad}} {sps:,d}",
                    f"{'episodes done:':>{pad}} {total_episodes:,d}",
                    f"{'mean episodic return:':>{pad}} {ep_ret:>10.3f}",
                    f"{'mean episode length:':>{pad}} {ep_len:>10.2f}",
                    f"{'qf1 / qf2 value:':>{pad}} {qf1_target_values.mean().item():>10.3f}  /  {qf2_target_values.mean().item():>10.3f}",
                    f"{'qf_loss:':>{pad}} {(qf_loss.item() / 2.0):>10.4f}",
                    f"{'actor_loss:':>{pad}} {actor_loss.item():>10.4f}",
                    f"{'log_pi (mean):':>{pad}} {log_pi.mean().item():>10.3f}",
                    f"{'kl_divergence:':>{pad}} {kl_old_policy.item():>10.5f}",
                    f"{'policy_entropy:':>{pad}} {policy_entropy.item():>10.4f}",
                    f"{'alpha:':>{pad}} {alpha:>10.4f}"
                    + (f"   alpha_loss={alpha_loss.item():.4f}" if args.autotune else ""),
                    bar,
                ]

                print("\n".join(lines), flush=True)

        # Periodic checkpoint save (after training starts so model isn't all-zeros)
        if args.save_model and global_step > args.learning_starts:
            steps_since_last = global_step % args.save_interval
            if steps_since_last < args.num_envs:  # crossed a save_interval boundary this iter
                ckpt_dir = f"runs/{run_name}"
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = f"{ckpt_dir}/model_{global_step:010d}.pt"
                ckpt = {
                    "actor": actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "qf1_target": qf1_target.state_dict(),
                    "qf2_target": qf2_target.state_dict(),
                    "actor_obs_normalizer": actor_obs_normalizer.state_dict() if args.obs_normalization else None,
                    "critic_obs_normalizer": critic_obs_normalizer.state_dict() if args.obs_normalization else None,
                    "proprio_obs_normalizer": (
                        proprio_obs_normalizer.state_dict() if (args.obs_normalization and has_proprio) else None
                    ),
                    "q_optimizer": q_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "global_step": global_step,
                    "args": vars(args),
                }
                if args.autotune:
                    ckpt["log_alpha"] = log_alpha.detach()
                    ckpt["a_optimizer"] = a_optimizer.state_dict()
                torch.save(ckpt, ckpt_path)
                print(f"[ckpt] saved {ckpt_path}")
                if args.track:
                    wandb.save(ckpt_path, base_path=ckpt_dir, policy="now")
        
        global_step += 1

    # Final checkpoint at end of training
    if args.save_model:
        ckpt_dir = f"runs/{run_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f"{ckpt_dir}/model_final.pt"
        ckpt = {
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
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
