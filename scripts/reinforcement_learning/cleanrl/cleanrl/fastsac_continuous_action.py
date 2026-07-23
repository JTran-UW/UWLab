# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
from collections import deque
import math
import os
import random
import statistics
import time
from dataclasses import dataclass

import gymnasium as gym
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
    num_steps: int = 1
    """n-step returns"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.125
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 16384
    """the batch size of sample from the reply memory"""
    learning_starts: int = 10
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    alpha_lr: float = 3e-4
    """the learning rate of alpha"""
    policy_frequency: int = 4
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
    obs_normalization: bool = True
    """use obs normalization"""
    num_updates: int = 8
    """Num updates per step"""
    target_entropy_ratio: float = 1.0
    """ratio of target entropy"""
    use_layer_norm: bool = True
    """use layer norm"""
    weight_decay: float = 0.001
    """the weight decay of the optimizer"""
    logging_interval: int = 100
    """the interval to log the metrics"""
    hessian_interval: int = 100 # 0
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


# ALGO LOGIC: initialize agent here:
class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        env,
        hidden_dim=768,
        num_atoms=101,
        v_min=-20.0,
        v_max=20.0,
        use_layer_norm=True
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space["critic"].shape).prod() + np.prod(env.single_action_space.shape),
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms)
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.q_support = torch.linspace(v_min, v_max, num_atoms, device="cuda")

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.net(x)
        return x

    def projection(
        self,
        obs,
        actions,
        rewards,
        bootstrap,
        discount
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

        next_dist = F.softmax(self(obs, actions), dim=1)
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


LOG_STD_MAX = 0
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, action_scale=None, action_bias=None, hidden_dim=512, use_layer_norm=False, use_tanh=False, init_trick=False):
        super().__init__()
        input_dim = np.array(env.single_observation_space["policy"].shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        self.use_layer_norm = use_layer_norm
        self.device = "cuda"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=self.device),
            nn.LayerNorm(hidden_dim, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=self.device),
            nn.LayerNorm(hidden_dim // 2, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=self.device),
            nn.LayerNorm(hidden_dim // 4, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, action_dim, device=self.device),
        )
        self.fc_logstd = nn.Linear(hidden_dim // 4, action_dim, device=self.device)
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)
        self.use_tanh = use_tanh

        # Initial exploration tricks
        if init_trick:
            nn.init.orthogonal_(self.fc_mu.weight, gain=0.01)
            nn.init.zeros_(self.fc_mu.bias)

            std_0 = 0.15
            nn.init.zeros_(self.fc_logstd.weight)
            nn.init.constant_(self.fc_logstd.bias, math.log(std_0))

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

    def forward(self, x):
        x = self.net(x)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
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

    max_action = float(envs.single_action_space.high[0])

    if args.obs_normalization:
        actor_obs_normalizer = EmpiricalNormalization(shape=envs.single_observation_space["policy"].shape, device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=envs.single_observation_space["critic"].shape, device=device)

    actor = Actor(envs, use_tanh=args.use_tanh, use_layer_norm=args.use_layer_norm).to(device)
    qf1 = DistributionalQNetwork(envs, use_layer_norm=args.use_layer_norm).to(device)
    qf2 = DistributionalQNetwork(envs, use_layer_norm=args.use_layer_norm).to(device)
    qf1_target = DistributionalQNetwork(envs).to(device)
    qf2_target = DistributionalQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.AdamW(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.policy_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item() * args.target_entropy_ratio
        # log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # alpha = log_alpha.exp().item()
        alpha = torch.tensor(args.alpha)
        log_alpha = torch.tensor([torch.log(alpha)], requires_grad=True, device=device)
        a_optimizer = optim.AdamW([log_alpha], lr=args.alpha_lr, betas=(0.9, 0.95))
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = AsymmetricReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        n_steps=args.num_steps,
    )
    start_time = time.time()

    if args.obs_normalization:
        normalize_obs = actor_obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward

    # Logging stuff
    rewbuffer = deque(maxlen=1000)
    lenbuffer = deque(maxlen=1000)
    total_episodes = 0
    num_success_episodes_log = 0
    num_episodes_log = 0
    # Rolling moving average, per termination term (e.g. success/time_out/off_table), of the
    # fraction of envs whose most recent completed episode ended via that term. Term names are
    # discovered dynamically from IsaacLab's TerminationManager ("Episode_Termination/<name>" in
    # infos["log"], refreshed whenever any env resets), so this works for any task's termination cfg.
    termination_buffers: dict[str, deque] = {}

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    while global_step < args.num_learning_iterations:
        # ALGO LOGIC: put action logic here
        # if global_step < args.learning_starts:
        #     actions = torch.tensor([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        # else:
        with torch.no_grad():
            policy_obs = obs["policy"]
            if args.obs_normalization:
                policy_obs = normalize_obs(policy_obs, update=False)

        actions, _, _ = actor.get_action(policy_obs)
        actions = actions.detach()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        rewbuffer.extend(infos["ep_reward"].cpu().tolist())
        lenbuffer.extend(infos["ep_length"].cpu().tolist())
        num_episodes_log += dones.int().sum()
        total_episodes += dones.int().sum()
        num_success_episodes_log += infos["ep_success"].sum()
        if dones.any():
            for key, val in infos.get("log", {}).items():
                if key.startswith("Episode_Termination/"):
                    term_name = key[len("Episode_Termination/"):]
                    termination_buffers.setdefault(term_name, deque(maxlen=1000)).append(val)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        real_next_obs[truncations.bool()] = infos["final_obs"]
        rb.add(obs, real_next_obs, actions, rewards, terminations, truncations, infos)
        # import pdb; pdb.set_trace()

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # Snapshot the policy on a reference batch before this epoch's updates, to
            # measure how far the policy moves (KL) over the course of the num_updates
            # gradient steps below.
            with torch.no_grad():
                kl_ref_obs = rb.sample(args.batch_size).policy_observations
                if args.obs_normalization:
                    kl_ref_obs = normalize_obs(kl_ref_obs, update=False)
                old_mean, old_log_std = actor(kl_ref_obs)
                old_mean = old_mean.detach()
                old_log_std = old_log_std.detach()

            for upd_i in range(args.num_updates):
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

                bootstrap = (~data.dones.bool()).float()

                with torch.no_grad():
                    next_state_actions, next_state_log_probs, _ = actor.get_action(next_policy_obs)
                    discount = args.gamma ** data.effective_n_steps

                    qf1_target_dist = qf1_target.projection(
                        next_critic_obs,
                        next_state_actions,
                        data.rewards.squeeze(-1) - discount * bootstrap.squeeze(-1) * log_alpha.exp() * next_state_log_probs.squeeze(-1),
                        bootstrap.squeeze(-1),
                        discount
                    )
                    qf2_target_dist = qf2_target.projection(
                        next_critic_obs,
                        next_state_actions,
                        data.rewards.squeeze(-1) - discount * bootstrap.squeeze(-1) * log_alpha.exp() * next_state_log_probs.squeeze(-1),
                        bootstrap.squeeze(-1),
                        discount,
                    )
                    qf1_target_values = qf1_target.get_value(qf1_target_dist)
                    qf2_target_values = qf2_target.get_value(qf2_target_dist)
                    target_value_max = torch.max(qf1_target_values, qf2_target_values)
                    target_value_min = torch.min(qf1_target_values, qf2_target_values)
                    
                qf1_outputs = qf1(critic_obs, data.actions)
                qf2_outputs = qf2(critic_obs, data.actions)
                qf1_log_probs = F.log_softmax(qf1_outputs, dim=-1)
                qf2_log_probs = F.log_softmax(qf2_outputs, dim=-1)
                qf1_loss = -torch.sum(qf1_target_dist * qf1_log_probs, dim=-1)
                qf2_loss = -torch.sum(qf2_target_dist * qf2_log_probs, dim=-1)
                qf_loss = torch.stack([qf1_loss, qf2_loss]).mean(dim=1).sum(dim=0)

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if args.num_updates > 1:
                    if upd_i % args.policy_frequency == 1:
                        pi, log_pi, _ = actor.get_action(policy_obs)
                        with torch.no_grad():
                            policy_entropy = -log_pi.mean()
                        qf1_pi = qf1(critic_obs, pi)
                        qf2_pi = qf2(critic_obs, pi)
                        qf1_probs = F.softmax(qf1_pi)
                        qf2_probs = F.softmax(qf2_pi)
                        qf1_value = qf1.get_value(qf1_probs)
                        qf2_value = qf2.get_value(qf2_probs)
                        mean_qf_pi = torch.stack([qf1_value, qf2_value]).mean(dim=0)
                        actor_loss = ((alpha * log_pi) - mean_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(policy_obs)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()
                else:
                    raise ValueError("Num updates <=1 is not supported")

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # KL(new policy || old policy) on the reference batch, measured over this epoch's
            # num_updates gradient steps. Closed-form KL between diagonal Gaussians, computed in
            # pre-tanh space (valid regardless of use_tanh, since KL is invariant under a shared
            # deterministic bijection).
            with torch.no_grad():
                new_mean, new_log_std = actor(kl_ref_obs)
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
            if global_step % args.hessian_interval == 0:
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

                h_bootstrap = (~h_data.dones.bool()).float()
                with torch.no_grad():
                    h_next_actions, h_next_log_probs, _ = actor.get_action(h_next_policy_obs)
                    h_discount = args.gamma ** h_data.effective_n_steps
                    qf1_target_dist_h = qf1_target.projection(
                        h_next_critic_obs,
                        h_next_actions,
                        h_data.rewards.squeeze(-1) - h_discount * h_bootstrap.squeeze(-1) * log_alpha.exp() * h_next_log_probs.squeeze(-1),
                        h_bootstrap.squeeze(-1),
                        h_discount,
                    )

                qf1_outputs_h = qf1(h_critic_obs, h_data.actions)
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
