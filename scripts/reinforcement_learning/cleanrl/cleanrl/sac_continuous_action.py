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
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
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
    obs_normalization: bool = True
    """use obs normalization"""
    num_updates: int = 1
    """Num updates per step"""
    target_entropy_ratio: float = 1.0
    """ratio of target entropy"""
    use_layer_norm: bool = False
    """use layer norm"""


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
class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256, use_layer_norm=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space["critic"].shape).prod() + np.prod(env.single_action_space.shape),
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.net(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, action_scale=None, action_bias=None, hidden_dim=256, use_layer_norm=False, use_tanh=False, init_trick=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space["policy"].shape).prod(), hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape))
        self.use_tanh = use_tanh

        # Initial exploration tricks
        if init_trick:
            nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
            nn.init.zeros_(self.fc_mean.bias)

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
        mean = self.fc_mean(x)
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
    qf1 = SoftQNetwork(envs, use_layer_norm=args.use_layer_norm).to(device)
    qf2 = SoftQNetwork(envs, use_layer_norm=args.use_layer_norm).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item() * args.target_entropy_ratio
        # log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # alpha = log_alpha.exp().item()
        alpha = torch.tensor(args.alpha)
        log_alpha = torch.tensor([torch.log(alpha)], requires_grad=True, device=device)
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = AsymmetricReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    if args.obs_normalization:
        normalize_obs = actor_obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward

    # Logging stuff
    rewbuffer = deque(maxlen=1000)
    lenbuffer = deque(maxlen=1000)
    successbuffer = deque(maxlen=1000)
    total_episodes = 0
    num_success_episodes_log = 0
    num_episodes_log = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    while global_step < args.total_timesteps:
        global_step += args.num_envs
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = torch.tensor([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            policy_obs = obs["policy"]
            if args.obs_normalization:
                policy_obs = normalize_obs(policy_obs, update=False)

            actions, _, _ = actor.get_action(policy_obs)
            actions = actions.detach()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            context_term = envs.env.env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
            is_env_success = getattr(context_term, "success")
            for i, info in enumerate(infos["final_info"]):
                if info is not None:
                    rewbuffer.extend([info["episode"]["r"]])
                    lenbuffer.extend([info["episode"]["l"]])
                    num_episodes_log += 1
                    num_success_episodes_log += 1 if is_env_success[i] else 0
                    total_episodes += 1

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
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

                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(next_policy_obs)
                    qf1_next_target = qf1_target(next_critic_obs, next_state_actions)
                    qf2_next_target = qf2_target(next_critic_obs, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(critic_obs, data.actions).view(-1)
                qf2_a_values = qf2(critic_obs, data.actions).view(-1)

                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % (args.policy_frequency * args.num_envs) == 0:  # TD 3 Delayed update support
                    # for _ in range(
                    #     args.policy_frequency
                    # ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(policy_obs)
                    qf1_pi = qf1(critic_obs, pi)
                    qf2_pi = qf2(critic_obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

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

                # update the target networks
                if global_step % (args.target_network_frequency * args.num_envs) == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % (25 * args.num_envs) == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
                samples_per_sec = sps * args.num_updates * args.batch_size
                writer.add_scalar("charts/samples_per_sec", samples_per_sec, global_step)

                if len(rewbuffer) > 0:
                    writer.add_scalar("charts/episodic_return", statistics.mean(rewbuffer), global_step)
                    writer.add_scalar("charts/episodic_length", statistics.mean(lenbuffer), global_step)
                
                writer.add_scalar("charts/success_rate", num_success_episodes_log / num_episodes_log, global_step)
                num_success_episodes_log = 0
                num_episodes_log = 0

                writer.add_scalar("charts/num_episodes", total_episodes, global_step)

                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                # ─── pretty console log ───
                elapsed = time.time() - start_time
                progress = global_step / args.total_timesteps
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
                    f"{'qf1 / qf2 value:':>{pad}} {qf1_a_values.mean().item():>10.3f}  /  {qf2_a_values.mean().item():>10.3f}",
                    f"{'qf_loss:':>{pad}} {(qf_loss.item() / 2.0):>10.4f}",
                    f"{'actor_loss:':>{pad}} {actor_loss.item():>10.4f}",
                    f"{'log_pi (mean):':>{pad}} {log_pi.mean().item():>10.3f}",
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
