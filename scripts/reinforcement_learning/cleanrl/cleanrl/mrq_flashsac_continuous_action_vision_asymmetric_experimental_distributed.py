# FlashSAC + MR.Q model-based auxiliary loss (arxiv.org/abs/2501.16142), ASYMMETRIC variant.
#
# Actor and critic see different observations:
#   actor  : three-camera grayscale + proprioception, encoded by MR.Q's f (CNN + fusion MLP)
#   critic : the non-privileged full state vector, straight into the FlashSAC C51 block
#
# Only the ACTOR branch is MR.Q. image+proprio -> f -> zs -> g -> zsa -> a linear head predicting
# the next latent, reward and done. That model-based loss is the ONLY thing that trains the vision
# encoder -- per the paper, "the value and policy networks are trained independently from the
# encoders", so the actor reads zs detached and never back-propagates into f. zsa exists purely to
# drive the auxiliary loss; nothing downstream consumes it.
#
# The critic is plain state-based FlashSAC (cf. flashsac_continuous_action.py): it concatenates the
# state observation with the action and runs the C51 trunk, trained by the distributional TD loss
# alone. There is no state encoder and no shared encoder, so the policy/critic shape assertion of
# the symmetric script does not apply -- the streams differ by design.
#
# DISTRIBUTED (multi-GPU) variant of the experimental testbed. Launch with torchrun, one process
# per GPU; docker/cluster/run_singularity.sh already does exactly that and appends --distributed.
#
# Data-parallel, replicated-agent design:
#   * every rank runs its OWN Isaac sim on cuda:{LOCAL_RANK} with its own --num_envs envs, its own
#     replay buffer of --buffer_size and its own copy of the expert buffer. Every hyperparameter is
#     PER-GPU: --buffer_size 1000000 means 1M transitions on each GPU, --batch_size 512 means 512
#     samples drawn per rank per update.
#   * the agent itself is replicated. Parameters start bitwise identical (broadcast from rank 0) and
#     stay identical because every gradient is averaged across ranks before the optimizer steps.
#
# Equivalence guarantee: one distributed update equals the corresponding single-GPU update on the
# CONCATENATED global batch (world_size * batch_size). That holds because
#   * all four losses (model / critic / actor / alpha) are means over the local batch, so the
#     average of the per-rank gradients IS the gradient of the global-batch mean;
#   * UnitBatchNorm normalizes with GLOBAL batch statistics under --sync_batchnorm (default on),
#     via a differentiable SUM all-reduce of the batch sum and sum-of-squares;
#   * EmpiricalNormalization already all-reduces its running statistics (cleanrl_utils/utils.py).
# tests/test_distributed_equivalence.py verifies this end-to-end against a world_size=1 reference.
#
# Logging, checkpointing and wandb happen on rank 0 only. Diagnostics that issue collectives (the
# Hessian probe) still RUN on every rank -- skipping them anywhere would desynchronize the
# collectives and hang the job -- but only rank 0 records the result.
#
# EXPERIMENTAL TESTBED -- torch.compile / CUDA graph work. Identical to the production script
# except for --compile_update, which routes the learning update through a single compiled region.
#
# Measured on an RTX 4090 at the L40S job's shapes (batch 512, UTD 256, 32x32 2-cam obs):
#   eager                                    7.23 ms/update
#   torch.compile, whole update, one region  5.07 ms/update   1.43x
#   torch.compile, per-loss regions          5.94 ms/update   1.18x
#   mode="reduce-overhead" (CUDA graphs)     unavailable -- inductor reports
#       "skipping cudagraphs due to mutated inputs (22 instances)"; the update mutates BatchNorm
#       running stats, EmpiricalNormalization buffers and (via normalize_parameters + the Polyak
#       EMA) the parameters themselves, all of which cudagraph trees refuse to capture.
#
# Pairs with the Grayscale-Asymmetric tasks in rl_state_cfg.py.
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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cleanrl_utils.buffers import AsymmetricReplayBuffer, cat_samples, load_expert_replay_buffer
from cleanrl_utils.utils import EmpiricalNormalization

from vecenv_wrapper import IsaacLabVectorEnv

_NULL_CTX = nullcontext()


# ────────────────────────────── distributed plumbing ──────────────────────────────
class _DistState:
    """Process-group facts, read from the torchrun environment once at startup.

    Module-level rather than passed around because UnitBatchNorm.forward has to consult it deep
    inside the networks, and threading a context object through every module signature would change
    the shapes of classes shared with the single-GPU script for no benefit.

    Defaults describe a plain single-process run, so importing this module (as the tests do) without
    torchrun leaves every collective a no-op.
    """

    enabled = False          # torch.distributed initialized (true even at world_size 1 under torchrun)
    rank = 0                 # global rank
    local_rank = 0           # GPU index within this node
    world_size = 1
    is_main = True           # rank 0: the only rank that logs, prints or checkpoints
    sync_bn = False          # --sync_batchnorm AND enabled


DIST = _DistState()


def rprint(*a, **kw):
    """print() on rank 0 only. Every rank runs identical setup, so N copies of it are pure noise."""
    if DIST.is_main:
        print(*a, **kw)


def milestone(msg: str):
    """Rank-tagged startup progress, printed by EVERY rank.

    Rank-0-only logging makes a multi-rank stall undiagnosable: if rank 0 blocks in a collective
    waiting for a rank that is still building its simulation, the log simply stops with no
    indication of which rank is where. These few lines cost nothing (startup only) and are the
    difference between "it hung somewhere" and knowing exactly which rank reached which step.
    """
    if DIST.enabled:
        print(f"[rank {DIST.rank}/{DIST.world_size}] {msg}", flush=True)
    else:
        print(f"[setup] {msg}", flush=True)


def init_distributed(backend: str = None) -> _DistState:
    """Join the process group described by torchrun's environment variables.

    run_singularity.sh launches `torch.distributed.run --nnodes .. --nproc_per_node ..`, which sets
    RANK / LOCAL_RANK / WORLD_SIZE. Absent those (a bare `python script.py`) this leaves DIST at its
    single-process defaults and every collective below degenerates to a no-op, so the same file runs
    unmodified on one GPU.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DIST

    DIST.rank = int(os.environ["RANK"])
    DIST.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    DIST.world_size = int(os.environ["WORLD_SIZE"])
    DIST.is_main = DIST.rank == 0

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    # set_device before init_process_group: NCCL binds a device per rank, and it also makes the
    # bare "cuda" device string (e.g. FlashSACQNetwork's q_support) resolve to this rank's GPU.
    if torch.cuda.is_available():
        torch.cuda.set_device(DIST.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    DIST.enabled = True
    return DIST


class _AllReduceSum(torch.autograd.Function):
    """SUM all-reduce that is differentiable, for statistics shared across ranks.

    Forward computes y = sum_r x_r, so dL/dx_r = dL/dy -- but on rank r autograd only holds
    dL_r/dy, the local term. All-reducing the incoming gradient turns that into sum_s dL_s/dy,
    which is what the chain rule wants for a quantity every rank's loss depends on. Without this
    the synced BatchNorm would have a correct forward and a silently wrong backward.
    """

    @staticmethod
    def forward(ctx, x):
        if DIST.world_size == 1:
            return x
        out = x.contiguous().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx, grad):
        if DIST.world_size == 1:
            return grad
        out = grad.contiguous().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


class GradSync:
    """Average one optimizer's gradients across ranks, in a single collective.

    Every loss in this script is a mean over the local batch, and every rank uses the same
    batch_size, so the mean of the per-rank gradients is exactly the gradient of the mean over the
    concatenated global batch. That is the whole equivalence argument -- it would break if the
    losses were sums, or if ranks had unequal batch sizes.

    Gradients are packed into one persistent flat buffer so each optimizer costs one all-reduce
    instead of one per parameter tensor; at UTD 256 that difference is thousands of launches per
    env step. ReduceOp.SUM followed by a divide rather than ReduceOp.AVG, because gloo (used by the
    CPU correctness tests) implements SUM but not AVG.
    """

    def __init__(self, params):
        self.params = [p for p in params if p.requires_grad]
        self._flat = None
        self._views = None
        self._grads = None

    def _build(self, grads):
        total = sum(g.numel() for g in grads)
        self._flat = torch.zeros(total, device=grads[0].device, dtype=grads[0].dtype)
        views, off = [], 0
        for g in grads:
            views.append(self._flat[off : off + g.numel()].view_as(g))
            off += g.numel()
        self._views = views
        self._grads = grads

    def __call__(self):
        if DIST.world_size == 1:
            return
        # Params with no grad are structurally unused (none are, in this model) and stay unused, so
        # the participating set is fixed after the first backward and the packing order is stable.
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return
        if self._flat is None:
            self._build(grads)
        elif len(grads) != len(self._views):
            raise RuntimeError(
                f"GradSync saw {len(grads)} gradients but bucketed {len(self._views)}: the set of "
                "parameters receiving gradients changed between updates, which would silently "
                "misalign the flat buffer."
            )
        torch._foreach_copy_(self._views, grads)
        dist.all_reduce(self._flat, op=dist.ReduceOp.SUM)
        self._flat.div_(DIST.world_size)
        torch._foreach_copy_(grads, self._views)


class UpdateSyncs:
    """The four GradSync buckets of one learning update, one per optimizer.

    Bundled so the eager, compiled and CUDA-graph paths take a single extra argument each and call
    the reductions at exactly the same four points.
    """

    def __init__(self, encoder_params, q_params, actor_params, alpha_params=None):
        self.encoder = GradSync(encoder_params)
        self.q = GradSync(q_params)
        self.actor = GradSync(actor_params)
        self._alpha = GradSync(alpha_params) if alpha_params is not None else None

    def alpha(self):
        if self._alpha is not None:
            self._alpha()


def broadcast_module_state(*modules_and_tensors):
    """Force every rank to rank 0's parameters and buffers.

    Identical seeds already produce identical initializations, but this run offsets the RNG per rank
    (so exploration noise decorrelates) and the guarantee that all replicas start bitwise equal is
    load-bearing for the whole design -- cheap insurance, paid once.
    """
    if DIST.world_size == 1:
        return
    for obj in modules_and_tensors:
        if obj is None:
            continue
        tensors = (
            list(obj.parameters()) + list(obj.buffers()) if isinstance(obj, nn.Module) else [obj]
        )
        for t in tensors:
            dist.broadcast(t.data, src=0)


class _NullWriter:
    """Stand-in for SummaryWriter on non-zero ranks: swallows everything, allocates nothing."""

    def add_scalar(self, *a, **kw):
        pass

    def add_text(self, *a, **kw):
        pass

    def close(self):
        pass


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
    critic_num_blocks: int = 2
    """number of FlashSAC residual blocks in each C51 critic trunk (hidden_dim 256)"""
    actor_num_blocks: int = 2
    """number of FlashSAC residual blocks in the actor trunk (hidden_dim 128)"""
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
    use_tanh: bool = False
    """use tanh layer after policy"""
    obs_normalization: bool = True
    """use obs normalization"""
    # ── MR.Q model-based auxiliary loss (arxiv.org/abs/2501.16142) ──
    zs_dim: int = 512
    """MR.Q state-embedding (zs) dimension"""
    za_dim: int = 256
    """MR.Q action-embedding (za) dimension inside the state-action encoder"""
    zsa_dim: int = 512
    """MR.Q state-action-embedding (zsa) dimension; the FlashSAC C51 critic operates on this"""
    encoder_hidden_dim: int = 512
    """hidden width of the MR.Q encoder MLPs (state MLP is unused for image obs; g-network MLP)"""
    encoder_lr: float = 3e-4
    """learning rate for the MR.Q encoder (state + state-action + MDP-predictor), trained only by
    the model-based auxiliary loss"""
    fused_optimizer: bool = True
    """use the fused AdamW kernel instead of the default multi-tensor (foreach) path. One kernel per
    optimizer step instead of several, which matters at high num_updates where the step runs
    num_updates times per env step. Requires fp32 CUDA params (true here). Set False to fall back if
    a build lacks the fused kernel."""
    tf32_matmul: bool = True
    """allow TF32 for fp32 matmuls / convolutions (Ampere and newer: a100, l40s, h200). Roughly free
    throughput on the MLP-heavy critic and actor at the cost of ~10 bits of mantissa on matmul
    accumulation. Set False if you suspect numerical trouble in the C51 head."""
    encoder_num_channels: int = 32
    """channels per conv layer in the MR.Q image state encoder"""
    encoder_conv_layers: int = 4
    """depth of the MR.Q image conv trunk: (n-1) stride-2 layers then one stride-1, kernel 3.
    4 suits 84x84 (-> 7x7 spatial, 1568 features). It is too deep for small images: at 32x32 it
    collapses to 1x1 (32 features), throwing away all spatial structure -- use 3 there (-> 5x5,
    800 features). The constructor prints the resulting spatial map and refuses a stack that
    collapses below 1x1."""
    proprio_fuse_layers: int = 2
    """number of MLP layers fusing [vision | proprio] into the final latent z (width zs_dim). Must be
    >= 1: MR.Q's MDP predictor is linear in zsa, so proprio needs a learned nonlinear embedding
    rather than being concatenated raw into the dynamics target. Ignored without a proprio group."""
    model_reward_coef: float = 1.0
    """weight on the reward-prediction term of the MR.Q auxiliary loss"""
    model_done_coef: float = 1.0
    """weight on the done-prediction term of the MR.Q auxiliary loss"""
    model_dynamics_coef: float = 1.0
    """weight on the next-latent-state prediction term of the MR.Q auxiliary loss"""
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
    profile_memory: bool = True
    """print a one-time memory breakdown after setup (envs / networks / optimizers / replay buffer,
    with the device each lives on) and log mem/* scalars every logging_interval. Cheap -- the
    breakdown is a few driver queries and a walk over already-allocated tensors."""
    sync_batchnorm: bool = True
    """compute UnitBatchNorm statistics over the GLOBAL batch (all ranks) instead of each rank's own
    shard. On by default because it is what makes a distributed update identical to the single-GPU
    update on the concatenated batch -- batch statistics are the only part of this model that data
    parallelism does not make equivalent for free.

    It costs one extra all-reduce per BatchNorm call that runs with training=True: 20 per update
    (qf1, qf2 and both targets, 5 layers each) plus 5 more on actor updates. The MR.Q encoder is
    LayerNorm-based and never participates. Ignored entirely at world_size 1.

    Set False for standard DDP behaviour -- each rank normalizes by its own batch -- which is faster
    but has two consequences worth knowing. The update is no longer comparable to a single-GPU run
    (measured: gradients diverge by ~1e-1 rather than ~1e-7). And the BatchNorm running_mean /
    running_var BUFFERS drift apart across ranks, because each rank accumulates them from its own
    data; parameters still stay bitwise identical, since those are driven by averaged gradients, but
    replicas stop being exact copies and each rank's behaviour policy differs slightly. Checkpoints
    remain self-consistent -- they are written by rank 0, whose buffers match its own weights."""
    compile_update: bool = False
    """route the learning update through a single torch.compile'd region (1.43x measured at batch
    512 / UTD 256). Mutually exclusive with --profile_timing, whose per-block cuda syncs would both
    defeat the point and force graph breaks. Costs ~25 s of one-time warmup."""
    cuda_graphs: bool = False
    """capture the learning update as CUDA graphs via tensordict.nn.CudaGraphModule, on top of
    torch.compile (measured 1.97x vs eager at batch 512, against 1.37x for compile alone).

    Implies --compile_update. Note inductor's own mode="reduce-overhead" does NOT work here -- it
    reports "skipping cudagraphs due to mutated inputs (22 instances)" because the update mutates
    BatchNorm running stats, EmpiricalNormalization buffers and the parameters themselves.
    CudaGraphModule captures the already-compiled (default-mode) function instead, handling warmup,
    static input buffers, output cloning and RNG generator state itself."""
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
        # training=False reads the running statistics, which are already identical on every rank,
        # so there is nothing to synchronize and the fast path is exact as-is.
        if training and DIST.sync_bn:
            return self._sync_forward(x)
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

    # Kept out of the compiled graph. Dynamo does not reliably trace a custom autograd.Function
    # wrapping a collective: measured against a world_size=1 reference, the compiled version of this
    # method produced gradients off by ~1e-2 (eager is ~1e-7), i.e. the synchronization was
    # effectively lost while every rank still agreed with the others -- silent, not a crash.
    # Forcing a graph break here keeps the collective in eager mode, where it is verified correct.
    # The surrounding update still compiles; only these ~25 small BN calls per update fall out.
    @torch._dynamo.disable
    def _sync_forward(self, x):
        """BatchNorm over the GLOBAL batch: what F.batch_norm would compute on the concatenation.

        Batch statistics are the one place data-parallelism is not automatically equivalent -- each
        rank would otherwise normalize by its own shard's mean/var. One SUM all-reduce of the batch
        sum and sum-of-squares (stacked into a single collective) yields the global moments, and
        _AllReduceSum keeps the backward correct.

        Sums are accumulated shifted by running_mean. That constant cancels exactly in both mean and
        variance, but it keeps E[x^2] - E[x]^2 away from the catastrophic cancellation it suffers
        when the mean is large relative to the spread. running_mean is a buffer (no grad) and is
        identical across ranks, so the shift changes nothing about the result or its gradient.
        """
        n = x.shape[0] * DIST.world_size
        shift = self.running_mean.detach()
        xs = x - shift
        local = torch.stack([xs.sum(0), (xs * xs).sum(0)])
        total = _AllReduceSum.apply(local)
        mean_shifted = total[0] / n
        # Biased (population) variance, matching what batch_norm normalizes with.
        var = total[1] / n - mean_shifted * mean_shifted
        mean = mean_shifted + shift

        with torch.no_grad():
            # Running stats take the UNBIASED variance, as torch's BatchNorm does, while the
            # normalization above uses the biased one. Mirroring both keeps a run that toggles
            # --sync_batchnorm off mid-experiment on the same footing.
            self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
            unbiased = var * (n / (n - 1))
            self.running_var.mul_(1 - self.momentum).add_(unbiased * self.momentum)

        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias

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


class MRQEncoder(nn.Module):
    """MR.Q encoder (arxiv.org/abs/2501.16142): a state encoder f, a state-action encoder g, and a
    linear MDP predictor, trained end-to-end as one network by the model-based auxiliary loss only.

      zs  = f(s)            state embedding      (used by the policy, gradients detached)
      zsa = g(zs, a)        state-action embed   (used by the value net, gradients detached)
      (next_zs_hat, r_hat, d_hat) = model(zsa)   one linear head predicting the next latent state,
                                                 reward, and terminal signal (the auxiliary targets)

    The image state encoder matches the paper: four 3x3 convs, 32 channels, strides (2, 2, 2, 1),
    ELU, then a linear -> LayerNorm -> ELU bottleneck to zs_dim. The flatten dim is computed from a
    dummy pass (84x84 gives 1568 as in the paper). We skip the paper's `state/255 - 0.5` rescale
    since the images are metric depth / normalized grayscale and already go through
    EmpiricalNormalization upstream.

    Proprioception (optional) is fused inside f, not bolted onto its output:

        vision = ln_elu(zs_lin(cnn(image)))
        z      = fuse_MLP([vision | proprio])        -> width zs_dim

    The MLP matters. MR.Q's MDP predictor is a single *linear* map off zsa, which is what forces the
    representation to be linear in the prediction step. Concatenating raw proprio straight into z
    would put un-embedded joint state in the dynamics target and require the model to predict
    proprioceptive dynamics linearly -- and those are not linear. Passing the fused vector through
    learned nonlinear layers lets the representation reshape itself so the linear head remains valid,
    which is the whole premise of the method.

    Because the fusion MLP outputs zs_dim, the latent width is zs_dim whether or not proprio is
    present, so g, the actor, and the MDP predictor are untouched by this option. proprio_dim=0
    skips the fusion entirely and reproduces the vision-only encoder exactly.
    """

    def __init__(self, obs_shape, action_dim, zs_dim=512, za_dim=256, zsa_dim=512, hidden_dim=512,
                 num_channels=32, proprio_dim=0, proprio_fuse_layers=2, conv_layers=4):
        super().__init__()
        in_channels = obs_shape[0] * obs_shape[1]  # history * C, folded into conv channels
        self.obs_hw = obs_shape[2:]
        self.zs_dim = zs_dim
        self.proprio_dim = proprio_dim

        # State encoder f (image): (conv_layers - 1) stride-2 layers then one stride-1, kernel 3.
        # conv_layers=4 reproduces the original fixed stack exactly.
        assert conv_layers >= 1, "the conv trunk needs at least one layer"
        convs, c_in = [], in_channels
        for i in range(conv_layers):
            stride = 2 if i < conv_layers - 1 else 1
            convs += [nn.Conv2d(c_in, num_channels, kernel_size=3, stride=stride), nn.ELU()]
            c_in = num_channels
        self.zs_cnn = nn.Sequential(*convs)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *self.obs_hw)
            feat = self.zs_cnn(dummy)
            flatten_dim = feat.flatten(1).shape[1]
        # A 1x1 map means the trunk is too deep for this input: every pixel has been pooled into a
        # single cell and all spatial structure is gone before zs_lin ever sees it.
        if min(feat.shape[2:]) < 2:
            raise ValueError(
                f"conv trunk of {conv_layers} layers collapses a {self.obs_hw[0]}x{self.obs_hw[1]} "
                f"input to {tuple(feat.shape[2:])} ({flatten_dim} features) -- reduce "
                f"--encoder_conv_layers (32x32 wants 3, 84x84 wants 4)."
            )
        rprint(
            f"[encoder] {conv_layers} conv layers on {self.obs_hw[0]}x{self.obs_hw[1]} "
            f"({in_channels} in-ch) -> {tuple(feat.shape[1:])} = {flatten_dim} features -> zs {zs_dim}"
        )
        self.zs_lin = nn.Linear(flatten_dim, zs_dim)

        # Fusion MLP: [vision | proprio] -> zs_dim. Every layer gets LayerNorm + ELU, matching the
        # paper's vector-input `mlp_forward` (ln_activ on all three layers including the last), so
        # the final z sits in the same normalized regime as the vision-only path.
        if proprio_dim > 0:
            assert proprio_fuse_layers >= 1, "proprio fusion needs at least one layer"
            layers, d_in = [], zs_dim + proprio_dim
            for i in range(proprio_fuse_layers):
                d_out = zs_dim if i == proprio_fuse_layers - 1 else hidden_dim
                layers.append(nn.Linear(d_in, d_out))
                d_in = d_out
            self.proprio_fuse = nn.ModuleList(layers)
        else:
            self.proprio_fuse = None

        # State-action encoder g
        self.za = nn.Linear(action_dim, za_dim)
        self.zsa1 = nn.Linear(zs_dim + za_dim, hidden_dim)
        self.zsa2 = nn.Linear(hidden_dim, hidden_dim)
        self.zsa3 = nn.Linear(hidden_dim, zsa_dim)

        # MDP predictor: next latent state (zs_dim) + reward (1) + done (1)
        self.model = nn.Linear(zsa_dim, zs_dim + 1 + 1)

    @staticmethod
    def _ln_elu(x):
        return F.elu(F.layer_norm(x, (x.shape[-1],)))

    def zs(self, obs_flat, proprio: torch.Tensor = None):
        """State embedding f(s). `obs_flat` is (B, prod(obs_shape)); reshaped to (B, hist*C, H, W).

        With proprio_dim > 0 the (already normalized) proprio vector is concatenated onto the conv
        trunk's output and the result is passed through the fusion MLP, so the returned latent is a
        learned joint embedding of vision and proprioception -- width zs_dim either way.
        """
        x = obs_flat.reshape(obs_flat.shape[0], -1, *self.obs_hw)
        h = self.zs_cnn(x).flatten(1)
        zs = self._ln_elu(self.zs_lin(h))  # LayerNorm + ELU on the final linear (paper's ln_activ)
        if self.proprio_fuse is not None:
            zs = torch.cat([zs, proprio], dim=1)
            for layer in self.proprio_fuse:
                zs = self._ln_elu(layer(zs))
        return zs

    def zsa(self, zs, action):
        """State-action embedding g(zs, a). Returned raw (no final activation), as in the paper."""
        za = F.elu(self.za(action))
        x = torch.cat([zs, za], dim=1)
        x = self._ln_elu(self.zsa1(x))
        x = self._ln_elu(self.zsa2(x))
        return self.zsa3(x)

    def predict(self, zsa):
        """MDP predictor head: (next_zs_hat, reward_hat, done_logit) from the state-action embed.

        next_zs_hat targets the fused latent (width zs_dim), i.e. the learned joint vision+proprio
        embedding -- never raw proprio, which this linear head could not model.
        """
        out = self.model(zsa)
        d = self.zs_dim
        next_zs_hat = out[:, :d]
        reward_hat = out[:, d : d + 1]
        done_logit = out[:, d + 1 : d + 2]
        return next_zs_hat, reward_hat, done_logit


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
    """Plain state-based FlashSAC C51 critic -- identical to flashsac_continuous_action.py.

    Consumes the raw critic observation concatenated with the action. Deliberately NOT MR.Q: no
    encoder, no zsa. The value function is trained by the distributional TD loss alone.
    """

    def __init__(
        self,
        critic_obs_dim,
        action_dim,
        hidden_dim=256,
        num_atoms=101,
        v_min=-20.0,
        v_max=20.0,
        num_blocks=2,
        device="cuda",
    ):
        super().__init__()

        input_dim = int(critic_obs_dim) + int(action_dim)
        self.norm1 = UnitBatchNorm(input_dim)
        self.linear1 = UnitLinear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.norm2 = UnitRMSNorm(hidden_dim)
        self.linear2 = UnitLinear(hidden_dim, num_atoms, bias=True)

        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        # Explicit device rather than a bare "cuda": on rank 1+ the default CUDA device is only this
        # rank's GPU after torch.cuda.set_device, and q_support is a plain attribute that .to(device)
        # would not move. Parameterizing it also lets the correctness tests run this network on CPU.
        self.q_support = torch.linspace(v_min, v_max, num_atoms, device=device)
        # Row offsets for the flattened scatter, rebuilt only when the batch size changes.
        # Plain attribute, not a buffer -- stays out of state_dict so checkpoints are unaffected.
        self._offset_cache = None

    def forward(self, x, a, training: bool):
        x = torch.cat([x, a], 1)
        x = self.linear1(self.norm1(x, training=training))
        for block in self.blocks:
            x = block(x, training=training) + x
        x = self.linear2(self.norm2(x))
        return x

    def _c51_index_math(self, rewards, bootstrap, discount):
        """Atom indices and interpolation weights for the C51 projection.

        Depends only on the TD target (rewards/bootstrap/discount) and the fixed support -- not on
        any critic's output distribution -- so an ensemble of critics sharing a support computes
        this once instead of per member.
        """
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

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

        offset = self._offset_cache
        if offset is None or offset.shape[0] != batch_size or offset.device != b.device:
            offset = (torch.arange(batch_size, device=b.device) * self.num_atoms).unsqueeze(1)
            self._offset_cache = offset

        max_index = batch_size * self.num_atoms - 1
        lower_indices = torch.clamp((lower + offset).view(-1), 0, max_index)
        upper_indices = torch.clamp((upper + offset).view(-1), 0, max_index)
        return lower_indices, upper_indices, upper.float() - b, b - lower.float()

    def project_ensemble(self, next_dists, rewards, bootstrap, discount):
        """Project K next-state distributions onto the atom support in one pass.

        next_dists: (K, B, num_atoms) -> (K, B, num_atoms). The index math runs once for all K, and
        the two scatters are done along dim 1 so all K members share a single kernel each.
        """
        k, batch_size, _ = next_dists.shape
        lower_idx, upper_idx, w_lower, w_upper = self._c51_index_math(rewards, bootstrap, discount)

        proj = torch.zeros(k, batch_size * self.num_atoms, device=next_dists.device, dtype=next_dists.dtype)
        proj.index_add_(1, lower_idx, (next_dists * w_lower).reshape(k, -1))
        proj.index_add_(1, upper_idx, (next_dists * w_upper).reshape(k, -1))
        return proj.view(k, batch_size, self.num_atoms)

    def projection(
        self,
        obs,
        actions,
        rewards,
        bootstrap,
        discount,
        training: bool,
        next_dist: torch.Tensor = None,
    ):
        if next_dist is None:
            next_dist = F.softmax(self(obs, actions, training=training), dim=1)
        return self.project_ensemble(next_dist.unsqueeze(0), rewards, bootstrap, discount).squeeze(0)
    
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
    """FlashSAC actor operating on the MR.Q state embedding zs (gradients detached from the encoder).

    Callers pass zs = f(s).detach(); this network never touches the raw obs or the encoder.
    """

    def __init__(self,
        env,
        zs_dim,
        action_scale=None,
        action_bias=None,
        hidden_dim=128,
        num_blocks=2,
        use_tanh=False,
    ):
        super().__init__()
        action_dim = np.prod(env.single_action_space.shape)

        self.norm1 = UnitBatchNorm(zs_dim)
        self.linear1 = UnitLinear(zs_dim, hidden_dim)
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

    def forward(self, zs, training: bool):
        x = self.linear1(self.norm1(zs, training=training))
        for block in self.blocks:
            x = block(x, training=training) + x
        x = self.norm2(x)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, training: bool):
        mean, log_std = self(x, training=training)
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


def _sample_rollout_action(actor, x, noise, cur_count, cur_n, zeta_cdf, training: bool):
    mean, log_std = actor(x, training=training)
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


def infer_num_blocks(state_dict, default=2):
    """Count FlashSAC residual blocks in a saved actor/critic.

    `num_blocks` is a constructor arg, so it never lands in the state dict -- rebuilding at the
    class default silently mismatches any checkpoint trained with --actor_num_blocks /
    --critic_num_blocks. Structure wins when there are blocks to count; pass the value recorded in
    ckpt["args"] as `default` so a zero-block trunk (which leaves no keys) still resolves.
    Checkpoints predating those args always had 2.
    """
    indices = {int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")}
    return max(indices) + 1 if indices else default


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


def build_graph_update(args, encoder, encoder_target, qf1, qf2, qf1_target, qf2_target, actor,
                       encoder_optimizer, q_optimizer, actor_optimizer, a_optimizer, log_alpha,
                       target_entropy, ema_target_params, ema_online_params,
                       normalize_obs, normalize_critic_obs, norm_proprio, syncs):
    """The update split into three separately-captured regions, preserving the eager ordering.

    A captured graph cannot branch on a Python bool, so the delayed actor update has to be its own
    region. The eager loop runs [main + critic step] -> [optional actor] -> [normalize] -> [EMA], so
    normalize/EMA go in a third region rather than being folded into `main`: folding them in would
    make the actor read unit-normalized critic weights and the Polyak EMA read un-normalized ones,
    both of which change the algorithm.

    Returns plain tuples rather than dicts -- CudaGraphModule handles tensors and tensordicts.
    """
    from tensordict.nn import CudaGraphModule

    def main(policy_raw, next_policy_raw, critic_raw, next_critic_raw, proprio_raw,
             next_proprio_raw, actions, rewards, dones, n_steps):
        policy_obs, next_policy_obs = policy_raw, next_policy_raw
        critic_obs, next_critic_obs = critic_raw, next_critic_raw
        if args.obs_normalization:
            policy_obs = normalize_obs(policy_raw)
            next_policy_obs = normalize_obs(next_policy_raw)
            critic_obs = normalize_critic_obs(critic_raw)
            next_critic_obs = normalize_critic_obs(next_critic_raw)
        proprio = norm_proprio(proprio_raw, update=True)
        next_proprio = norm_proprio(next_proprio_raw, update=True)

        bootstrap = (~dones.bool()).float()
        discount = args.gamma ** n_steps

        zs = encoder.zs(policy_obs, proprio=proprio)
        zsa = encoder.zsa(zs, actions)
        next_zs_hat, reward_hat, done_logit = encoder.predict(zsa)
        with torch.no_grad():
            next_zs_target = encoder_target.zs(next_policy_obs, proprio=next_proprio)

        dyn_loss = F.mse_loss(next_zs_hat, next_zs_target)
        reward_loss = F.mse_loss(reward_hat, rewards.reshape(-1, 1))
        done_loss = F.binary_cross_entropy_with_logits(done_logit, dones.reshape(-1, 1).float())
        model_loss = (args.model_dynamics_coef * dyn_loss
                      + args.model_reward_coef * reward_loss
                      + args.model_done_coef * done_loss)
        encoder_optimizer.zero_grad(set_to_none=False)
        model_loss.backward()
        syncs.encoder()
        encoder_optimizer.step()

        with torch.no_grad():
            nsa, nslp, _ = actor.get_action(next_zs_target, training=False)
            d1 = F.softmax(qf1_target(next_critic_obs, nsa, training=True), dim=-1)
            d2 = F.softmax(qf2_target(next_critic_obs, nsa, training=True), dim=-1)
            reward_term = (rewards.squeeze(-1)
                           - discount * bootstrap.squeeze(-1) * log_alpha.exp() * nslp.squeeze(-1))
            target_dists = qf1_target.project_ensemble(
                torch.stack([d1, d2]), reward_term, bootstrap.squeeze(-1), discount)
            target_values = torch.sum(target_dists * qf1_target.q_support, dim=-1)
            use_qf1 = (target_values[0] <= target_values[1]).unsqueeze(-1)
            qf_target_dist = torch.where(use_qf1, target_dists[0], target_dists[1])

        qf1_log_probs = F.log_softmax(qf1(critic_obs, actions, training=True), dim=-1)
        qf2_log_probs = F.log_softmax(qf2(critic_obs, actions, training=True), dim=-1)
        qf1_loss = -torch.sum(qf_target_dist * qf1_log_probs, dim=-1)
        qf2_loss = -torch.sum(qf_target_dist * qf2_log_probs, dim=-1)
        qf_loss = torch.stack([qf1_loss, qf2_loss]).mean(dim=1).sum(dim=0)
        q_optimizer.zero_grad(set_to_none=False)
        qf_loss.backward()
        syncs.q()
        q_optimizer.step()

        return (zs.detach(), critic_obs.detach(), qf_loss.detach(), qf1_loss.detach().mean(),
                qf2_loss.detach().mean(), target_values.detach(), model_loss.detach(),
                dyn_loss.detach(), reward_loss.detach(), done_loss.detach())

    def actor_step(zs_pi, critic_obs):
        pi, log_pi, _ = actor.get_action(zs_pi, training=True)
        with torch.no_grad():
            policy_entropy = -log_pi.mean()
            log_pi_mean = log_pi.mean()
        alpha_val = log_alpha.detach().exp().squeeze()
        qf1_pi = qf1.get_value(F.softmax(qf1(critic_obs, pi, training=False), dim=-1))
        qf2_pi = qf2.get_value(F.softmax(qf2(critic_obs, pi, training=False), dim=-1))
        actor_loss = ((alpha_val * log_pi) - torch.min(qf1_pi, qf2_pi)).mean()
        actor_optimizer.zero_grad(set_to_none=False)
        actor_loss.backward()
        syncs.actor()
        actor_optimizer.step()

        alpha_loss = actor_loss.new_zeros(())
        if args.autotune:
            with torch.no_grad():
                _, log_pi2, _ = actor.get_action(zs_pi, training=False)
            alpha_loss = (-log_alpha.exp() * (log_pi2 + target_entropy)).mean()
            a_optimizer.zero_grad(set_to_none=False)
            alpha_loss.backward()
            syncs.alpha()
            a_optimizer.step()
        return actor_loss.detach(), alpha_loss.detach(), policy_entropy, log_pi_mean

    def tail(anchor):
        # `anchor` is unused; CudaGraphModule needs at least one tensor input to key the capture on.
        actor.normalize_parameters()
        qf1.normalize_parameters()
        qf2.normalize_parameters()
        with torch.no_grad():
            torch._foreach_lerp_(ema_target_params, ema_online_params, args.tau)
        return anchor.new_zeros(())

    fns = [main, actor_step, tail]
    fns = [torch.compile(f) for f in fns]
    if args.cuda_graphs:
        fns = [CudaGraphModule(f) for f in fns]
    return fns

def build_compiled_update(args, encoder, encoder_target, qf1, qf2, qf1_target, qf2_target, actor,
                          encoder_optimizer, q_optimizer, actor_optimizer, a_optimizer, log_alpha,
                          target_entropy, ema_target_params, ema_online_params,
                          normalize_obs, normalize_critic_obs, norm_proprio, syncs):
    """The whole learning update as ONE callable, so torch.compile sees a single region.

    Compiling the four loss computations separately only reaches 1.18x: each becomes its own graph,
    and the eager optimizer steps between them force materialization at every boundary. Folding
    forwards, backwards and optimizer steps together measured 1.43x.

    zero_grad(set_to_none=False) keeps gradient storage addresses stable across iterations, which
    costs nothing here and is a prerequisite for any future CUDA-graph capture.
    """
    def full_update(data, do_actor: bool):
        policy_obs = data.policy_observations
        next_policy_obs = data.next_policy_observations
        critic_obs = data.critic_observations
        next_critic_obs = data.next_critic_observations
        if args.obs_normalization:
            policy_obs = normalize_obs(data.policy_observations)
            next_policy_obs = normalize_obs(data.next_policy_observations)
            critic_obs = normalize_critic_obs(data.critic_observations)
            next_critic_obs = normalize_critic_obs(data.next_critic_observations)
        proprio = norm_proprio(data.proprio_observations, update=True)
        next_proprio = norm_proprio(data.next_proprio_observations, update=True)

        bootstrap = (~data.dones.bool()).float()
        discount = args.gamma ** data.effective_n_steps

        zs = encoder.zs(policy_obs, proprio=proprio)
        zsa = encoder.zsa(zs, data.actions)
        next_zs_hat, reward_hat, done_logit = encoder.predict(zsa)
        with torch.no_grad():
            next_zs_target = encoder_target.zs(next_policy_obs, proprio=next_proprio)

        dyn_loss = F.mse_loss(next_zs_hat, next_zs_target)
        reward_loss = F.mse_loss(reward_hat, data.rewards.reshape(-1, 1))
        done_loss = F.binary_cross_entropy_with_logits(done_logit, data.dones.reshape(-1, 1).float())
        model_loss = (args.model_dynamics_coef * dyn_loss
                      + args.model_reward_coef * reward_loss
                      + args.model_done_coef * done_loss)
        encoder_optimizer.zero_grad(set_to_none=False)
        model_loss.backward()
        syncs.encoder()
        encoder_optimizer.step()

        with torch.no_grad():
            next_state_actions, next_state_log_probs, _ = actor.get_action(next_zs_target, training=False)
            qf1_next = F.softmax(qf1_target(next_critic_obs, next_state_actions, training=True), dim=-1)
            qf2_next = F.softmax(qf2_target(next_critic_obs, next_state_actions, training=True), dim=-1)
            reward_term = (data.rewards.squeeze(-1)
                           - discount * bootstrap.squeeze(-1) * log_alpha.exp() * next_state_log_probs.squeeze(-1))
            target_dists = qf1_target.project_ensemble(
                torch.stack([qf1_next, qf2_next]), reward_term, bootstrap.squeeze(-1), discount)
            target_values = torch.sum(target_dists * qf1_target.q_support, dim=-1)
            use_qf1 = (target_values[0] <= target_values[1]).unsqueeze(-1)
            qf_target_dist = torch.where(use_qf1, target_dists[0], target_dists[1])

        qf1_log_probs = F.log_softmax(qf1(critic_obs, data.actions, training=True), dim=-1)
        qf2_log_probs = F.log_softmax(qf2(critic_obs, data.actions, training=True), dim=-1)
        qf1_loss = -torch.sum(qf_target_dist * qf1_log_probs, dim=-1)
        qf2_loss = -torch.sum(qf_target_dist * qf2_log_probs, dim=-1)
        qf_loss = torch.stack([qf1_loss, qf2_loss]).mean(dim=1).sum(dim=0)
        q_optimizer.zero_grad(set_to_none=False)
        qf_loss.backward()
        syncs.q()
        q_optimizer.step()

        actor_loss = qf_loss.new_zeros(())
        alpha_loss = qf_loss.new_zeros(())
        policy_entropy = qf_loss.new_zeros(())
        log_pi_mean = qf_loss.new_zeros(())
        if do_actor:
            zs_pi = zs.detach()
            pi, log_pi, _ = actor.get_action(zs_pi, training=True)
            with torch.no_grad():
                policy_entropy = -log_pi.mean()
                log_pi_mean = log_pi.mean()
            alpha_val = log_alpha.detach().exp().squeeze()
            qf1_pi = qf1.get_value(F.softmax(qf1(critic_obs, pi, training=False), dim=-1))
            qf2_pi = qf2.get_value(F.softmax(qf2(critic_obs, pi, training=False), dim=-1))
            actor_loss = ((alpha_val * log_pi) - torch.min(qf1_pi, qf2_pi)).mean()
            actor_optimizer.zero_grad(set_to_none=False)
            actor_loss.backward()
            syncs.actor()
            actor_optimizer.step()

            if args.autotune:
                with torch.no_grad():
                    _, log_pi2, _ = actor.get_action(zs_pi, training=False)
                alpha_loss = (-log_alpha.exp() * (log_pi2 + target_entropy)).mean()
                a_optimizer.zero_grad(set_to_none=False)
                alpha_loss.backward()
                syncs.alpha()
                a_optimizer.step()

        actor.normalize_parameters()
        qf1.normalize_parameters()
        qf2.normalize_parameters()
        with torch.no_grad():
            torch._foreach_lerp_(ema_target_params, ema_online_params, args.tau)

        return {
            "qf_loss": qf_loss.detach(), "qf1_loss": qf1_loss.detach().mean(),
            "qf2_loss": qf2_loss.detach().mean(), "target_values": target_values.detach(),
            "actor_loss": actor_loss.detach(), "alpha_loss": alpha_loss.detach(),
            "policy_entropy": policy_entropy.detach(), "log_pi": log_pi_mean.detach(),
            "model_loss": model_loss.detach(), "dyn_loss": dyn_loss.detach(),
            "reward_loss": reward_loss.detach(), "done_loss": done_loss.detach(),
        }

    return torch.compile(full_update) if args.compile_update else full_update


if __name__ == "__main__":

    args, launcher_args = tyro.cli(Args, return_unknown_args=True)

    # Join the process group FIRST: rank identity decides who logs, which GPU this process owns and
    # what seed offsets apply, all of which the setup below depends on.
    init_distributed()
    DIST.sync_bn = args.sync_batchnorm and DIST.enabled
    if DIST.enabled:
        rprint(
            f"[dist] world_size={DIST.world_size} backend={dist.get_backend()} "
            f"sync_batchnorm={DIST.sync_bn} | per-GPU: num_envs={args.num_envs} "
            f"batch_size={args.batch_size} buffer_size={args.buffer_size:,} "
            f"-> global batch {args.batch_size * DIST.world_size}"
        )

    run_name = args.run_name if args.run_name else f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track and DIST.is_main:
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
    # Non-zero ranks get a writer that discards everything, so the logging call sites stay identical
    # across ranks and cannot drift apart.
    writer = SummaryWriter(f"runs/{run_name}") if DIST.is_main else _NullWriter()
    if args.save_depth_video and DIST.is_main:
        depth_video_dir = f"runs/{run_name}/depth_videos"
        os.makedirs(depth_video_dir, exist_ok=True)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    # Offset by rank so the ranks explore differently -- identical streams would make every replica
    # collect near-identical data and waste the extra GPUs. Network initialization is put back in
    # lockstep by the broadcast after construction, so the replicas still start bitwise equal.
    seed = args.seed + DIST.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        f"cuda:{DIST.local_rank}" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # TF32 for fp32 matmul/conv on Ampere+ (a100 / l40s / h200). The critic and actor are MLP-heavy
    # and run num_updates times per env step, so this is one of the cheapest throughput knobs
    # available. It costs mantissa bits on matmul accumulation only -- master weights stay fp32.
    if args.tf32_matmul and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        rprint("[perf] TF32 enabled for fp32 matmul/conv")
    rprint(f"[perf] fused AdamW: {args.fused_optimizer}")

    # ── memory accounting: baseline before Isaac Sim starts ──
    # Taken here so the env delta below includes Isaac's PhysX/rendering allocations, which live
    # outside PyTorch's allocator. The baseline itself is mostly this process's CUDA context.
    mem_baseline_gpu = cuda_used_bytes()
    mem_baseline_host = _host_rss_bytes()

    # env setup
    envs = IsaacLabVectorEnv(
        args.env_id, args.num_envs, launcher_args=launcher_args, device=device,
        keep_distributed_flag=DIST.enabled, sim_device=str(device),
    )
    milestone("Isaac env ready")
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    mem_after_env_gpu = cuda_used_bytes()
    mem_after_env_host = _host_rss_bytes()

    policy_obs_shape = envs.single_observation_space["policy"].shape   # (history, C, H, W) image
    critic_obs_shape = envs.single_observation_space["critic"].shape   # flat state vector
    critic_obs_dim = int(np.prod(critic_obs_shape))
    # No shape assertion here: unlike the symmetric script there is no shared encoder, so the actor
    # (image) and critic (state) streams are expected to differ.
    rprint(f"[obs] asymmetric: actor {tuple(policy_obs_shape)} image | critic {critic_obs_dim}-d state")

    action_dim = int(np.prod(envs.single_action_space.shape))

    # Optional proprioception group (e.g. the grayscale reaching task). Tasks that only define
    # policy/critic leave proprio_dim at 0, which reproduces the vision-only encoder exactly.
    has_proprio = "proprio" in envs.single_observation_space.spaces
    proprio_dim = int(np.prod(envs.single_observation_space["proprio"].shape)) if has_proprio else 0
    rprint(f"[obs] proprio group: {'present' if has_proprio else 'absent'} (dim={proprio_dim})")

    if args.obs_normalization:
        policy_obs_dim = int(np.prod(envs.single_observation_space["policy"].shape))
        critic_obs_dim = int(np.prod(envs.single_observation_space["critic"].shape))
        actor_obs_normalizer = EmpiricalNormalization(shape=(policy_obs_dim,), device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=(critic_obs_dim,), device=device)
        if has_proprio:
            proprio_obs_normalizer = EmpiricalNormalization(shape=(proprio_dim,), device=device)

    mem_before_nets_gpu = cuda_used_bytes()

    # MR.Q encoder (state f + state-action g + MDP predictor), trained only by the auxiliary loss.
    # A target copy provides the next-state embeddings for both the RL target and the dynamics target.
    def make_encoder():
        return MRQEncoder(
            policy_obs_shape, action_dim,
            zs_dim=args.zs_dim, za_dim=args.za_dim, zsa_dim=args.zsa_dim,
            hidden_dim=args.encoder_hidden_dim, num_channels=args.encoder_num_channels,
            proprio_dim=proprio_dim, proprio_fuse_layers=args.proprio_fuse_layers,
            conv_layers=args.encoder_conv_layers,
        ).to(device)

    encoder = make_encoder()
    encoder_target = make_encoder()
    encoder_target.load_state_dict(encoder.state_dict())

    qf1 = FlashSACQNetwork(critic_obs_dim, action_dim, num_atoms=args.num_atoms, num_blocks=args.critic_num_blocks, device=device).to(device)
    qf2 = FlashSACQNetwork(critic_obs_dim, action_dim, num_atoms=args.num_atoms, num_blocks=args.critic_num_blocks, device=device).to(device)
    # The fused latent is zs_dim wide whether or not proprio is present.
    actor = FlashSACActor(envs, args.zs_dim, num_blocks=args.actor_num_blocks, use_tanh=args.use_tanh).to(device)
    zeta_cdf = _build_truncated_zeta_cdf(args.noise_repeat_zeta_mu, args.noise_repeat_zeta_max, device)
    noise_repeat_n = torch.ones((), dtype=torch.int32, device=device)
    noise_repeat_count = torch.zeros((), dtype=torch.int32, device=device)
    cached_noise = torch.randn((args.num_envs,) + envs.single_action_space.shape, device=device)
    qf1_target = FlashSACQNetwork(critic_obs_dim, action_dim, num_atoms=args.num_atoms, num_blocks=args.critic_num_blocks, device=device).to(device)
    qf2_target = FlashSACQNetwork(critic_obs_dim, action_dim, num_atoms=args.num_atoms, num_blocks=args.critic_num_blocks, device=device).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # Flat (online, target) lists for the Polyak update, built once so the per-update EMA is a
    # couple of _foreach_ kernels instead of ~3 launches per parameter tensor.
    _ema_pairs = unique_param_pairs((qf1, qf1_target), (qf2, qf2_target))
    _ema_pairs += list(zip(encoder.parameters(), encoder_target.parameters()))
    ema_online_params = [p for p, _ in _ema_pairs]
    ema_target_params = [tp for _, tp in _ema_pairs]

    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=args.encoder_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=args.fused_optimizer)
    q_optimizer = optim.AdamW(unique_params(qf1, qf2), lr=args.q_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=args.fused_optimizer)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.policy_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=args.fused_optimizer)

    # Normalize after init (FlashSAC unit-norm heads only; the MR.Q encoder uses plain params)
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
        alpha = torch.tensor(args.alpha, device=device)
        log_alpha = torch.tensor([torch.log(alpha)], requires_grad=True, device=device)
        a_optimizer = optim.AdamW([log_alpha], lr=args.alpha_lr, betas=(0.9, 0.95), fused=args.fused_optimizer)
    else:
        alpha = args.alpha

    # ── put every replica in lockstep, then wire up the gradient reductions ──
    # Broadcast covers parameters AND buffers, so the BatchNorm running stats and the (still-empty)
    # observation normalizers also start equal. From here on the replicas only ever see averaged
    # gradients and identical deterministic post-steps, so they stay bitwise equal for the run.
    milestone("networks built; entering broadcast (first collective -- a stall here means some "
              "rank has not reached it yet)")
    broadcast_module_state(encoder, encoder_target, qf1, qf2, qf1_target, qf2_target, actor)
    if args.obs_normalization:
        broadcast_module_state(actor_obs_normalizer, critic_obs_normalizer)
        if has_proprio:
            broadcast_module_state(proprio_obs_normalizer)
    if args.autotune:
        broadcast_module_state(log_alpha)

    milestone("broadcast done; replicas in lockstep")

    syncs = UpdateSyncs(
        encoder_params=list(encoder.parameters()),
        q_params=unique_params(qf1, qf2),
        actor_params=list(actor.parameters()),
        alpha_params=[log_alpha] if args.autotune else None,
    )

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
        rprint(
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
            "encoder (f+g+model)": module_bytes_by_device(encoder),
            "encoder_target": module_bytes_by_device(encoder_target),
            "qf1 + qf2": module_bytes_by_device(qf1, qf2),
            "qf1_target + qf2_target": module_bytes_by_device(qf1_target, qf2_target),
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
        rprint("\n".join(lines), flush=True)

    start_time = time.time()

    # Bound unconditionally: build_compiled_update() takes them as arguments, so they must exist
    # even when --no-obs_normalization leaves them unused.
    normalize_obs = normalize_critic_obs = None
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

    if (args.compile_update or args.cuda_graphs) and not args.autotune:
        # log_alpha only exists under --autotune; the C51 target reads it unconditionally, so the
        # eager path has the same constraint -- fail loudly rather than with a NameError.
        raise SystemExit("[error] --compile_update / --cuda_graphs require --autotune")
    if (args.compile_update or args.cuda_graphs) and args.profile_timing:
        raise SystemExit("[error] --compile_update and --profile_timing are mutually exclusive: the "
                         "per-block cuda syncs force graph breaks and make the timings meaningless.")
    if args.cuda_graphs and DIST.world_size > 1:
        # Refused rather than shipped untested. Two independent problems, both of which would
        # corrupt training silently rather than crash:
        #   1. GradSync's all_reduce sits INSIDE the region CudaGraphModule captures. NCCL
        #      collectives are capturable only under specific conditions, and a mis-capture
        #      replays a stale reduction every update -- numerically plausible, completely wrong.
        #   2. synced UnitBatchNorm carries a deliberate @torch._dynamo.disable graph break (it has
        #      to: compiling that collective produced ~1e-2 gradient errors), and a graph break
        #      inside a captured region is not something CudaGraphModule can honour.
        # The single-GPU path is unaffected -- this only fires at world_size > 1.
        raise SystemExit(
            "[error] --cuda_graphs is not supported at world_size > 1 (this run has "
            f"{DIST.world_size}). The gradient all-reduce and the synced-BatchNorm graph break "
            "both sit inside the captured region, and a mis-capture there fails silently rather "
            "than loudly. Use --compile_update instead (verified equivalent to single-GPU by "
            "tests/test_distributed_equivalence.py --compile), or run --cuda_graphs on 1 GPU."
        )
    graph_fns = None
    if args.cuda_graphs:
        graph_fns = build_graph_update(
            args, encoder, encoder_target, qf1, qf2, qf1_target, qf2_target, actor,
            encoder_optimizer, q_optimizer, actor_optimizer,
            a_optimizer if args.autotune else None, log_alpha,
            target_entropy if args.autotune else 0.0,
            ema_target_params, ema_online_params,
            normalize_obs, normalize_critic_obs, norm_proprio, syncs)
        rprint("[perf] --cuda_graphs on: update captured as 3 CUDA graphs over compiled regions "
              "(main / actor / normalize+EMA). Expect ~40 s warmup.")

    compiled_update = None
    if args.compile_update and not args.cuda_graphs:
        compiled_update = build_compiled_update(
            args, encoder, encoder_target, qf1, qf2, qf1_target, qf2_target, actor,
            encoder_optimizer, q_optimizer, actor_optimizer,
            a_optimizer if args.autotune else None, log_alpha,
            target_entropy if args.autotune else 0.0,
            ema_target_params, ema_online_params,
            normalize_obs, normalize_critic_obs, norm_proprio, syncs)
        rprint("[perf] --compile_update on: update runs as one torch.compile region "
              "(expect ~25 s warmup on the first updates)")

    timer = BlockTimer(args.profile_timing, device)
    if args.profile_timing:
        rprint("[profile] --profile_timing on: CUDA syncs added around every block; SPS is not "
              "comparable to a normal run.")

    # TRY NOT TO MODIFY: start the game
    # Rank-offset seed: each replica must roll out a DIFFERENT trajectory, or the extra GPUs would
    # fill their buffers with near-duplicate data and add nothing.
    milestone("setup complete; starting rollout")
    obs, _ = envs.reset(seed=seed)
    global_step = 0
    while global_step < args.num_learning_iterations:
        timer.start("total/iteration")
        with timer.block("rollout/act"):
            with torch.no_grad():
                policy_obs = obs["policy"].reshape(obs["policy"].shape[0], -1)
                if args.obs_normalization:
                    policy_obs = normalize_obs(policy_obs, update=False)

            with torch.no_grad():
                policy_zs = encoder.zs(policy_obs, proprio=get_proprio(obs, update=False))
                actions, cached_noise, noise_repeat_count, noise_repeat_n = _sample_rollout_action(
                    actor, policy_zs, cached_noise, noise_repeat_count, noise_repeat_n, zeta_cdf, training=False
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
                        encoder.zs(kl_ref_obs, proprio=kl_ref_proprio), training=False
                    )
                    old_mean = old_mean.detach()
                    old_log_std = old_log_std.detach()

            for upd_i in range(n_updates_this_step):
                # rb.sample() gathers on the buffer device then moves the batch host->device when
                # --buffer_on_cpu; for image observations that copy usually dwarfs the indexing.
                with timer.block("buffer/sample_H2D"):
                    data = sample_batch(args.batch_size)

                if graph_fns is not None:
                    g_main, g_actor, g_tail = graph_fns
                    (zs_d, critic_obs_d, qf_loss, qf1_loss, qf2_loss, target_values,
                     model_loss, dyn_loss, reward_loss, done_loss) = g_main(
                        data.policy_observations, data.next_policy_observations,
                        data.critic_observations, data.next_critic_observations,
                        data.proprio_observations, data.next_proprio_observations,
                        data.actions, data.rewards, data.dones, data.effective_n_steps)
                    qf1_target_values, qf2_target_values = target_values[0], target_values[1]
                    if update_step % args.policy_frequency == 0:
                        actor_loss, alpha_loss, policy_entropy, log_pi = g_actor(zs_d, critic_obs_d)
                        alpha = log_alpha.detach().exp().squeeze()
                    # normalize_parameters + Polyak EMA, kept after the actor update as in the
                    # eager path (see build_graph_update docstring).
                    g_tail(zs_d)
                    update_step += 1
                    continue

                if compiled_update is not None:
                    out = compiled_update(data, update_step % args.policy_frequency == 0)
                    qf_loss = out["qf_loss"]; qf1_loss = out["qf1_loss"]; qf2_loss = out["qf2_loss"]
                    target_values = out["target_values"]
                    qf1_target_values, qf2_target_values = target_values[0], target_values[1]
                    if update_step % args.policy_frequency == 0:
                        # Critic-only updates leave these untouched in the eager path; overwriting
                        # them with the compiled function's zero placeholders would log 0.0000
                        # whenever the last update of a step was not an actor update.
                        actor_loss = out["actor_loss"]; alpha_loss = out["alpha_loss"]
                        policy_entropy = out["policy_entropy"]; log_pi = out["log_pi"]
                    model_loss = out["model_loss"]; dyn_loss = out["dyn_loss"]
                    reward_loss = out["reward_loss"]; done_loss = out["done_loss"]
                    alpha = log_alpha.detach().exp().squeeze()
                    update_step += 1
                    continue

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

                    # Proprio joins the state embedding inside encoder.zs (concatenated after the conv
                    # trunk); it is otherwise untouched by the image path.
                    proprio = norm_proprio(data.proprio_observations, update=True)
                    next_proprio = norm_proprio(data.next_proprio_observations, update=True)

                bootstrap = (~data.dones.bool()).float()

                with torch.no_grad():
                    discount = args.gamma ** data.effective_n_steps

                # ── MR.Q on the ACTOR/vision stream: f/g/model, trained by the auxiliary loss only ──
                timer.start("model/encoder_fwd")
                zs = encoder.zs(policy_obs, proprio=proprio)             # fused vision+proprio latent
                zsa = encoder.zsa(zs, data.actions)                      # only feeds the MDP head
                next_zs_hat, reward_hat, done_logit = encoder.predict(zsa)
                timer.stop("model/encoder_fwd")

                with timer.block("model/target_encoder_fwd"), torch.no_grad():
                    # Target latent of the (n-step) next observation, on the same vision stream.
                    # Reused below as the actor's input for the TD target action.
                    next_zs_target = encoder_target.zs(next_policy_obs, proprio=next_proprio)

                # data.rewards / data.dones are (B,) 1-D; reshape to (B, 1) to match the predictions
                # (otherwise MSE/BCE would broadcast (B,1) vs (B,) into a (B, B) matrix).
                with timer.block("model/aux_loss_bwd"):
                    dyn_loss = F.mse_loss(next_zs_hat, next_zs_target)
                    reward_loss = F.mse_loss(reward_hat, data.rewards.reshape(-1, 1))
                    done_loss = F.binary_cross_entropy_with_logits(done_logit, data.dones.reshape(-1, 1).float())
                    model_loss = (
                        args.model_dynamics_coef * dyn_loss
                        + args.model_reward_coef * reward_loss
                        + args.model_done_coef * done_loss
                    )
                    encoder_optimizer.zero_grad(set_to_none=False)
                    model_loss.backward()
                    syncs.encoder()
                    encoder_optimizer.step()

                # ── C51 target on the STATE critic (no encoder in this path) ──
                with timer.block("critic/c51_target"), torch.no_grad():
                    # Action comes from the actor, which reads the vision latent; the critic that
                    # scores it reads the state observation.
                    next_state_actions, next_state_log_probs, _ = actor.get_action(next_zs_target, training=False)
                    qf1_target_next_dist = F.softmax(
                        qf1_target(next_critic_obs, next_state_actions, training=True), dim=-1
                    )
                    qf2_target_next_dist = F.softmax(
                        qf2_target(next_critic_obs, next_state_actions, training=True), dim=-1
                    )
                    reward_term = (
                        data.rewards.squeeze(-1)
                        - discount * bootstrap.squeeze(-1) * log_alpha.exp() * next_state_log_probs.squeeze(-1)
                    )
                    # qf1_target/qf2_target share a support, so the atom indices are identical --
                    # project both in one call rather than repeating ~20 elementwise kernels twice.
                    target_dists = qf1_target.project_ensemble(
                        torch.stack([qf1_target_next_dist, qf2_target_next_dist]),
                        reward_term, bootstrap.squeeze(-1), discount,
                    )
                    target_values = torch.sum(target_dists * qf1_target.q_support, dim=-1)
                    qf1_target_values, qf2_target_values = target_values[0], target_values[1]
                    use_qf1 = (qf1_target_values <= qf2_target_values).unsqueeze(-1)
                    qf_target_dist = torch.where(use_qf1, target_dists[0], target_dists[1])

                # Critic loss straight on the state observation -- nothing to detach, the critic owns
                # every parameter on this path.
                with timer.block("critic/loss_bwd"):
                    qf1_log_probs = F.log_softmax(qf1(critic_obs, data.actions, training=True), dim=-1)
                    qf2_log_probs = F.log_softmax(qf2(critic_obs, data.actions, training=True), dim=-1)
                    qf1_loss = -torch.sum(qf_target_dist * qf1_log_probs, dim=-1)
                    qf2_loss = -torch.sum(qf_target_dist * qf2_log_probs, dim=-1)
                    qf_loss = torch.stack([qf1_loss, qf2_loss]).mean(dim=1).sum(dim=0)

                    # optimize the model
                    q_optimizer.zero_grad(set_to_none=False)
                    qf_loss.backward()
                    syncs.q()
                    q_optimizer.step()

                if update_step % args.policy_frequency == 0:
                    # Decoupled policy update: the actor reads the vision latent DETACHED, so the
                    # actor loss never trains the encoder (MR.Q: "the value and policy networks are
                    # trained independently from the encoders"). The critic scores the proposed
                    # action against the state observation.
                    with timer.block("actor/update"):
                        zs_pi = zs.detach()
                        pi, log_pi, _ = actor.get_action(zs_pi, training=True)
                        with torch.no_grad():
                            policy_entropy = -log_pi.mean()
                        qf1_pi = qf1(critic_obs, pi, training=False)
                        qf2_pi = qf2(critic_obs, pi, training=False)
                        qf1_probs = F.softmax(qf1_pi, dim=-1)
                        qf2_probs = F.softmax(qf2_pi, dim=-1)
                        qf1_value = qf1.get_value(qf1_probs)
                        qf2_value = qf2.get_value(qf2_probs)
                        min_qf_pi = torch.min(qf1_value, qf2_value)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad(set_to_none=False)
                        actor_loss.backward()
                        syncs.actor()
                        actor_optimizer.step()

                    if args.autotune:
                        with timer.block("actor/alpha_update"):
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(zs_pi, training=False)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad(set_to_none=False)
                            alpha_loss.backward()
                            syncs.alpha()
                            a_optimizer.step()
                            # Kept as a detached 0-dim DEVICE tensor rather than .item(): .item() is a
                            # device->host sync, and at high UTD it runs once per policy update
                            # (num_updates/policy_frequency times per env step -- 128x at UTD 256),
                            # stalling the CPU until the GPU drains each time. detach() preserves the
                            # old semantics, since the actor loss must not backprop into log_alpha.
                            alpha = log_alpha.detach().exp().squeeze()

                update_step += 1

                # Normalize the params
                with timer.block("update/param_normalize"):
                    actor.normalize_parameters()
                    qf1.normalize_parameters()
                    qf2.normalize_parameters()

                # update the target networks (critics + MR.Q encoder)
                if global_step % args.target_network_frequency == 0:
                    with timer.block("update/target_ema"), torch.no_grad():
                        # lerp_(online, tau) == (1 - tau) * target + tau * online, i.e. exactly the
                        # Polyak update this replaces -- but batched over every parameter tensor.
                        torch._foreach_lerp_(ema_target_params, ema_online_params, args.tau)
            
        
            # KL(new policy || old policy) on the reference batch, measured over this epoch's
            # num_updates gradient steps. Closed-form KL between diagonal Gaussians, computed in
            # pre-tanh space (valid regardless of use_tanh, since KL is invariant under a shared
            # deterministic bijection).
            if args.enable_kl_diag:
                with timer.block("diag/kl"), torch.no_grad():
                    new_mean, new_log_std = actor(
                        encoder.zs(kl_ref_obs, proprio=kl_ref_proprio), training=False
                    )
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
                # Split into setup vs the Lanczos iteration itself: the latter is the double-backward
                # loop (hessian_lanczos_iters HVPs, each a full backward through the retained graph)
                # and is normally the whole cost of this diagnostic.
                timer.start("total/hessian_diag")
                with timer.block("diag/hessian_setup"):
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
                        h_next_zs_pi = encoder_target.zs(h_next_policy_obs, proprio=h_next_proprio)
                        h_next_actions, h_next_log_probs, _ = actor.get_action(h_next_zs_pi, training=False)
                        h_discount = args.gamma ** h_data.effective_n_steps
                        qf1_target_dist_h = qf1_target.projection(
                            h_next_critic_obs,
                            h_next_actions,
                            h_data.rewards.squeeze(-1) - h_discount * h_bootstrap.squeeze(-1) * log_alpha.exp() * h_next_log_probs.squeeze(-1),
                            h_bootstrap.squeeze(-1),
                            h_discount,
                            training=True,
                        )

                    qf1_outputs_h = qf1(h_critic_obs, h_data.actions, training=False)
                    qf1_log_probs_h = F.log_softmax(qf1_outputs_h, dim=-1)
                    qf1_loss_h = (-torch.sum(qf1_target_dist_h * qf1_log_probs_h, dim=-1)).mean()
                    qf1_params = list(qf1.parameters())

                with timer.block("diag/hessian_lanczos"):
                    hess_lambda_max, hess_lambda_min = lanczos_extreme_eigenvalues(
                        qf1_loss_h, qf1_params, num_iters=args.hessian_lanczos_iters, device=device
                    )
                hess_cond_number = abs(hess_lambda_max) / max(abs(hess_lambda_min), 1e-12)
                timer.stop("total/hessian_diag")

                # log10 versions for log-scale plotting: these metrics (per arxiv.org/abs/2509.25174)
                # commonly span several orders of magnitude, so a linear-axis chart of the raw value
                # is hard to read. top_eig can in principle be <= 0 (non-convex loss), so we take
                # log10 of its magnitude and clamp both away from 0 to avoid log(0).
                writer.add_scalar("metrics/critic_hessian_top_eig_log10", math.log10(max(abs(hess_lambda_max), 1e-12)), global_step)
                writer.add_scalar("metrics/critic_hessian_cond_number_log10", math.log10(max(hess_cond_number, 1e-12)), global_step)

            # Rank 0 owns all logging. The reported episode statistics are therefore rank 0's own
            # envs (num_envs of the world_size * num_envs total) -- a fair sample of the same
            # policy, since every replica runs identical weights against identically-configured
            # envs, but a 1/world_size sample, so it is noisier than a single-GPU run's.
            # Deliberately NOT all-reduced: these are pure diagnostics, and a collective here would
            # be one more place for the ranks to fall out of step.
            if not DIST.is_main and global_step % args.logging_interval == 0:
                num_success_episodes_log = 0
                num_episodes_log = 0
            if DIST.is_main and global_step % args.logging_interval == 0:
                # Every .item() below is a device->host sync; with many scalars this is not free.
                timer.start("io/logging")
                writer.add_scalar("losses/qf1_values", qf1_target_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_target_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.mean().item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/model_loss", model_loss.item(), global_step)
                writer.add_scalar("losses/model_dynamics_loss", dyn_loss.item(), global_step)
                writer.add_scalar("losses/model_reward_loss", reward_loss.item(), global_step)
                writer.add_scalar("losses/model_done_loss", done_loss.item(), global_step)
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
                    f"{'qf1 / qf2 value:':>{pad}} {qf1_target_values.mean().item():>10.3f}  /  {qf2_target_values.mean().item():>10.3f}",
                    f"{'qf_loss:':>{pad}} {(qf_loss.item() / 2.0):>10.4f}",
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

        # Periodic checkpoint save (after training starts so model isn't all-zeros).
        # Rank 0 only: every replica holds bitwise-identical weights, so N copies would be N-1
        # redundant multi-GB writes racing for the same path.
        if args.save_model and DIST.is_main and global_step > args.learning_starts:
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
                    "encoder_target": encoder_target.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "qf1_target": qf1_target.state_dict(),
                    "qf2_target": qf2_target.state_dict(),
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

    # Final checkpoint at end of training (rank 0 only, as above)
    if args.save_model and DIST.is_main:
        ckpt_dir = f"{args.ckpt_dir}/{run_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f"{ckpt_dir}/model_final.pt"
        ckpt = {
            "actor": actor.state_dict(),
            "encoder": encoder.state_dict(),
            "encoder_target": encoder_target.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
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

    # Collectives are finished with BEFORE the environments are torn down. Ordering matters: the
    # barrier exists so no rank destroys its communicator while rank 0 is still writing the final
    # checkpoint (that would abort the write), but putting it *after* envs.close() would mean one
    # rank's slow Isaac shutdown blocks every other rank inside the barrier indefinitely. Sim
    # teardown needs no coordination, so each rank does it independently afterwards.
    if DIST.enabled:
        dist.barrier()
        dist.destroy_process_group()

    envs.close()
    writer.close()
