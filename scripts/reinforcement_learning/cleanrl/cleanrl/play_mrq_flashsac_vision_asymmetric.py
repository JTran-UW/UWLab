# Copyright (c) 2024-2025, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an MR.Q + FlashSAC checkpoint on the asymmetric vision peg-insertion task.

The generic ``play.py`` only knows the flat state-based SAC/PPO actors; these checkpoints pair an
MR.Q image encoder with a FlashSAC actor over the latent, plus a separate state critic, so they need
their own loader. Networks are rebuilt from the ``args`` dict saved inside the checkpoint rather
than from this script's flags, which keeps width/depth hyperparameters in sync with training
automatically.

Point ``--env_id`` at the Play task matching the render mode the checkpoint was trained under:

    OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-FastRender-Play-v0
    OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-Play-v0

Example:

    python play_mrq_flashsac_vision_asymmetric.py \
        --checkpoint checkpoints/MRQ-GS-Asym-NU5-BS1024-L40S/model_final.pt \
        --num_envs 32 --num_episodes 200 --enable_cameras --headless
"""

from dataclasses import dataclass
import random
import statistics

import numpy as np
import torch
import tyro

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cleanrl_utils.utils import EmpiricalNormalization

from vecenv_wrapper import IsaacLabVectorEnv


@dataclass
class Args:
    checkpoint: str = ""
    """path to a model_*.pt written by mrq_flashsac_continuous_action_vision_asymmetric.py"""
    env_id: str = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Grayscale-Asymmetric-FastRender-Play-v0"
    """id of the eval env; must match the render mode the checkpoint was trained on"""
    num_envs: int = 1
    """number of parallel envs"""
    num_episodes: int = 100
    """stop once this many episodes have finished across all envs"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """use or do not use cuda"""
    stochastic: bool = False
    """sample from the policy instead of taking the distribution mean"""
    report_q: bool = True
    """also load qf1 and report its mean value estimate over the rollout"""
    video: bool = False
    """record a video: the viewport camera as the main pane, with the policy's own downsampled
    camera observations tiled down the right-hand side. Requires --enable_cameras."""
    video_path: str = "play_video.mp4"
    """where to write the video"""
    video_fps: int = 20
    """frames per second; one frame is captured per env step"""
    video_env: int = 0
    """which env's observations to show in the side tiles (the viewport shows the whole scene)"""
    video_max_frames: int = 600
    """stop recording after this many frames (the rollout itself continues)"""
    saliency: bool = False
    """add a saliency column to the video: |d(||a||^2)/dx| per input pixel, i.e. which pixels the
    action is most sensitive to. Costs one extra forward+backward per recorded frame."""
    saliency_blur: int = 1
    """box-blur radius (in downsampled pixels) applied to the saliency map. Raw per-pixel gradients
    are noisy at 32x32; 1-2 makes the structure readable without inventing detail."""
    camera_names: str = ""
    """comma-separated tile labels, outermost-first. Empty = infer from the camera count
    (2 -> side,wrist ; 3 -> front,side,wrist), matching the task cfgs' sensor_cfgs order."""


# Imported for its network definitions only; the module guards its training loop behind __main__.
import mrq_flashsac_continuous_action_vision_asymmetric as mrq


def _infer_camera_names(n: int) -> list[str]:
    """Tile labels in sensor_cfgs order, which is how the channels are concatenated."""
    return {2: ["side", "wrist"], 3: ["front", "side", "wrist"]}.get(n, [f"cam{i}" for i in range(n)])


def compute_saliency(encoder, actor, normalizer, policy_obs_flat, proprio, obs_shape, blur=1):
    """Vanilla-gradient saliency: |d(||a||^2)/dx| for a single env, shaped like the observation.

    Must run OUTSIDE torch.no_grad(). The actor consumes zs without a detach in this script -- the
    detach during training is what keeps the actor loss from reaching the encoder, it is not part of
    the forward pass -- so the gradient flows encoder -> actor all the way back to the raw pixels.

    The deterministic mean action is used as the target, so the map does not depend on the policy's
    sampling noise. proprio is passed detached: this asks what the IMAGE contributes, not the
    proprioceptive channel.
    """
    x = policy_obs_flat.detach().clone().requires_grad_(True)
    if normalizer is not None:
        # EmpiricalNormalization.forward is decorated @torch.no_grad(), so calling it here would
        # sever the graph at the first op and autograd would report x as unused. Apply the same
        # affine transform inline instead -- mean/std are constants w.r.t. x, so the gradient is
        # simply scaled by 1/std per dimension, which is part of the true pixel sensitivity.
        xn = (x - normalizer.mean) / (normalizer.std + normalizer.eps)
    else:
        xn = x
    zs = encoder.zs(xn, proprio=None if proprio is None else proprio.detach())
    _, _, mean_action = actor.get_action(zs, training=False)
    (g,) = torch.autograd.grad(mean_action.pow(2).sum(), x)
    sal = g.abs().reshape(obs_shape).detach()
    if blur and blur > 0:
        k = 2 * int(blur) + 1
        flat = sal.reshape(-1, 1, *obs_shape[-2:])
        sal = torch.nn.functional.avg_pool2d(flat, k, stride=1, padding=k // 2).reshape(obs_shape)
    return sal


def _gray_tile(img, size):
    """Min-max stretched grayscale tile as RGB uint8."""
    from PIL import Image

    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    g = (img * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(g).resize((size, size), Image.NEAREST).convert("RGB"))


def _heat_tile(img, size, vmax=None):
    """Saliency as a perceptually-uniform heat tile (dark = insensitive, bright = sensitive).

    ``vmax`` is supplied by the caller so every camera in a frame shares one colour scale -- without
    it each tile self-normalizes and a camera the policy barely uses looks as hot as one it depends
    on, which is exactly the comparison the panel is meant to support.
    """
    import matplotlib
    from PIL import Image

    hi = float(vmax if vmax is not None else img.max())
    norm = img / hi if hi > 0 else np.zeros_like(img)
    rgb = (matplotlib.colormaps["inferno"](np.clip(norm, 0, 1))[..., :3] * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.NEAREST))


def compose_frame(viewport, policy_obs_env, cam_names, hud=None, saliency=None):
    """Viewport as the main pane; the policy's own camera observations tiled down the right.

    ``policy_obs_env`` is the RAW (pre-normalization) observation for one env, shaped
    (history, cameras, H, W) -- exactly the pixels the encoder consumes after grayscale conversion
    and downsampling. The most recent history frame is shown.

    With ``saliency`` supplied the panel gains a second column: |d(||a||^2)/dx| for the same frame,
    so each row reads "what the camera sees" next to "what the action responds to".

    Observation tiles are min-max stretched per camera: a 32x32 grayscale crop of a mostly dark
    scene is unreadable at true contrast, and the pane exists to show WHAT the policy sees, not to
    judge absolute brightness. Saliency tiles are normalized per camera for the same reason -- so
    brightness is comparable WITHIN a tile, never across tiles. Saliency tiles instead share ONE
    scale across cameras, and each is labelled with its share of the frame's total gradient
    magnitude -- so "which camera is the action most sensitive to" is answerable both visually and
    numerically, the latter independent of any normalization choice.
    """
    from PIL import Image, ImageDraw

    vp = np.ascontiguousarray(viewport[..., :3]).astype(np.uint8)
    vh, vw = vp.shape[:2]

    frames = policy_obs_env[-1]                      # most recent history frame -> (cams, H, W)
    sal = saliency[-1] if saliency is not None else None
    n = frames.shape[0]
    cols = 2 if sal is not None else 1
    tile = max(1, vh // max(n, 1))                   # tiles stack to fill the viewport height
    panel = np.zeros((vh, tile * cols, 3), dtype=np.uint8)

    # One colour scale for every camera in this frame, so tile brightness is comparable ACROSS
    # cameras. `share` is the fraction of the frame's total gradient magnitude landing on each
    # camera -- scale-invariant, so it stays comparable across frames as well as across cameras.
    vmax = float(sal.max()) if sal is not None else None
    total = float(sal.sum()) if sal is not None else 0.0
    shares = [float(sal[i].sum()) / total if total > 0 else 0.0 for i in range(n)] if sal is not None else []

    for i in range(n):
        y0 = i * tile
        panel[y0:y0 + tile, 0:tile] = _gray_tile(frames[i].astype(np.float32), tile)
        if sal is not None:
            panel[y0:y0 + tile, tile:2 * tile] = _heat_tile(sal[i].astype(np.float32), tile, vmax=vmax)

    canvas = np.concatenate([vp, panel], axis=1)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    for i in range(n):
        y0 = i * tile
        label = cam_names[i] if i < len(cam_names) else f"cam{i}"
        for c in range(cols):
            x0 = vw + c * tile
            d.rectangle([x0, y0, x0 + tile - 1, y0 + tile - 1], outline=(60, 60, 60))
        d.text((vw + 5, y0 + 4), f"{label}  {frames.shape[-2]}x{frames.shape[-1]}", fill=(255, 255, 255))
        if sal is not None:
            d.text((vw + tile + 5, y0 + 4),
                   f"{label}  saliency  {shares[i]*100:4.1f}%", fill=(255, 255, 255))
    if hud:
        d.text((8, 8), hud, fill=(255, 255, 255))
    return np.asarray(im)


def build_agent(envs, ckpt, device, report_q, env_id):
    """Rebuild encoder/actor/critic/normalizers from a checkpoint and its embedded training args."""
    targs = ckpt["args"]

    policy_obs_shape = envs.single_observation_space["policy"].shape
    action_dim = int(np.prod(envs.single_action_space.shape))
    critic_obs_dim = int(np.prod(envs.single_observation_space["critic"].shape))
    policy_obs_dim = int(np.prod(policy_obs_shape))

    has_proprio = "proprio" in envs.single_observation_space.spaces
    proprio_dim = int(np.prod(envs.single_observation_space["proprio"].shape)) if has_proprio else 0
    print(f"[obs] actor {tuple(policy_obs_shape)} image | critic {critic_obs_dim}-d state | proprio {proprio_dim}")

    encoder = mrq.MRQEncoder(
        policy_obs_shape,
        action_dim,
        zs_dim=targs["zs_dim"],
        za_dim=targs["za_dim"],
        zsa_dim=targs["zsa_dim"],
        hidden_dim=targs["encoder_hidden_dim"],
        num_channels=targs["encoder_num_channels"],
        proprio_dim=proprio_dim,
        proprio_fuse_layers=targs["proprio_fuse_layers"],
        # Read from the checkpoint, not the constructor default: the trunk depth has to match what
        # the weights were trained with. Checkpoints predating this arg used the fixed 4-layer stack.
        conv_layers=targs.get("encoder_conv_layers", 4),
    ).to(device)
    actor = mrq.FlashSACActor(
        envs,
        targs["zs_dim"],
        num_blocks=mrq.infer_num_blocks(ckpt["actor"], default=targs.get("actor_num_blocks", 2)),
        use_tanh=targs["use_tanh"],
    ).to(device)

    # A mismatch here almost always means --env_id disagrees with the training task (different
    # camera stack, proprio group or critic terms), so say that rather than surfacing a raw
    # size-mismatch traceback.
    try:
        encoder.load_state_dict(ckpt["encoder"])
        actor.load_state_dict(ckpt["actor"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint does not fit env '{env_id}'. It was trained on '{targs.get('env_id')}' -- pass the Play "
            f"task matching that env's camera stack and render mode. Original error:\n{e}"
        ) from e
    encoder.eval()
    actor.eval()

    qf1 = None
    if report_q:
        # Read the atom count off the checkpoint rather than trusting the constructor default,
        # so checkpoints trained with --num_atoms still load.
        num_atoms = ckpt["qf1"]["linear2.linear.weight"].shape[0]
        qf1 = mrq.FlashSACQNetwork(
            critic_obs_dim,
            action_dim,
            num_atoms=num_atoms,
            num_blocks=mrq.infer_num_blocks(ckpt["qf1"], default=targs.get("critic_num_blocks", 2)),
        ).to(device)
        qf1.load_state_dict(ckpt["qf1"])
        qf1.eval()

    normalizers = {"actor": None, "critic": None, "proprio": None}
    if targs["obs_normalization"]:
        normalizers["actor"] = EmpiricalNormalization(shape=(policy_obs_dim,), device=device)
        normalizers["actor"].load_state_dict(ckpt["actor_obs_normalizer"])
        normalizers["critic"] = EmpiricalNormalization(shape=(critic_obs_dim,), device=device)
        normalizers["critic"].load_state_dict(ckpt["critic_obs_normalizer"])
        if has_proprio and ckpt.get("proprio_obs_normalizer") is not None:
            normalizers["proprio"] = EmpiricalNormalization(shape=(proprio_dim,), device=device)
            normalizers["proprio"].load_state_dict(ckpt["proprio_obs_normalizer"])

    return encoder, actor, qf1, normalizers, has_proprio


if __name__ == "__main__":
    args, launcher_args = tyro.cli(Args, return_unknown_args=True)
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = IsaacLabVectorEnv(args.env_id, args.num_envs, launcher_args=launcher_args,
                             render_mode="rgb_array" if args.video else None)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"[ckpt] {args.checkpoint} | global_step={ckpt.get('global_step')} | trained on {ckpt['args'].get('env_id')}")
    encoder, actor, qf1, normalizers, has_proprio = build_agent(envs, ckpt, device, args.report_q, args.env_id)

    def get_proprio(obs):
        if not has_proprio:
            return None
        p = obs["proprio"].reshape(obs["proprio"].shape[0], -1)
        return normalizers["proprio"](p, update=False) if normalizers["proprio"] is not None else p

    returns: list[float] = []
    lengths: list[float] = []
    q_values: list[float] = []
    num_episodes = 0
    num_successes = 0.0

    video_frames: list = []
    cam_names: list[str] = []
    if args.video:
        n_cams = envs.single_observation_space["policy"].shape[1]
        cam_names = ([c.strip() for c in args.camera_names.split(",") if c.strip()]
                     if args.camera_names else _infer_camera_names(n_cams))
        print(f"[video] recording -> {args.video_path} | tiles: {cam_names} | max {args.video_max_frames} frames")

    obs, _ = envs.reset(seed=args.seed)
    while num_episodes < args.num_episodes:
        with torch.no_grad():
            policy_obs = obs["policy"].reshape(obs["policy"].shape[0], -1)
            if normalizers["actor"] is not None:
                policy_obs = normalizers["actor"](policy_obs, update=False)
            zs = encoder.zs(policy_obs, proprio=get_proprio(obs))
            # get_action returns (sample, log_prob, mean); index 2 is the deterministic action.
            sampled, _, mean_action = actor.get_action(zs, training=False)
            actions = sampled if args.stochastic else mean_action

            if qf1 is not None:
                critic_obs = obs["critic"].reshape(obs["critic"].shape[0], -1)
                if normalizers["critic"] is not None:
                    critic_obs = normalizers["critic"](critic_obs, update=False)
                # C51 critic: logits over atoms, so take the expectation under the softmax.
                logits = qf1(critic_obs, actions, training=False)
                q = (torch.softmax(logits, dim=1) * qf1.q_support.to(logits.device)).sum(dim=1)
                q_values.append(q.mean().item())

        if args.video and len(video_frames) < args.video_max_frames:
            vp = envs.env.render()
            if vp is not None:
                # obs["policy"] here is the RAW observation the actor just consumed, before
                # normalization -- which is what makes the tiles a faithful view of its input.
                po = obs["policy"][args.video_env].detach().float().cpu().numpy()
                sal_np = None
                if args.saliency:
                    # Separate grad-enabled pass on this env alone; the rollout above stays no_grad.
                    e = args.video_env
                    pr = get_proprio(obs)
                    sal_np = compute_saliency(
                        encoder, actor, normalizers["actor"],
                        obs["policy"][e:e + 1].reshape(1, -1),
                        None if pr is None else pr[e:e + 1],
                        tuple(obs["policy"].shape[1:]),
                        blur=args.saliency_blur,
                    ).float().cpu().numpy()
                hud = f"step {len(video_frames)}  ep {num_episodes}/{args.num_episodes}"
                video_frames.append(compose_frame(vp, po, cam_names, hud, saliency=sal_np))

        obs, rewards, terminations, truncations, infos = envs.step(actions)

        finished = int((terminations | truncations).int().sum().item())
        if finished:
            returns.extend(infos["ep_reward"].cpu().tolist())
            lengths.extend(infos["ep_length"].cpu().tolist())
            num_episodes += finished
            # Set by the env's progress context, aligned with the envs that just reset.
            ep_success = infos.get("ep_success", None)
            if ep_success is not None:
                num_successes += float(ep_success.float().sum().item())
            print(f"[eval] {num_episodes}/{args.num_episodes} episodes", flush=True)

    print("\n" + "─" * 60)
    print(f"checkpoint         : {args.checkpoint}")
    print(f"env                : {args.env_id}")
    print(f"episodes           : {len(returns)}")
    print(f"action             : {'sampled' if args.stochastic else 'deterministic (mean)'}")
    print(f"success rate       : {num_successes / max(len(returns), 1):.3f}")
    print(f"mean return        : {statistics.fmean(returns):.4f}"
          + (f"  (std {statistics.pstdev(returns):.4f})" if len(returns) > 1 else ""))
    print(f"mean episode length: {statistics.fmean(lengths):.1f}")
    if q_values:
        print(f"mean Q estimate    : {statistics.fmean(q_values):.4f}")
    print("─" * 60)

    if args.video and video_frames:
        import imageio.v2 as imageio

        # macro_block_size=1 keeps the exact canvas size instead of padding to a multiple of 16.
        imageio.mimsave(args.video_path, video_frames, fps=args.video_fps, macro_block_size=1)
        h, w = video_frames[0].shape[:2]
        print(f"[video] wrote {len(video_frames)} frames ({w}x{h} @ {args.video_fps} fps) -> {args.video_path}")
    elif args.video:
        print("[video] no frames captured -- env.render() returned None (is --enable_cameras set?)")

    envs.close()
