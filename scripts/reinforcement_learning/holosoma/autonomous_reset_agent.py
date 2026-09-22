# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FastSAC finetuning with AUTONOMOUS resets driven by a goal-conditioned PPO policy.

The cycle, per iteration:

    [collect] N_c steps   FastSAC acts, replay buffer ON,  no gradient updates
    [update]  K gradient updates (critic; actor every ``policy_frequency``-th)
    [reset]   N_r steps   GC policy acts, replay buffer OFF, no gradient updates

The point is that nothing teleports. A scripted reset is not available on real hardware, so the
goal-conditioned policy physically drives the arm back to a state drawn from the task's own reset
distribution, and only then does the next collection episode begin. The single exception is an
``abnormal_robot`` termination, which the env resets immediately as an unrecoverable state.

This requires the ``...-GC-AutoReset-v0`` task, whose ``time_out`` is effectively disabled (an env
auto-reset at a phase boundary WOULD teleport) and which carries a ``gc`` observation group holding
the goal-conditioned policy's own inputs.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
import torch.nn as nn
import tqdm
from tensordict import TensorDict

from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from holosoma.utils.logging import logger

# Phase banner colours: green = SAC assembly rollout, orange = gradient updates, red = GC
# disassembly rollout. Chosen to read at a glance in a screen recording, not from a palette.
PHASE_LABELS = {
    "collect": ((34, 160, 60), "SAC ASSEMBLY POLICY  -  collecting"),
    "update": ((235, 140, 20), "TRAINING  -  gradient updates"),
    "reset": ((200, 40, 40), "GC DISASSEMBLY POLICY  -  autonomous reset"),
}

MIN_EPISODES_TO_LOG = 10

COLLECT, RESET = 0, 1


class GCResetPolicy(nn.Module):
    """Deterministic actor rebuilt straight from an rsl_rl PPO checkpoint.

    Constructing an ``OnPolicyRunner`` here would need a VecEnv whose observation groups match the
    GC task, which this env deliberately does not have -- it carries the GC inputs as one extra
    group. Rebuilding the MLP from the state dict avoids that plumbing entirely and cannot silently
    disagree with the checkpoint: every dimension is read from the stored weights.

    Empirical observation normalization is part of the policy (rsl_rl stores it in the same
    checkpoint), so it is applied here rather than left to the caller.
    """

    def __init__(self, checkpoint_path: str, device: str):
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]

        lin_idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
        mods: list[nn.Module] = []
        for n, i in enumerate(lin_idx):
            w = sd[f"actor.{i}.weight"]
            mods.append(nn.Linear(w.shape[1], w.shape[0]))
            if n < len(lin_idx) - 1:
                mods.append(nn.ELU())
        self.mlp = nn.Sequential(*mods)
        self.mlp.load_state_dict({k[len("actor.") :]: v for k, v in sd.items() if k.startswith("actor.")})

        if "actor_obs_normalizer._mean" in sd:
            self.register_buffer("obs_mean", sd["actor_obs_normalizer._mean"].to(device).float())
            self.register_buffer("obs_std", sd["actor_obs_normalizer._std"].to(device).float())
        else:
            self.register_buffer("obs_mean", torch.zeros(self.obs_dim, device=device))
            self.register_buffer("obs_std", torch.ones(self.obs_dim, device=device))

        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def obs_dim(self) -> int:
        return self.mlp[0].in_features

    @property
    def action_dim(self) -> int:
        return self.mlp[-1].out_features

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.obs_mean) / (self.obs_std + 1e-8))


class AutonomousResetFastSACAgent(FastSACAgent):
    """``FastSACAgent`` whose rollout alternates collection with GC-policy-driven resets."""

    def configure_autonomous_reset(
        self,
        gc_checkpoint: str,
        collect_steps: int = 160,
        reset_steps: int = 160,
        gc_obs_group: str = "gc",
    ) -> None:
        self.gc_policy = GCResetPolicy(gc_checkpoint, self.device)
        self.collect_steps = int(collect_steps)
        self.reset_steps = int(reset_steps)
        self.gc_obs_group = gc_obs_group

        base = self.env.unwrapped
        gc_dim = base.observation_manager.group_obs_dim[gc_obs_group][0]
        if gc_dim != self.gc_policy.obs_dim:
            raise ValueError(
                f"GC policy expects {self.gc_policy.obs_dim}-dim observations but the env's "
                f"'{gc_obs_group}' group is {gc_dim}-dim. The checkpoint and the task's GC "
                "observation group must match term-for-term."
            )
        act_dim = base.action_manager.total_action_dim
        if act_dim != self.gc_policy.action_dim:
            raise ValueError(
                f"GC policy outputs {self.gc_policy.action_dim} actions, env expects {act_dim}."
            )
        logger.info(
            f"autonomous reset: collect {self.collect_steps} steps -> "
            f"{int(self.config.num_updates)} update(s) -> GC reset {self.reset_steps} steps; "
            f"GC policy {gc_checkpoint} ({gc_dim} obs -> {act_dim} act)"
        )

    # ------------------------------------------------------------------ video / markers
    def configure_phase_video(
        self, path: str, steps: int = 640, update_frames: int = 30, fps: int = 30, camera_env: int = 0
    ) -> None:
        """Record an annotated mp4 of the collect -> update -> reset cycle.

        Written here rather than with ``gym.wrappers.RecordVideo`` because the update phase
        consumes no env steps: there is no frame to trigger on, so the orange segment has to be
        injected by holding a rendered frame for ``update_frames``. Frames are labelled as they are
        captured, then muxed once at the end.
        """
        import atexit

        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Fragmented MP4: frames are muxed as they arrive and the file stays playable even if the
        # process dies without finalising, so Ctrl+C (or a hard kill) still leaves a usable video
        # instead of a zero-byte stub waiting on a moov atom.
        writer = imageio.get_writer(
            path, fps=int(fps), codec="libx264", quality=9, macro_block_size=1,
            output_params=["-pix_fmt", "yuv420p", "-crf", "16",
                           "-movflags", "frag_keyframe+empty_moov+default_base_moof"],
        )
        self._video = {
            "path": path,
            "steps": int(steps),  # 0 = record for the whole run
            "update_frames": int(update_frames),
            "fps": int(fps),
            "camera_env": int(camera_env),
            "writer": writer,
            "count": 0,
            "done": False,
        }
        atexit.register(self._finish_video)
        limit = "until the run ends" if int(steps) <= 0 else f"{steps} env steps"
        logger.info(f"phase video: recording {limit} -> {path}")

    def _label_frame(self, rgb, phase_key: str):
        from PIL import Image, ImageDraw, ImageFont

        color, text = PHASE_LABELS[phase_key]
        img = Image.fromarray(rgb[:, :, :3].astype("uint8"))
        draw = ImageDraw.Draw(img)
        bar = max(28, img.height // 14)
        draw.rectangle([0, 0, img.width, bar], fill=color)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(bar * 0.55))
        except Exception:  # noqa: BLE001 - any font failure falls back to the bitmap default
            font = ImageFont.load_default()
        draw.text((int(bar * 0.4), int(bar * 0.2)), text, fill=(255, 255, 255), font=font)
        return img

    def _capture(self, phase_key: str, repeat: int = 1) -> None:
        v = getattr(self, "_video", None)
        if v is None or v["done"]:
            return
        rgb = self.env.unwrapped.render()
        if rgb is None:
            return
        import numpy as np

        img = self._label_frame(rgb, phase_key)
        # Even dimensions: libx264 rejects odd width/height.
        w, h = img.size
        arr = np.asarray(img.crop((0, 0, w - (w % 2), h - (h % 2))))
        for _ in range(repeat):
            v["writer"].append_data(arr)
        v["count"] += repeat

    def _finish_video(self) -> None:
        """Close the streaming writer. Safe to call repeatedly (atexit + the end of learn())."""
        v = getattr(self, "_video", None)
        if v is None or v["done"]:
            return
        v["done"] = True
        try:
            v["writer"].close()
        except Exception as e:  # noqa: BLE001 - never let video teardown mask a training error
            logger.warning(f"phase video close failed: {e}")
        logger.info(f"phase video written ({v['count']} frames): {v['path']}")

    def _setup_goal_markers(self) -> None:
        """Blue spheres at the disassembly policy's target keypoints, orange at the peg's own.

        The finetune task's ``progress_context`` is the plain (peghole-relative) one, so the GC
        keypoint markers wired into GCProgressContext are unavailable here -- these are drawn
        directly from the sampled ``goal_state`` instead.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.rewards import axis_keypoints_local

        def sphere(path, color, radius=0.008):
            return VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=path,
                    markers={
                        "p": sim_utils.SphereCfg(
                            radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
                        )
                    },
                )
            )

        self._kp_local = axis_keypoints_local(0.03, 4, self.device)
        self._goal_marker = sphere("/Visuals/AutoReset/goal", (0.0, 0.45, 1.0))
        self._peg_marker = sphere("/Visuals/AutoReset/peg", (1.0, 0.45, 0.0), radius=0.006)

    def _draw_goal_markers(self, reset_term, show: bool) -> None:
        if not hasattr(self, "_goal_marker"):
            return
        if not show:
            self._goal_marker.set_visibility(False)
            self._peg_marker.set_visibility(False)
            return
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp.rewards import transform_keypoints

        base = self.env.unwrapped
        origins = base.scene.env_origins
        goal = reset_term.goal_state["rigid_object"]["insertive_object"]["root_pose"]
        peg = base.scene["insertive_object"]
        gk = transform_keypoints(goal[:, :3] + origins, goal[:, 3:7], self._kp_local)
        pk = transform_keypoints(peg.data.root_pos_w, peg.data.root_quat_w, self._kp_local)
        self._goal_marker.set_visibility(True)
        self._peg_marker.set_visibility(True)
        self._goal_marker.visualize(translations=gk.reshape(-1, 3))
        self._peg_marker.visualize(translations=pk.reshape(-1, 3))

    def learn(self) -> None:  # noqa: C901
        args = self.config
        device = self.device
        update_main = self._update_main
        update_pol = self._update_pol
        policy = self.policy
        normalize_obs = self.obs_normalizer.forward
        normalize_critic_obs = self.critic_obs_normalizer.forward
        qnet, qnet_target = self.qnet, self.qnet_target
        env, rb = self.env, self.rb
        base_env = env.unwrapped
        reset_term = base_env.event_manager.get_term_cfg("reset_from_reset_states").func
        if not hasattr(reset_term, "resample_goals_uniform"):
            raise ValueError(
                "the reset event term must be a GoalConditionedMultiResetManager (use the "
                "...-GC-AutoReset-v0 task); got " + type(reset_term).__name__
            )
        all_envs = torch.arange(base_env.num_envs, device=device)
        # Episode bookkeeping has to be done here rather than read off `infos`: with time_out
        # disabled the env never observes the synthetic collect-episode boundary, so its own
        # ep_success/ep_counted only ever report abnormal terminations. The progress context's
        # per-env success flag is the ground truth we accumulate instead.
        progress_ctx = base_env.reward_manager.get_term_cfg("progress_context").func

        start_time = time.time()

        tau_eff = args.tau
        if args.no_learning:
            for _opt in (self.actor_optimizer, self.q_optimizer, self.alpha_optimizer):
                for _g in _opt.param_groups:
                    _g["lr"] = 0.0
            tau_eff = 0.0

        obs, critic_obs = env.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
        dones = None

        num_updates = max(int(args.num_updates), 1)
        critic_updates = 0
        # Phases are GLOBAL: "K updates at the end of the episode" is a single event, so every env
        # shares one counter. An env that trips abnormal_robot is teleported by the env itself and
        # simply carries on in whatever phase is current.
        phase, phase_step = COLLECT, 0
        collect_steps_total = 0

        ep_return = torch.zeros(base_env.num_envs, device=device)
        ep_length = torch.zeros(base_env.num_envs, device=device)
        rewbuffer: deque = deque(maxlen=1000)
        lenbuffer: deque = deque(maxlen=1000)
        total_episodes = 0
        num_success_episodes_log = 0
        num_episodes_log = 0
        ep_success_acc = torch.zeros(base_env.num_envs, device=device)
        curriculum_log: dict[str, float] = {}
        last_update_stats: dict[str, float] = {}
        pbar = tqdm.tqdm(total=args.num_learning_iterations, initial=self.global_step)

        while self.global_step <= args.num_learning_iterations:
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            collecting = phase == COLLECT
            with torch.no_grad(), self._maybe_amp():
                if collecting:
                    actions = policy(obs=normalize_obs(obs, update=False), dones=dones)
                else:
                    gc_obs = base_env.observation_manager.compute_group(self.gc_obs_group)
                    actions = self.gc_policy(gc_obs)

            next_obs, rewards, dones, infos = env.step(actions.float())
            next_critic_obs = infos["observations"]["critic"]
            env_done = dones.bool()  # time_out is disabled, so this is abnormal_robot only

            for key, value in (infos.get("episode") or {}).items():
                if key.startswith(("Curriculum/", "metrics/")):
                    curriculum_log[key] = value.item() if torch.is_tensor(value) else float(value)
                elif key.startswith("Episode_Termination/"):
                    term_name = key[len("Episode_Termination/") :]
                    curriculum_log[f"charts/termination_{term_name}"] = (
                        value.item() if torch.is_tensor(value) else float(value)
                    )

            if collecting:
                last_collect_step = phase_step == self.collect_steps - 1
                # The collection episode ends by fiat, not by env time_out, so the truncation flag
                # has to be synthesized. With handle_truncations=true this bootstraps the value at
                # the artificial horizon and stops the n-step return from reaching into the
                # GC-controlled steps that follow, which are never stored.
                synth = torch.full_like(dones, int(last_collect_step))
                truncations = (infos["time_outs"].long() | synth) & (~env_done).long()
                buffer_dones = (dones.long() | synth).clamp(max=1)

                ep_return += rewards
                ep_length += 1
                ep_success_acc = torch.maximum(
                    ep_success_acc, progress_ctx.success.to(device=device, dtype=torch.float)
                )
                boundary = env_done | last_collect_step
                if bool(boundary.any()):
                    rewbuffer.extend(ep_return[boundary].cpu().tolist())
                    lenbuffer.extend(ep_length[boundary].cpu().tolist())
                    num_episodes_log += int(boundary.sum().item())
                    total_episodes += int(boundary.sum().item())
                    num_success_episodes_log += int(ep_success_acc[boundary].sum().item())
                    ep_return[boundary] = 0
                    ep_length[boundary] = 0
                    ep_success_acc[boundary] = 0

                # Nothing teleports at the synthetic boundary, so next_obs already IS the true next
                # state there; only a real (abnormal) reset needs the pre-reset observation.
                true_next_obs = torch.where(env_done[:, None], infos["observations"]["final"]["actor_obs"], next_obs)
                true_next_critic_obs = torch.where(
                    env_done[:, None], infos["observations"]["final"]["critic_obs"], next_critic_obs
                )

                store_rewards = rewards
                if self.sgft_enabled and self.sgft_shaping:
                    store_rewards = self.sgft_shaped_rewards(
                        obs, critic_obs, true_next_obs, true_next_critic_obs,
                        rewards, buffer_dones, truncations, args.gamma,
                    )

                transition = TensorDict(
                    {
                        "observations": obs,
                        "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                        "next": {
                            "observations": true_next_obs,
                            "rewards": torch.as_tensor(store_rewards, device=device, dtype=torch.float),
                            "truncations": truncations.long(),
                            "dones": buffer_dones.long(),
                        },
                    },
                    batch_size=(base_env.num_envs,),
                    device=device,
                )
                transition["critic_observations"] = critic_obs
                transition["next"]["critic_observations"] = true_next_critic_obs
                rb.extend(transition if rb.n_env == base_env.num_envs else transition[: rb.n_env])
                collect_steps_total += 1

            obs = next_obs
            critic_obs = next_critic_obs
            phase_step += 1

            if getattr(self, "_video", None) is not None and not self._video["done"]:
                # Markers show only while the disassembly policy is driving: during collection the
                # goal is a leftover from the previous cycle and would misread as a live target.
                # Markers live on the reset term (created at env construction -- markers made
                # mid-run do not render). Shown only while the disassembly policy drives: during
                # collection the goal is last cycle's leftover and would misread as a live target.
                reset_term.visualize_goal_keypoints(show=not collecting)
                self._capture("collect" if collecting else "reset")
                if self._video["steps"] > 0 and self._video["count"] >= self._video["steps"]:
                    self._finish_video()

            # ---- phase transitions -------------------------------------------------------
            if collecting and phase_step >= self.collect_steps:
                # K gradient updates, once, at the end of the collection episode.
                if rb.ptr > args.learning_starts:
                    if getattr(self, "_video", None) is not None and not self._video["done"]:
                        # Updates consume no env steps, so hold a frame to make the phase visible.
                        # Inside the learning_starts guard on purpose: an orange banner must mean
                        # gradients actually ran, not merely that a boundary was reached.
                        self._capture("update", repeat=self._video["update_frames"])
                    batch_size = max(args.batch_size // rb.n_env // self.gpu_world_size, 1)
                    prepared = self._sample_and_prepare_batches(
                        batch_size, num_updates, normalize_obs, normalize_critic_obs
                    )
                    for data in prepared:
                        (
                            buffer_rewards, critic_grad_norm, qf_loss, q_values,
                            qf_max, qf_min, alpha_loss, bc_critic_loss,
                        ) = update_main(data)
                        if self.expert_critic is not None:
                            self.lambda_bc_critic *= 0.999
                        critic_updates += 1
                        if critic_updates % args.policy_frequency == 0:
                            actor_grad_norm, actor_loss, policy_entropy, action_std, bc_policy_loss = update_pol(data)
                            if self.expert_policy is not None:
                                self.lambda_bc_policy *= 0.999
                        with torch.no_grad():
                            src_ps = [p.data for p in qnet.parameters()]
                            tgt_ps = [p.data for p in qnet_target.parameters()]
                            torch._foreach_mul_(tgt_ps, 1.0 - tau_eff)
                            torch._foreach_add_(tgt_ps, src_ps, alpha=tau_eff)
                    last_update_stats["losses/qf1_values"] = q_values[0].mean().item()
                    last_update_stats["losses/qf2_values"] = q_values[1].mean().item()
                    last_update_stats["losses/qf_loss"] = qf_loss.item() / 2.0
                    last_update_stats["losses/alpha"] = float(self.log_alpha.exp())
                    if critic_updates >= args.policy_frequency:
                        last_update_stats["losses/actor_loss"] = actor_loss.item()
                    if self.config.use_autotune:
                        last_update_stats["losses/alpha_loss"] = alpha_loss.item()

                # A fresh reset target, drawn from the task's own reset distribution; the GC policy
                # now has `reset_steps` to physically get there.
                reset_term.resample_goals_uniform(all_envs)
                phase, phase_step = RESET, 0

            elif not collecting and phase_step >= self.reset_steps:
                phase, phase_step = COLLECT, 0

            # ---- logging / checkpointing -------------------------------------------------
            if self.global_step % args.logging_interval == 0:
                for _k, _v in last_update_stats.items():
                    self.writer.add_scalar(_k, _v, self.global_step)
                sps = self.global_step / max(time.time() - start_time, 1e-6)
                self.writer.add_scalar("charts/SPS", int(sps), self.global_step)
                self.writer.add_scalar("Perf/total_fps", int(sps * base_env.num_envs), self.global_step)
                self.writer.add_scalar("charts/collect_steps", collect_steps_total, self.global_step)
                self.writer.add_scalar("charts/phase", phase, self.global_step)
                if len(rewbuffer) >= MIN_EPISODES_TO_LOG:
                    self.writer.add_scalar("charts/episodic_return", statistics.mean(rewbuffer), self.global_step)
                    self.writer.add_scalar("charts/episodic_length", statistics.mean(lenbuffer), self.global_step)
                if num_episodes_log >= MIN_EPISODES_TO_LOG:
                    self.writer.add_scalar(
                        "charts/success_rate", num_success_episodes_log / num_episodes_log, self.global_step
                    )
                    num_success_episodes_log = 0
                    num_episodes_log = 0
                self.writer.add_scalar("charts/num_episodes", total_episodes, self.global_step)
                for key, value in curriculum_log.items():
                    self.writer.add_scalar(key, value, self.global_step)

            if args.save_interval > 0 and self.global_step > 0 and self.global_step % args.save_interval == 0:
                if self.is_main_process:
                    logger.info(f"Saving model at global step {self.global_step}")
                    self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
            if (
                args.save_replay_buffer_interval > 0
                and self.global_step > 0
                and self.global_step % args.save_replay_buffer_interval == 0
            ):
                if self.is_main_process:
                    self.save_replay_buffer(
                        os.path.join(self.log_dir, f"replay_buffer_{self.global_step:07d}.pt")
                    )

            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1
            pbar.update(1)

        self._finish_video()
        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
