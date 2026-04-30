# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper around Isaac Lab environments for use with Holosoma's FastSACAgent.

This wrapper makes an Isaac Lab ManagerBasedRLEnv look like Holosoma's BaseTask
so it can be passed directly to FastSACEnv and FastSACAgent.
"""

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Stub objects that mimic Holosoma's BaseTask attributes
# ---------------------------------------------------------------------------

class _InitState:
    """Stub for robot_config.init_state."""
    def __init__(self):
        self.default_joint_angles: dict[str, float] = {}


class _Control:
    """Stub for robot_config.control."""
    def __init__(self):
        self.action_scale: float = 1.0


class _RobotConfig:
    """Stub for BaseTask.robot_config.

    Provides the minimal interface that FastSACEnv._compute_action_boundaries
    and FastSACAgent.setup() require. Since the UWLab task uses Cartesian OSC
    (not joint-position control), action boundaries are meaningless and we
    populate dummy values that produce torch.ones(num_actions).
    """

    def __init__(self, num_actions: int):
        self.actions_dim: int = num_actions
        self.dof_names: list[str] = [f"action_{i}" for i in range(num_actions)]
        self.dof_pos_lower_limit_list: list[float] = [-1.0] * num_actions
        self.dof_pos_upper_limit_list: list[float] = [1.0] * num_actions
        self.init_state = _InitState()
        self.control = _Control()


class _ObsGroupCfg:
    """Stub for a single observation group config entry."""

    def __init__(self, term_names: list[str], history_length: int = 1):
        # terms must be a dict-like object that is iterable over keys
        self.terms: dict[str, None] = {name: None for name in term_names}
        self.history_length: int = history_length


class _ObsManagerCfg:
    """Stub for observation_manager.cfg with a .groups dict."""

    def __init__(self, groups: dict[str, _ObsGroupCfg]):
        self.groups = groups


class _ObservationManager:
    """Stub for BaseTask.observation_manager.

    Provides get_obs_dims() and cfg.groups so that FastSACAgent.setup() can
    compute actor/critic observation dimensions.
    """

    def __init__(self, obs_dims: dict[str, int], groups: dict[str, _ObsGroupCfg]):
        self._obs_dims = obs_dims
        self.cfg = _ObsManagerCfg(groups)

    def get_obs_dims(self) -> dict[str, int]:
        return dict(self._obs_dims)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class HolosomaVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for the Holosoma library.

    Exposes the BaseTask-like interface that FastSACEnv and FastSACAgent expect
    while delegating simulation to the Isaac Lab environment underneath.

    .. caution::
        This class must be the last wrapper in the wrapper chain. This is because
        the wrapper does not follow the :class:`gym.Wrapper` interface. Any
        subsequent wrappers will need to be modified to work with this wrapper.
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # modify the action space to the clip range
        self._modify_action_space()

        # ----- BaseTask-compatible stubs -----
        self.robot_config = _RobotConfig(self.num_actions)

        # Build observation manager stub from Isaac Lab's observation space
        isaaclab_obs_mgr = self.unwrapped.observation_manager
        obs_dims: dict[str, int] = {}
        obs_groups: dict[str, _ObsGroupCfg] = {}
        for group_name in isaaclab_obs_mgr.active_terms:
            # Total dim for this group (concatenated)
            group_shape = isaaclab_obs_mgr.group_obs_dim[group_name]
            if isinstance(group_shape, tuple):
                # Concatenated group: shape is a single tuple, product gives total dim
                total_dim = 1
                for d in group_shape:
                    total_dim *= d
            else:
                # Non-concatenated: list of tuples
                total_dim = sum(
                    1 if not isinstance(s, tuple) else s[0] for s in group_shape
                )
            obs_dims[group_name] = total_dim
            # Term names for this group
            term_names = isaaclab_obs_mgr.active_terms[group_name]
            obs_groups[group_name] = _ObsGroupCfg(term_names, history_length=1)

        self.observation_manager = _ObservationManager(obs_dims, obs_groups)

        # Evaluation flag (no-op in Isaac Lab)
        self.is_evaluating = False

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP (VecEnv interface)
    """

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def step(self, actions):
        """Step the environment.

        Accepts either a raw action tensor (VecEnv interface) or a dict
        ``{"actions": tensor}`` (BaseTask interface used by FastSACEnv).
        When called with a dict, returns plain dicts instead of TensorDicts
        and builds a BaseTask-style info_dict.
        """
        basetask_mode = isinstance(actions, dict)
        if basetask_mode:
            actions = actions["actions"]

        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        if basetask_mode:
            # BaseTask interface: plain dicts, info_dict with expected keys
            plain_obs = dict(obs_dict) if isinstance(obs_dict, TensorDict) else obs_dict
            info_dict = {
                "time_outs": extras.get(
                    "time_outs", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                ),
                "episode": extras.get("log", {}) if dones.any() else {},
                "episode_all": {},
                "to_log": {},
            }
            return plain_obs, rew, dones, info_dict

        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self):
        return self.env.close()

    """
    Operations - BaseTask interface (for FastSACEnv / FastSACAgent)
    """

    def reset_all(self) -> dict[str, torch.Tensor]:
        """Reset all environments. Returns obs dict keyed by group name.

        Matches BaseTask.reset_all() signature that FastSACEnv.reset() calls.
        """
        obs_td, _extras = self.reset()
        # Convert TensorDict to plain dict for FastSACEnv
        return {k: obs_td[k] for k in obs_td.keys()}

    def set_is_evaluating(self) -> None:
        """No-op. Isaac Lab does not have an eval toggle."""
        self.is_evaluating = True

    def get_checkpoint_state(self) -> dict:
        """No env-specific state to checkpoint."""
        return {}

    def load_checkpoint_state(self, state: dict | None) -> None:
        """No env-specific state to restore."""
        pass

    def synchronize_curriculum_state(self, *, device: str, world_size: int) -> None:
        """No-op for curriculum sync."""
        pass

    """
    Helper functions
    """

    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
