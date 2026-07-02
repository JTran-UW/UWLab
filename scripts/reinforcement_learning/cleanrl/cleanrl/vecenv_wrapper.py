import torch

import gymnasium as gym
from gymnasium.spaces import Space
from gymnasium.vector.vector_env import VectorEnv
import argparse
from gymnasium.vector.utils import (
    batch_differing_spaces,
    batch_space,
    concatenate,
    create_empty_array,
    iterate,
)

import numpy as np

class IsaacLabVectorEnv(VectorEnv):
    """Vectorized env for IsaacLab tasks"""

    copy: bool
    num_envs: int
    single_action_space: Space
    action_space: Space
    observation_space: Space

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        copy: bool = True,
        launcher_args: list[str] = None,
        device = "cuda"
    ):
        """Vector env for IsaacLab tasks"""

        super().__init__()

        self.copy = copy
        self.num_envs = num_envs
        
        # Launch Isaac
        from isaaclab.app import AppLauncher
        parser = argparse.ArgumentParser(description="Isaac Lab Launcher")
        AppLauncher.add_app_launcher_args(parser)
        launcher_args_parsed = parser.parse_args(launcher_args) 

        app_launcher = AppLauncher(launcher_args_parsed)
        simulation_app = app_launcher.app

        from isaaclab_tasks.utils import parse_env_cfg
        import isaaclab_tasks  # noqa: F401
        import uwlab_tasks  # noqa: F401

        # Create the env
        env_cfg = parse_env_cfg(env_id)
        env_cfg.scene.num_envs = num_envs
        self.env = gym.make(env_id, cfg=env_cfg)

        self.single_action_space = self.env.unwrapped.single_action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.single_observation_space = self.env.unwrapped.single_observation_space
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.ep_return = torch.zeros(self.num_envs, device=device)
        self.ep_length = torch.zeros(self.num_envs, device=device)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        next_obs, reward, terminations, truncations, infos = self.env.step(actions)

        self.ep_return += reward
        self.ep_length += 1
        done = terminations | truncations
        infos["ep_reward"] = self.ep_return[done]
        infos["ep_length"] = self.ep_length[done]

        self.ep_return[done] = 0
        self.ep_length[done] = 0

        return next_obs, reward, terminations, truncations, infos

