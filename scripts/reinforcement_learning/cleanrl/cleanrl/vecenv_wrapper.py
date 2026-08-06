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
        device = "cuda",
        render_mode: str | None = None,
        keep_distributed_flag: bool = False,
        sim_device: str | None = None,
    ):
        """Vector env for IsaacLab tasks"""

        super().__init__()

        self.copy = copy
        self.num_envs = num_envs
        
        # Launch Isaac
        from isaaclab.app import AppLauncher
        parser = argparse.ArgumentParser(description="Isaac Lab Launcher")
        AppLauncher.add_app_launcher_args(parser)

        # docker/cluster/run_singularity.sh appends --distributed to every job unconditionally. It is
        # an rsl_rl / holosoma train.py flag, NOT an AppLauncher one, so passing it through here makes
        # argparse abort with "unrecognized arguments". These CleanRL scripts have no DDP path and are
        # always launched with nproc_per_node=1, so the flag is a genuine no-op: drop it rather than
        # fail. Anything else unknown is still surfaced by argparse as usual.
        #
        # The multi-GPU script passes keep_distributed_flag=True: there the flag IS meaningful, and
        # AppLauncher consumes it to bind this process to cuda:${LOCAL_RANK}. AppLauncher does not
        # define the flag itself -- add_app_launcher_args() never registers it, and the rsl_rl /
        # holosoma entry points declare it on their own parsers and hand it over in the namespace --
        # so declare it here, and inject it when absent so the rank binding does not depend on the
        # caller having appended it.
        if keep_distributed_flag:
            parser.add_argument(
                "--distributed", action="store_true", default=False,
                help="Run training with multiple GPUs or nodes.",
            )
            launcher_args = list(launcher_args or [])
            if "--distributed" not in launcher_args:
                launcher_args.append("--distributed")
        if launcher_args and not keep_distributed_flag:
            dropped = [a for a in launcher_args if a == "--distributed"]
            if dropped:
                print("[launcher] ignoring --distributed (cluster-injected; CleanRL scripts are single-process)")
            launcher_args = [a for a in launcher_args if a != "--distributed"]

        launcher_args_parsed = parser.parse_args(launcher_args)

        app_launcher = AppLauncher(launcher_args_parsed)
        simulation_app = app_launcher.app
        self._simulation_app = simulation_app  # kept so close() can shut Isaac down

        from isaaclab_tasks.utils import parse_env_cfg
        import isaaclab_tasks  # noqa: F401
        import uwlab_tasks  # noqa: F401

        # Create the env
        env_cfg = parse_env_cfg(env_id)
        env_cfg.scene.num_envs = num_envs
        # Under torchrun each rank must simulate on its OWN GPU; parse_env_cfg otherwise defaults
        # every rank to cuda:0, which would put N sims on one card and silently thrash it.
        if sim_device is not None:
            env_cfg.sim.device = sim_device
        # render_mode="rgb_array" makes env.render() return the viewport image (rendered from
        # cfg.viewer.cam_prim_path at cfg.viewer.resolution). Needs --enable_cameras when headless.
        self.env = gym.make(env_id, cfg=env_cfg, render_mode=render_mode)

        self.single_action_space = self.env.unwrapped.single_action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.single_observation_space = self.env.unwrapped.single_observation_space

        # Tasks whose critic observation would be identical to the policy one omit the `critic`
        # group entirely, so ObservationManager does not recompute the same images a second time
        # (it evaluates every group independently). Consumers still expect obs["critic"], so expose
        # it here as an alias of the policy stream -- a view, never a copy.
        self._alias_critic_obs = (
            "critic" not in self.single_observation_space.spaces
            and "policy" in self.single_observation_space.spaces
        )
        if self._alias_critic_obs:
            spaces_dict = dict(self.single_observation_space.spaces)
            spaces_dict["critic"] = spaces_dict["policy"]
            self.single_observation_space = gym.spaces.Dict(spaces_dict)
            print("[obs] no critic group in task; aliasing obs['critic'] -> obs['policy'] (no copy)")

        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.ep_return = torch.zeros(self.num_envs, device=device)
        self.ep_length = torch.zeros(self.num_envs, device=device)

    def _with_critic_alias(self, obs):
        """Add obs['critic'] as an alias of obs['policy'] when the task defines no critic group."""
        if self._alias_critic_obs and obs is not None and "critic" not in obs:
            obs["critic"] = obs["policy"]
        return obs

    def close(self, **kwargs):
        """Shut the Isaac app down so the process can actually exit.

        gymnasium's VectorEnv.close() only calls close_extras(), which knows nothing about the
        wrapped Isaac env -- so without this the simulation app keeps its threads alive and the
        process hangs after training instead of exiting (observed: a run that finished every
        iteration then sat until its timeout killed it). Under SLURM that hang burns the rest of
        the allocation; under torchrun it also holds up every other rank.
        """
        try:
            self.env.close()
        finally:
            # simulation_app.close() is what actually releases Isaac's threads. Held from __init__
            # because AppLauncher does not expose it afterwards.
            if getattr(self, "_simulation_app", None) is not None:
                self._simulation_app.close()
                self._simulation_app = None

    def reset(self, *, seed=None, options=None):
        obs, extras = self.env.reset(seed=seed, options=options)
        return self._with_critic_alias(obs), extras

    def step(self, actions):
        next_obs, reward, terminations, truncations, infos = self.env.step(actions)
        next_obs = self._with_critic_alias(next_obs)
        # The training loop splices infos["final_obs"] into a copy of next_obs on truncated steps,
        # so it must carry the same key set or that assignment mismatches.
        if self._alias_critic_obs and isinstance(infos, dict) and "final_obs" in infos:
            self._with_critic_alias(infos["final_obs"])

        self.ep_return += reward
        self.ep_length += 1
        done = terminations | truncations
        infos["ep_reward"] = self.ep_return[done]
        infos["ep_length"] = self.ep_length[done]

        self.ep_return[done] = 0
        self.ep_length[done] = 0

        return next_obs, reward, terminations, truncations, infos

