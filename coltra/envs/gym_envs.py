from typing import Dict, Any, Optional, Callable, List, Type

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np

from coltra.buffers import Observation, Action
from coltra.envs.spaces import ObservationSpace, ActionSpace
from coltra.utils import np_float
from coltra.envs.base_env import MultiAgentEnv
from coltra.envs.subproc_vec_env import SubprocVecEnv, SequentialVecEnv


def import_bullet():
    # noinspection PyUnresolvedReferences
    import pybullet_envs


class MultiGymEnv(MultiAgentEnv):
    """
    A wrapper for environments that can be `gym.make`'d
    """

    s_env: gym.Env

    def __init__(
        self,
        env_name: str,
        name: str = "agent",
        import_fn: Callable = lambda: None,
        seed: Optional[int] = None,
        wrappers: Optional[list[Type[Wrapper]]] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        gym.logger.set_level(gym.logger.ERROR)
        if wrappers is None:
            wrappers = []
        if "Bullet" in env_name:
            import_fn = import_bullet

        import_fn()

        self.s_env = gym.make(
            env_name, disable_env_checker=True, render_mode=render_mode, **kwargs
        )
        self.s_env.reset(seed=seed)
        self.name = name
        self.wrappers = wrappers

        for wrapper in self.wrappers:
            self.s_env = wrapper(self.s_env)

        self.is_discrete_action = isinstance(
            self.s_env.action_space, gym.spaces.Discrete
        )

        self.is_continuous_action = isinstance(self.s_env.action_space, gym.spaces.Box)

        self.observation_space = ObservationSpace(vector=self.s_env.observation_space)

        action_space_key = "discrete" if self.is_discrete_action else "continuous"
        self.action_space = ActionSpace({action_space_key: self.s_env.action_space})

        self.total_reward = 0

    def reset(self, *args, **kwargs):
        obs, info = self.s_env.reset()
        obs = Observation(vector=obs.astype(np.float32))
        self.total_reward = 0

        return self._dict(obs)

    def step(self, action_dict: dict[str, Action]):
        action = action_dict[self.name]
        if self.is_discrete_action:
            action = action.discrete
        elif self.is_continuous_action:
            action = action.continuous
        else:
            pass

        obs, reward, terminated, truncated, info = self.s_env.step(action)
        done = terminated or truncated
        self.total_reward += reward

        if done:
            info["final_obs"] = Observation(vector=obs.astype(np.float32))
            info["e_episode_reward"] = np_float(self.total_reward)
            self.total_reward = 0
            obs, _ = self.s_env.reset()

        obs = Observation(vector=obs.astype(np.float32))
        return self._dict(obs), self._dict(reward), self._dict(done), info

    def render(self, *args, **kwargs):
        return self.s_env.render()

    def _dict(self, val: Any) -> dict[str, Any]:
        return {self.name: val}

    @classmethod
    def get_venv(
        cls, workers: int = 8, seed: Optional[int] = None, **env_kwargs
    ) -> SequentialVecEnv:
        if seed is None:
            seeds = [None] * workers
        else:
            seeds = [seed + i for i in range(workers)]

        venv = SequentialVecEnv(
            [cls.get_env_creator(seed=seeds[i], **env_kwargs) for i in range(workers)]
        )
        return venv
