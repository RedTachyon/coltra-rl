from typing import Dict, Any, Optional, Callable, List, Type

import gym
from gym import Wrapper
import numpy as np

from coltra.buffers import Observation, Action
from coltra.utils import np_float
from coltra.envs.base_env import MultiAgentEnv
from coltra.envs.subproc_vec_env import VecEnv, SubprocVecEnv


def import_bullet():
    # noinspection PyUnresolvedReferences
    import pybullet_envs


class MultiGymEnv(MultiAgentEnv):
    """
    A wrapper for environments that can be `gym.make`'d
    """

    def __init__(
        self,
        env_name: str,
        name: str = "agent",
        import_fn: Callable = lambda: None,
        seed: Optional[int] = None,
        wrappers: Optional[list[Type[Wrapper]]] = None,
        **kwargs
    ):
        super().__init__(seed)
        if wrappers is None:
            wrappers = []
        if "Bullet" in env_name:
            import_fn = import_bullet

        import_fn()

        self.s_env = gym.make(env_name, **kwargs)
        self.s_env.seed(seed)
        self.name = name
        self.wrappers = wrappers

        for wrapper in self.wrappers:
            self.s_env = wrapper(self.s_env)

        self.observation_space = self.s_env.observation_space
        self.action_space = self.s_env.action_space

        self.is_discrete_action = isinstance(
            self.s_env.action_space, gym.spaces.Discrete
        )

        self.total_reward = 0

    def reset(self, *args, **kwargs):
        obs = self.s_env.reset()
        obs = Observation(vector=obs.astype(np.float32))
        self.total_reward = 0

        return self._dict(obs)

    def step(self, action_dict: Dict[str, Action]):
        action = action_dict[self.name]
        if self.is_discrete_action:
            action = action.discrete
        else:
            action = action.continuous

        obs, reward, done, info = self.s_env.step(action)
        self.total_reward += reward

        if done:
            info["final_obs"] = Observation(vector=obs.astype(np.float32))
            info["e_episode_reward"] = np_float(self.total_reward)
            self.total_reward = 0
            obs = self.s_env.reset()

        obs = Observation(vector=obs.astype(np.float32))
        return self._dict(obs), self._dict(reward), self._dict(done), info

    def render(self, *args, **kwargs):
        return self.s_env.render(*args, **kwargs)

    def _dict(self, val: Any) -> Dict[str, Any]:
        return {self.name: val}

    @classmethod
    def get_venv(
        cls, workers: int = 8, seed: Optional[int] = None, **env_kwargs
    ) -> SubprocVecEnv:
        if seed is None:
            seeds = [None] * workers
        else:
            seeds = [seed + i for i in range(workers)]

        venv = SubprocVecEnv(
            [cls.get_env_creator(seed=seeds[i], **env_kwargs) for i in range(workers)]
        )
        return venv
