from typing import Callable

import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo.utils.env import ParallelEnv

from coltra.buffers import Observation
from coltra.envs import MultiAgentEnv, SubprocVecEnv
from coltra.envs.base_env import ActionDict, StepReturn, ObsDict


def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


class PettingZooEnv(MultiAgentEnv):
    def __init__(
        self, env_creator: Callable[..., ParallelEnv], sigmoid: bool = False, **kwargs
    ):
        super().__init__()
        self.pz_env = env_creator(**kwargs)
        self.active_agents = self.pz_env.possible_agents
        self.sigmoid = sigmoid

        agent_name = self.pz_env.possible_agents[0]
        self.action_space = self.pz_env.action_spaces[agent_name]
        self.observation_space = self.pz_env.observation_spaces[agent_name]

        self.is_discrete_action = isinstance(self.action_space, Discrete)

    def _embed_observation(self, obs: np.ndarray) -> Observation:
        shape = obs.shape
        if len(shape) == 1:
            return Observation(vector=obs)
        elif len(shape) == 2:
            return Observation(buffer=obs)
        elif len(shape) == 3:
            return Observation(image=obs)
        else:
            raise ValueError(f"Observation shape {obs.shape} not supported")

    def reset(self, *args, **kwargs) -> ObsDict:
        obs = self.pz_env.reset()
        return {key: self._embed_observation(obs[key]) for key in obs}

    def step(self, action_dict: ActionDict) -> StepReturn:
        if self.is_discrete_action:
            action = {
                agent_id: action_dict[agent_id].discrete for agent_id in action_dict
            }
        else:
            action = {
                agent_id: sigmoid(action_dict[agent_id].continuous)
                if self.sigmoid
                else action_dict[agent_id].continuous
                for agent_id in action_dict
            }
        obs, reward, done, info = self.pz_env.step(action)

        if all(done.values()):
            obs = self.pz_env.reset()

        obs = {key: self._embed_observation(obs[key]) for key in obs}
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        return self.pz_env.render(mode)

    @classmethod
    def get_venv(cls, workers: int = 8, **env_kwargs) -> SubprocVecEnv:

        venv = SubprocVecEnv(
            [cls.get_env_creator(**env_kwargs) for i in range(workers)]
        )
        return venv
