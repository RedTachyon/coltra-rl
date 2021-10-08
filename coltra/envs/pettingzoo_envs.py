from typing import Callable, Any
import numpy as np

from pettingzoo.utils.env import ParallelEnv

from coltra.buffers import Observation
from coltra.envs import MultiAgentEnv
from coltra.envs.base_env import VecEnv, ActionDict, StepReturn, ObsDict


# class PettingZooEnv(MultiAgentEnv):
#     """WIP"""
#
#     def __init__(self, env_creator: Callable[[Any], ParallelEnv], **kwargs):
#         super().__init__(**kwargs)
#         self.s_env = env_creator(**kwargs)
#
#     def _embed_observation(self, obs: np.ndarray) -> Observation:
#         pass
#
#     def reset(self, *args, **kwargs) -> ObsDict:
#         obs = self.s_env.reset()
#         return {key: Observation(vector=obs[key]) for key in obs}
#
#     def step(self, action_dict: ActionDict) -> StepReturn:
#         pass
#
#     def render(self, mode="rgb_array"):
#         pass
#
#     @classmethod
#     def get_venv(cls, workers: int = 8, *args, **kwargs) -> VecEnv:
#         pass
