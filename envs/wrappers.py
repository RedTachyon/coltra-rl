import copy
import pickle
from typing import Dict, Any, Union

import numpy as np
import gym

from .base_env import MultiAgentEnv, VecEnvWrapper, StepReturn
from .subproc_vec_env import VecEnv, SubprocVecEnv
from coltra.buffers import Observation, Action


# class MultiAgentWrapper(MultiAgentEnv):
#     """
#     A simple wrapper converting any instance of a single agent gym environment, into one compatible with my MultiAgentEnv approach.
#     Might be useful for benchmarking.
#     """
#
#     def __init__(self, env: gym.Env, name: str = "agent", *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.s_env = env
#         self.name = name
#
#         self.observation_space = self.s_env.observation_space
#         self.action_space = self.s_env.action_space
#
#         self.is_discrete_action = isinstance(self.s_env.action_space, gym.spaces.Discrete)
#
#     def reset(self, *args, **kwargs):
#         obs = self.s_env.reset()
#         obs = Observation(vector=obs.astype(np.float32))
#         return self._dict(obs)
#
#     def step(self, action: Dict[str, Action], *args, **kwargs):
#         action = action[self.name]
#         if self.is_discrete_action:
#             action = action.discrete
#         else:
#             action = action.continuous
#
#         obs, reward, done, info = self.s_env.step(action, *args, **kwargs)
#         obs = Observation(vector=obs.astype(np.float32))
#         return self._dict(obs), self._dict(reward), self._dict(done), info
#
#     def render(self, *args, **kwargs):
#         return self.s_env.render(*args, **kwargs)
#
#     def _dict(self, val: Any) -> Dict[str, Any]:
#         return {self.name: val}


class MultiGymEnv(MultiAgentEnv):
    """
    A wrapper for environments that can be `gym.make`'d
    """

    def __init__(self, env_name: str, name: str = "agent", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_env = gym.make(env_name, **kwargs)
        self.name = name

        self.observation_space = self.s_env.observation_space
        self.action_space = self.s_env.action_space

        self.is_discrete_action = isinstance(self.s_env.action_space, gym.spaces.Discrete)

    def reset(self, *args, **kwargs):
        obs = self.s_env.reset()
        obs = Observation(vector=obs.astype(np.float32))
        return self._dict(obs)

    def step(self, action: Dict[str, Action], *args, **kwargs):
        action = action[self.name]
        if self.is_discrete_action:
            action = action.discrete
        else:
            action = action.continuous

        obs, reward, done, info = self.s_env.step(action, *args, **kwargs)

        if done:
            obs = self.s_env.reset()

        obs = Observation(vector=obs.astype(np.float32))
        return self._dict(obs), self._dict(reward), self._dict(done), info

    def render(self, *args, **kwargs):
        return self.s_env.render(*args, **kwargs)

    def _dict(self, val: Any) -> Dict[str, Any]:
        return {self.name: val}

    @classmethod
    def get_env_creator(cls, env_name: str, *args, **kwargs):
        def _inner():
            env = cls(env_name, **kwargs)
            return env
        return _inner

    @classmethod
    def get_venv(cls, workers: int = 8, *args, **kwargs) -> VecEnv:
        venv = SubprocVecEnv([
            cls.get_env_creator(*args, **kwargs)
            for i in range(workers)
        ])
        return venv


# This piece of dead code is for me to use as a reference to implement agent-level normalization

# class VecNormalize(VecEnvWrapper):
#     """
#     A moving average, normalizing wrapper for vectorized environment.
#     has support for saving/loading moving average,
#     :param venv: the vectorized environment to wrap
#     :param training: Whether to update or not the moving average
#     :param norm_obs: Whether to normalize observation or not (default: True)
#     :param norm_reward: Whether to normalize rewards or not (default: True)
#     :param clip_obs: Max absolute value for observation
#     :param clip_reward: Max value absolute for discounted reward
#     :param gamma: discount factor
#     :param epsilon: To avoid division by zero
#     """
#
#     def __init__(
#         self,
#         venv: VecEnv,
#         training: bool = True,
#         norm_obs: bool = True,
#         norm_reward: bool = True,
#         clip_obs: float = 10.0,
#         clip_reward: float = 10.0,
#         gamma: float = 0.99,
#         epsilon: float = 1e-8,
#     ):
#         VecEnvWrapper.__init__(self, venv)
#
#         assert isinstance(
#             self.observation_space, (gym.spaces.Box, gym.spaces.Dict)
#         ), "VecNormalize only support `gym.spaces.Box` and `gym.spaces.Dict` observation spaces"
#
#         if isinstance(self.observation_space, gym.spaces.Dict):
#             self.obs_keys = set(self.observation_space.spaces.keys())
#             self.obs_spaces = self.observation_space.spaces
#             self.obs_rms = {key: RunningMeanStd(shape=space.shape) for key, space in self.obs_spaces.items()}
#         else:
#             self.obs_keys, self.obs_spaces = None, None
#             self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
#
#         self.ret_rms = RunningMeanStd(shape=())
#         self.clip_obs = clip_obs
#         self.clip_reward = clip_reward
#         # Returns: discounted rewards
#         self.ret = np.zeros(self.num_envs)
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.training = training
#         self.norm_obs = norm_obs
#         self.norm_reward = norm_reward
#         self.old_obs = np.array([])
#         self.old_reward = np.array([])
#
#     def __getstate__(self) -> Dict[str, Any]:
#         """
#         Gets state for pickling.
#         Excludes self.venv, as in general VecEnv's may not be pickleable."""
#         state = self.__dict__.copy()
#         # these attributes are not pickleable
#         del state["venv"]
#         del state["class_attributes"]
#         # these attributes depend on the above and so we would prefer not to pickle
#         del state["ret"]
#         return state
#
#     def __setstate__(self, state: Dict[str, Any]) -> None:
#         """
#         Restores pickled state.
#         User must call set_venv() after unpickling before using.
#         :param state:"""
#         self.__dict__.update(state)
#         assert "venv" not in state
#         self.venv = None
#
#     def set_venv(self, venv: VecEnv) -> None:
#         """
#         Sets the vector environment to wrap to venv.
#         Also sets attributes derived from this such as `num_env`.
#         :param venv:
#         """
#         if self.venv is not None:
#             raise ValueError("Trying to set venv of already initialized VecNormalize wrapper.")
#         VecEnvWrapper.__init__(self, venv)
#
#         # Check only that the observation_space match
#         # utils.check_for_correct_spaces(venv, self.observation_space, venv.action_space)
#         self.ret = np.zeros(self.num_envs)
#
#     def step_wait(self) -> StepReturn:
#         """
#         Apply sequence of actions to sequence of environments
#         actions -> (observations, rewards, dones)
#         where ``dones`` is a boolean vector indicating whether each element is new.
#         """
#         obs, rewards, dones, infos = self.venv.step_wait()
#         self.old_obs = obs
#         self.old_reward = rewards
#
#         if self.training:
#             if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
#                 for key in self.obs_rms.keys():
#                     self.obs_rms[key].update(obs[key])
#             else:
#                 self.obs_rms.update(obs)
#
#         obs = self.normalize_obs(obs)
#
#         if self.training:
#             self._update_reward(rewards)
#         rewards = self.normalize_reward(rewards)
#
#         # Normalize the terminal observations
#         for idx, done in enumerate(dones):
#             if not done:
#                 continue
#             if "terminal_observation" in infos[idx]:
#                 infos[idx]["terminal_observation"] = self.normalize_obs(infos[idx]["terminal_observation"])
#
#         self.ret[dones] = 0
#         return obs, rewards, dones, infos
#
#     def _update_reward(self, reward: np.ndarray) -> None:
#         """Update reward normalization statistics."""
#         self.ret = self.ret * self.gamma + reward
#         self.ret_rms.update(self.ret)
#
#     def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
#         """
#         Helper to normalize observation.
#         :param obs:
#         :param obs_rms: associated statistics
#         :return: normalized observation
#         """
#         return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
#
#     def _unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
#         """
#         Helper to unnormalize observation.
#         :param obs:
#         :param obs_rms: associated statistics
#         :return: unnormalized observation
#         """
#         return (obs * np.sqrt(obs_rms.var + self.epsilon)) + obs_rms.mean
#
#     def normalize_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
#         """
#         Normalize observations using this VecNormalize's observations statistics.
#         Calling this method does not update statistics.
#         """
#         # Avoid modifying by reference the original object
#         obs_ = copy.deepcopy(obs)
#         if self.norm_obs:
#             if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
#                 for key in self.obs_rms.keys():
#                     obs_[key] = self._normalize_obs(obs[key], self.obs_rms[key]).astype(np.float32)
#             else:
#                 obs_ = self._normalize_obs(obs, self.obs_rms).astype(np.float32)
#         return obs_
#
#     def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
#         """
#         Normalize rewards using this VecNormalize's rewards statistics.
#         Calling this method does not update statistics.
#         """
#         if self.norm_reward:
#             reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
#         return reward
#
#     def unnormalize_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
#         # Avoid modifying by reference the original object
#         obs_ = copy.deepcopy(obs)
#         if self.norm_obs:
#             if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
#                 for key in self.obs_rms.keys():
#                     obs_[key] = self._unnormalize_obs(obs[key], self.obs_rms[key])
#             else:
#                 obs_ = self._unnormalize_obs(obs, self.obs_rms)
#         return obs_
#
#     def unnormalize_reward(self, reward: np.ndarray) -> np.ndarray:
#         if self.norm_reward:
#             return reward * np.sqrt(self.ret_rms.var + self.epsilon)
#         return reward
#
#     def get_original_obs(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
#         """
#         Returns an unnormalized version of the observations from the most recent
#         step or reset.
#         """
#         return copy.deepcopy(self.old_obs)
#
#     def get_original_reward(self) -> np.ndarray:
#         """
#         Returns an unnormalized version of the rewards from the most recent step.
#         """
#         return self.old_reward.copy()
#
#     def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
#         """
#         Reset all environments
#         :return: first observation of the episode
#         """
#         obs = self.venv.reset()
#         self.old_obs = obs
#         self.ret = np.zeros(self.num_envs)
#         if self.training:
#             self._update_reward(self.ret)
#         return self.normalize_obs(obs)
#
#     @staticmethod
#     def load(load_path: str, venv: VecEnv) -> "VecNormalize":
#         """
#         Loads a saved VecNormalize object.
#         :param load_path: the path to load from.
#         :param venv: the VecEnv to wrap.
#         :return:
#         """
#         with open(load_path, "rb") as file_handler:
#             vec_normalize = pickle.load(file_handler)
#         vec_normalize.set_venv(venv)
#         return vec_normalize
#
#     def save(self, save_path: str) -> None:
#         """
#         Save current VecNormalize object with
#         all running statistics and settings (e.g. clip_obs)
#         :param save_path: The path to save to
#         """
#         with open(save_path, "wb") as file_handler:
#             pickle.dump(self, file_handler)