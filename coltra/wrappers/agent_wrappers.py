import copy

from typing import Tuple, Dict

import numpy as np
import torch
from torch import Tensor

from coltra.agents import Agent
from coltra.buffers import Observation, Action
from coltra.wrappers.base_wrappers import AgentWrapper


class ObsVecNormWrapper(AgentWrapper):
    def __init__(self, agent: Agent, eps: float = 1e-8):
        super().__init__(agent)

        self._obs_mean = None
        self._obs_var = None
        self._obs_eps = eps

        self._obs_count = 0
        self._initialized = False

    def update_obs_norm(self, obs_batch: Observation):
        batch_count = obs_batch.batch_size
        vec_batch = obs_batch.vector
        if not self._initialized:
            self._obs_mean = vec_batch.mean(axis=0)
            self._obs_var = vec_batch.var(axis=0)
            self._initialized = True
            self._obs_count = batch_count
        else:
            delta = vec_batch.mean(axis=0) - self._obs_mean
            tot_count = self._obs_count + batch_count

            m_a = self._obs_var * self._obs_count
            m_b = vec_batch.var(0) * batch_count

            M2 = m_a + m_b + delta**2 * self._obs_count * batch_count / tot_count

            self._obs_mean = self._obs_mean + delta * batch_count / tot_count
            self._obs_var = M2 / tot_count
            self._obs_count = tot_count

    def normalize_obs(self, obs_batch: Observation):
        norm_obs = copy.copy(obs_batch)  # shallow copy
        norm_obs.vector = (obs_batch.vector - self._obs_mean) / (
            np.sqrt(self._obs_var + self._obs_eps)
        )
        return norm_obs

    def act(
        self,
        obs_batch: Observation,
        state_batch: Tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        update_obs_norm: bool = True,
        **kwargs,
    ) -> Tuple[Action, Tuple, Dict]:

        if update_obs_norm:
            self.update_obs_norm(obs_batch)

        norm_obs = self.normalize_obs(obs_batch)
        return self.agent.act(norm_obs, state_batch, deterministic, get_value, **kwargs)

    def value(
        self, obs_batch: Observation, state_batch: tuple = (), **kwargs
    ) -> tuple[Tensor, tuple]:
        return self.agent.value(
            self.normalize_obs(obs_batch), state_batch=state_batch, **kwargs
        )

    def evaluate(
        self, obs_batch: Observation, action_batch: Action
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.agent.evaluate(self.normalize_obs(obs_batch), action_batch)


class RetNormWrapper(AgentWrapper):
    def __init__(self, agent: Agent, eps: float = 1e-8):
        super().__init__(agent)

        self._ret_mean = torch.tensor(0)
        self._ret_var = torch.tensor(1)
        self._ret_eps = eps

        self._ret_count = 0
        self._initialized = False

    def update_ret_norm(self, returns: Tensor):
        batch_count = len(returns)
        if not self._initialized:
            self._ret_mean = torch.mean(returns).cpu()
            self._ret_var = torch.var(returns).cpu()
            self._initialized = True
            self._ret_count = batch_count
        else:
            delta = torch.mean(returns) - self._ret_mean
            tot_count = self._ret_count + batch_count

            m_a = self._ret_var * self._ret_count
            m_b = returns.var() * batch_count

            M2 = m_a + m_b + delta**2 * self._ret_count * batch_count / tot_count

            self._ret_mean = self._ret_mean + delta * batch_count / tot_count
            self._ret_var = M2 / tot_count
            self._ret_count = tot_count

    def normalize_ret(self, returns: Tensor):
        return (returns - self._ret_mean) / np.sqrt(self._ret_var + self._ret_eps)

    def unnormalize_value(self, value: Tensor):
        return self._ret_var * value + self._ret_mean

    def value(
        self,
        obs_batch: Observation,
        state_batch: tuple = (),
        real_value: bool = False,
        **kwargs,
    ) -> tuple[Tensor, tuple]:
        value, state = self.agent.value(obs_batch, state_batch=state_batch, **kwargs)
        if real_value:
            value = self.unnormalize_value(value)
        return value, state
