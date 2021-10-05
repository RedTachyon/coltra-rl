import copy

from typing import Tuple, Dict

import numpy as np
from torch import Tensor

from coltra.agents import Agent
from coltra.buffers import Observation, Action
from coltra.wrappers.base_wrappers import AgentWrapper


class ObsVecNormWrapper(AgentWrapper):
    def __init__(self, agent: Agent, eps: float = 1e-8):
        super().__init__(agent)

        self.mean = None
        self.var = None
        self.eps = eps

        self.count = 0
        self._initialized = False

    def update(self, obs_batch: Observation):
        batch_count = obs_batch.batch_size
        vec_batch = obs_batch.vector
        if not self._initialized:
            self.mean = obs_batch.vector.mean(axis=0)
            self.var = obs_batch.vector.var(axis=0)
            self._initialized = True
            self.count = batch_count
        else:
            delta = vec_batch - self.mean
            tot_count = self.count + batch_count

            m_a = self.var * self.count
            m_b = obs_batch.vector.var(0) * batch_count

            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count

            self.mean = self.mean + delta * batch_count / tot_count
            self.var = M2 / tot_count
            self.count = tot_count

    def normalize(self, obs_batch: Observation):
        norm_obs = copy.copy(obs_batch)  # shallow copy
        norm_obs.vector = (obs_batch.vector - self.mean) / (
            np.sqrt(self.var + self.eps)
        )
        return norm_obs

    def act(
        self,
        obs_batch: Observation,
        state_batch: Tuple = (),
        deterministic: bool = False,
        get_value: bool = False,
        update: bool = True,
    ) -> Tuple[Action, Tuple, Dict]:

        if update:
            self.update(obs_batch)

        norm_obs = self.normalize(obs_batch)
        return self.agent.act(norm_obs, state_batch, deterministic, get_value)

    def value(self, obs_batch: Observation) -> Tensor:
        return self.agent.value(self.normalize(obs_batch))

    def evaluate(
        self, obs_batch: Observation, action_batch: Action
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.agent.evaluate(self.normalize(obs_batch), action_batch)
