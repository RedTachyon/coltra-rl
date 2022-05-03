from typing import List, Dict, Optional

from gym.spaces import Box, Discrete
import numpy as np

from coltra.buffers import Observation, Action
from coltra.envs.base_env import MultiAgentEnv
from coltra.envs.spaces import ObservationSpace
from coltra.envs.subproc_vec_env import SubprocVecEnv
from coltra.utils import np_float


class ConstRewardEnv(MultiAgentEnv):
    def __init__(self, num_agents: int = 1, seed: Optional[int] = None):
        super().__init__()
        self.num_agents = num_agents
        self.active_agents = [f"Agent{i}" for i in range(num_agents)]
        self.rng = np.random.default_rng(seed)

        self.obs_vector_size = 1
        self.action_vector_size = 1

        self.observation_space = ObservationSpace({"vector": Box(-1, 1, (1,))})
        self.action_space = Box(-1, 1, (1,))

    def reset(self, *args, **kwargs):
        if num_agents := kwargs.get("num_agents"):
            self.num_agents = num_agents
            self.active_agents = [f"Agent{i}" for i in range(num_agents)]

        zero_obs = Observation(vector=np.ones((1,), dtype=np.float32))
        return {agent_id: zero_obs for agent_id in self.active_agents}

    def step(self, actions: dict[str, Action]):
        zero_obs = {
            agent_id: Observation(vector=np.ones((1,), dtype=np.float32))
            for agent_id in self.active_agents
        }
        reward = {agent_id: np.float32(1.0) for agent_id in self.active_agents}
        done = {agent_id: True for agent_id in self.active_agents}
        info = {
            "m_stat": np_float(1),
            "m_another_stat": np_float(2),
            "m_random_stat": np.random.randn(1),
        }
        return zero_obs, reward, done, info

    def render(self, mode="human"):
        return 0

    @classmethod
    def get_venv(cls, workers: int = 8, **kwargs) -> SubprocVecEnv:
        venv = SubprocVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


class ObsDependentRewardEnv(MultiAgentEnv):
    def __init__(self, num_agents: int = 1, seed: Optional[int] = None):
        super().__init__()
        self.num_agents = num_agents
        self.active_agents = [f"Agent{i}" for i in range(num_agents)]
        self.rng = np.random.default_rng(seed)

        self.obs_vector_size = 1
        self.action_vector_size = 1

        self.observation_space = ObservationSpace({"vector": Box(-1, 1, (1,))})
        self.action_space = Box(-1, 1, (1,))

        self.current_obs = {}

    def reset(self, *args, **kwargs):
        if num_agents := kwargs.get("num_agents"):
            self.num_agents = num_agents
            self.active_agents = [f"Agent{i}" for i in range(num_agents)]

        obs = self.rng.choice([-1, 1])
        random_obs = Observation(vector=np_float(obs))

        self.current_obs = {agent_id: random_obs for agent_id in self.active_agents}
        return self.current_obs

    def step(self, actions: dict[str, Action]):
        obs = {
            agent_id: Observation(vector=np_float(self.rng.choice([-1, 1])))
            for agent_id in self.active_agents
        }
        reward = {
            agent_id: np.float32(1.0 if self.current_obs[agent_id].vector > 0 else -1.0)
            for agent_id in self.active_agents
        }
        done = {agent_id: True for agent_id in self.active_agents}
        info = {
            "m_stat": np_float(1),
            "m_another_stat": np_float(2),
            "m_random_stat": np.random.randn(1),
        }
        self.current_obs = obs
        return obs, reward, done, info

    def render(self, mode="human"):
        return 0

    @classmethod
    def get_venv(cls, workers: int = 8, **kwargs) -> SubprocVecEnv:
        venv = SubprocVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


class ActionDependentRewardEnv(MultiAgentEnv):
    def __init__(self, num_agents: int = 1, seed: Optional[int] = None):
        super().__init__()
        self.num_agents = num_agents
        self.active_agents = [f"Agent{i}" for i in range(num_agents)]
        self.rng = np.random.default_rng(seed)

        self.obs_vector_size = 1
        self.action_vector_size = 1

        self.observation_space = ObservationSpace({"vector": Box(-1, 1, (1,))})
        # self.action_space = Box(-1, 1, (1,))
        self.action_space = Discrete(2)

    def reset(self, *args, **kwargs):
        if num_agents := kwargs.get("num_agents"):
            self.num_agents = num_agents
            self.active_agents = [f"Agent{i}" for i in range(num_agents)]

        zero_obs = Observation(vector=np.ones((1,), dtype=np.float32))
        return {agent_id: zero_obs for agent_id in self.active_agents}

    def step(self, actions: dict[str, Action]):
        zero_obs = {
            agent_id: Observation(vector=np.ones((1,), dtype=np.float32))
            for agent_id in self.active_agents
        }
        reward = {
            agent_id: np.float32(1.0 if action.discrete > 0 else -1.0)
            for agent_id, action in actions.items()
        }
        done = {agent_id: True for agent_id in self.active_agents}
        info = {
            "m_stat": np_float(1),
        }
        return zero_obs, reward, done, info

    def render(self, mode="human"):
        return 0

    @classmethod
    def get_venv(cls, workers: int = 8, **kwargs) -> SubprocVecEnv:
        venv = SubprocVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


class StateActionDependentRewardEnv(MultiAgentEnv):
    def __init__(
        self, num_agents: int = 1, ep_end_prob: float = 0.1, seed: Optional[int] = None
    ):
        super().__init__()
        self.num_agents = num_agents
        self.ep_end_prob = ep_end_prob
        self.active_agents = [f"Agent{i}" for i in range(num_agents)]
        self.rng = np.random.default_rng(seed)

        self.obs_vector_size = 1
        self.action_vector_size = 1

        self.observation_space = ObservationSpace({"vector": Box(-1, 1, (1,))})
        # self.action_space = Box(-1, 1, (1,))
        self.action_space = Discrete(2)

        self.last_obs = {}

    def reset(self, *args, **kwargs):
        if num_agents := kwargs.get("num_agents"):
            self.num_agents = num_agents
            self.active_agents = [f"Agent{i}" for i in range(num_agents)]

        self.last_obs = {
            agent_id: np_float(self.rng.integers(2)) for agent_id in self.active_agents
        }
        obs = {
            agent_id: Observation(vector=agent_obs)
            for agent_id, agent_obs in self.last_obs.items()
        }

        return obs

    def step(self, actions: dict[str, Action]):

        reward = {
            agent_id: np.float32(
                1.0 if np.allclose(action.discrete, self.last_obs[agent_id]) else -1.0
            )
            for agent_id, action in actions.items()
        }
        done = {
            agent_id: self.rng.random() < self.ep_end_prob
            for agent_id in self.active_agents
        }

        self.last_obs = {
            agent_id: np_float(self.rng.integers(2)) for agent_id in self.active_agents
        }
        obs = {
            agent_id: Observation(vector=agent_obs)
            for agent_id, agent_obs in self.last_obs.items()
        }

        info = {
            "m_stat": np_float(1),
        }
        return obs, reward, done, info

    def render(self, mode="human"):
        return 0

    @classmethod
    def get_venv(cls, workers: int = 8, **kwargs) -> SubprocVecEnv:
        venv = SubprocVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


class EpisodicMetricEnv(MultiAgentEnv):
    def __init__(
        self, num_agents: int = 1, ep_len: int = 2, seed: Optional[int] = None
    ):
        super().__init__()
        self.num_agents = num_agents
        self.active_agents = [f"Agent{i}" for i in range(num_agents)]
        self.rng = np.random.default_rng(seed)
        self.ep_len = ep_len

        self.obs_vector_size = 1
        self.action_vector_size = 1

        self.observation_space = ObservationSpace({"vector": Box(0, self.ep_len, (1,))})
        self.action_space = Box(-1, 1, (1,))

        self.current_obs = {}
        self.time = 0

    def reset(self, *args, **kwargs):
        if num_agents := kwargs.get("num_agents"):
            self.num_agents = num_agents
            self.active_agents = [f"Agent{i}" for i in range(num_agents)]
        self.time = 0

        obs = self.time
        random_obs = Observation(vector=np_float(obs))

        self.current_obs = {agent_id: random_obs for agent_id in self.active_agents}
        return self.current_obs

    def step(self, actions: dict[str, Action]):
        self.time += 1
        _done = self.time >= self.ep_len
        _obs = self.time
        obs = {
            agent_id: Observation(vector=np_float(_obs))
            for agent_id in self.active_agents
        }
        reward = {
            agent_id: np.float32(1.0 if self.current_obs[agent_id].vector > 0 else -1.0)
            for agent_id in self.active_agents
        }
        done = {agent_id: _done for agent_id in self.active_agents}
        info = {
            "m_stat": np_float(1),
            "m_another_stat": np_float(2),
            "m_random_stat": np.random.randn(1),
        }
        if _done:
            info["e_ep_len"] = np_float(self.time)
            info["e_episodic_stat"] = np.random.rand(1)
            obs = self.reset()

        self.current_obs = obs
        return obs, reward, done, info

    def render(self, mode="human"):
        return 0

    @classmethod
    def get_venv(cls, workers: int = 8, **kwargs) -> SubprocVecEnv:
        venv = SubprocVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


probe_env_classes = [
    ConstRewardEnv,
    ObsDependentRewardEnv,
    ActionDependentRewardEnv,
    StateActionDependentRewardEnv,
    EpisodicMetricEnv,
]
