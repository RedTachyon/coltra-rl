from typing import List, Dict, Optional

from gymnasium.spaces import Box, Discrete
import numpy as np

from coltra.buffers import Observation, Action
from coltra.envs.base_env import MultiAgentEnv
from coltra.envs.spaces import ObservationSpace, ActionSpace
from coltra.envs.subproc_vec_env import SubprocVecEnv, SequentialVecEnv
from coltra.utils import np_float


class ConstRewardEnv(MultiAgentEnv):  # 0
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
    def get_venv(cls, workers: int = 8, **kwargs) -> MultiAgentEnv:
        venv = SequentialVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


class ObsDependentRewardEnv(MultiAgentEnv):  # 1
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
    def get_venv(cls, workers: int = 8, **kwargs) -> MultiAgentEnv:
        venv = SequentialVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


class ActionDependentRewardEnv(MultiAgentEnv):  # 2
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
    def get_venv(cls, workers: int = 8, **kwargs) -> MultiAgentEnv:
        venv = SequentialVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


class StateActionDependentRewardEnv(MultiAgentEnv):  # 3
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
    def get_venv(cls, workers: int = 8, **kwargs) -> MultiAgentEnv:
        venv = SequentialVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


class EpisodicMetricEnv(MultiAgentEnv):  # 4
    def __init__(
        self, num_agents: int = 1, ep_len: int = 4, seed: Optional[int] = None
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
    def get_venv(cls, workers: int = 8, **kwargs) -> MultiAgentEnv:
        venv = SequentialVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


class MemoryEnv(MultiAgentEnv):  # 5
    def __init__(
        self,
        num_agents: int = 1,
        delay: int = 0,
        cooloff: int = 9,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.active_agents = [f"Agent{i}" for i in range(num_agents)]
        self.rng = np.random.default_rng(seed)

        self.delay = delay
        self.cooloff = cooloff

        self.observation_space = ObservationSpace({"vector": Box(0, 1, (2,))})
        self.action_space = ActionSpace({"discrete": Discrete(2)})

        self.initial_obs = {}
        self.step_count = {}

    def reset(self, *args, **kwargs):
        if num_agents := kwargs.get("num_agents"):
            self.num_agents = num_agents
            self.active_agents = [f"Agent{i}" for i in range(num_agents)]

        # initial_obs_value = self.rng.choice([[1, 0], [0, 1]], size=(1, 2))
        initial_obs_value = {
            agent_id: (
                np.array([1, 0], dtype=np.float32)
                if self.rng.random() < 0.5
                else np.array([0, 1], dtype=np.float32)
            )
            for agent_id in self.active_agents
        }
        initial_obs = {
            agent_id: Observation(vector=initial_obs_value[agent_id])
            for agent_id in self.active_agents
        }

        self.initial_obs = initial_obs_value
        self.step_count = {agent_id: 0 for agent_id in self.active_agents}

        return initial_obs

    def step(self, actions: dict[str, Action]):
        obs = {}
        reward = {}
        done = {}

        for agent_id in self.active_agents:
            self.step_count[agent_id] += 1

            if self.step_count[agent_id] <= self.delay:
                obs[agent_id] = Observation(vector=np.array([0, 0], dtype=np.float32))
                reward[agent_id] = np.float32(0)
            elif self.step_count[agent_id] == self.delay + 1:
                obs[agent_id] = Observation(vector=np.array([0, 0], dtype=np.float32))
                if np.argmax(self.initial_obs[agent_id]) == actions[agent_id].discrete:
                    reward[agent_id] = np.float32(1)
                else:
                    reward[agent_id] = np.float32(-1)
            else:
                obs[agent_id] = Observation(vector=np.array([0, 0], dtype=np.float32))

                reward[agent_id] = np.float32(0)

            done[agent_id] = self.step_count[agent_id] >= self.delay + self.cooloff

        info = {"m_stat": np_float(1)}

        return obs, reward, done, info

    @classmethod
    def get_venv(cls, workers: int = 8, **kwargs) -> MultiAgentEnv:
        venv = SequentialVecEnv([cls.get_env_creator(**kwargs) for _ in range(workers)])
        return venv


probe_env_classes = [
    ConstRewardEnv,
    ObsDependentRewardEnv,
    ActionDependentRewardEnv,
    StateActionDependentRewardEnv,
    EpisodicMetricEnv,
    MemoryEnv,
]
