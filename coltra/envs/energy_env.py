import gymnasium as gym
import numpy as np

from coltra import MultiAgentEnv, Action, Observation
from coltra.envs.spaces import ActionSpace, ObservationSpace


class EnergyEnv(MultiAgentEnv):
    e_s = 2.23
    e_w = 1.26
    max_accel = 5.0
    max_speed = 2.0
    drag = max_accel / max_speed

    def __init__(
        self,
        num_agents: int,
        time_limit: int = 100,
        env_scale: float = 2.0,
        dt: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.time_limit = time_limit
        self.env_scale = env_scale
        self.dt = dt

        self.observation_space = ObservationSpace(
            vector=gym.spaces.Box(-np.inf, np.inf, shape=(6,))
        )
        self.action_space = ActionSpace(
            {"continuous": gym.spaces.Box(-np.inf, np.inf, shape=(2,))}
        )

        self.pos = None
        self.vel = None
        self.goal = None
        self.live = []
        self.time = 0

        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self.pos = self.rng.uniform(
            low=0, high=self.env_scale, size=(self.num_agents, 2)
        ).astype(np.float32)
        self.vel = np.zeros_like(self.pos).astype(np.float32)
        self.goal = self.rng.uniform(
            low=0, high=self.env_scale, size=(self.num_agents, 2)
        ).astype(np.float32)
        self.live = [1 for _ in range(self.num_agents)]
        self.time = 0
        return self.get_observation()

    def step(self, actions: dict[str, Action]):
        self.time += 1

        actions = np.array([action.continuous for action in actions.values()])
        actions = self.normalize_action(actions)

        force = actions * self.max_accel

        self.vel = self.vel + (force - self.drag * self.vel) * self.dt
        self.pos = self.pos + self.vel * self.dt
        for i in range(self.num_agents):
            if np.linalg.norm(self.pos[i] - self.goal[i]) < 0.1:
                self.live[i] = 0

        done = {
            f"agent{i}": (self.time >= self.time_limit) for i in range(self.num_agents)
        }
        energies = (self.e_s + self.e_w * np.linalg.norm(self.vel, axis=-1)) * self.dt

        rewards = {
            f"agent{i}": -energies[i] * self.live[i] for i in range(self.num_agents)
        }

        info = {
            "m_mean_velocity": np.linalg.norm(self.vel, axis=-1),
            "m_mean_distance": np.linalg.norm(self.pos - self.goal, axis=-1),
        }

        if all(done.values()):
            info["e_success"] = np.array([1 - np.mean(self.live)])

        return self.get_observation(), rewards, done, info

    def get_observation(self):
        full_obs = np.concatenate([self.pos, self.goal, self.vel], axis=1)
        return {
            f"agent{i}": Observation(vector=full_obs[i]) for i in range(self.num_agents)
        }

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(action, axis=-1, keepdims=True)
        direction = action / speed
        magnitude = np.tanh(speed)
        return magnitude * direction


def straight_policy(obs: Observation) -> Action:
    obs = obs.vector
    pos, goal, vel = obs[:2], obs[2:4], obs[4:]
    action = goal - pos
    return Action(continuous=action * 10)


class LineEnergyEnv(MultiAgentEnv):
    e_s = 2.23
    e_w = 1.26
    max_accel = 5.0
    max_speed = 2.0
    drag = max_accel / max_speed

    def __init__(
        self,
        num_agents: int,
        time_limit: int = 100,
        env_scale: float = 2.0,
        dt: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.time_limit = time_limit
        self.env_scale = env_scale
        self.dt = dt

        self.observation_space = ObservationSpace(
            vector=gym.spaces.Box(-np.inf, np.inf, shape=(3,))
        )
        self.action_space = ActionSpace(
            {"continuous": gym.spaces.Box(-np.inf, np.inf, shape=(1,))}
        )

        self.pos = None
        self.vel = None
        self.goal = None
        self.live = []
        self.time = 0

        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self.pos = self.rng.uniform(
            low=0, high=self.env_scale, size=(self.num_agents, 1)
        ).astype(np.float32)
        self.vel = np.zeros_like(self.pos).astype(np.float32)
        self.goal = self.rng.uniform(
            low=0, high=self.env_scale, size=(self.num_agents, 1)
        ).astype(np.float32)
        self.live = [1 for _ in range(self.num_agents)]
        self.time = 0
        return self.get_observation()

    def step(self, actions: dict[str, Action]):
        self.time += 1

        actions = np.array([action.continuous for action in actions.values()])
        actions = self.normalize_action(actions)

        force = actions * self.max_accel

        self.vel = self.vel + (force - self.drag * self.vel) * self.dt
        self.pos = self.pos + self.vel * self.dt
        for i in range(self.num_agents):
            if np.linalg.norm(self.pos[i] - self.goal[i]) < 0.1:
                self.live[i] = 0

        done = {
            f"agent{i}": (self.time >= self.time_limit) for i in range(self.num_agents)
        }
        energies = (self.e_s + self.e_w * np.linalg.norm(self.vel, axis=-1)) * self.dt

        rewards = {
            f"agent{i}": -energies[i] * self.live[i] for i in range(self.num_agents)
        }
        info = {
            "m_mean_velocity": np.linalg.norm(self.vel, axis=-1),
            "m_mean_distance": np.linalg.norm(self.pos - self.goal, axis=-1),
        }

        return self.get_observation(), rewards, done, info

    def get_observation(self):
        full_obs = np.concatenate([self.pos, self.goal, self.vel], axis=1)
        return {
            f"agent{i}": Observation(vector=full_obs[i]) for i in range(self.num_agents)
        }

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(action, axis=-1, keepdims=True)
        direction = action / speed
        magnitude = np.tanh(speed)
        return magnitude * direction
