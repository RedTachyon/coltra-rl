import gymnasium as gym
import numpy as np

from coltra import MultiAgentEnv, Observation, Action
from coltra.envs.spaces import ObservationSpace, ActionSpace


class DelayedEnv(MultiAgentEnv):
    """
    Basic version:
    1. Take an action
    2. Take 100 idle steps
    """

    def __init__(self, num_agents: int, steps: int = 100):
        super().__init__()

        self.num_agents = num_agents
        self.active_agents = [f"agent{i}" for i in range(self.num_agents)]
        self.steps = steps

        self.observation_space = ObservationSpace(
            {"vector": gym.spaces.Box(low=0, high=1, shape=(2,))}
        )
        self.action_space = ActionSpace({"discrete": gym.spaces.Discrete(2)})

        self.timer = 0

    def reset(self, **kwargs):
        self.timer = 0
        return self.get_obs()

    def step(self, action_dict):
        rewards = {agent_id: np.float32(0) for agent_id in self.active_agents}
        done = {agent_id: False for agent_id in self.active_agents}
        info = {}
        if self.timer == 0:
            self.choices = {
                agent_id: 1 + 0.75 * (1 - act.discrete)
                for agent_id, act in action_dict.items()
            }
            # self.choices = {agent_id: np.float32(1 + 0.75 * act.discrete) for agent_id, act in action_dict.items()}
            self.timer += 1
        elif self.timer < self.steps - 1:
            self.timer += 1
        elif self.timer == self.steps - 1:
            rewards = self.choices
            done = {agent_id: True for agent_id in self.active_agents}
            self.reset()
        # elif self.timer == STEPS:
        #     self.reset()
        #     self.timer -= 1  # Dirty hack
        else:
            raise ValueError("Something's wrong")

        return self.get_obs(), rewards, done, info

    def get_obs(self):
        if self.timer == 0:
            obs = np.array([1, 0], dtype=np.float32)
        else:
            obs = np.array([0, 1], dtype=np.float32)

        return {agent_id: Observation(obs.copy()) for agent_id in self.active_agents}


class ReversalEnv(MultiAgentEnv):
    """
    Everything calibrated for k=0.01 => mu ~= 0.99
    Overall idea:
    At step 0, make a "commitment" - either SER (+1 after 100 steps) or LLR (+1.75 after 200 steps)
    At step 100:
    - if chosen SER, either take 1, or switch to LLR and take +1.7 later
    - if chosen LLR, either take 1.75 later, or switch to SER and take +0.95
    At step 200:
    - if chosen SER, take nothing
    - if chosen LLR, take 1.75 or 1.7

    Ideal outcome:
    - At step 0, commit to LLR (expected discounted value: Gamma(200) * 1.75)
    - At step 100, switch to SER (expected discounted value: 0.95)

    Do I even need to include a tax? I guess it highlights the suboptimality of switching

    This happend if the value of
    """

    def __init__(
        self, num_agents: int, dt: int = 100, t2: int = 100, tax: float = 0.01
    ):
        super().__init__()

        self.num_agents = num_agents
        self.active_agents = [f"agent{i}" for i in range(self.num_agents)]
        self.dt = dt
        self.t1 = 0
        self.t2 = t2

        self.steps = self.dt + self.t2

        self.v1 = 1
        self.v2 = 1.75

        self.tax = tax

        self.observation_space = ObservationSpace(
            {"vector": gym.spaces.Box(low=0, high=1, shape=(4,))}
        )
        self.action_space = ActionSpace({"discrete": gym.spaces.Discrete(2)})

        self.timer = 0
        self.initial_decisions = {}
        self.secondary_decisions = {}

    def reset(self, **kwargs):
        self.timer = 0
        self.initial_decisions = {}
        return self.get_obs()

    def step(self, action_dict):
        rewards = {agent_id: np.float32(0) for agent_id in self.active_agents}
        done = {agent_id: False for agent_id in self.active_agents}
        info = {}
        if self.timer == 0:
            # Choices: either SER == 0 or LLR == 1
            self.initial_decisions = {
                agent_id: "SER" if act.discrete == 0 else "LLR"
                for agent_id, act in action_dict.items()
            }
            self.timer += 1
        elif self.timer == self.dt - 1:
            # If SER, either take 1, or switch to LLR
            # If LLR, either take 1.75, or switch to SER

            self.secondary_decisions = {
                agent_id: "SER" if act.discrete == 0 else "LLR"
                for agent_id, act in action_dict.items()
            }

            for agent_id, initial_decision in self.initial_decisions.items():
                secondary_decision = self.secondary_decisions[agent_id]
                if (
                    initial_decision == "SER" and secondary_decision == "SER"
                ):  # Committed to SER, sticking with it
                    rewards[agent_id] = self.v1
                elif (
                    initial_decision == "LLR" and secondary_decision == "SER"
                ):  # Committed to LLR, switching to SER
                    rewards[agent_id] = self.v1 - self.tax
                elif (
                    initial_decision == "SER" and secondary_decision == "LLR"
                ):  # Committed to SER, switching to LLR - not allowed
                    rewards[agent_id] = self.v1
                else:
                    rewards[agent_id] = 0.0

            self.timer += 1
        elif self.timer == self.dt + self.t2 - 1:
            for agent_id, initial_decision in self.initial_decisions.items():
                secondary_decision = self.secondary_decisions[agent_id]

                if (
                    initial_decision == "LLR" and secondary_decision == "LLR"
                ):  # Committed to LLR, sticking with it
                    rewards[agent_id] = self.v2
                elif (
                    initial_decision == "SER" and secondary_decision == "LLR"
                ):  # Committed to SER, switching to LLR
                    rewards[agent_id] = 0.0
                else:
                    rewards[agent_id] = 0.0

            done = {agent_id: True for agent_id in self.active_agents}
            self.reset()
        else:
            self.timer += 1

        return self.get_obs(), rewards, done, info

    def get_obs(self):
        if self.timer == 0:  # First decision step
            obs = np.array([1, 0, 0, 0], dtype=np.float32)
            obs_dict = {
                agent_id: Observation(obs.copy()) for agent_id in self.active_agents
            }
        elif self.timer == self.dt - 1:  # Second decision step, counting the increment
            obs1 = np.array([0, 1, 0, 0], dtype=np.float32)
            obs2 = np.array([0, 0, 1, 0], dtype=np.float32)
            obs_dict = {
                agent_id: Observation(
                    obs1.copy()
                    if self.initial_decisions[agent_id] == "SER"
                    else obs2.copy()
                )
                for agent_id in self.active_agents
            }
        else:  # idle step
            obs = np.array([0, 0, 0, 1], dtype=np.float32)
            obs_dict = {
                agent_id: Observation(obs.copy()) for agent_id in self.active_agents
            }

        return obs_dict


class ReversalEnv2(MultiAgentEnv):
    # TODO debugging: train with a reward after a fixed time, reward is given in the first step
    def __init__(self, num_agents: int, t_max: int = 400, v_max: float = 5):
        super().__init__()

        self.num_agents = num_agents
        self.active_agents = [f"agent{i}" for i in range(self.num_agents)]

        self.t_max = t_max
        self.v_max = v_max

        self.v1 = np.nan
        self.v2 = np.nan

        self.t1 = np.nan
        self.t2 = np.nan

        self.observation_space = ObservationSpace(
            {"vector": gym.spaces.Box(low=0, high=1, shape=(4,))}
        )
        self.action_space = ActionSpace({"discrete": gym.spaces.Discrete(2)})

        self.timer = 0

        self.initial_decisions = {}

    def reset(self):
        self.timer = 0
        self.initial_decisions = {}
        self.v1 = {
            agent_id: np.random.uniform(0, self.v_max)
            for agent_id in self.active_agents
        }
        # self.v2 = {agent_id: np.random.uniform(self.v1[agent_id], self.v_max) for agent_id in self.active_agents}
        self.v2 = {
            agent_id: np.random.uniform(0, self.v_max)
            for agent_id in self.active_agents
        }
        # self.t1 = {agent_id: self.t_max - 1 for agent_id in self.active_agents}
        # self.t2 = {agent_id: self.t_max - 1 for agent_id in self.active_agents}
        self.t1 = {
            agent_id: np.random.randint(0, self.t_max)
            for agent_id in self.active_agents
        }
        self.t2 = {
            agent_id: np.random.randint(0, self.t_max)
            for agent_id in self.active_agents
        }

        return self.get_obs()

    def step(self, action_dict: dict[str, Action]):
        rewards = {agent_id: np.float32(0) for agent_id in self.active_agents}
        done = {agent_id: False for agent_id in self.active_agents}
        if self.timer == 0:
            self.initial_decisions = {
                agent_id: act.discrete for agent_id, act in action_dict.items()
            }

        for agent_id, initial_decision in self.initial_decisions.items():
            if self.timer == self.t1[agent_id] and initial_decision == 0:
                rewards[agent_id] = self.v1[agent_id]
            elif self.timer == self.t2[agent_id] and initial_decision == 1:
                rewards[agent_id] = self.v2[agent_id]
            else:
                rewards[agent_id] = 0.0

        self.timer += 1

        if self.timer == self.t_max:
            done = {agent_id: True for agent_id in self.active_agents}
            self.reset()

        return self.get_obs(), rewards, done, {}

    def get_obs(self):
        if self.timer == 0:
            obs = {
                agent_id: np.array(
                    [
                        self.v1[agent_id],
                        self.v2[agent_id],
                        self.t1[agent_id] / self.t_max,
                        self.t2[agent_id] / self.t_max,
                    ],
                    dtype=np.float32,
                )
                for agent_id in self.active_agents
            }
        else:
            obs = {
                agent_id: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                for agent_id in self.active_agents
            }
            # obs = {
            #     agent_id: np.array([self.v1[agent_id],# / self.v_max,
            #              self.t1[agent_id] / self.t_max,
            #              self.v2[agent_id],# / self.v_max,
            #              self.t2[agent_id] / self.t_max,
            #              self.initial_decisions[agent_id],
            #              1.],
            #              dtype=np.float32)
            #     for agent_id in self.active_agents}

        return {agent_id: Observation(obs[agent_id]) for agent_id in self.active_agents}

    def _dict(self, val):
        return {agent_id: val for agent_id in self.active_agents}


"""
TODO: Different environment design idea:
Train the agent on various dilemmas, to choose the best one given (v, t)
Show that this will lead to a reversal

"""
