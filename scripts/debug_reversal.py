import numpy as np
import coltra
from coltra import MultiAgentEnv, Observation, Action, DAgent, HomogeneousGroup
from coltra.models import MLPModel
from coltra.envs.spaces import ObservationSpace, ActionSpace
from coltra.trainers import PPOCrowdTrainer

from coltra.discounting import (
    discount_experience,
    get_episode_rewards,
    get_episode_lengths,
)

import torch

from tqdm import tqdm, trange
import gymnasium as gym

STEPS = 5


class DelayedEnv(MultiAgentEnv):
    """
    Basic version:
    1. Take an action
    2. Take 100 idle steps
    """

    def __init__(self, num_agents: int):
        super().__init__()

        self.num_agents = num_agents
        self.active_agents = [f"agent{i}" for i in range(self.num_agents)]

        self.observation_space = ObservationSpace(
            {"vector": gym.spaces.Box(low=0, high=1, shape=(2,))}
        )
        self.action_space = ActionSpace({"discrete": gym.spaces.Discrete(2)})

    def reset(self, **kwargs):
        self.timer = 0
        return self.get_obs()

    def step(self, action_dict):
        rewards = {agent_id: 0 for agent_id in self.active_agents}
        done = {agent_id: False for agent_id in self.active_agents}
        if self.timer == 0:
            # self.choices = {agent_id: 1 + 0.75 * (1-act.discrete) for agent_id, act in action_dict.items()}
            self.choices = {
                agent_id: 1 + 0.75 * act.discrete
                for agent_id, act in action_dict.items()
            }

            # rewards = self.choices
        elif self.timer < STEPS - 1:
            pass
            # rewards = self.choices
        elif self.timer == STEPS - 1:
            rewards = self.choices
            done = {agent_id: True for agent_id in self.active_agents}
            # self.reset()
        elif self.timer == STEPS:
            self.reset()
            self.timer -= 1  # Dirty hack
        else:
            raise ValueError("Something's wrong")

        self.timer += 1

        return self.get_obs(), rewards, done, {}

    def get_obs(self):
        if self.timer == 0:
            obs = np.array([1, 0], dtype=np.float32)
        else:
            obs = np.array([0, 1], dtype=np.float32)

        return {agent_id: Observation(obs.copy()) for agent_id in self.active_agents}


env = DelayedEnv(100)

model = MLPModel(
    config={}, observation_space=env.observation_space, action_space=env.action_space
)
agent = DAgent(model)
group = HomogeneousGroup(agent)

trainer_config = {
    "steps": STEPS,
    "workers": 1,
    "tensorboard_name": None,
    "save_freq": 0,
    "PPOConfig": {
        "gamma": 0.99,
        "minibatch_size": 100,
    },
}

trainer = PPOCrowdTrainer(group, env, trainer_config)

metrics = trainer.train(1)

data, metrics, shape = coltra.collect_crowd_data(group, env, STEPS, deterministic=False)
data = data["crowd"]
obs = data.obs
actions = data.action
rewards = data.reward
dones = data.done
last_values = data.last_value

ep_rewards = get_episode_rewards(rewards.numpy(), dones.numpy(), shape)

print(ep_rewards.mean())

ret, adv = discount_experience(
    data.reward, data.value, data.done, data.last_value, use_ugae=True, γ=0.99, η=0, λ=0
)

print(ret.reshape(data.last_value.shape + (-1,)))
