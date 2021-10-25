import os
import shutil

import gym
import torch

from coltra.buffers import discrete
from coltra.envs import MultiGymEnv
from coltra.wrappers.env_wrappers import LastRewardWrapper
from coltra.models import MLPModel
from coltra.agents import CAgent
from coltra.wrappers.agent_wrappers import RetNormWrapper


def test_reward_wrapper():
    env = MultiGymEnv.get_venv(8, env_name="CartPole-v1", wrappers=[LastRewardWrapper])

    ref_env = gym.make("CartPole-v1")

    assert env.observation_space.shape[0] == ref_env.observation_space.shape[0] + 1

    env.seed(0)
    obs = env.reset()

    for agent_id in obs:
        assert obs[agent_id].vector[-1] == 0.0
        assert obs[agent_id].vector.shape == env.observation_space.shape

    for _ in range(10):
        obs, reward, done, info = env.step(
            {agent_id: discrete(env.action_space.sample()) for agent_id in obs}
        )

        for agent_id in obs:
            assert obs[agent_id].vector[-1] == reward[agent_id] * (1 - done[agent_id])
            assert obs[agent_id].vector.shape == env.observation_space.shape

def test_agent_wrapper_save():
    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.mkdir("temp")
    agent = RetNormWrapper(CAgent(MLPModel({"input_size": 5, "num_actions": 2, "discrete": False})))
    agent.save("temp")

    loaded = CAgent.load("temp")
    assert loaded._ret_mean == agent._ret_mean
    assert isinstance(loaded._ret_mean, torch.Tensor)

    shutil.rmtree("temp")
