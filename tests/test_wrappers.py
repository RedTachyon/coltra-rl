import os
import shutil

import gym
import numpy as np
import torch
from gym.spaces import Box

from coltra.buffers import discrete
from coltra.envs import MultiGymEnv
from coltra.envs.spaces import ObservationSpace
from coltra.wrappers.env_wrappers import LastRewardWrapper
from coltra.models import MLPModel
from coltra.agents import CAgent
from coltra.wrappers.agent_wrappers import RetNormWrapper


def test_reward_wrapper():
    env = MultiGymEnv.get_venv(8, env_name="CartPole-v1", wrappers=[LastRewardWrapper])

    ref_env = gym.make("CartPole-v1")

    assert (
        env.observation_space["vector"].shape[0]
        == ref_env.observation_space.shape[0] + 1
    )

    # env.seed(0)
    obs = env.reset(seed=0)

    for agent_id in obs:
        assert obs[agent_id].vector[-1] == 0.0
        assert obs[agent_id].vector.shape == env.observation_space["vector"].shape

    for _ in range(10):
        obs, reward, done, info = env.step(
            {agent_id: env.action_space.sample() for agent_id in obs}
        )

        for agent_id in obs:
            assert obs[agent_id].vector[-1] == reward[agent_id] * (1 - done[agent_id])
            assert obs[agent_id].vector.shape == env.observation_space["vector"].shape


def test_agent_wrapper_save():
    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.mkdir("temp")
    agent = RetNormWrapper(
        CAgent(
            MLPModel(
                {},
                observation_space=ObservationSpace(vector=Box(-np.inf, np.inf, (5,))),
                action_space=Box(
                    low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
                ),
            )
        )
    )
    agent.save("temp")

    loaded = CAgent.load("temp")
    assert loaded._ret_mean == agent._ret_mean
    assert isinstance(loaded._ret_mean, torch.Tensor)

    shutil.rmtree("temp")
