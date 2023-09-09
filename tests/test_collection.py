import torch
from torch import Tensor

from coltra.groups import HomogeneousGroup
from coltra.models import LSTMModel
from coltra.models.mlp_models import MLPModel
from coltra.agents import CAgent, DAgent
from coltra.collectors import collect_crowd_data
from coltra.envs.probe_envs import ConstRewardEnv, ActionDependentRewardEnv


def test_const_reward():
    env = ConstRewardEnv(num_agents=10)
    model = MLPModel(
        {}, observation_space=env.observation_space, action_space=env.action_space
    )
    agents = HomogeneousGroup(CAgent(model))

    data_dict, stats, shape = collect_crowd_data(agents, env, num_steps=100)
    data = data_dict[agents.policy_name]

    assert data.obs.vector.shape == (1000, 1)
    assert torch.allclose(data.obs.vector, torch.ones(1000, 1))
    assert torch.allclose(
        data.reward,
        torch.ones(
            1000,
        ),
    )

    assert all(data.done)
    assert env.render() == 0
    assert stats["m_stat"].shape == (100,)

    assert isinstance(data.obs.vector, Tensor)
    assert isinstance(data.action.continuous, Tensor)
    # assert data.action.discrete is None
    assert isinstance(data.reward, Tensor)
    assert isinstance(data.done, Tensor)
    assert isinstance(data.value, Tensor)


def test_lstm_collection():
    env = ActionDependentRewardEnv(num_agents=10)
    model = LSTMModel(
        {}, observation_space=env.observation_space, action_space=env.action_space
    )
    agents = HomogeneousGroup(DAgent(model))

    data_dict, stats, shape = collect_crowd_data(agents, env, num_steps=100)
    data = data_dict[agents.policy_name]

    assert data.obs.vector.shape == (1000, 1)
    assert torch.allclose(data.obs.vector, torch.ones(1000, 1))
    assert torch.allclose(
        data.reward.abs(),
        torch.ones(
            1000,
        ),
    )

    assert all(data.done)
    assert env.render() == 0
    assert stats["m_stat"].shape == (100,)

    assert isinstance(data.obs.vector, Tensor)
    assert isinstance(data.action.discrete, Tensor)
    # assert data.action.discrete is None
    assert isinstance(data.reward, Tensor)
    assert isinstance(data.done, Tensor)
    assert isinstance(data.value, Tensor)
