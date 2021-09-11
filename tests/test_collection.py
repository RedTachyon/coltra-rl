import torch
from torch import Tensor

from coltra.models.mlp_models import FancyMLPModel
from coltra.agents import CAgent
from coltra.collectors import collect_crowd_data
from coltra.envs.probe_envs import ConstRewardEnv


def test_const_reward():
    env = ConstRewardEnv(num_agents=10)
    model = FancyMLPModel({"input_size": env.obs_vector_size})
    agent = CAgent(model)

    data, stats = collect_crowd_data(agent, env, num_steps=100)

    assert data.obs.vector.shape == (1000, 1)
    assert torch.allclose(data.obs.vector, torch.ones(1000, 1))
    assert torch.allclose(data.reward, torch.ones(1000, ))

    assert all(data.done)
    assert env.render() == 0
    assert stats["stat"].shape == (100, 3)

    assert isinstance(data.obs.vector, Tensor)
    assert isinstance(data.action.continuous, Tensor)
    assert data.action.discrete is None
    assert isinstance(data.reward, Tensor)
    assert isinstance(data.done, Tensor)
    assert isinstance(data.value, Tensor)
