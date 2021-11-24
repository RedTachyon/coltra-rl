import torch
from torch import Tensor

from coltra.groups import HomogeneousGroup
from coltra.models.mlp_models import MLPModel
from coltra.agents import CAgent
from coltra.collectors import collect_crowd_data
from coltra.envs.probe_envs import ConstRewardEnv


def test_const_reward():
    env = ConstRewardEnv(num_agents=10)
    model = MLPModel({"input_size": env.obs_vector_size}, action_space=env.action_space)
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
    assert stats["stat"].shape == (100,)

    assert isinstance(data.obs.vector, Tensor)
    assert isinstance(data.action.continuous, Tensor)
    # assert data.action.discrete is None
    assert isinstance(data.reward, Tensor)
    assert isinstance(data.done, Tensor)
    assert isinstance(data.value, Tensor)
