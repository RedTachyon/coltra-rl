from typing import List

import torch
from torch import Tensor
from torch.distributions import Normal
from typarse import BaseConfig

from coltra.agents import CAgent
from coltra.buffers import Observation
from coltra.models import FCNetwork
from coltra.models.raycast_models import LeeNetwork, LeeModel
from coltra.models.relational_models import RelationNetwork, RelationModel


def test_fc():
    torch.manual_seed(0)

    network = FCNetwork(input_size=10,
                        output_sizes=[2, 2],
                        hidden_sizes=[64, 64],
                        activation='tanh',
                        initializer='kaiming_uniform',
                        is_policy=True)

    inp = torch.zeros(5, 10)
    [out1, out2] = network(inp)

    assert isinstance(out1, Tensor)
    assert isinstance(out2, Tensor)
    assert torch.allclose(out1, out2)
    assert torch.allclose(out1, torch.zeros((5, 2)))

    inp = torch.randn(5, 10)
    [out1, out2] = network(inp)

    assert isinstance(out1, Tensor)
    assert isinstance(out2, Tensor)
    assert not torch.allclose(out1, out2)
    assert not torch.allclose(out1, torch.zeros((5, 2)))


def test_empty_fc():
    network = FCNetwork(input_size=10,
                        output_sizes=[32],
                        hidden_sizes=[],
                        activation='elu',
                        initializer='kaiming_uniform',
                        is_policy=False)

    inp = torch.randn(5, 10)
    [out] = network(inp)

    assert isinstance(out, Tensor)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_lee():
    network = LeeNetwork(input_size=4,
                         output_sizes=[2, 4],
                         rays_input_size=126,
                         conv_filters=2)

    obs = Observation(vector=torch.randn(10, 4), rays=torch.randn(10, 126))

    [out1, out2] = network(obs)

    assert out1.shape == (10, 2)
    assert out2.shape == (10, 4)

    model = LeeModel({})
    agent = CAgent(model)

    action, state, extra = agent.act(obs, get_value=True)

    assert action.continuous.shape == (10, 2)
    assert state == ()
    assert extra["value"].shape == (10,)


def test_relnet():
    class Config(BaseConfig):
        vec_input_size: int = 4
        rel_input_size: int = 5
        vec_hidden_layers: List[int] = [32, 32]
        rel_hidden_layers: List[int] = [32, 32]
        com_hidden_layers: List[int] = [32, 32]
        num_action: int = 2
        activation: str = "tanh"
        initializer: str = "kaiming_uniform"

    config = Config.to_dict()
    model = RelationModel(config)

    obs = Observation(
        vector=torch.rand(7, 4),
        buffer=torch.rand(7, 11, 5)
    )

    action, state, extra = model(obs, get_value=True)

    assert isinstance(action, Normal)
    assert action.loc.shape == (7, 2)
    assert action.scale.shape == (7, 2)
    assert state == ()
    assert extra["value"].shape == (7, 1)