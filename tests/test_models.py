from typing import List

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import Tensor
from torch.distributions import Normal
from typarse import BaseConfig

from coltra.agents import CAgent
from coltra.buffers import Observation
from coltra.envs.spaces import ObservationSpace
from coltra.models import FCNetwork, MLPModel

from coltra.models.relational_models import RelationNetwork, RelationModel
from coltra.utils import AffineBeta


def test_fc():
    torch.manual_seed(0)

    network = FCNetwork(
        input_size=10,
        output_sizes=[2, 2],
        hidden_sizes=[64, 64],
        activation="tanh",
        initializer="kaiming_uniform",
        is_policy=True,
    )

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
    network = FCNetwork(
        input_size=10,
        output_sizes=[32],
        hidden_sizes=[],
        activation="elu",
        initializer="kaiming_uniform",
        is_policy=False,
    )

    inp = torch.randn(5, 10)
    [out] = network(inp)

    assert isinstance(out, Tensor)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_relnet():
    class Config(BaseConfig):
        sigma0: float = 0.0

        vec_hidden_layers: List[int] = [32, 32]
        rel_hidden_layers: List[int] = [32, 32]
        com_hidden_layers: List[int] = [32, 32]

        activation: str = "tanh"
        initializer: str = "orthogonal"

    config = Config.to_dict()
    model = RelationModel(
        config,
        observation_space=ObservationSpace(
            vector=Box(-np.inf, np.inf, shape=(4,)),
            buffer=Box(-np.inf, np.inf, shape=(11, 5)),
        ),
        action_space=Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        ),
    )

    obs = Observation(vector=torch.rand(7, 4), buffer=torch.rand(7, 11, 5))

    action, state, extra = model(obs, get_value=True)

    assert isinstance(action, Normal)
    assert action.loc.shape == (7, 2)
    assert action.scale.shape == (7, 2)
    assert state == ()
    assert extra["value"].shape == (7, 1)


def test_multiple_mlps():
    config1 = {}

    mlp1 = MLPModel(
        config1,
        observation_space=ObservationSpace(vector=Box(-np.inf, np.inf, shape=(3,))),
        action_space=Box(
            low=-np.ones(1, dtype=np.float32), high=np.ones(1, dtype=np.float32)
        ),
    )

    assert mlp1.discrete is False
    assert mlp1.policy_network.hidden_layers[0].in_features == 3

    config2 = {}

    mlp2 = MLPModel(
        config2,
        observation_space=ObservationSpace(vector=Box(-np.inf, np.inf, shape=(5,))),
        action_space=Discrete(2),
    )

    assert mlp2.discrete is True
    assert mlp2.policy_network.hidden_layers[0].in_features == 5


def test_beta_mlp():
    config = {"mode": "beta"}

    mlp = MLPModel(
        config,
        observation_space=ObservationSpace(vector=Box(-np.inf, np.inf, shape=(5,))),
        action_space=Box(
            low=-np.ones(1, dtype=np.float32), high=np.ones(1, dtype=np.float32)
        ),
    )

    dummy_input = Observation(torch.randn(5, 5))
    action_dist, state, extra = mlp(dummy_input, get_value=True)

    assert isinstance(action_dist, AffineBeta)
    assert torch.allclose(action_dist.low, -torch.ones(5, 2))
    assert torch.allclose(action_dist.high, torch.ones(5, 2))
