import numpy as np
import torch
from gym.spaces import Box
from torch.distributions import Normal
from typarse import BaseConfig

from coltra import Observation
from coltra.envs.spaces import ObservationSpace
from coltra.models.attention_models import AttentionNetwork, AttentionModel


def test_attention_network():
    net = AttentionNetwork(
        vec_input_size=8,
        rel_input_size=8,
        vec_hidden_layers=(32,),
        rel_hidden_layers=(32,),
        com_hidden_layers=(32,),
        output_sizes=(2,),
        emb_size=32,
        attention_heads=4,
        is_policy=True,
    )

    obs = Observation(vector=torch.rand(7, 8), buffer=torch.rand(7, 11, 8))
    out, attention = net(obs)


def test_attention_model():
    class Config(BaseConfig):
        activation: str = "tanh"
        initializer: str = "orthogonal"

        sigma0: float = 1.0

        beta: bool = False

        vec_hidden_layers: list[int] = [32, 32]
        rel_hidden_layers: list[int] = [32, 32]
        com_hidden_layers: list[int] = [32, 32]
        emb_size: int = 64
        attention_heads: int = 8

    config = Config.to_dict()
    model = AttentionModel(
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
    assert extra["attention"] is not None
    assert extra["attention"].shape == (7, 8, 11)
