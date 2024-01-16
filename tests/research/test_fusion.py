from itertools import chain

import numpy as np
import torch
from gymnasium import Space
from gymnasium.spaces import Box, Discrete
from torch import nn
from torch.distributions import Distribution

from coltra.buffers import Observation
from coltra.envs.spaces import ObservationSpace
from coltra.models import MLPModel
import pytest

from coltra.research.policy_fusion.policy_fusion import JointModel


def assert_models_equal(model1: nn.Module, model2: nn.Module):
    for (p1, p2) in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


ACTION_SPACE_BOX_2 = Box(
    low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
)

ACTION_SPACE_BOX_1 = Box(
    low=-np.ones(1, dtype=np.float32), high=np.ones(1, dtype=np.float32)
)

ACTION_SPACE_DISCRETE_1 = Discrete(1)

ACTION_SPACE_DISCRETE_2 = Discrete(2)


@pytest.mark.parametrize(
    "input_size,action_space,hidden_sizes1,hidden_sizes2",
    [
        (5, ACTION_SPACE_BOX_2, [32, 32], [32, 32]),
        (3, ACTION_SPACE_BOX_1, [32, 64], [128, 16]),
        (3, ACTION_SPACE_BOX_2, [32, 64, 16], [128, 16]),
        (5, ACTION_SPACE_DISCRETE_2, [64, 64], [128, 128, 25]),
        (3, ACTION_SPACE_DISCRETE_2, [64, 64], [64, 64, 17]),
        (3, ACTION_SPACE_DISCRETE_1, [64, 64], [64, 64]),
    ],
)
def test_constructor(
    input_size: int,
    action_space: Space,
    hidden_sizes1: list[int],
    hidden_sizes2: list[int],
):
    model = MLPModel(
        {
            "hidden_sizes": hidden_sizes1,
        },
        observation_space=ObservationSpace(vector=Box(-np.inf, np.inf, (input_size,))),
        action_space=action_space,
    )
    model2 = MLPModel(
        {
            "hidden_sizes": hidden_sizes2,
        },
        observation_space=ObservationSpace(vector=Box(-np.inf, np.inf, (input_size,))),
        action_space=action_space,
    )

    models = [model, model2]
    for _model in models:
        assert _model.input_size == input_size
        assert (
            _model.num_actions == action_space.n
            if isinstance(action_space, Discrete)
            else action_space.shape[0]
        )
        assert _model.discrete == isinstance(action_space, Discrete)

    assert model.latent_size == hidden_sizes1[-1]
    assert model2.latent_size == hidden_sizes2[-1]

    jmodel = JointModel(config={}, action_space=action_space, models=models)
    inp = Observation(torch.ones(input_size))
    out, _, extra = jmodel.forward(inp)
    action = out.sample()

    assert jmodel.latent_size == hidden_sizes1[-1] + hidden_sizes2[-1]
    assert isinstance(out, Distribution)
    assert_models_equal(model, jmodel.models[0])
    assert_models_equal(model2, jmodel.models[1])

    state_dict = model.state_dict()
    state_dict2 = model2.state_dict()
    j_state_dict = jmodel.state_dict()

    for k in state_dict:
        assert torch.allclose(state_dict[k], j_state_dict[f"models.0.{k}"])

    for k in state_dict2:
        assert torch.allclose(state_dict2[k], j_state_dict[f"models.1.{k}"])


@pytest.mark.parametrize(
    "input_size,action_space,hidden_sizes",
    [
        (5, ACTION_SPACE_BOX_2, [32, 32]),
        (3, ACTION_SPACE_BOX_1, [32, 64]),
        (3, ACTION_SPACE_BOX_2, [32, 64, 16]),
        (5, ACTION_SPACE_DISCRETE_2, [64, 64]),
        (3, ACTION_SPACE_DISCRETE_2, [64, 64]),
        (3, ACTION_SPACE_DISCRETE_1, [64, 64]),
        (5, ACTION_SPACE_BOX_2, [10, 10, 10, 10]),
    ],
)
def test_clone(
    input_size: int,
    action_space: Space,
    hidden_sizes: list[int],
):
    model = MLPModel(
        {
            "hidden_sizes": hidden_sizes,
        },
        observation_space=ObservationSpace(vector=Box(-np.inf, np.inf, (input_size,))),
        action_space=action_space,
    )

    assert model.input_size == input_size
    assert (
        model.num_actions == action_space.n
        if isinstance(action_space, Discrete)
        else action_space.shape[0]
    )
    assert model.discrete == isinstance(action_space, Discrete)

    assert model.latent_size == hidden_sizes[-1]

    # jmodel = JointModel(models=models, num_actions=num_actions, discrete=discrete)
    jmodel = JointModel.clone_model(model=model, num_clones=1)
    inp = Observation(torch.ones(input_size))
    out, _, extra = jmodel.forward(inp)
    action = out.sample()

    assert jmodel.latent_size == hidden_sizes[-1] * 2
    assert isinstance(out, Distribution)
    assert_models_equal(model, jmodel.models[0])

    state_dict = model.state_dict()
    j_state_dict = jmodel.state_dict()

    for k in state_dict:
        assert torch.allclose(state_dict[k], j_state_dict[f"models.0.{k}"])

    for k in state_dict:
        assert torch.allclose(state_dict[k], j_state_dict[f"models.1.{k}"])
