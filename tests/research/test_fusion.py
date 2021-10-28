from itertools import chain

import torch
from torch import nn
from torch.distributions import Distribution

from coltra.buffers import Observation
from coltra.models import MLPModel
import pytest

from coltra.research.policy_fusion import JointModel


def assert_models_equal(model1: nn.Module, model2: nn.Module):
    for (p1, p2) in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


@pytest.mark.parametrize(
    "input_size,num_actions,discrete,hidden_sizes1,hidden_sizes2",
    [
        (5, 2, False, [32, 32], [32, 32]),
        (3, 1, False, [32, 64], [128, 16]),
        (3, 1, False, [32, 64, 16], [128, 16]),
        (5, 2, True, [64, 64], [128, 128, 25]),
        (3, 1, True, [64, 64], [64, 64, 17]),
        (3, 1, True, [64, 64], [64, 64]),
    ],
)
def test_constructor(
    input_size: int,
    num_actions: int,
    discrete: bool,
    hidden_sizes1: list[int],
    hidden_sizes2: list[int],
):
    model = MLPModel(
        {
            "input_size": input_size,
            "num_actions": num_actions,
            "discrete": discrete,
            "hidden_sizes": hidden_sizes1,
        }
    )
    model2 = MLPModel(
        {
            "input_size": input_size,
            "num_actions": num_actions,
            "discrete": discrete,
            "hidden_sizes": hidden_sizes2,
        }
    )

    models = [model, model2]
    for _model in models:
        assert _model.input_size == input_size
        assert _model.num_actions == num_actions
        assert _model.discrete == discrete

    assert model.latent_size == hidden_sizes1[-1]
    assert model2.latent_size == hidden_sizes2[-1]

    jmodel = JointModel(models=models, num_actions=num_actions, discrete=discrete)
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
    "input_size,num_actions,discrete,hidden_sizes",
    [
        (5, 2, False, [32, 32]),
        (3, 1, False, [32, 64]),
        (5, 2, True, [64, 64]),
        (3, 1, True, [64, 64]),
        (3, 1, True, [64, 64, 15]),
        (3, 1, True, [13]),
        (5, 2, False, [10, 10, 10, 10]),
    ],
)
def test_clone(
    input_size: int,
    num_actions: int,
    discrete: bool,
    hidden_sizes: list[int],
):
    model = MLPModel(
        {
            "input_size": input_size,
            "num_actions": num_actions,
            "discrete": discrete,
            "hidden_sizes": hidden_sizes,
        }
    )

    assert model.input_size == input_size
    assert model.num_actions == num_actions
    assert model.discrete == discrete

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
