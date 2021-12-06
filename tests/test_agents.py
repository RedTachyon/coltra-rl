import os
import shutil

import numpy as np
import torch
from gym.spaces import Box, Discrete
from torch import Tensor

from coltra.agents import ConstantAgent, CAgent, DAgent, Agent
from coltra.wrappers import ObsVecNormWrapper
from coltra.models.mlp_models import MLPModel
from coltra.buffers import Observation


def test_constant_agent():
    obs = Observation(vector=np.random.randn(5, 81), buffer=np.random.randn(5, 10, 4))
    agent = ConstantAgent(np.array([1.0, 1.0], dtype=np.float32))

    actions, _, _ = agent.act(obs_batch=obs)

    assert actions.continuous.shape == (5, 2)
    # assert actions.discrete is None
    assert np.allclose(actions.continuous, np.ones_like(actions.continuous))

    logprobs, values, entropies = agent.evaluate(obs, actions)

    assert logprobs.shape == (5,)
    assert values.shape == (5,)
    assert entropies.shape == (5,)

    assert isinstance(logprobs, Tensor)
    assert isinstance(values, Tensor)
    assert isinstance(entropies, Tensor)

    assert torch.allclose(logprobs, torch.zeros_like(logprobs))
    assert torch.allclose(values, torch.zeros_like(values))
    assert torch.allclose(entropies, torch.zeros_like(entropies))


# def test_mlp_agent():
#     obs = Observation(vector=np.random.randn(5, 81).astype(np.float32),
#                       buffer=np.random.randn(5, 10, 4).astype(np.float32))
#
#     for sep_value in [True, False]:
#         model = MLPModel({"input_size": 81, "hidden_sizes": [32, 32], "separate_value": sep_value})
#
#         assert len(model.policy_network.hidden_layers) == 2
#
#         agent = CAgent(model)
#         actions, _, extra = agent.act(obs_batch=obs, get_value=True)
#
#         assert actions.continuous.shape == (5, 2)
#         assert actions.discrete is None
#         assert extra["value"].shape == (5,)
#
#         actions, _, extra = agent.act(obs_batch=obs, get_value=True, deterministic=True)
#
#         assert actions.continuous.shape == (5, 2)
#         assert actions.discrete is None
#         assert extra["value"].shape == (5,)
#
#         logprobs, values, entropies = agent.evaluate(obs, actions)
#
#         assert logprobs.shape == (5,)
#         assert values.shape == (5,)
#         assert entropies.shape == (5,)
#
#         assert isinstance(logprobs, Tensor)
#         assert isinstance(values, Tensor)
#         assert isinstance(entropies, Tensor)
#
#         assert agent.get_initial_state() == ()
#
#         values = model.value(obs.tensor())
#
#         assert isinstance(values, Tensor)
#         assert values.shape == (5, 1)
#
#     if torch.cuda.is_available():
#         model.cuda()
#         assert model.device == "cuda"
#     model.cpu()
#     assert model.device == "cpu"


def test_fancy_mlp_agent():
    obs = Observation(
        vector=np.random.randn(5, 81).astype(np.float32),
        buffer=np.random.randn(5, 10, 4).astype(np.float32),
    )

    model = MLPModel(
        {
            "input_size": 81,
            "num_actions": 2,
            "discrete": False,
            "hidden_sizes": [32, 32],
        },
        action_space=Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        ),
    )

    assert len(model.policy_network.hidden_layers) == 2
    assert not model.discrete
    assert len(model.value_network.hidden_layers) == 2

    agent = CAgent(model)
    actions, _, extra = agent.act(obs_batch=obs, get_value=True)

    assert actions.continuous.shape == (5, 2)
    # assert actions.discrete is None
    assert extra["value"].shape == (5,)

    actions, _, extra = agent.act(obs_batch=obs, get_value=True, deterministic=True)

    assert actions.continuous.shape == (5, 2)
    # assert actions.discrete is None
    assert extra["value"].shape == (5,)

    logprobs, values, entropies = agent.evaluate(obs, actions)

    assert logprobs.shape == (5,)
    assert values.shape == (5,)
    assert entropies.shape == (5,)

    assert isinstance(logprobs, Tensor)
    assert isinstance(values, Tensor)
    assert isinstance(entropies, Tensor)

    assert agent.get_initial_state() == ()

    values = model.value(obs.tensor())

    assert isinstance(values, Tensor)
    assert values.shape == (5, 1)  # Squeezing happens in the agent

    if torch.cuda.is_available():
        model.cuda()
        assert model.device == "cuda"
    model.cpu()
    assert model.device == "cpu"


def test_discrete_fancy_mlp_agent():
    obs = Observation(
        vector=np.random.randn(5, 81).astype(np.float32),
        buffer=np.random.randn(5, 10, 4).astype(np.float32),
    )

    model = MLPModel(
        {
            "input_size": 81,
            "hidden_sizes": [32, 32],
            "num_actions": 2,
            "discrete": True,
        },
        action_space=Discrete(2),
    )

    assert len(model.policy_network.hidden_layers) == 2
    assert model.discrete
    assert len(model.value_network.hidden_layers) == 2

    agent = DAgent(model)
    actions, _, extra = agent.act(obs_batch=obs, get_value=True)

    assert actions.discrete.shape == (5,)
    # assert actions.continuous is None
    assert extra["value"].shape == (5,)

    actions, _, extra = agent.act(obs_batch=obs, get_value=True, deterministic=True)

    assert actions.discrete.shape == (5,)
    # assert actions.continuous is None
    assert extra["value"].shape == (5,)

    logprobs, values, entropies = agent.evaluate(obs, actions)

    assert logprobs.shape == (5,)
    assert values.shape == (5,)
    assert entropies.shape == (5,)

    assert isinstance(logprobs, Tensor)
    assert isinstance(values, Tensor)
    assert isinstance(entropies, Tensor)

    assert agent.get_initial_state() == ()

    values = model.value(obs.tensor())

    assert isinstance(values, Tensor)
    assert values.shape == (5, 1)  # Squeezing happens in the agent

    if torch.cuda.is_available():
        model.cuda()
        assert model.device == "cuda"
    model.cpu()
    assert model.device == "cpu"


def test_saving():
    obs = Observation(
        vector=np.random.randn(5, 81).astype(np.float32),
    )

    model = MLPModel(
        {
            "input_size": 81,
            "hidden_sizes": [32, 32],
        },
        action_space=Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        ),
    )

    assert len(model.policy_network.hidden_layers) == 2
    assert not model.discrete
    assert len(model.value_network.hidden_layers) == 2

    agent = CAgent(model)
    actions, _, extra = agent.act(obs_batch=obs, get_value=True)

    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.mkdir("temp")
    agent.save("temp")
    loaded_agent = CAgent.load("temp")

    for param, l_param in zip(
        agent.model.parameters(), loaded_agent.model.parameters()
    ):
        assert torch.allclose(param, l_param)

    shutil.rmtree("temp")


def test_saving_wrapper():
    obs = Observation(
        vector=np.random.randn(5, 81).astype(np.float32),
    )

    model = MLPModel(
        {
            "input_size": 81,
            "hidden_sizes": [32, 32],
        },
        action_space=Box(
            low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
        ),
    )

    assert len(model.policy_network.hidden_layers) == 2
    assert not model.discrete
    assert len(model.value_network.hidden_layers) == 2

    agent = CAgent(model)
    agent = ObsVecNormWrapper(agent)

    actions, _, extra = agent.act(obs_batch=obs, get_value=True)

    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.mkdir("temp")
    agent.save("temp")
    loaded_agent: ObsVecNormWrapper = Agent.load("temp")

    for param, l_param in zip(
        agent.model.parameters(), loaded_agent.model.parameters()
    ):
        assert torch.allclose(param, l_param)

    assert np.allclose(agent._obs_mean, loaded_agent._obs_mean)
    assert np.allclose(agent._obs_var, loaded_agent._obs_var)

    shutil.rmtree("temp")
