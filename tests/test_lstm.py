import torch

from coltra import Observation
from coltra.models.base_models import LSTMNetwork
from coltra.models import LSTMModel

from gymnasium.spaces import Box


def test_forward_pass():
    model = LSTMNetwork(
        input_size=4,
        output_sizes=[2, 3],
        pre_hidden_sizes=[64, 64],
        post_hidden_sizes=[64, 64],
        lstm_hidden_size=16,
        activation="tanh",
    )
    x = torch.randn(5, 4)  # 5 samples, each of dimension 4
    state = model.get_initial_state(batch_size=5)
    outputs, new_state = model.forward(x, state)

    assert len(outputs) == 2  # Because we have two output heads
    assert outputs[0].shape == (5, 2)  # The size of the first output head is [2]
    assert outputs[1].shape == (5, 3)  # The size of the second output head is [3]
    assert new_state[0].shape == (5, 16)  # Hidden state shape
    assert new_state[1].shape == (5, 16)  # Cell state shape


# Initialization
def test_lstm_model_initialization():
    config = {
        "mode": "head",
        "sigma0": 0.5,
        "activation": "tanh",
        "pre_hidden_sizes": [128],
        "post_hidden_sizes": [64],
        "lstm_hidden_size": 32,
    }
    obs_space = Box(low=0, high=1, shape=(4,))
    act_space = Box(low=-1, high=1, shape=(2,))

    model = LSTMModel(config, obs_space, act_space)

    assert model.input_size == 4
    assert model.action_mode == "head"
    assert model.latent_size == 64


def test_lstm_model_forward():
    config = {
        "mode": "head",
        "sigma0": 0.5,
        "activation": "tanh",
        "pre_hidden_sizes": [128],
        "post_hidden_sizes": [64],
        "lstm_hidden_size": 32,
    }
    obs_space = Box(low=0, high=1, shape=(4,))
    act_space = Box(low=-1, high=1, shape=(2,))

    model = LSTMModel(config, obs_space, act_space)
    obs_tensor = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    obs = Observation(vector=obs_tensor)
    state = model.get_initial_state(batch_size=1)

    action_dist, new_state, extra_outputs = model.forward(obs, state, get_value=True)

    assert isinstance(action_dist, torch.distributions.Normal)
    assert "value" in extra_outputs
    assert new_state[0][0].shape == (1, 32)
    assert new_state[0][1].shape == (1, 32)


# Latent representation
def test_lstm_model_latent():
    config = {
        "mode": "head",
        "sigma0": 0.5,
        "activation": "tanh",
        "pre_hidden_sizes": [128],
        "post_hidden_sizes": [64],
        "lstm_hidden_size": 32,
    }
    obs_space = Box(low=0, high=1, shape=(4,))
    act_space = Box(low=-1, high=1, shape=(2,))

    model = LSTMModel(config, obs_space, act_space)
    obs_tensor = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    obs = Observation(vector=obs_tensor)
    state = model.get_initial_state()

    latent_vector = model.latent(obs, state[0])

    assert latent_vector.shape == (1, 64)
