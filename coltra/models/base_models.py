from __future__ import annotations

from typing import List, Callable, Union, Tuple, Dict, Sequence

import torch
from gymnasium import Space
from gymnasium.spaces import Discrete, Box
from torch import nn, Tensor
from torch.distributions import Distribution

from coltra.buffers import Observation
from coltra.envs.spaces import ActionSpace
from coltra.utils import get_activation, get_initializer


class BaseModel(nn.Module):
    """
    A base class for any NN-based models, stateful or not, following a common convention:
    Each model in its forward pass takes an input and the previous recurrent state.
    If the state is not used in that specific model, it will just be discarded

    The output of each model is an action distribution, the next recurrent state,
    and a dictionary with any extra outputs like the value
    """

    input_size: int
    latent_size: int
    num_actions: int
    discrete: bool
    activation: Callable
    device: str

    def __init__(self, config: dict, observation_space: Space, action_space: Space):
        super().__init__()
        self._stateful = False
        self.raw_config = config
        self.device = "cpu"

        if isinstance(action_space, ActionSpace):
            self.action_space = action_space.space
        else:
            self.action_space = action_space

        self.discrete = isinstance(self.action_space, Discrete)

        if self.discrete:
            assert isinstance(self.action_space, Discrete)
            self.num_actions = self.action_space.n
            self.action_low, self.action_high = None, None
        elif isinstance(self.action_space, Box):
            assert isinstance(self.action_space, Box)
            self.num_actions = self.action_space.shape[0]
            self.action_low, self.action_high = (
                torch.tensor(self.action_space.low),
                torch.tensor(self.action_space.high),
            )
        else:
            self.num_actions = None
            self.action_low, self.action_high = None, None

        self.observation_space = observation_space

    # TO IMPLEMENT
    def forward(
        self, x: Observation, state: Tuple, get_value: bool
    ) -> Tuple[Distribution, tuple, dict[str, Tensor]]:
        # Output: action_dist, state, {value, whatever else}
        raise NotImplementedError

    def value(self, x: Observation, state: Tuple) -> tuple[Tensor, tuple]:
        raise NotImplementedError

    def latent(self, x: Observation, state: Tuple) -> Tensor:
        raise NotImplementedError

    def latent_value(self, x: Observation, state: Tuple) -> Tensor:
        raise NotImplementedError

    # Built-ins
    def get_initial_state(
        self, batch_size: int = 1, requires_grad: bool = True
    ) -> Tuple:
        return ()

    @property
    def stateful(self):
        return self._stateful

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        if not self.discrete:
            self.action_low = self.action_low.to("cuda")
            self.action_high = self.action_high.to("cuda")
        self.device = "cuda"

    def cpu(self):
        super().cpu()
        if not self.discrete:
            self.action_low = self.action_low.to("cpu")
            self.action_high = self.action_high.to("cpu")
        self.device = "cpu"


class BaseQModel(nn.Module):
    def __init__(self, config: dict, action_space: Space):
        super().__init__()
        self.raw_config = config
        self.device = "cpu"

        self.action_space = action_space
        assert isinstance(self.action_space, Discrete)

        # DQN can only handle discrete actions
        self.discrete = True
        self.num_actions = self.action_space.n
        self.action_low, self.action_high = None, None

    def forward(self, x: Observation, state: tuple) -> tuple[Tensor, tuple]:
        raise NotImplementedError

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.device = "cuda"

    def cpu(self):
        super().cpu()
        self.device = "cpu"


class FCNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        hidden_sizes: Sequence[int],
        activation: str,
        initializer: str = "kaiming_uniform",
        is_policy: Union[bool, Sequence[bool]] = False,
    ):
        super().__init__()

        self.activation: Callable = get_activation(activation)
        layer_sizes = (input_size,) + tuple(hidden_sizes)

        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
            ]
        )

        self.heads = nn.ModuleList(
            [nn.Linear(layer_sizes[-1], output_size) for output_size in output_sizes]
        )

        if initializer:
            # If given an initializer, initialize all weights using it, and all biases with 0's
            initializer_ = get_initializer(initializer)

            for layer in self.hidden_layers:
                initializer_(layer.weight)
                nn.init.zeros_(layer.bias)

            for i, head in enumerate(self.heads):
                initializer_(head.weight)
                if isinstance(is_policy, list):
                    divide = is_policy[i]
                else:
                    divide = is_policy
                if divide:
                    head.weight.data /= 100.0
                nn.init.zeros_(head.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        return [head(x) for head in self.heads]

    def latent(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if i < len(self.hidden_layers) - 1:
                x = self.activation(x)

        return x


class LSTMNetwork(nn.Module):
    # TODO: states need to be recomputed every time in evaluation
    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        lstm_hidden_size: int,
        pre_hidden_sizes: Sequence[int],
        post_hidden_sizes: Sequence[int],
        activation: str,
        initializer: str = "kaiming_uniform",
        is_policy: Union[bool, Sequence[bool]] = False,
    ):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size

        self.activation = get_activation(activation)

        # Define Pre-MLP layers
        pre_mlp_layer_sizes = (input_size,) + tuple(pre_hidden_sizes)
        self.pre_mlp = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(
                    pre_mlp_layer_sizes, pre_mlp_layer_sizes[1:]
                )
            ]
        )

        # Define LSTM layer
        self.lstm = nn.LSTMCell(pre_mlp_layer_sizes[-1], lstm_hidden_size)

        # Define Post-MLP layers
        post_mlp_layer_sizes = (lstm_hidden_size,) + tuple(post_hidden_sizes)
        self.post_mlp = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(
                    post_mlp_layer_sizes, post_mlp_layer_sizes[1:]
                )
            ]
        )

        # Define output heads
        self.heads = nn.ModuleList(
            [
                nn.Linear(post_mlp_layer_sizes[-1], output_size)
                for output_size in output_sizes
            ]
        )

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # Weight Initialization
        if initializer:
            initializer_ = get_initializer(initializer)
            for layer in self.pre_mlp + self.post_mlp:
                initializer_(layer.weight)
                nn.init.zeros_(layer.bias)

            for i, head in enumerate(self.heads):
                initializer_(head.weight)
                if isinstance(is_policy, list):
                    divide = is_policy[i]
                else:
                    divide = is_policy
                if divide:
                    head.weight.data /= 100.0
                nn.init.zeros_(head.bias)

    def forward(
        self, x: Tensor, hidden_state: tuple[Tensor, Tensor]
    ) -> Tuple[List[Tensor], Tuple[Tensor, Tensor]]:
        # Pre-MLP
        for layer in self.pre_mlp:
            x = self.activation(layer(x))

        # LSTM
        hidden_state, cell_state = self.lstm(x, hidden_state)

        x = hidden_state
        # Post-MLP
        for layer in self.post_mlp:
            x = self.activation(layer(x))

        # Output heads
        return [head(x) for head in self.heads], (hidden_state, cell_state)

    def latent(self, x: Tensor, hidden_state: Tuple[Tensor, Tensor]) -> Tensor:
        # Pre-MLP
        for layer in self.pre_mlp:
            x = self.activation(layer(x))

        # LSTM
        hidden_state, cell_state = self.lstm(x, hidden_state)

        x = hidden_state

        # Post-MLP
        for layer in self.post_mlp:
            x = self.activation(layer(x))

        return x

    def get_initial_state(
        self, batch_size: int = 1, requires_grad: bool = True
    ) -> Tuple[Tensor, Tensor]:
        return (
            torch.zeros(
                batch_size,
                self.lstm_hidden_size,
                requires_grad=requires_grad,
                device=self.device,
            ),
            torch.zeros(
                batch_size,
                self.lstm_hidden_size,
                requires_grad=requires_grad,
                device=self.device,
            ),
        )
