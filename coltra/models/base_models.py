from typing import List, Callable, Union, Tuple, Dict

import torch
from torch import nn, Tensor
from torch.distributions import Distribution

from coltra.buffers import Observation
from coltra.utils import get_activation, get_initializer


class BaseModel(nn.Module):
    """
    A base class for any NN-based models, stateful or not, following a common convention:
    Each model in its forward pass takes an input and the previous recurrent state.
    If the state is not used in that specific model, it will just be discarded

    The output of each model is an action distribution, the next recurrent state,
    and a dictionary with any extra outputs like the value
    """

    def __init__(self):
        super().__init__()
        self._stateful = False
        # self.config = config
        self.device = 'cpu'

    # TO IMPLEMENT
    def forward(self, x: Observation,
                state: Tuple,
                get_value: bool) -> Tuple[Distribution, Tuple, Dict[str, Tensor]]:
        # Output: action_dist, state, {value, whatever else}
        raise NotImplementedError

    def value(self, x: Observation,
              state: Tuple) -> Tensor:
        raise NotImplementedError

    # Built-ins
    def get_initial_state(self, requires_grad=True) -> Tuple:
        return ()

    @property
    def stateful(self):
        return self._stateful

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.device = 'cuda'

    def cpu(self):
        super().cpu()
        self.device = 'cpu'



class FCNetwork(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_sizes: List[int],
                 hidden_sizes: List[int],
                 activation: str,
                 initializer: str = "kaiming_uniform",
                 is_policy: Union[bool, List[bool]] = False):
        super().__init__()

        self.activation: Callable = get_activation(activation)
        layer_sizes = [input_size] + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.heads = nn.ModuleList([
            nn.Linear(layer_sizes[-1], output_size)
            for output_size in output_sizes
        ])

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
                if divide: head.weight.data /= 100.
                nn.init.zeros_(head.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        return [head(x) for head in self.heads]