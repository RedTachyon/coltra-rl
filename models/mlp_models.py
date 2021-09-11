from typing import Dict, Tuple, Callable, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Distribution, Normal, Categorical
from typarse import BaseConfig

from coltra.buffers import Observation
from coltra.utils import get_activation
from .base_models import FCNetwork, BaseModel


class MLPModel(BaseModel):
    """DEPRECATED (mostly)"""
    def __init__(self, config: Dict):
        super().__init__()

        class Config(BaseConfig):
            input_size: int = 94
            num_actions: int = 2
            activation: str = "leaky_relu"

            hidden_sizes: List[int] = [64, 64]
            separate_value: bool = False

            sigma0: float = 0.3

            initializer: str = "kaiming_uniform"

        Config.update(config)
        self.config = Config

        self.activation: Callable = get_activation(self.config.activation)
        self.separate_value = self.config.separate_value

        if not self.separate_value:
            self.policy_network = FCNetwork(
                input_size=self.config.input_size,
                output_sizes=[self.config.num_actions, 1],
                hidden_sizes=self.config.hidden_sizes,
                activation=self.config.activation,
                initializer=self.config.initializer,
                is_policy=[True, False]
            )

            self.value_network = None
        else:
            self.policy_network = FCNetwork(
                input_size=self.config.input_size,
                output_sizes=[self.config.num_actions],
                hidden_sizes=self.config.hidden_sizes,
                activation=self.config.activation,
                initializer=self.config.initializer,
                is_policy=True
            )

            self.value_network = FCNetwork(
                input_size=self.config.input_size,
                output_sizes=[1],
                hidden_sizes=self.config.hidden_sizes,
                activation=self.config.activation,
                initializer=self.config.initializer,
                is_policy=False
            )

        self.std = nn.Parameter(torch.tensor(self.config.sigma0) * torch.ones(1, self.config.num_actions))

        self.config = self.config.to_dict()  # Convert to a dictionary for pickling

    def forward(self, x: Observation,
                state: Tuple = (),
                get_value: bool = True) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:
        if self.separate_value:
            [action_mu] = self.policy_network(x.vector)
            value = None
        else:
            [action_mu, value] = self.policy_network(x.vector)

        action_distribution = Normal(loc=action_mu, scale=self.std)

        extra_outputs = {}

        if get_value:
            if self.separate_value:
                [value] = self.value_network(x.vector)

            extra_outputs["value"] = value

        return action_distribution, state, extra_outputs

    def value(self, x: Observation,
              state: Tuple = ()) -> Tensor:
        if self.separate_value:
            [value] = self.value_network(x.vector)
        else:
            [_, value] = self.policy_network(x.vector)

        return value


class FancyMLPModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__()

        class Config(BaseConfig):
            input_size: int = 94
            num_actions: int = 2
            activation: str = "leaky_relu"

            hidden_sizes: List[int] = [64, 64]

            discrete: bool = False

            initializer: str = "kaiming_uniform"

        Config.update(config)
        self.config = Config
        self.discrete = self.config.discrete

        self.activation: Callable = get_activation(self.config.activation)

        heads = [self.config.num_actions] \
            if self.discrete \
            else [self.config.num_actions, self.config.num_actions]

        # Create the policy network
        self.policy_network = FCNetwork(
            input_size=self.config.input_size,
            output_sizes=heads,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            initializer=self.config.initializer,
            is_policy=[True, False]
        )

        self.value_network = FCNetwork(
            input_size=self.config.input_size,
            output_sizes=[1],
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            initializer=self.config.initializer,
            is_policy=False
        )

        self.config = self.config.to_dict()  # Convert to a dictionary for pickling

    def forward(self, x: Observation,
                state: Tuple = (),
                get_value: bool = True) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:

        if self.discrete:
            [action_logits] = self.policy_network(x.vector)
            action_distribution = Categorical(logits=action_logits)
        else:
            [action_mu, action_std] = self.policy_network(x.vector)
            action_std = F.softplus(action_std - 0.5)

            action_distribution = Normal(loc=action_mu, scale=action_std)

        extra_outputs = {}

        if get_value:
            [value] = self.value_network(x.vector)
            extra_outputs["value"] = value

        return action_distribution, state, extra_outputs

    def value(self, x: Observation,
              state: Tuple = ()) -> Tensor:
        [value] = self.value_network(x.vector)
        return value
