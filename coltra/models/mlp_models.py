import copy
from typing import Dict, Tuple, Callable, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Distribution, Normal, Categorical
from typarse import BaseConfig

from coltra.buffers import Observation
from coltra.utils import get_activation
from coltra.models.base_models import FCNetwork, BaseModel
from coltra.configs import MLPConfig


class MLPModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__()

        Config: MLPConfig = MLPConfig.clone()

        Config.update(config)
        self.config = Config

        assert (
            self.config.input_size > 0
        ), "Model config invalid, input_size must be > 0"
        assert (
            self.config.num_actions > 0
        ), "Model config invalid, num_actions must be > 0"

        self.discrete = self.config.discrete
        self.std_head = self.config.std_head
        self.sigma0 = self.config.sigma0

        self.activation: Callable = get_activation(self.config.activation)

        heads: List[int]
        is_policy: List[bool]
        if self.discrete:
            heads = [self.config.num_actions]
            is_policy = [True]
        elif self.std_head:
            heads = [self.config.num_actions, self.config.num_actions]
            is_policy = [True, False]
        else:  # not discrete, not std_head
            heads = [self.config.num_actions]
            is_policy = [True]

        # Create the policy network
        self.policy_network = FCNetwork(
            input_size=self.config.input_size,
            output_sizes=heads,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            initializer=self.config.initializer,
            is_policy=is_policy,
        )

        self.value_network = FCNetwork(
            input_size=self.config.input_size,
            output_sizes=[1],
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            initializer=self.config.initializer,
            is_policy=is_policy,
        )

        if self.std_head:
            self.logstd = None
        else:
            self.logstd = nn.Parameter(
                torch.tensor(self.config.sigma0)
                * torch.ones(1, self.config.num_actions)
            )
        self.config = self.config.to_dict()  # Convert to a dictionary for pickling

    def forward(
        self, x: Observation, state: Tuple = (), get_value: bool = True
    ) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:

        action_distribution: Distribution

        if self.discrete:
            [action_logits] = self.policy_network(x.vector)
            action_distribution = Categorical(logits=action_logits)
        elif self.std_head:
            [action_mu, action_std] = self.policy_network(x.vector)
            action_std = F.softplus(action_std - self.sigma0)

            action_distribution = Normal(loc=action_mu, scale=action_std)
        else:
            [action_mu] = self.policy_network(x.vector)

            action_std = torch.exp(self.logstd)
            action_distribution = Normal(loc=action_mu, scale=action_std)

        extra_outputs = {}

        if get_value:
            value = self.value(x)
            extra_outputs["value"] = value

        return action_distribution, state, extra_outputs

    def value(self, x: Observation, state: Tuple = ()) -> Tensor:
        [value] = self.value_network(x.vector)
        return value
