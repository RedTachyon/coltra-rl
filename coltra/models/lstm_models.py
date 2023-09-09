import copy
from typing import Dict, Tuple, Callable, List

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import Space
from gymnasium.spaces import Discrete, Box
from torch import nn, Tensor
from torch.distributions import (
    Distribution,
    Normal,
    Categorical,
    Beta,
    TransformedDistribution,
    AffineTransform,
)
from typarse import BaseConfig

from coltra.buffers import Observation
from coltra.envs.spaces import ObservationSpace
from coltra.utils import get_activation, AffineBeta
from coltra.models.base_models import FCNetwork, BaseModel, BaseQModel, LSTMNetwork
from coltra.configs import MLPConfig, QMLPConfig, LSTMConfig
from .model_utils import ContCategorical


LSTMState = tuple[Tensor, Tensor]


class LSTMModel(BaseModel):
    def __init__(self, config: dict, observation_space: Space, action_space: Space):
        super().__init__(config, observation_space, action_space)
        self._stateful = True

        Config: LSTMConfig = LSTMConfig.clone()

        Config.update(config)
        self.config = Config

        if isinstance(observation_space, ObservationSpace):
            self.input_size = observation_space.vector.shape[0]
        else:
            self.input_size = observation_space.shape[0]

        assert self.config.mode in [
            "head",
            "logstd",
            "beta",
        ], "Model config invalid, mode must be either 'head', 'logstd', 'beta' or None"

        self.action_mode = self.config.mode

        self.sigma0 = self.config.sigma0
        # self.input_size = self.config.input_size
        self.latent_size = self.config.post_hidden_sizes[-1]

        self.activation: Callable = get_activation(self.config.activation)

        heads: tuple[int, ...]
        is_policy: tuple[bool, ...]
        if self.discrete:
            heads = (self.num_actions,)
            is_policy = (True,)
        elif self.action_mode == "head":
            heads = (self.num_actions, self.num_actions)
            is_policy = (True, False)
        elif self.action_mode == "logstd":
            heads = (self.num_actions,)
            is_policy = (True,)
        elif self.action_mode == "beta":
            heads = (self.num_actions, self.num_actions)
            is_policy = (True, True)
        else:
            raise ValueError(
                "Mode invalid, must be passed if the action space is discrete."
            )

        self.policy_network = LSTMNetwork(
            input_size=self.input_size,
            output_sizes=heads,
            pre_hidden_sizes=self.config.pre_hidden_sizes,
            post_hidden_sizes=self.config.post_hidden_sizes,
            lstm_hidden_size=self.config.lstm_hidden_size,
            activation=self.config.activation,
            initializer=self.config.initializer,
            is_policy=is_policy,
        )

        self.value_network = LSTMNetwork(
            input_size=self.input_size,
            output_sizes=[1],
            pre_hidden_sizes=self.config.pre_hidden_sizes,
            post_hidden_sizes=self.config.post_hidden_sizes,
            lstm_hidden_size=self.config.lstm_hidden_size,
            activation=self.config.activation,
            initializer=self.config.initializer,
            is_policy=False,
        )

        if self.action_mode == "logstd":
            self.logstd = nn.Parameter(
                torch.tensor(self.config.sigma0) * torch.ones(1, self.num_actions)
            )
        else:
            self.logstd = None

        self.config = self.config.to_dict()  # Convert to a dictionary for pickling

    def forward(
        self,
        x: Observation,
        state: Tuple[LSTMState, LSTMState],
        get_value: bool = False,
    ) -> Tuple[Distribution, tuple[LSTMState, LSTMState], dict[str, Tensor]]:

        action_distribution: Distribution

        policy_out, new_state = self.policy_network.forward(x.vector, state[0])

        if self.discrete:
            [action_logits] = policy_out
            action_distribution = Categorical(logits=action_logits)
        elif self.action_mode == "head":
            [action_mu, action_std] = policy_out
            action_std = F.softplus(action_std - self.sigma0)

            action_distribution = Normal(loc=action_mu, scale=action_std)
        elif self.action_mode == "logstd":
            [action_mu] = policy_out

            action_std = torch.exp(self.logstd)
            action_distribution = Normal(loc=action_mu, scale=action_std)
        elif self.action_mode == "beta":
            [action_a, action_b] = policy_out
            action_a, action_b = action_a.exp(), action_b.exp()
            action_distribution = AffineBeta(
                action_a, action_b, self.action_low, self.action_high
            )
        else:
            raise ValueError(
                "Mode invalid, must be passed if the action space is discrete."
            )

        extra_outputs = {}

        if get_value:
            value, value_state = self.value(x, state[1])
            extra_outputs["value"] = value
        else:
            value_state = ()

        return action_distribution, (new_state, value_state), extra_outputs

    def latent(self, x: Observation, state: LSTMState) -> Tensor:
        return self.policy_network.latent(x.vector, state)

    def value(self, x: Observation, state: LSTMState) -> tuple[Tensor, LSTMState]:
        [value], new_state = self.value_network(x.vector, state)
        return value, new_state

    def latent_value(self, x: Observation, state: LSTMState) -> Tensor:
        return self.value_network.latent(x.vector, state)

    def get_initial_state(
        self, batch_size: int = 1, requires_grad: bool = True
    ) -> Tuple:
        return self.policy_network.get_initial_state(
            batch_size, requires_grad
        ), self.value_network.get_initial_state(batch_size, requires_grad)
