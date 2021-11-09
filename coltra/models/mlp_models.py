import copy
from typing import Dict, Tuple, Callable, List

import torch
import torch.nn.functional as F
from gym import Space
from gym.spaces import Discrete, Box
from torch import nn, Tensor
from torch.distributions import Distribution, Normal, Categorical, Beta
from typarse import BaseConfig

from coltra.buffers import Observation
from coltra.utils import get_activation
from coltra.models.base_models import FCNetwork, BaseModel
from coltra.configs import MLPConfig


class MLPModel(BaseModel):
    def __init__(self, config: Dict, action_space: Space):
        super().__init__()

        Config: MLPConfig = MLPConfig.clone()

        Config.update(config)
        self.config = Config

        assert (
            self.config.input_size > 0
        ), "Model config invalid, input_size must be > 0"

        assert self.config.mode in ["head", "logstd", "beta"], \
            "Model config invalid, mode must be either 'head', 'logstd', 'beta' or None"

        self.action_space = action_space
        self.discrete = isinstance(self.action_space, Discrete)
        self.action_mode = self.config.mode

        if self.discrete:
            assert isinstance(self.action_space, Discrete)
            self.num_actions = self.action_space.n
            self.action_low, self.action_high = None, None
        else:
            assert isinstance(self.action_space, Box)
            self.num_actions = self.action_space.shape[0]
            self.action_low, self.action_high = torch.tensor(self.action_space.low), torch.tensor(self.action_space.high)

        self.sigma0 = self.config.sigma0
        self.input_size = self.config.input_size
        self.latent_size = self.config.hidden_sizes[-1]

        self.activation: Callable = get_activation(self.config.activation)

        heads: List[int]
        is_policy: List[bool]
        if self.discrete:
            heads = [self.num_actions]
            is_policy = [True]
        elif self.action_mode == "head":
            heads = [self.num_actions, self.num_actions]
            is_policy = [True, False]
        elif self.action_mode == "logstd":
            heads = [self.num_actions]
            is_policy = [True]
        elif self.action_mode == "beta":
            heads = [self.num_actions, self.num_actions]
            is_policy = [True, True]
        else:
            raise ValueError("Mode invalid, must be passed if the action space is discrete.")

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

        if self.action_mode == "logstd":
            self.logstd = nn.Parameter(
                torch.tensor(self.config.sigma0)
                * torch.ones(1, self.num_actions)
            )
        else:
            self.logstd = None

        self.config = self.config.to_dict()  # Convert to a dictionary for pickling

    def forward(
        self, x: Observation, state: Tuple = (), get_value: bool = True
    ) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:

        action_distribution: Distribution

        if self.discrete:
            [action_logits] = self.policy_network(x.vector)
            action_distribution = Categorical(logits=action_logits)
        elif self.action_mode == "head":
            [action_mu, action_std] = self.policy_network(x.vector)
            action_std = F.softplus(action_std - self.sigma0)

            action_distribution = Normal(loc=action_mu, scale=action_std)
        elif self.action_mode == "logstd":
            [action_mu] = self.policy_network(x.vector)

            action_std = torch.exp(self.logstd)
            action_distribution = Normal(loc=action_mu, scale=action_std)
        elif self.action_mode == "beta":  # TODO: add appropriate transformations
            [action_mu, action_eta] = self.policy_network(x.vector)
            action_distribution = Beta(action_mu, action_eta)
        else:
            raise ValueError("Mode invalid, must be passed if the action space is discrete.")

        extra_outputs = {}

        if get_value:
            value = self.value(x)
            extra_outputs["value"] = value

        return action_distribution, state, extra_outputs

    def latent(self, x: Observation, state: Tuple) -> Tensor:
        return self.policy_network.latent(x.vector)

    def value(self, x: Observation, state: Tuple = ()) -> Tensor:
        [value] = self.value_network(x.vector)
        return value

    def latent_value(self, x: Observation, state: Tuple) -> Tensor:
        return self.value_network.latent(x.vector)


class ImageMLPModel(MLPModel):
    def __init__(self, config: Dict):
        super().__init__(config)

    def _flatten(self, obs: Observation):
        image: torch.Tensor = obs.image
        if image.shape == 3:  # no batch
            vector = torch.flatten(image)
        else:  # image.shape == 4, batch
            vector = torch.flatten(image, start_dim=1)

        return Observation(vector=vector)

    def forward(
        self, x: Observation, state: Tuple = (), get_value: bool = True
    ) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:
        return super().forward(self._flatten(x), state, get_value)

    def latent(self, x: Observation, state: Tuple) -> Tensor:
        return super().latent(self._flatten(x), state)

    def value(self, x: Observation, state: Tuple = ()) -> Tensor:
        if x.image is not None and x.vector is None:
            x = self._flatten(x)
        return super().value(x, state)

    def latent_value(self, x: Observation, state: Tuple) -> Tensor:
        return super().latent_value(self._flatten(x), state)
