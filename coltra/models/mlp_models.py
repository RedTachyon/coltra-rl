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
from coltra.models.base_models import FCNetwork, BaseModel, BaseQModel
from coltra.configs import MLPConfig, QMLPConfig
from .model_utils import ContCategorical


class MLPModel(BaseModel):
    def __init__(self, config: dict, observation_space: Space, action_space: Space):
        super().__init__(config, observation_space, action_space)

        Config: MLPConfig = MLPConfig.clone()

        Config.update(config)
        self.config = Config

        if isinstance(observation_space, ObservationSpace):
            self.input_size = observation_space.vector.shape[0]
        else:
            self.input_size = observation_space.shape[0]

        # assert (
        #     input_size > 0
        # ), "Model config invalid, input_size must be > 0"

        assert self.config.mode in [
            "head",
            "logstd",
            "beta",
        ], "Model config invalid, mode must be either 'head', 'logstd', 'beta' or None"

        self.action_mode = self.config.mode

        self.sigma0 = self.config.sigma0
        # self.input_size = self.config.input_size
        self.latent_size = self.config.hidden_sizes[-1]

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

        # Create the policy network
        self.policy_network = FCNetwork(
            input_size=self.input_size,
            output_sizes=heads,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            initializer=self.config.initializer,
            is_policy=is_policy,
        )

        self.value_network = FCNetwork(
            input_size=self.input_size,
            output_sizes=[1],
            hidden_sizes=self.config.hidden_sizes,
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
        self, x: Observation, state: Tuple = (), get_value: bool = False
    ) -> Tuple[Distribution, tuple, dict[str, Tensor]]:

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
        elif self.action_mode == "beta":
            [action_a, action_b] = self.policy_network(x.vector)
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
            value, _ = self.value(x)
            extra_outputs["value"] = value

        return action_distribution, state, extra_outputs

    def latent(self, x: Observation, state: Tuple) -> Tensor:
        return self.policy_network.latent(x.vector)

    def value(self, x: Observation, state: Tuple = ()) -> tuple[Tensor, tuple]:
        [value] = self.value_network(x.vector)
        return value, state

    def latent_value(self, x: Observation, state: Tuple) -> Tensor:
        return self.value_network.latent(x.vector)


class FlattenMLPModel(MLPModel):
    """
    An abstraction to create models that flatten several inputs into a single vector.
    Need to implement `_flatten` for any new models, and the result will be a fully connected
    model that only has a `vector` entry, flattened according to the custom model.
    """

    def _flatten(self, obs: Observation) -> Observation:
        raise NotImplementedError

    def forward(
        self, x: Observation, state: Tuple = (), get_value: bool = False
    ) -> Tuple[Distribution, tuple, dict[str, Tensor]]:
        return super().forward(self._flatten(x), state, get_value)

    def latent(self, x: Observation, state: Tuple = ()) -> Tensor:
        return super().latent(self._flatten(x), state)

    def value(self, x: Observation, state: Tuple = ()) -> tuple[Tensor, tuple]:
        return super().value(self._flatten(x), state)

    def latent_value(self, x: Observation, state: Tuple) -> Tensor:
        return super().latent_value(self._flatten(x), state)


class ImageMLPModel(FlattenMLPModel):
    def __init__(
        self, config: dict, observation_space: ObservationSpace, action_space: Space
    ):
        assert (
            "image" in observation_space.spaces
        ), "ImageMLPModel requires an observation space with image"

        vector_size = (
            observation_space.vector.shape[0]
            if "vector" in observation_space.spaces
            else 0
        )
        image_size = np.prod(observation_space.spaces["image"].shape)
        new_vector_size = vector_size + image_size

        new_observation_space = ObservationSpace(
            {"vector": Box(-np.inf, np.inf, (new_vector_size,))}
        )

        super().__init__(config, new_observation_space, action_space)

    def _flatten(self, obs: Observation) -> Observation:
        if not hasattr(obs, "image"):
            return obs
        image: torch.Tensor = obs.image

        if len(image.shape) == 3:  # no batch
            dim = 0
        else:  # image.shape == 4, batch
            dim = 1

        vector = torch.flatten(image, start_dim=dim)

        if hasattr(obs, "vector"):
            vector = torch.cat([obs.vector, vector], dim=dim)

        return Observation(vector=vector)


class RayMLPModel(FlattenMLPModel):
    def __init__(
        self, config: dict, observation_space: ObservationSpace, action_space: Space
    ):
        assert (
            "rays" in observation_space.spaces
        ), "RayMLPModel requires an observation space with rays"

        new_vector_size = (
            observation_space.vector.shape[0]
            if "vector" in observation_space.spaces
            else 0
        ) + np.prod(observation_space.spaces["rays"].shape)

        new_observation_space = ObservationSpace(
            {"vector": Box(-np.inf, np.inf, (new_vector_size,))}
        )

        super().__init__(config, new_observation_space, action_space)

    def _flatten(self, obs: Observation) -> Observation:
        if not hasattr(obs, "rays"):
            return obs

        rays = obs.rays
        if len(rays.shape) == 1:  # no batch
            dim = 0
        else:  # rays.shape == 2, batch
            dim = 1

        if hasattr(obs, "vector"):
            vector = torch.cat([obs.vector, rays], dim=dim)
        else:
            vector = rays

        return Observation(vector=vector)


class BufferMLPModel(FlattenMLPModel):
    def __init__(
        self, config: dict, observation_space: ObservationSpace, action_space: Space
    ):
        assert (
            "buffer" in observation_space.spaces
        ), "BufferMLPModel requires an observation space with buffer"

        new_vector_size = (
            observation_space.vector.shape[0]
            if "vector" in observation_space.spaces
            else 0
        ) + np.prod(observation_space.spaces["buffer"].shape)

        new_observation_space = ObservationSpace(
            {"vector": Box(-np.inf, np.inf, (new_vector_size,))}
        )

        super().__init__(config, new_observation_space, action_space)

    def _flatten(self, obs: Observation) -> Observation:
        if not hasattr(obs, "buffer"):
            return obs
        buffer: torch.Tensor = obs.buffer

        if len(buffer.shape) == 2:  # no batch
            dim = 0
        else:  # image.shape == 3, batch
            dim = 1

        vector = torch.flatten(buffer, start_dim=dim)

        if hasattr(obs, "vector"):
            vector = torch.cat([obs.vector, vector], dim=dim)

        return Observation(vector=vector)


class MLPQModel(BaseQModel):
    def __init__(self, config: dict, action_space: Space):
        super().__init__(config, action_space)

        Config: QMLPConfig = QMLPConfig.clone()

        Config.update(config)
        self.config = Config

        self.q_network = FCNetwork(
            input_size=self.config.input_size,
            output_sizes=[self.num_actions],
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            initializer=self.config.initializer,
            is_policy=False,
        )

    def forward(
        self, obs: Observation, state: tuple = ()
    ) -> tuple[torch.Tensor, tuple]:
        return self.q_network(obs.vector), ()
