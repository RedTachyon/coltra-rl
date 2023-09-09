from __future__ import annotations

from typing import Dict, Any, List, Tuple, Union, Sequence

import numpy as np
import torch
from gymnasium import Space
from gymnasium.spaces import Box
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal
from typarse import BaseConfig

from coltra.buffers import Observation
from coltra.configs import RelationConfig, AttentionConfig
from coltra.envs.spaces import ObservationSpace
from coltra.models.base_models import FCNetwork, BaseModel
from coltra.utils import AffineBeta


class AttentionNetwork(nn.Module):
    """Basic attention network, with Q=K=V ?"""

    def __init__(
        self,
        vec_input_size: int = 4,
        rel_input_size: int = 4,
        vec_hidden_layers: Sequence[int] = (32,),
        rel_hidden_layers: Sequence[int] = (32,),
        com_hidden_layers: Sequence[int] = (32,),
        output_sizes: Sequence[int] = (2, 2),
        emb_size: int = 32,
        attention_heads: int = 4,
        is_policy: Union[bool, Sequence[bool]] = (True, False),
        activation: str = "tanh",
        initializer: str = "kaiming_uniform",
    ):
        super().__init__()

        self.vec_mlp = FCNetwork(
            input_size=vec_input_size,
            output_sizes=(emb_size,),
            hidden_sizes=vec_hidden_layers,
            activation=activation,
            initializer=initializer,
            is_policy=False,
        )

        self.rel_mlp = FCNetwork(
            input_size=rel_input_size,
            output_sizes=(emb_size,),
            hidden_sizes=rel_hidden_layers,
            activation=activation,
            initializer=initializer,
            is_policy=False,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=emb_size, num_heads=attention_heads, batch_first=True
        )

        self.com_mlp = FCNetwork(
            input_size=emb_size + emb_size,
            output_sizes=output_sizes,
            hidden_sizes=com_hidden_layers,
            activation=activation,
            initializer=initializer,
            is_policy=is_policy,
        )

    def forward(self, x: Observation) -> Tuple[List[Tensor], Tensor]:
        x_vector = x.vector
        assert len(x_vector.shape) == 2  # [B, N]
        [x_vector] = self.vec_mlp(x_vector)

        x_buffer = x.buffer
        assert len(x_buffer.shape) == 3  # [B, A, N]

        [x_buffer] = self.rel_mlp(x_buffer)  # [B, A, N]; K = V in attention

        Q = x_vector.unsqueeze(1)  # [B, 1, N]
        K = V = x_buffer  # [B, A, N]

        if hasattr(self, "buffer_mask"):
            buffer_mask = self.buffer_mask
        else:
            buffer_mask = None

        x_attn, attention = self.attention(
            Q, K, V, average_attn_weights=False, key_padding_mask=buffer_mask
        )  # [B, 1, N], [B, H, 1, A]

        attention = attention.squeeze(2)  # [B, H, A]
        # num_agents = V.shape[-2]

        # attention = torch.einsum("ban,bna->baa", Q, K) / torch.sqrt(num_agents)  # [B, A, A]
        # attention = F.softmax(attention, dim=-1)  # [B, A]

        # x_com = torch.cat([x_vector, attention @ V], dim=-1)

        x_attn = x_attn.squeeze(1)  # [B, N]
        x_com = torch.cat([x_vector, x_attn], dim=-1)  # [B, N]
        # x_com = torch.squeeze(x_com, dim=1)  # [B, N]

        x_com = self.com_mlp(x_com)

        return x_com, attention

    def latent(self, x: Observation) -> Tensor:
        x_vector = x.vector
        assert len(x_vector.shape) == 2  # [B, N]
        [x_vector] = self.vec_mlp(x_vector)

        x_buffer = x.buffer
        assert len(x_buffer.shape) == 3  # [B, A, N]

        [x_buffer] = self.rel_mlp(x_buffer)  # [B, A, N]; Q = K = V in attention

        Q = x_vector.unsqueeze(1)  # [B, 1, N]
        K = V = x_buffer

        x_attn, attention = self.attention(
            Q, K, V, average_attn_weights=False
        )  # [B, 1, N]

        # Manual computation of attention - might not be necessary if torch attention behaves well
        # num_agents = V.shape[-2]
        # attention = torch.einsum("ban,bna->baa", Q, K) / torch.sqrt(num_agents)  # [B, A, A]
        # attention = F.softmax(attention, dim=-1)  # [B, A]

        # x_com = torch.cat([x_vector, attention @ V], dim=-1)

        x_attn = x_attn.squeeze(1)  # [B, N]
        x_com = torch.cat([x_vector, x_attn], dim=-1)  # [B, N]
        # x_com = torch.squeeze(x_com, dim=1)  # [B, N]

        x_com = self.com_mlp.latent(x_com)

        return x_com


class AttentionModel(BaseModel):
    def __init__(
        self, config: dict, observation_space: ObservationSpace, action_space: Space
    ):
        super().__init__(
            config, observation_space=observation_space, action_space=action_space
        )

        Config: AttentionConfig = AttentionConfig.clone()

        Config.update(config)
        self.config = Config

        assert not self.discrete
        # self.discrete = False  # TODO: add support for discrete heads
        # self.std_head = self.config.std_head
        self.sigma0 = self.config.sigma0
        self.input_size = observation_space.vector.shape[0]
        self.rel_input_size = observation_space.buffer.shape[-1]

        self.latent_size = self.config.com_hidden_layers[-1]
        self.beta = self.config.beta

        if self.beta:
            heads = (self.num_actions, self.num_actions)
            is_policy = (True, True)
        else:
            heads = (self.num_actions,)
            is_policy = (True,)

        self.policy_network = AttentionNetwork(
            vec_input_size=self.input_size,
            rel_input_size=self.rel_input_size,
            vec_hidden_layers=self.config.vec_hidden_layers,
            rel_hidden_layers=self.config.rel_hidden_layers,
            com_hidden_layers=self.config.com_hidden_layers,
            output_sizes=heads,
            emb_size=self.config.emb_size,
            attention_heads=self.config.attention_heads,
            is_policy=is_policy,
            activation=self.config.activation,
            initializer=self.config.initializer,
        )

        self.value_network = AttentionNetwork(
            vec_input_size=self.input_size,
            rel_input_size=self.rel_input_size,
            vec_hidden_layers=self.config.vec_hidden_layers,
            rel_hidden_layers=self.config.rel_hidden_layers,
            com_hidden_layers=self.config.com_hidden_layers,
            output_sizes=(1,),
            emb_size=self.config.emb_size,
            attention_heads=self.config.attention_heads,
            is_policy=False,
            activation=self.config.activation,
            initializer=self.config.initializer,
        )

        if self.config.beta:
            self.logstd = None
        else:
            self.logstd = nn.Parameter(
                torch.tensor(self.config.sigma0) * torch.ones(1, self.num_actions)
            )

        self.config = self.config.to_dict()

    def forward(
        self, x: Observation, state: Tuple = (), get_value: bool = False
    ) -> Tuple[Distribution, Tuple, dict[str, Tensor]]:
        action_distribution: Distribution
        if self.beta:
            [action_a, action_b], attention = self.policy_network(x)
            action_a, action_b = action_a.exp() + 1, action_b.exp() + 1
            action_distribution = AffineBeta(
                action_a, action_b, self.action_low, self.action_high
            )
        else:
            [action_mu], attention = self.policy_network(x)
            action_std = torch.exp(self.logstd)
            action_distribution = Normal(loc=action_mu, scale=action_std)

        extra_outputs = {"attention": attention}

        if get_value:
            [value], v_attention = self.value_network(x)
            extra_outputs["value"] = value
            extra_outputs["value_attention"] = v_attention

        return action_distribution, (), extra_outputs

    def value(self, x: Observation, state: Tuple = ()) -> tuple[Tensor, tuple]:
        [value], attn = self.value_network(x)
        return value, attn

    def latent(self, x: Observation, state: Tuple) -> Tensor:
        latent = self.policy_network.latent(x)
        return latent

    def latent_value(self, x: Observation, state: Tuple) -> Tensor:
        latent = self.value_network.latent(x)
        return latent


#
# class FlattenRelationModel(RelationModel):
#     """
#     An abstraction to create models that flatten select inputs into a single vector.
#     Basically just the equivalent of `FlattenMLPModel`, I should combine them into a parametrized version.
#     Need to implement `_flatten` for any new models, and the result will be a fully connected
#     model that only has a `vector` entry, flattened according to the custom model.
#     """
#
#     def _flatten(self, obs: Observation) -> Observation:
#         raise NotImplementedError
#
#     def forward(
#         self, x: Observation, state: Tuple = (), get_value: bool = False
#     ) -> Tuple[Distribution, Tuple[Tensor, Tensor], dict[str, Tensor]]:
#         return super().forward(self._flatten(x), state, get_value)
#
#     def latent(self, x: Observation, state: Tuple = ()) -> Tensor:
#         return super().latent(self._flatten(x), state)
#
#     def value(self, x: Observation, state: Tuple = ()) -> Tensor:
#         return super().value(self._flatten(x), state)
#
#     def latent_value(self, x: Observation, state: Tuple) -> Tensor:
#         return super().latent_value(self._flatten(x), state)
#
#
# class RayRelationModel(FlattenRelationModel):
#     def __init__(
#         self, config: dict, observation_space: ObservationSpace, action_space: Space
#     ):
#         assert (
#             "rays" in observation_space.spaces
#         ), "RayRelationModel requires an observation space with rays"
#
#         vector_size = (
#             observation_space.vector.shape[0]
#             if "vector" in observation_space.spaces
#             else 0
#         )
#         image_size = np.prod(observation_space.spaces["rays"].shape)
#         new_vector_size = vector_size + image_size
#
#         other_spaces = {
#             k: v
#             for k, v in observation_space.spaces.items()
#             if k not in ("rays", "vector")
#         }
#
#         new_observation_space = ObservationSpace(
#             {"vector": Box(-np.inf, np.inf, (new_vector_size,)), **other_spaces}
#         )
#
#         super().__init__(config, new_observation_space, action_space)
#
#     def _flatten(self, obs: Observation) -> Observation:
#         if not hasattr(obs, "rays"):
#             return obs
#
#         rays = obs.rays
#         if rays.shape == 1:  # no batch
#             dim = 0
#         else:  # rays.shape == 2, batch
#             dim = 1
#
#         if hasattr(obs, "vector"):
#             vector = torch.cat([obs.vector, rays], dim=dim)
#         else:
#             vector = rays
#
#         return Observation(vector=vector, buffer=obs.buffer)
