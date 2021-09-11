from typing import Dict, Any, List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal
from typarse import BaseConfig

from coltra.buffers import Observation
from .base_models import FCNetwork, BaseModel


class RelationNetwork(nn.Module):
    def __init__(self,
                 vec_input_size: int = 4,
                 rel_input_size: int = 4,
                 vec_hidden_layers: List[int] = [32, 32],
                 rel_hidden_layers: List[int] = [32, 32],
                 com_hidden_layers: List[int] = [32, 32],
                 output_sizes: List[int] = [2, 2],
                 is_policy: Union[bool, List[bool]] = [True, False],
                 activation: str = "tanh",
                 initializer: str = "kaiming_uniform"
                 ):
        super().__init__()

        *vec_hidden, vec_head = vec_hidden_layers
        self.vec_mlp = FCNetwork(
            input_size=vec_input_size,
            output_sizes=[vec_head],
            hidden_sizes=vec_hidden,
            activation=activation,
            initializer=initializer,
            is_policy=False
        )

        *rel_hidden, rel_head = rel_hidden_layers
        self.rel_mlp = FCNetwork(
            input_size=rel_input_size,
            output_sizes=[rel_head],
            hidden_sizes=rel_hidden,
            activation=activation,
            initializer=initializer,
            is_policy=False
        )

        self.com_mlp = FCNetwork(
            input_size=rel_head + vec_head,
            output_sizes=output_sizes,
            hidden_sizes=com_hidden_layers,
            activation=activation,
            initializer=initializer,
            is_policy=is_policy
        )

    def forward(self, x: Observation) -> List[Tensor]:
        x_vector = x.vector
        assert len(x_vector.shape) == 2  # [B, N]
        [x_vector] = self.vec_mlp(x_vector)

        x_buffer = x.buffer
        assert len(x_buffer.shape) == 3  # [B, A, N]

        [x_buffer] = self.rel_mlp(x_buffer)
        x_buffer = torch.mean(x_buffer, dim=-2)  # [B, N]

        x_com = torch.cat([x_vector, x_buffer], dim=-1)

        x_com = self.com_mlp(x_com)

        return x_com


class RelationModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__()

        class Config(BaseConfig):
            vec_input_size: int = 4
            rel_input_size: int = 4
            vec_hidden_layers: List[int] = [32, 32]
            rel_hidden_layers: List[int] = [32, 32]
            com_hidden_layers: List[int] = [32, 32]
            num_actions: int = 2
            activation: str = "tanh"
            initializer: str = "kaiming_uniform"

        Config.update(config)
        self.config = Config

        self.policy_network = RelationNetwork(
            vec_input_size=self.config.vec_input_size,
            rel_input_size=self.config.rel_input_size,
            vec_hidden_layers=self.config.vec_hidden_layers,
            rel_hidden_layers=self.config.rel_hidden_layers,
            com_hidden_layers=self.config.com_hidden_layers,
            output_sizes=[self.config.num_actions, self.config.num_actions],
            is_policy=[True, False],
            activation=self.config.activation,
            initializer=self.config.initializer
        )

        self.value_network = RelationNetwork(
            vec_input_size=self.config.vec_input_size,
            rel_input_size=self.config.rel_input_size,
            vec_hidden_layers=self.config.vec_hidden_layers,
            rel_hidden_layers=self.config.rel_hidden_layers,
            com_hidden_layers=self.config.com_hidden_layers,
            output_sizes=[1],
            is_policy=False,
            activation=self.config.activation,
            initializer=self.config.initializer
        )

        self.config = self.config.to_dict()

    def forward(self, x: Observation,
                state: Tuple = (),
                get_value: bool = True) -> Tuple[Distribution, Tuple, Dict[str, Tensor]]:

        [action_mu, action_std] = self.policy_network(x)
        action_std = F.softplus(action_std)  # TODO.txt: Figure out the low entropy

        action_distribution = Normal(loc=action_mu, scale=action_std)

        extra_outputs = {}

        if get_value:
            [value] = self.value_network(x)
            extra_outputs["value"] = value

        return action_distribution, (), extra_outputs

    def value(self, x: Observation,
              state: Tuple = ()) -> Tensor:
        [value] = self.value_network(x)
        return value
