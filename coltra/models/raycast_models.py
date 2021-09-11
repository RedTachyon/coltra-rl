from typing import Dict, List, Tuple

from torch.distributions import Distribution, Normal
from typarse import BaseConfig

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from coltra.buffers import Observation
from .base_models import FCNetwork, BaseModel


class LeeNetwork(nn.Module):
    def __init__(self,
                 input_size: int = 4,
                 output_sizes: List[int] = [2, 2],
                 rays_input_size: int = 126,
                 conv_filters: int = 2):
        super().__init__()

        self.stacked_rays = 3
        self.ray_values = 2

        self.activation = F.elu
        self.int_fc1 = nn.Linear(input_size, 32)
        self.int_fc2 = nn.Linear(32, 64)

        self.ext_conv = nn.Conv2d(in_channels=2,
                                  out_channels=conv_filters,
                                  kernel_size=(3, 3))
        conv_size = (rays_input_size // (self.stacked_rays * self.ray_values)) - 2

        self.ext_fc = nn.Linear(conv_size * conv_filters, 32)

        combined_size = 32 + 64

        self.combined_fc = nn.Linear(combined_size, 32)

        self.heads = nn.ModuleList([nn.Linear(32, size) for size in output_sizes])

    def forward(self, x: Observation):
        x_vector = x.vector
        x_vector = self.activation(self.int_fc1(x_vector))
        x_vector = self.activation(self.int_fc2(x_vector))

        batch_size = x.batch_size
        x_rays = x.rays.view((batch_size, self.stacked_rays, self.ray_values, -1))
        x_rays = torch.transpose(x_rays, 1, 2)
        x_rays = self.activation(self.ext_conv(x_rays))
        x_rays = x_rays.view((batch_size, -1))
        x_rays = self.activation(self.ext_fc(x_rays))

        x_combined = torch.cat([x_vector, x_rays], dim=1)
        x_combined = self.activation(self.combined_fc(x_combined))

        return [head(x_combined) for head in self.heads]


class LeeModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__()

        class Config(BaseConfig):
            input_size: int = 4
            rays_input_size: int = 126

            conv_filters: int = 2

        Config.update(config)
        self.config = Config

        self.policy_network = LeeNetwork(input_size=self.config.input_size,
                                         output_sizes=[2, 2],
                                         rays_input_size=self.config.rays_input_size,
                                         conv_filters=self.config.conv_filters)

        self.value_network = LeeNetwork(input_size=self.config.input_size,
                                        output_sizes=[1],
                                        rays_input_size=self.config.rays_input_size,
                                        conv_filters=self.config.conv_filters)

        self.config = self.config.to_dict()

    def forward(self, x: Observation,
                state: Tuple = (),
                get_value: bool = False) -> Tuple[Distribution, Tuple, Dict[str, Tensor]]:

        [action_mu, action_std] = self.policy_network(x)
        action_std = F.softplus(action_std - 0.5)

        action_distribution = Normal(loc=action_mu, scale=action_std)

        extra_outputs = {}
        if get_value:
            [value] = self.value_network(x)
            extra_outputs["value"] = value

        return action_distribution, state, extra_outputs

    def value(self, x: Observation, state: Tuple = ()):
        [value] = self.value_network(x)
        return value
