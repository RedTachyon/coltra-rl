from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Categorical, Normal

from coltra.buffers import Observation
from coltra.models import MLPModel, BaseModel
from coltra.utils import get_activation


class JointModel(BaseModel):
    def __init__(
        self,
        models: list[MLPModel],
        num_actions: int,
        discrete: bool,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        assert len(models) > 0, "JointModel needs at least 1 model"
        self.models = models
        self.num_actions = num_actions
        self.discrete = discrete
        self.activation = get_activation(activation)

        if self.discrete:
            self.logstd = None
        else:
            self.logstd = nn.Parameter(
                torch.tensor(self.config.sigma0)
                * torch.ones(1, self.config.num_actions)
            )

        total_latent_size = sum([model.latent_size for model in models])
        self.policy_head = nn.Linear(total_latent_size, num_actions)
        self.value_head = nn.Linear(total_latent_size, 1)

    def forward(
        self, x: Observation, state: tuple, get_value: bool = True
    ) -> tuple[Distribution, Tuple, dict[str, Tensor]]:

        latent = [model.latent(x, state) for model in self.models]
        latent = torch.cat(latent, dim=-1)

        latent = self.activation(latent)
        latent = self.policy_head(latent)

        action_distribution: Distribution
        if self.discrete:
            action_distribution = Categorical(logits=latent)
        else:
            action_distribution = Normal(loc=latent, scale=torch.exp(self.logstd))

        extra_outputs = {}
        if get_value:
            extra_outputs["value"] = self.value(x, state)

        return action_distribution, state, extra_outputs

    def latent(self, x: Observation, state: Tuple) -> Tensor:
        x = [model.latent(x, state) for model in self.models]
        x = torch.cat(x, dim=-1)
        return x

    def value(self, x: Observation, state: Tuple) -> Tensor:
        x = [model.latent_value(x, state) for model in self.models]
        x = torch.cat(x, dim=-1)
        x = self.activation(x)
        return self.value_head(x)

    def latent_value(self, x: Observation, state: Tuple) -> Tensor:
        x = [model.latent_value(x, state) for model in self.models]
        x = torch.cat(x, dim=-1)
        return x

    def freeze_models(self, freeze_list: list[bool]):
        assert len(freeze_list) == len(self.models)
        for model, freeze in zip(self.models, freeze_list):
            model.requires_grad_(not freeze)
