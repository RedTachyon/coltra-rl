import copy
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Categorical, Normal

from coltra.buffers import Observation
from coltra.models import MLPModel, BaseModel
from coltra.utils import get_activation


class JointModel(BaseModel):
    def __init__(
        self,
        models: list[BaseModel],
        num_actions: int,
        discrete: bool,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        assert len(models) > 0, "JointModel needs at least 1 model"
        assert (
            len(set([model.input_size for model in models])) == 1
        ), "Constituent models must have the same input size"

        self.device = models[0].device

        self.models = nn.ModuleList(models)
        self.input_size = self.models[0].input_size
        self.num_actions = num_actions
        self.discrete = discrete
        self.latent_size = sum([model.latent_size for model in models])
        self.activation = get_activation(activation)

        if self.discrete:
            self.logstd = None
        else:
            self.logstd = nn.Parameter(torch.ones(1, self.num_actions))

        self.policy_head = nn.Linear(self.latent_size, num_actions).to(self.device)
        self.value_head = nn.Linear(self.latent_size, 1).to(self.device)

    def forward(
        self, x: Observation, state: tuple = (), get_value: bool = True
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

    def latent(self, x: Observation, state: Tuple = ()) -> Tensor:
        x = [model.latent(x, state) for model in self.models]
        x = torch.cat(x, dim=-1)
        return x

    def value(self, x: Observation, state: Tuple = ()) -> Tensor:
        x = [model.latent_value(x, state) for model in self.models]
        x = torch.cat(x, dim=-1)
        x = self.activation(x)
        return self.value_head(x)

    def latent_value(self, x: Observation, state: Tuple = ()) -> Tensor:
        x = [model.latent_value(x, state) for model in self.models]
        x = torch.cat(x, dim=-1)
        return x

    def freeze_models(self, freeze_list: list[bool]):
        assert len(freeze_list) == len(self.models)
        for model, freeze in zip(self.models, freeze_list):
            model.requires_grad_(not freeze)

    @classmethod
    def clone_model(
        cls, model: BaseModel, num_clones: int = 1, activation: str = "leaky_relu"
    ) -> "JointModel":
        device = model.device
        model_list = [model] + [copy.deepcopy(model).to(device) for _ in range(num_clones)]
        return cls(
            models=model_list,
            num_actions=model.num_actions,
            discrete=model.discrete,
            activation=activation,
        )
