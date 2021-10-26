from typing import Dict, Tuple

from torch import Tensor
from torch.distributions import Distribution

from coltra.buffers import Observation
from coltra.models import MLPModel, BaseModel


class JointModel(BaseModel):
    def __init__(self, models: list[MLPModel]):
        super().__init__()
        self.models = models

    def forward(
        self, x: Observation, state: tuple, get_value: bool
    ) -> tuple[Distribution, Tuple, dict[str, Tensor]]:
        pass

    def value(self, x: Observation, state: Tuple) -> Tensor:
        pass

    def freeze_models(self, freeze_list: list[bool]):
        assert len(freeze_list) == len(self.models)
        for model, freeze in zip(self.models, freeze_list):
            model.requires_grad_(not freeze)
