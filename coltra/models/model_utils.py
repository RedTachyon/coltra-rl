import torch
from torch.distributions import Categorical, Normal
from torch.distributions.distribution import Distribution

from coltra.buffers import Action


class ContCategorical(Distribution):
    def __init__(self, categorical: Categorical, normal: Normal):
        self.categorical = categorical
        self.normal = normal
        super().__init__(normal.batch_shape, validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        category = self.categorical.sample(sample_shape)[:, None]
        parameter = self.normal.sample(sample_shape)
        return torch.cat([category, parameter], dim=-1)

    def log_prob(self, value: Action):
        category = value.discrete.long()
        parameter = value.continuous[:, None]
        return self.categorical.log_prob(category) + self.normal.log_prob(parameter).squeeze(-1)

    def entropy(self):
        return self.categorical.entropy() + self.normal.entropy().squeeze(1)

    def deterministic_sample(self):
        category = self.categorical.probs.argmax(dim=-1)
        parameter = self.normal.loc
        return torch.stack([category, parameter], dim=-1)
