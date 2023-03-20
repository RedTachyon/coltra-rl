import torch
from torch.distributions import Categorical, Normal
from torch.distributions.distribution import Distribution


class ContCategorical(Distribution):
    def __init__(self, categorical: Categorical, normal: Normal):
        self.categorical = categorical
        self.normal = normal
        super().__init__(normal.batch_shape, validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        category = self.categorical.sample(sample_shape)
        parameter = self.normal.sample(sample_shape)
        # import pdb; pdb.set_trace()
        return torch.stack([category, parameter], dim=-1)

    def log_prob(self, value):
        category = value[..., 0].long()
        parameter = value[..., 1]
        return self.categorical.log_prob(category) + self.normal.log_prob(parameter)

    def entropy(self):
        return self.categorical.entropy() + self.normal.entropy()

    def deterministic_sample(self):
        category = self.categorical.probs.argmax(dim=-1)
        parameter = self.normal.loc
        return torch.stack([category, parameter], dim=-1)
