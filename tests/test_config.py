import numpy as np
from gym.spaces import Box, Discrete

from coltra.configs import MLPConfig
from coltra.envs.spaces import ObservationSpace
from coltra.models import MLPModel


def test_configs():

    model = MLPModel(
        {
            "discrete": False,
            "hidden_sizes": [32, 32, 32],
        },
        observation_space=ObservationSpace(vector=Box(-np.inf, np.inf, (2,))),
        action_space=Box(
            low=-np.ones(1, dtype=np.float32), high=np.ones(1, dtype=np.float32)
        ),
    )

    assert len(model.policy_network.hidden_layers) == 3

    model = MLPModel(
        {"discrete": True},
        observation_space=ObservationSpace(vector=Box(-np.inf, np.inf, (5,))),
        action_space=Discrete(2),
    )

    assert len(model.policy_network.hidden_layers) == 2  # Default value
