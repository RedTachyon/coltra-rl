from coltra.configs import MLPConfig
from coltra.models import MLPModel


def test_configs():

    model = MLPModel(
        {
            "input_size": 2,
            "num_actions": 1,
            "discrete": False,
            "hidden_sizes": [32, 32, 32],
        }
    )

    assert len(model.policy_network.hidden_layers) == 3

    model = MLPModel({"input_size": 5, "num_actions": 2, "discrete": True})

    assert len(model.policy_network.hidden_layers) == 2  # Default value
