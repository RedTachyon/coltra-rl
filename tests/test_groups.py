import numpy as np
from gym.spaces import Box

from coltra.envs.spaces import ObservationSpace
from coltra.groups import MacroAgent, HomogeneousGroup
from coltra.models import MLPModel
from coltra.agents import Agent, CAgent, DAgent
import os
import shutil


def test_policy_mapping():
    group = HomogeneousGroup(
        CAgent(
            MLPModel(
                {},
                observation_space=ObservationSpace(vector=Box(-np.inf, np.inf, (5,))),
                action_space=Box(
                    low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
                ),
            )
        )
    )

    assert group.get_policy_name("anything") == "crowd" == group.policy_name
    assert group.get_policy_name("") == "crowd" == group.policy_name
    assert group.get_policy_name("pursuer_1?env=5&id=0") == "crowd" == group.policy_name


def test_save():
    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.mkdir("temp")
    group = HomogeneousGroup(
        CAgent(
            MLPModel(
                {},
                observation_space=ObservationSpace(
                    vector=Box(-np.inf, np.inf, shape=(5,))
                ),
                action_space=Box(
                    low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32)
                ),
            )
        )
    )

    group.save("temp")
    group.save_state("temp", 0)

    loaded = group.load(base_path="temp", weight_idx=0)
    loaded.load_state(base_path="temp", idx=0)
