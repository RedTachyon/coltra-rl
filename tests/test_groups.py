from coltra.groups import MacroAgent, HomogeneousGroup
from coltra.models import MLPModel
from coltra.agents import Agent, CAgent, DAgent
import os
import shutil


def test_policy_mapping():
    group = HomogeneousGroup(
        CAgent(MLPModel({"input_size": 5, "num_actions": 2, "discrete": False}))
    )

    assert group.get_policy_name("anything") == "crowd" == group.policy_name
    assert group.get_policy_name("") == "crowd" == group.policy_name
    assert group.get_policy_name("pursuer_1?env=5&id=0") == "crowd" == group.policy_name


def test_save():
    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.mkdir("temp")
    group = HomogeneousGroup(
        CAgent(MLPModel({"input_size": 5, "num_actions": 2, "discrete": False}))
    )

    group.save("temp")
    group.save_state("temp", 0)

    loaded = group.load(base_path="temp", weight_idx=0)
    loaded.load_state(base_path="temp", idx=0)
