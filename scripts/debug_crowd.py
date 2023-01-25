from tqdm import trange

import coltra
from coltra import (
    UnitySimpleCrowdEnv,
    Action,
    Observation,
    collect_crowd_data,
    HomogeneousGroup,
    CAgent,
    PPOCrowdTrainer,
)
from coltra.policy_optimization import CrowdPPOptimizer
from coltra.models import AttentionModel
from coltra.agents import ConstantAgent
from coltra.collectors import collect_renders
from coltra.training_utils import evaluate

import numpy as np

# coltra.disable_unity_logs()

import torch

import yaml


path = "../scripts/v7-configs/crowd_config.yaml"

with open(path, "r") as f:
    config = yaml.load(f, yaml.Loader)

unity_path = "../../CrowdAI/Builds/Mac/crowd-v7rc3.app"
# unity_path = None
env = UnitySimpleCrowdEnv(
    file_name=unity_path, no_graphics=False, extra_params=config["environment"]
)

# env.set_timescale()
model = AttentionModel(
    config={}, observation_space=env.observation_space, action_space=env.action_space
)

# torch.nn.init.zeros_(model.policy_network.com_mlp.heads[0].weight)
# torch.nn.init.zeros_(model.policy_network.com_mlp.heads[0].bias)

agent = CAgent(model)

group = HomogeneousGroup(agent)

optimizer = CrowdPPOptimizer(group, config["trainer"]["PPOConfig"])

data, metrics, shape = collect_crowd_data(
    group, env, 1000, disable_tqdm=False, deterministic=False
)

optimizer.train_on_data(data, shape)

env.close()

print(shape)
print(shape)

# TODO: debug this in a notebook

# TODO plan: check if the semantic I'm using for crowds makes sense.
# If so, rewrite gym envs to use it, and apply it consistenly to the rest of the codebase.
# General idea is that an episode starts at step 0, the sequence of everything is
# Can I "desync" it between the collection and the saved data?
