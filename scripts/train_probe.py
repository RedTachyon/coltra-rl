from typing import Optional

import gymnasium as gym
import torch
import yaml
from typarse import BaseParser

from coltra.agents import CAgent, DAgent, Agent
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.envs.probe_envs import ConstRewardEnv, ObsDependentRewardEnv
from coltra.groups import HomogeneousGroup
from coltra.models.mlp_models import MLPModel
from coltra.models.relational_models import RelationModel
from coltra.trainers import PPOCrowdTrainer
from coltra.envs import probe_env_classes

import wandb


class Parser(BaseParser):
    config: str = "configs/probe_config.yaml"
    probe: int = 0
    iters: int = 500
    name: str

    _help = {
        "config": "Config file for the coltra",
        "probe": "The index of the probe environment",
        "iters": "Number of coltra iterations",
        "name": "Name of the tb directory to store the logs",
    }

    _abbrev = {
        "config": "c",
        "probe": "p",
        "iters": "i",
        "name": "n",
    }


if __name__ == "__main__":
    CUDA = torch.cuda.is_available()

    args = Parser()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), yaml.Loader)

    trainer_config = config["trainer"]
    model_config = config["model"]

    trainer_config["tensorboard_name"] = args.name
    trainer_config["PPOConfig"]["use_gpu"] = CUDA

    workers = trainer_config.get("workers") or 8  # default value

    # Initialize the environment
    env_cls = probe_env_classes[args.probe]
    env = env_cls.get_venv(workers)
    action_space = env.action_space

    print(f"{env.observation_space=}")
    print(f"{action_space=}")

    model_cls = MLPModel
    agent_cls = CAgent if isinstance(env.action_space, gym.spaces.Box) else DAgent

    agent: Agent
    model = model_cls(model_config, env.observation_space, action_space)
    agent = agent_cls(model)

    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    trainer = PPOCrowdTrainer(agents, env, trainer_config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)
