import json
from typing import Optional

import torch
import wandb
import yaml
from typarse import BaseParser

import coltra
from coltra.agents import CAgent, DAgent, Agent
from coltra.envs.energy_env import EnergyEnv
from coltra.envs.spaces import ActionSpace
from coltra.groups import HomogeneousGroup
from coltra.models.mlp_models import MLPModel
from coltra.trainers import PPOCrowdTrainer


class Parser(BaseParser):
    config: str = "configs/base_config.yaml"
    iters: int = 100
    name: str
    project: str = "coltra"
    extra_config: Optional[str] = None

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "name": "Name of the tb directory to store the logs",
        "project": "Name of the wandb project to use",
        "extra_config": "Extra config items to override the config file. Should be passed in a json format.",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "name": "n",
        "project": "p",
        "extra_config": "ec",
    }


if __name__ == "__main__":
    CUDA = torch.cuda.is_available()

    args = Parser()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), yaml.Loader)

    if args.extra_config is not None:
        extra_config = json.loads(args.extra_config)
        extra_config = coltra.utils.undot_dict(extra_config)
        coltra.utils.update_dict(target=config, source=extra_config)

        from pprint import pprint

        print("Extra config:")
        pprint(extra_config)

    trainer_config = config["trainer"]
    model_config = config["model"]

    trainer_config["tensorboard_name"] = args.name
    trainer_config["PPOConfig"]["use_gpu"] = CUDA

    wandb.init(
        project=args.project,
        entity="redtachyon",
        sync_tensorboard=True,
        config=config,
        name=args.name,
    )

    workers = trainer_config["workers"]

    # Initialize the environment
    env = EnergyEnv(num_agents=workers)

    action_space: ActionSpace = env.action_space
    observation_space = env.observation_space

    print(f"{observation_space=}")
    print(f"{action_space=}")

    model_cls = MLPModel
    agent_cls = CAgent if "continuous" in action_space.spaces else DAgent

    agent: Agent
    model = model_cls(model_config, observation_space, action_space)
    agent = agent_cls(model)

    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    trainer = PPOCrowdTrainer(agents, env, trainer_config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)
