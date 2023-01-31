from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import yaml
from typarse import BaseParser

from coltra.agents import CAgent, DAgent
from coltra.envs.pettingzoo_envs import PettingZooEnv
from coltra.groups import HomogeneousGroup
from coltra.models.mlp_models import MLPModel, ImageMLPModel
from coltra.trainers import PPOCrowdTrainer
from coltra.envs import MultiGymEnv
from coltra.wrappers import ObsVecNormWrapper
from coltra.wrappers.agent_wrappers import RetNormWrapper

import wandb

from pettingzoo.sisl import pursuit_v3


class Parser(BaseParser):
    config: str = "configs/pursuit_config.yaml"
    iters: int = 500
    name: Optional[str] = None
    start_dir: Optional[str]
    start_idx: Optional[int] = -1
    normalize: bool = False

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "name": "Name of the tb directory to store the logs",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
        "normalize": "Whether to use the obs and return normalizing wrappers",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "name": "n",
        "start_dir": "sd",
        "start_idx": "si",
        "normalize": "norm",
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

    wandb.init(
        project="coltra", sync_tensorboard=True, config=config
    )

    workers = trainer_config["workers"]

    # Initialize the environment
    env = PettingZooEnv.get_venv(8, env_creator=pursuit_v3.parallel_env)
    action_space = env.action_space
    observation_space = env.observation_space

    print(f"{observation_space=}")
    print(f"{action_space=}")

    is_discrete_action = isinstance(action_space, gym.spaces.Discrete)
    if is_discrete_action:
        action_shape = action_space.n
    else:
        action_shape = action_space.shape[0]

    model_cls = ImageMLPModel
    agent_cls = CAgent if isinstance(action_space, gym.spaces.Box) else DAgent

    if args.start_dir:
        agent = agent_cls.load(args.start_dir, weight_idx=args.start_idx)
    else:
        model = model_cls(model_config, env.observation_space, action_space)
        agent = agent_cls(model)

    if args.normalize:
        agent = ObsVecNormWrapper(agent)
        agent = RetNormWrapper(agent)

    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    trainer = PPOCrowdTrainer(agents, env, trainer_config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)

    env.close()
