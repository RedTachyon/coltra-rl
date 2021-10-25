from typing import Optional, Type

import gym
import torch
import yaml
from typarse import BaseParser

from coltra.agents import CAgent, DAgent, Agent
from coltra.groups import HomogeneousGroup
from coltra.models.mlp_models import MLPModel
from coltra.trainers import PPOCrowdTrainer
from coltra.envs import MultiGymEnv

import wandb

import pybullet_envs

from coltra.wrappers import ObsVecNormWrapper, LastRewardWrapper
from coltra.wrappers.agent_wrappers import RetNormWrapper


class Parser(BaseParser):
    config: str = "configs/base_config.yaml"
    iters: int = 500
    env_name: str
    name: str
    start_dir: Optional[str]
    start_idx: Optional[int] = -1
    normalize: bool = False
    reward_wrapper: bool = False

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env_name": "Environment gym name",
        "name": "Name of the tb directory to store the logs",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
        "normalize": "Whether to use the obs and return normalizing wrappers",
        "reward_wrapper": "Whether env should use the reward wrapper",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env_name": "e",
        "name": "n",
        "start_dir": "sd",
        "start_idx": "si",
        "normalize": "norm",
        "reward_wrapper": "rw",
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
        project="coltra", entity="redtachyon", sync_tensorboard=True, config=config
    )

    workers = trainer_config["workers"]

    wrappers = []
    if args.reward_wrapper:
        wrappers.append(LastRewardWrapper)
    # Initialize the environment
    env = MultiGymEnv.get_venv(
        workers=workers, env_name=args.env_name, wrappers=wrappers
    )
    action_space = env.action_space
    observation_space = env.observation_space

    print(f"{observation_space=}")
    print(f"{action_space=}")

    is_discrete_action = isinstance(action_space, gym.spaces.Discrete)
    if is_discrete_action:
        action_shape = action_space.n
    else:
        action_shape = action_space.shape[0]

    model_config["input_size"] = observation_space.shape[0]
    model_config["num_actions"] = action_shape
    model_config["discrete"] = is_discrete_action

    model_cls = MLPModel
    agent_cls = CAgent if isinstance(action_space, gym.spaces.Box) else DAgent

    agent: Agent
    if args.start_dir:
        agent = agent_cls.load(args.start_dir, weight_idx=args.start_idx)
    else:
        model = model_cls(model_config)
        agent = agent_cls(model)

    if args.normalize:
        agent = ObsVecNormWrapper(agent)
        agent = RetNormWrapper(agent)

    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    trainer = PPOCrowdTrainer(agents, env, trainer_config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)
