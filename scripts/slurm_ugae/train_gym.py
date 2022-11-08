import json
from typing import Optional, Type

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.spaces import Box
from typarse import BaseParser

import coltra
from coltra.agents import CAgent, DAgent, Agent
from coltra.envs.spaces import ActionSpace
from coltra.groups import HomogeneousGroup
from coltra.models.mlp_models import MLPModel
from coltra.trainers import PPOCrowdTrainer
from coltra.envs import MultiGymEnv

import wandb


from coltra.wrappers import ObsVecNormWrapper, LastRewardWrapper
from coltra.wrappers.agent_wrappers import RetNormWrapper
from coltra.wrappers.env_wrappers import TimeFeatureWrapper


class Parser(BaseParser):
    config: str = "ugae_gym_config.yaml"
    iters: int = 1000
    env_name: str
    name: str = "ugae"
    project: str = "coltra"
    seed: Optional[int] = None
    # normalize: bool = False
    time_feature_wrapper: bool = False
    normalize_env: bool = False
    extra_config: Optional[str] = None
    tb_path: Optional[str] = None

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env_name": "Environment gym name",
        "name": "Name of the tb directory to store the logs",
        "project": "Name of the wandb project to use",
        "seed": "Seed for the random number generator",
        # "normalize": "Whether to use the obs and return normalizing wrappers",
        "time_feature_wrapper": "Whether env should use the time feature wrapper",
        "normalize_env": "Whether to normalize the env obs and returns",
        "extra_config": "Extra config items to override the config file. Should be passed in a json format.",
        "tb_path": "Root location for the tb_logs directory",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env_name": "e",
        "name": "n",
        "project": "p",
        "seed": "s",
        # "normalize": "norm",
        "time_feature_wrapper": "tf",
        "normalize_env": "norme",
        "extra_config": "ec",
        "tb_path": "tb",
    }


if __name__ == "__main__":
    CUDA = torch.cuda.is_available()

    args = Parser()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), yaml.Loader)

    if args.extra_config is not None:
        config_str = args.extra_config
        if config_str[0] == "'" and config_str[-1] == "'":
            config_str = config_str[1:-1]
        extra_config = json.loads(config_str)
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

    wrappers = []
    if args.time_feature_wrapper:
        wrappers.append(TimeFeatureWrapper)

    if args.normalize_env:
        wrappers.append(
            lambda e: gym.wrappers.TransformObservation(e, lambda obs: obs / 10.0)
        )
        wrappers.append(
            lambda e: gym.wrappers.TransformReward(e, lambda reward: reward / 10.0)
        )
        _env = gym.make(args.env_name)
        if isinstance(_env.action_space, Box):
            wrappers.append(gym.wrappers.ClipAction)
        _env.close()

    # Initialize the environment
    env = MultiGymEnv.get_venv(
        workers=workers, env_name=args.env_name, wrappers=wrappers, seed=args.seed
    )
    action_space: ActionSpace = env.action_space
    observation_space = env.observation_space

    print(f"{observation_space=}")
    print(f"{action_space=}")

    model_cls = MLPModel
    agent_cls = CAgent if "continuous" in action_space.spaces else DAgent

    model = model_cls(model_config, observation_space, action_space)
    agent = agent_cls(model)

    # if args.normalize:
    #     agent = ObsVecNormWrapper(agent)
    #     agent = RetNormWrapper(agent)

    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    trainer = PPOCrowdTrainer(agents, env, trainer_config, seed=args.seed, use_uuid=True, save_path=args.tb_path)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)
