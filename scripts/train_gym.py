import json
from typing import Optional, Type

import gymnasium as gym
import numpy as np
import torch
import yaml
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
    config: str = "configs/base_config.yaml"
    iters: int = 500
    env_name: str
    name: str
    start_dir: Optional[str]
    start_idx: Optional[int] = -1
    project: str = "coltra"
    seed: Optional[int] = None
    normalize: bool = False
    reward_wrapper: bool = False
    time_feature_wrapper: bool = False
    normalize_env: bool = False
    extra_config: Optional[str] = None

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env_name": "Environment gym name",
        "name": "Name of the tb directory to store the logs",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
        "project": "Name of the wandb project to use",
        "seed": "Seed for the random number generator",
        "normalize": "Whether to use the obs and return normalizing wrappers",
        "reward_wrapper": "Whether env should use the reward wrapper",
        "time_feature_wrapper": "Whether env should use the time feature wrapper",
        "normalize_env": "Whether to normalize the env obs and returns",
        "extra_config": "Extra config items to override the config file. Should be passed in a json format.",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env_name": "e",
        "name": "n",
        "start_dir": "sd",
        "start_idx": "si",
        "project": "p",
        "seed": "s",
        "normalize": "norm",
        "reward_wrapper": "rw",
        "time_feature_wrapper": "tf",
        "normalize_env": "norme",
        "extra_config": "ec",
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
        sync_tensorboard=True,
        config=config,
        name=args.name,
    )

    workers = trainer_config["workers"]

    wrappers = []
    if args.reward_wrapper:
        wrappers.append(LastRewardWrapper)
    if args.time_feature_wrapper:
        wrappers.append(TimeFeatureWrapper)

    if args.normalize_env:
        wrappers.append(
            lambda e: gym.wrappers.TransformObservation(e, lambda obs: obs / 10.0)
        )
        wrappers.append(
            lambda e: gym.wrappers.TransformReward(e, lambda reward: reward / 10.0)
        )
        wrappers.append(gym.wrappers.ClipAction)
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

    agent: Agent
    if args.start_dir:
        agent = agent_cls.load(args.start_dir, weight_idx=args.start_idx)
    else:
        model = model_cls(model_config, observation_space, action_space)
        agent = agent_cls(model)

    if args.normalize:
        agent = ObsVecNormWrapper(agent)
        agent = RetNormWrapper(agent)

    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    trainer = PPOCrowdTrainer(agents, env, trainer_config, seed=args.seed)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)
