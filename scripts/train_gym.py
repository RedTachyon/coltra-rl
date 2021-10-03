from typing import Optional

import gym
import torch
import yaml
from typarse import BaseParser

from coltra.agents import CAgent, DAgent
from coltra.models.mlp_models import FancyMLPModel
from coltra.trainers import PPOCrowdTrainer
from coltra.envs import MultiGymEnv

import wandb

import pybullet_envs


class Parser(BaseParser):
    config: str = "configs/base_config.yaml"
    iters: int = 500
    env_name: str
    name: str
    workers: int = 8
    start_dir: Optional[str]
    start_idx: Optional[int] = -1

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env_name": "Environment gym name",
        "name": "Name of the tb directory to store the logs",
        "workers": "Number of parallel collection envs to use",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env_name": "e",
        "name": "n",
        "workers": "w",
        "start_dir": "sd",
        "start_idx": "si",
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

    workers = trainer_config.get("workers") or 8  # default value

    # Initialize the environment
    env = MultiGymEnv.get_venv(workers=workers, env_name=args.env_name)
    action_space = env.action_space

    print(f"{env.observation_space=}")
    print(f"{action_space=}")

    is_discrete_action = isinstance(action_space, gym.spaces.Discrete)
    if is_discrete_action:
        action_shape = action_space.n
    else:
        action_shape = action_space.shape[0]

    # Initialize the agent
    sample_obs = next(iter(env.reset().values()))
    obs_size = sample_obs.vector.shape[0]
    ray_size = sample_obs.rays.shape[0] if sample_obs.rays is not None else None

    model_config["input_size"] = obs_size
    model_config["rays_input_size"] = ray_size
    model_config["discrete"] = is_discrete_action
    model_config["num_actions"] = action_shape

    model_cls = FancyMLPModel
    agent_cls = CAgent if isinstance(action_space, gym.spaces.Box) else DAgent

    if args.start_dir:
        agent = agent_cls.load_agent(args.start_dir, weight_idx=args.start_idx)
    else:
        model = model_cls(model_config)
        agent = agent_cls(model)

    if CUDA:
        agent.cuda()

    trainer = PPOCrowdTrainer(agent, env, trainer_config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)
