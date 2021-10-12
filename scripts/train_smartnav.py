from typing import Optional

import numpy as np
import torch
import yaml
from typarse import BaseParser
import gym

from coltra.agents import CAgent, DAgent
from coltra.envs.smartnav_envs import SmartNavEnv
from coltra.models.mlp_models import MLPModel
from coltra.models.relational_models import RelationModel
from coltra.trainers import PPOCrowdTrainer
from coltra.models.raycast_models import LeeModel


class Parser(BaseParser):
    config: str = "configs/base_config.yaml"
    iters: int = 500
    env: str
    name: str
    start_dir: Optional[str]
    start_idx: Optional[int] = -1

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env": "Path to the Unity environment binary",
        "name": "Name of the tb directory to store the logs",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env": "e",
        "name": "n",
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

    workers = trainer_config.get("workers")

    # Initialize the environment
    # env = SmartNavEnv.get_venv(workers, file_name=args.env)

    METRICS = [
        "success_rate",
        "num_steps_not_progressing",
        "current_map",
        "goal_distance",
    ]

    env = SmartNavEnv(file_name=args.env, metrics=METRICS)
    action_space = env.action_space
    observation_space = env.observation_space

    print(f"{observation_space=}")
    print(f"{action_space=}")

    is_discrete_action = isinstance(action_space, gym.spaces.Discrete)
    if is_discrete_action:
        action_shape = action_space.n
    else:
        action_shape = action_space.shape[0]

    model_config["input_size"] = np.product(observation_space.shape) - len(METRICS)
    model_config["num_actions"] = action_shape
    model_config["discrete"] = is_discrete_action

    model_cls = MLPModel
    agent_cls = CAgent if isinstance(action_space, gym.spaces.Box) else DAgent

    if args.start_dir:
        agent = agent_cls.load(args.start_dir, weight_idx=args.start_idx)
    else:
        model = model_cls(model_config)
        agent = agent_cls(model)

    if CUDA:
        agent.cuda()

    # env = SubprocVecEnv([
    #     get_env_creator(file_name=args.env, no_graphics=True, worker_id=i, seed=i)
    #     for i in range(workers)
    # ])

    trainer = PPOCrowdTrainer(agent, env, trainer_config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)
