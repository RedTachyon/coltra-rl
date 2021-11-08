from typing import Optional

import gym
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
from coltra.models.raycast_models import LeeModel
from coltra.envs import probe_env_classes


class Parser(BaseParser):
    config: str = "configs/probe_config.yaml"
    probe: int = 0
    iters: int = 500
    name: str
    workers: int = 8
    start_dir: Optional[str]
    start_idx: Optional[int] = -1

    _help = {
        "config": "Config file for the coltra",
        "probe": "The index of the probe environment",
        "iters": "Number of coltra iterations",
        "name": "Name of the tb directory to store the logs",
        "workers": "Number of parallel collection envs to use",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
    }

    _abbrev = {
        "config": "c",
        "probe": "p",
        "iters": "i",
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

    workers = trainer_config.get("workers") or 8  # default value

    # Initialize the environment
    env_cls = probe_env_classes[args.probe]
    env = env_cls.get_venv(workers)
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

    model_config["input_size"] = obs_size
    model_config["discrete"] = is_discrete_action
    model_config["num_actions"] = action_shape

    model_cls = MLPModel
    agent_cls = CAgent if isinstance(action_space, gym.spaces.Box) else DAgent

    agent: Agent
    if args.start_dir:
        agent = agent_cls.load(args.start_dir, weight_idx=args.start_idx)
    else:
        model = model_cls(model_config)
        agent = agent_cls(model)

    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    trainer = PPOCrowdTrainer(agents, env, trainer_config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)
