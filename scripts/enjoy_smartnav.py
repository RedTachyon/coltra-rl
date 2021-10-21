from typing import Optional

import numpy as np
import torch
import yaml
from typarse import BaseParser
import gym

from coltra.agents import CAgent, DAgent, RandomGymAgent
from coltra.collectors import collect_crowd_data
from coltra.discounting import get_episode_rewards
from coltra.envs.smartnav_envs import SmartNavEnv
from coltra.models.mlp_models import MLPModel
from coltra.models.relational_models import RelationModel
from coltra.trainers import PPOCrowdTrainer
from coltra.models.raycast_models import LeeModel


class Parser(BaseParser):
    steps: int = 500
    env: Optional[str] = None
    start_dir: str
    start_idx: Optional[int] = -1
    deterministic: bool = False

    _help = {
        "steps": "Number of environment steps",
        "env": "Path to the Unity environment binary",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
        "deterministic": "Whether action selection should be greedy",
    }

    _abbrev = {
        "steps": "s",
        "env": "e",
        "start_dir": "sd",
        "start_idx": "si",
        "deterministic": "d",
    }


if __name__ == "__main__":
    CUDA = torch.cuda.is_available()

    args = Parser()

    # Initialize the environment
    # env = SmartNavEnv.get_venv(workers, file_name=args.env)

    METRICS = [
        "success_rate",
        "num_steps_not_progressing",
        "current_map",
        "goal_distance",
    ]

    env = SmartNavEnv(file_name=args.env, metrics=METRICS)
    env.engine_channel.set_configuration_parameters(time_scale=1)
    action_space = env.action_space
    observation_space = env.observation_space

    print(f"{observation_space=}")
    print(f"{action_space=}")

    is_discrete_action = isinstance(action_space, gym.spaces.Discrete)
    if is_discrete_action:
        action_shape = action_space.n
    else:
        action_shape = action_space.shape[0]

    model_cls = MLPModel
    agent_cls = CAgent if isinstance(action_space, gym.spaces.Box) else DAgent

    if args.env == "RANDOM":
        agent = RandomGymAgent(env.action_space)
    else:
        agent_cls = CAgent if isinstance(action_space, gym.spaces.Box) else DAgent
        agent = agent_cls.load(args.start_dir, args.start_idx)

    data, metrics, shape = collect_crowd_data(
        agent, env, args.steps, deterministic=args.deterministic, disable_tqdm=False
    )

    ep_rewards = get_episode_rewards(data.reward.numpy(), data.done.numpy(), shape)
    print(
        f"Mean episode reward: {ep_rewards.mean()} ± {ep_rewards.std()}; "
        f"[{ep_rewards.min()}, {ep_rewards.max()}]. n={len(ep_rewards)}"
    )
    env.close()
    # if CUDA:
    #     agent.cuda()

    # trainer = PPOCrowdTrainer(agent, env, trainer_config)
    # trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)
