from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
import yaml
from typarse import BaseParser

from coltra.agents import CAgent, DAgent, RandomGymAgent
from coltra.collectors import collect_renders
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
    path: str
    video: str = "output"
    idx: Optional[int] = -1
    num_steps: Optional[int] = 1
    deterministic: bool = False

    _help = {
        "path": "Path to the saved model",
        "video": "Filename of the output video, without the extension",
        "idx": "Which training iteration should be used",
        "num_steps": "Number of steps to record",
        "deterministic": "Whether the agent should be greedy",
    }

    _abbrev = {
        "path": "p",
        "video": "v",
        "idx": "i",
        "num_steps": "n",
        "deterministic": "d",
    }


if __name__ == "__main__":
    args = Parser()

    # Initialize the environment
    env = PettingZooEnv.get_venv(1, env_creator=pursuit_v3.parallel_env)
    action_space = env.action_space
    observation_space = env.observation_space

    print(f"{observation_space=}")
    print(f"{action_space=}")

    is_discrete_action = isinstance(action_space, gym.spaces.Discrete)
    if is_discrete_action:
        action_shape = action_space.n
    else:
        action_shape = action_space.shape[0]

    if args.path == "RANDOM":
        agent = RandomGymAgent(env.action_space)
    else:
        agent_cls = CAgent if isinstance(action_space, gym.spaces.Box) else DAgent
        agent = agent_cls.load(args.path, args.idx)
    agents = HomogeneousGroup(agent)

    renders, returns = collect_renders(
        agents=agents,
        env=env,
        num_steps=args.num_steps,
        deterministic=args.deterministic,
    )

    print(
        f"Mean return: {returns.mean()} ± {returns.std()} with {returns.shape[0]} samples"
    )

    frame_size = renders.shape[1:3]

    out = cv2.VideoWriter(
        f"{args.video}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, frame_size[::-1]
    )
    for frame in renders[..., ::-1]:
        out.write(frame)

    out.release()

    print(f"Video saved to {args.video}.mp4")

    env.close()
