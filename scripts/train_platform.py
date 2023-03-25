import json
from typing import Optional, Type

import gym as ogym
import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.spaces import Tuple, Discrete, Box
from gymnasium.wrappers import TimeLimit
from shimmy import GymV21CompatibilityV0
from typarse import BaseParser

import coltra
from coltra import Action
from coltra.agents import CAgent, DAgent, Agent, MixedAgent
from coltra.envs.spaces import ActionSpace
from coltra.groups import HomogeneousGroup
from coltra.models.mlp_models import MLPModel, PlatformMLPModel
from coltra.trainers import PPOCrowdTrainer
from coltra.envs import MultiGymEnv

import wandb


from coltra.wrappers import ObsVecNormWrapper, LastRewardWrapper
from coltra.wrappers.agent_wrappers import RetNormWrapper
from coltra.wrappers.env_wrappers import TimeFeatureWrapper


class Parser(BaseParser):
    config: str = "configs/base_config.yaml"
    iters: int = 500
    name: str
    project: str = "instadeep"
    seed: Optional[int] = None
    extra_config: Optional[str] = None

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "name": "Name of the tb directory to store the logs",
        "project": "Name of the wandb project to use",
        "seed": "Seed for the random number generator",
        "extra_config": "Extra config items to override the config file. Should be passed in a json format.",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "name": "n",
        "project": "p",
        "seed": "s",
        "extra_config": "ec",
    }


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class PlatformerWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, shape=(10,), dtype=np.float32)
        # self.action_space = self.env.action_space
        # self.action_space = Tuple((Discrete(3), Box(-np.inf, np.inf, (1,), np.float32)))
        self.action_space = Tuple((Discrete(3), Box(-np.inf, np.inf, (3,), np.float32)))
        # self.action_space = Box(np.array([0, 0]), np.array([3, 1]), dtype=np.float32)

        self.param_scales = np.array([30, 720, 430], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, *rest, info = self.env.step(self.action(action))
        info["m_original_reward"] = np.array([reward])
        return self.observation(obs), reward, *rest, info

    def observation(self, obs):
        (state, time) = obs
        new_obs = np.append(state, time)
        # print(state)
        # print(time)
        # print(new_obs.shape)
        return new_obs.astype(np.float32)

    def action(self, action: Action):
        disc, cont = int(action.discrete), action.continuous
        # param = np.zeros((3, 1))
        param = (self.param_scales * sigmoid(cont - 2.5))[:, None]

        return (disc, param)


def register_platform():
    import gym_platform
    def create_platform_env(render_mode):
        env = ogym.make("Platform-v0", disable_env_checker=True, apply_api_compatibility=False, max_episode_steps=None)
        env = GymV21CompatibilityV0(env=env.env)
        env = TimeLimit(env, 200)
        env = PlatformerWrapper(env)

        return env

    gym.register("PlatformEnv-v0", entry_point=create_platform_env, order_enforce=False, disable_env_checker=True)



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
    # Initialize the environment

    env = MultiGymEnv.get_venv(
        workers=workers, env_name="PlatformEnv-v0", wrappers=wrappers, seed=args.seed, import_fn=register_platform
    )
    action_space: ActionSpace = env.action_space
    observation_space = env.observation_space

    print(f"{observation_space=}")
    print(f"{action_space=}")


    model = PlatformMLPModel(model_config, observation_space, action_space)
    agent = MixedAgent(model)
    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    trainer = PPOCrowdTrainer(agents, env, trainer_config, seed=args.seed)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)
