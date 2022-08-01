import copy
import os
from logging import ERROR

import cv2
import gym
import numpy as np
import optuna
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import yaml
from mlagents_envs.logging_util import set_log_level
from typarse import BaseParser
import wandb

from coltra import PPOCrowdTrainer, CAgent, HomogeneousGroup, collect_renders
from coltra.envs import UnitySimpleCrowdEnv, MultiGymEnv
from coltra.envs.spaces import ActionSpace
from coltra.models import RelationModel, MLPModel
from coltra.wrappers.env_wrappers import TimeFeatureWrapper

CUDA = torch.cuda.is_available()
set_log_level(ERROR)


class Parser(BaseParser):
    config: str = "base.yaml"
    n_trials: int = 1
    optuna_name: str = "optuna"
    wandb_project: str = "jeanzay-sweep"

    _abbrev = {
        "config": "c",
        "n_trials": "n",
        "optuna_name": "o",
        "wandb_project": "wp",
    }

    _help = {
        "config": "Path to the config file",
        "n_trials": "Number of trials",
        "optuna_name": "Name of the optuna study",
        "wandb_project": "Name of the wandb project",
    }


def train_one(
    trial: optuna.Trial, config: dict, wandb_project: str
):

    config = copy.deepcopy(config)

    wrappers = []
    wrappers.append(TimeFeatureWrapper)

    wrappers.append(
        lambda e: gym.wrappers.TransformObservation(e, lambda obs: obs / 1e3)
    )
    wrappers.append(
        lambda e: gym.wrappers.TransformReward(e, lambda reward: reward / 1e5)
    )
    wrappers.append(gym.wrappers.ClipAction)

    env = MultiGymEnv.get_venv(
        workers=config["trainer"]["workers"], env_name="HumanoidStandup-v4", wrappers=wrappers
    )
    action_space: ActionSpace = env.action_space
    observation_space = env.observation_space
    # env.reset()

    # Initialize the agent
    trainer_config = config["trainer"]
    model_config = config["model"]

    wandb.init(
        project=wandb_project,
        entity="redtachyon",
        sync_tensorboard=True,
        config=config,
        name=f"trial{trial.number}",
    )

    model_cls = MLPModel
    agent_cls = CAgent

    model = model_cls(model_config, observation_space, action_space)
    agent = agent_cls(model)
    agents = HomogeneousGroup(agent)

    if CUDA:
        agent.cuda()

    trainer = PPOCrowdTrainer(
        agents=agents, env=env, config=config["trainer"], use_uuid=True, save_path="/gpfswork/rech/nbk/utu66tc/"
    )

    final_metrics = trainer.train(
        num_iterations=2000,
        disable_tqdm=False,
        save_path=trainer.path,
        # trial=trial,
    )


    # all_returns = []
    # for _ in range(10):
    #     _, returns = collect_renders(agents, env, num_steps=1000, deterministic=False)
    #     all_returns.append(returns)

    # all_returns = np.array(all_returns)
    # mean_reward = np.mean(all_returns)
    mean_reward = final_metrics["crowd/mean_episode_reward"]

    env.close()

    wandb.finish()

    return mean_reward


def objective(
    trial: optuna.Trial, config_path: str, wandb_project: str
) -> float:
    # Get some parameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    n_episodes = 1

    steps = n_episodes * 1000

    optuna_PPO_kwargs = {
        # "OptimizerKwargs": {
        #     "lr": lr,
        # },
        "gamma": 1 - trial.suggest_loguniform("1-gamma", 1e-4, 1e-1),
        "gae_lambda": trial.suggest_uniform("gae_lambda", 0.8, 1.0),
        "eps": trial.suggest_uniform("eps", 0.05, 0.2),
        "target_kl": trial.suggest_uniform("target_kl", 0.01, 0.05),
        "entropy_coeff": trial.suggest_uniform("entropy_coeff", 0, 0.05),
        "ppo_epochs": trial.suggest_int("ppo_epochs", 15, 40),
    }

    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "tanh"])

    LAYER_OPTIONS = [
        [128, 128],
        [256, 256],
        [256, 256, 256],
    ]

    LAYER_IDX = list(range(len(LAYER_OPTIONS)))

    vec_hidden_layers = trial.suggest_categorical("vec_hidden_layers", LAYER_IDX)
    vec_hidden_layers = LAYER_OPTIONS[vec_hidden_layers]


    optuna_model_kwargs = {
        "hidden_sizes": vec_hidden_layers,
        "activation": activation,
    }

    # Read the main config

    with open(config_path, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # Update the config
    config["trainer"]["PPOConfig"]["OptimizerKwargs"]["lr"] = lr
    config["trainer"]["steps"] = steps

    for key, value in optuna_PPO_kwargs.items():
        config["trainer"]["PPOConfig"][key] = value

    for key, value in optuna_model_kwargs.items():
        config["model"][key] = value

    config["trainer"]["tensorboard_name"] = f"trial{trial.number}"
    config["trainer"]["PPOConfig"]["use_gpu"] = CUDA

    mean_reward = train_one(trial, config, wandb_project)

    return mean_reward


if __name__ == "__main__":
    args = Parser()

    study = optuna.load_study(
        storage=f"sqlite:///{args.optuna_name}.db", study_name=args.optuna_name
    )

    study.optimize(
        lambda trial: objective(
            trial, args.config, args.wandb_project
        ),
        n_trials=args.n_trials,
    )

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
