import copy
import os
from logging import ERROR

import cv2
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
from coltra.envs import UnitySimpleCrowdEnv
from coltra.models import RelationModel
import data_utils_optimize as du

CUDA = torch.cuda.is_available()
set_log_level(ERROR)


class Parser(BaseParser):
    env: str
    config: str = "base.yaml"
    worker_id: int = 0
    n_trials: int = 50
    optuna_name: str = "optuna"
    wandb_project: str = "jeanzay-sweep"

    _abbrev = {
        "env": "e",
        "config": "c",
        "worker_id": "w",
        "n_trials": "n",
        "optuna_name": "o",
        "wandb_project": "wp",
    }

    _help = {
        "env": "Path to the environment",
        "config": "Path to the config file",
        "worker_id": "Worker ID to start from",
        "n_trials": "Number of trials",
        "optuna_name": "Name of the optuna study",
        "wandb_project": "Name of the wandb project",
    }


def train_one(
    trial: optuna.Trial, worker_id: int, path: str, config: dict, wandb_project: str
):

    config = copy.deepcopy(config)

    env = UnitySimpleCrowdEnv.get_venv(
        file_name=path,
        workers=config["trainer"]["workers"],
        base_worker_id=worker_id,
        no_graphics=True,
    )
    env.reset(save_trajectory=0.0)

    # Initialize the agent

    wandb.init(
        project=wandb_project,
        entity="redtachyon",
        sync_tensorboard=True,
        config=config,
        name=f"trial{trial.number}",
    )

    model = RelationModel(config["model"], observation_space=env.observation_space, action_space=env.action_space)
    agent = CAgent(model)
    agents = HomogeneousGroup(agent)

    if CUDA:
        agent.cuda()

    trainer = PPOCrowdTrainer(
        agents=agents, env=env, config=config["trainer"], use_uuid=True
    )

    final_metrics = trainer.train(
        num_iterations=1000,
        disable_tqdm=False,
        save_path=trainer.path,
        collect_kwargs=config["environment"],
        # trial=trial,
    )

    env.close()

    mean_reward = final_metrics["crowd/mean_episode_reward"]

    # EVALUATION
    env = UnitySimpleCrowdEnv(
        file_name=args.env,
        no_graphics=True,
        worker_id=worker_id + 5,
    )
    config["environment"]["evaluation_mode"] = 1.0
    env.reset(**config["environment"])

    os.mkdir(os.path.join(trainer.path, "trajectories"))
    os.mkdir(os.path.join(trainer.path, "videos"))
    os.mkdir(os.path.join(trainer.path, "images"))

    sns.set()
    UNIT_SIZE = 3
    plt.rcParams["figure.figsize"] = (8 * UNIT_SIZE, 4 * UNIT_SIZE)

    mode = "circle"
    for i in range(4):
        idx = i % 2
        d = idx == 0
        if i == 2:
            mode = "json"
            config["environment"]["mode"] = "json"

        trajectory_path = os.path.join(
            trainer.path,
            "trajectories",
            f"trajectory_{mode}_{'det' if d else 'rnd'}_{idx}.json",
        )

        dashboard_path = os.path.join(
            trainer.path,
            "images",
            f"dashboard_{mode}_{'det' if d else 'rnd'}_{idx}.png",
        )

        env.reset(save_path=trajectory_path, **config["environment"])
        print(
            f"Collecting data for {'' if d else 'non'}deterministic {mode} video number {idx}"
        )

        renders, returns = collect_renders(
            agents,
            env,
            num_steps=200,
            disable_tqdm=False,
            env_kwargs=config["environment"],
            deterministic=d,
        )

        print(f"Mean return: {np.mean(returns)}")

        # Generate the dashboard
        print("Generating dashboard")
        print("Skipping dashboard")
        trajectory = du.read_trajectory(trajectory_path)

        plt.clf()
        du.make_dashboard(trajectory, save_path=dashboard_path)

        # Upload to wandb
        print("Uploading dashboard")
        wandb.log(
            {
                "dashboard": wandb.Image(
                    dashboard_path,
                    caption=f"Dashboard {mode} {'det' if d else 'rng'} {i}",
                )
            }
        )

        print("Skipping video")

        trajectory_artifact = wandb.Artifact(
            name=f"trajectory_{mode}_{'det' if d else 'rnd'}_{idx}", type="json"
        )
        trajectory_artifact.add_file(trajectory_path)
        wandb.log_artifact(trajectory_artifact)

    env.close()

    wandb.finish()

    return mean_reward


def objective(
    trial: optuna.Trial, worker_id: int, path: str, config_path: str, wandb_project: str
) -> float:
    # Get some parameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    n_episodes = 1

    steps = n_episodes * 200

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
        [32, 32],
        [64, 64],
        [128, 128],
        [32, 32, 32],
        [64, 64, 64],
    ]

    LAYER_IDX = list(range(len(LAYER_OPTIONS)))

    vec_hidden_layers = trial.suggest_categorical("vec_hidden_layers", LAYER_IDX)
    vec_hidden_layers = LAYER_OPTIONS[vec_hidden_layers]

    rel_hidden_layers = trial.suggest_categorical("rel_hidden_layers", LAYER_IDX)
    rel_hidden_layers = LAYER_OPTIONS[rel_hidden_layers]

    com_hidden_layers = trial.suggest_categorical("com_hidden_layers", LAYER_IDX)
    com_hidden_layers = LAYER_OPTIONS[com_hidden_layers]

    optuna_model_kwargs = {
        "vec_hidden_layers": vec_hidden_layers,
        "rel_hidden_layers": rel_hidden_layers,
        "com_hidden_layers": com_hidden_layers,
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

    # TODO: run this several times and average the results

    mean_reward = train_one(trial, worker_id, path, config, wandb_project)

    return mean_reward


if __name__ == "__main__":
    args = Parser()

    study = optuna.load_study(
        storage=f"sqlite:///{args.optuna_name}.db", study_name=args.optuna_name
    )

    study.optimize(
        lambda trial: objective(
            trial, args.worker_id, args.env, args.config, args.wandb_project
        ),
        n_trials=args.n_trials,
    )

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
