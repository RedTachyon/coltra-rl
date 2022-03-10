import numpy as np
import optuna
from optuna.integration import WeightsAndBiasesCallback
import torch
import yaml
from typarse import BaseParser
import wandb

from coltra import PPOCrowdTrainer, CAgent, HomogeneousGroup
from coltra.envs import UnitySimpleCrowdEnv
from coltra.models import RelationModel

CUDA = torch.cuda.is_available()


class Parser(BaseParser):
    env: str
    worker_id: int = 0
    n_trials: int = 50
    optuna_name: str = "optuna"

    _abbrev = {"env": "e", "worker_id": "w", "n_trials": "n", "optuna_name": "o"}

    _help = {
        "env": "Path to the environment",
        "worker_id": "Worker ID to start from",
        "n_trials": "Number of trials",
        "optuna_name": "Name of the optuna study",
    }


def objective(trial: optuna.Trial, worker_id: int, path: str) -> float:
    # Get some parameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    n_episodes = trial.suggest_int("n_episodes", 1, 5)

    steps = n_episodes * 200

    optuna_PPO_kwargs = {
        # "OptimizerKwargs": {
        #     "lr": lr,
        # },
        "gamma": trial.suggest_loguniform("gamma", 0.9, 0.9999),
        "gae_lambda": trial.suggest_uniform("gae_lambda", 0.8, 1.0),
        "eps": trial.suggest_uniform("eps", 0.05, 0.2),
        "target_kl": trial.suggest_uniform("target_kl", 0.01, 0.05),
        "entropy_coeff": trial.suggest_loguniform("entropy_coeff", 0.01, 0.05),
        "ppo_epochs": trial.suggest_int("ppo_epochs", 5, 20),
        "minibatch_size": trial.suggest_int("minibatch_size", 512, 4096),
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

    vec_hidden_layer_sizes = trial.suggest_categorical(
        "vec_hidden_layer_sizes", LAYER_IDX
    )
    vec_hidden_layer_sizes = LAYER_OPTIONS[vec_hidden_layer_sizes]

    rel_hidden_layer_sizes = trial.suggest_categorical(
        "rel_hidden_layer_sizes", LAYER_IDX
    )
    rel_hidden_layer_sizes = LAYER_OPTIONS[rel_hidden_layer_sizes]

    com_hidden_layer_sizes = trial.suggest_categorical(
        "com_hidden_layer_sizes", LAYER_IDX
    )
    com_hidden_layer_sizes = LAYER_OPTIONS[com_hidden_layer_sizes]

    optuna_model_kwargs = {
        "vec_hidden_layer_sizes": vec_hidden_layer_sizes,
        "rel_hidden_layer_sizes": rel_hidden_layer_sizes,
        "com_hidden_layer_sizes": com_hidden_layer_sizes,
        "activation": activation,
    }

    # Read the main config

    with open("config.yml", "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # Update the config
    config["trainer"]["PPOConfig"]["OptimizerKwargs"]["lr"] = lr
    config["trainer"]["steps"] = steps

    for key, value in optuna_PPO_kwargs:
        config["trainer"]["PPOConfig"][key] = value

    for key, value in optuna_model_kwargs:
        config["model"][key] = value

    config["trainer"]["tensorboard_name"] = f"trial {trial.number}"
    config["trainer"]["PPOConfig"]["use_gpu"] = CUDA

    env = UnitySimpleCrowdEnv.get_venv(
        file_name=path,
        workers=config["trainer"]["workers"],
        worker_id=worker_id,
        no_graphics=True,
    )
    env.reset(save_trajectory=0.0)

    # Initialize the agent
    obs_size = env.observation_space.shape[0]
    buffer_size = env.get_attr("obs_buffer_size")[0]
    action_size = env.action_space.shape[0]

    config["model"]["input_size"] = obs_size
    config["model"]["rel_input_size"] = buffer_size
    config["model"]["num_actions"] = action_size

    wandb.init(
        project="optuna-sweep",
        entity="redtachyon",
        sync_tensorboard=True,
        config=config,
        name=f"trial{trial.number}",
    )

    model = RelationModel(config["model"], action_space=env.action_space)
    agent = CAgent(model)
    agents = HomogeneousGroup(agent)

    if CUDA:
        agent.cuda()

    trainer = PPOCrowdTrainer(
        agents=agents,
        env=env,
        config=config["trainer"],
    )

    final_metrics = trainer.train(
        num_iterations=1000,
        disable_tqdm=False,
        save_path=trainer.path,
        collect_kwargs=config["environment"],
        trial=trial,
    )

    env.close()

    mean_reward = final_metrics["crowd/mean_episode_reward"]
    wandb.finish()

    return mean_reward


if __name__ == "__main__":
    args = Parser()

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(
        lambda trial: objective(trial, args.worker_id, args.env), n_trials=args.n_trials
    )

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
