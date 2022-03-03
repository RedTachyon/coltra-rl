import numpy as np
import optuna
import torch
import yaml
from typarse import BaseParser
import wandb

from coltra.envs import UnitySimpleCrowdEnv

data_x = np.linspace(0, 1, 1000)
data_y: np.ndarray = 2*data_x + 5 + np.random.randn(1000) * 0.1
CUDA = torch.cuda.is_available()

# TODO: Finish optuna optimization

class Parser(BaseParser):
    worker_id: int = 0
    n_trials: int = 1000

    _abbrev = {
        'worker_id': 'w',
        'n_trials': 'n'
    }

    _help = {
        'worker_id': 'Worker ID to start from',
        'n_trials': 'Number of trials'
    }

def objective(trial: optuna.Trial, worker_id: int) -> float:
    # Get some parameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)

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

    vec_hidden_layer_sizes = trial.suggest_categorical("vec_hidden_layer_sizes", LAYER_IDX)
    vec_hidden_layer_sizes = LAYER_OPTIONS[vec_hidden_layer_sizes]

    rel_hidden_layer_sizes = trial.suggest_categorical("rel_hidden_layer_sizes", LAYER_IDX)
    rel_hidden_layer_sizes = LAYER_OPTIONS[rel_hidden_layer_sizes]

    com_hidden_layer_sizes = trial.suggest_categorical("com_hidden_layer_sizes", LAYER_IDX)
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

    for key, value in optuna_PPO_kwargs:
        config["trainer"]["PPOConfig"][key] = value

    for key, value in optuna_model_kwargs:
        config["model"][key] = value

    config["trainer"]["tensorboard_name"] = f"trial {trial.number}"
    config["trainer"]["PPOConfig"]["use_gpu"] = CUDA

    wandb.init(
        project="optuna-sweep",
        entity="redtachyon",
        sync_tensorboard=True,
        config=config,
        name=f"trial{trial.number}",
    )

    env = UnitySimpleCrowdEnv.get_venv(workers=8, worker_id=worker_id)

    return 0.0


if __name__ == "__main__":
    args = Parser()
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(lambda trial: objective(trial, args.worker_id), n_trials=1000)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)