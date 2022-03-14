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
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    n_episodes = 1

    steps = n_episodes * 200

    optuna_PPO_kwargs = {
        # "OptimizerKwargs": {
        #     "lr": lr,
        # },
        "gamma": 1-trial.suggest_loguniform("1-gamma", 1e-4, 1e-1),
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
    #
    # vec_hidden_layer_sizes = trial.suggest_categorical(
    #     "vec_hidden_layer_sizes", LAYER_IDX
    # )
    # vec_hidden_layer_sizes = LAYER_OPTIONS[vec_hidden_layer_sizes]
    #
    # rel_hidden_layer_sizes = trial.suggest_categorical(
    #     "rel_hidden_layer_sizes", LAYER_IDX
    # )
    # rel_hidden_layer_sizes = LAYER_OPTIONS[rel_hidden_layer_sizes]
    #
    # com_hidden_layer_sizes = trial.suggest_categorical(
    #     "com_hidden_layer_sizes", LAYER_IDX
    # )
    # com_hidden_layer_sizes = LAYER_OPTIONS[com_hidden_layer_sizes]
    #
    # optuna_model_kwargs = {
    #     "vec_hidden_layer_sizes": vec_hidden_layer_sizes,
    #     "rel_hidden_layer_sizes": rel_hidden_layer_sizes,
    #     "com_hidden_layer_sizes": com_hidden_layer_sizes,
    #     "activation": activation,
    # }


    vec_hidden_layers = trial.suggest_categorical(
        "vec_hidden_layers", LAYER_IDX
    )
    vec_hidden_layers = LAYER_OPTIONS[vec_hidden_layers]

    rel_hidden_layers = trial.suggest_categorical(
        "rel_hidden_layers", LAYER_IDX
    )
    rel_hidden_layers = LAYER_OPTIONS[rel_hidden_layers]

    com_hidden_layers = trial.suggest_categorical(
        "com_hidden_layers", LAYER_IDX
    )
    com_hidden_layers = LAYER_OPTIONS[com_hidden_layers]

    optuna_model_kwargs = {
        "vec_hidden_layers": vec_hidden_layers,
        "rel_hidden_layers": rel_hidden_layers,
        "com_hidden_layers": com_hidden_layers,
        "activation": activation,
    }


    # Read the main config

    with open("base.yaml", "r") as f:
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

    env = UnitySimpleCrowdEnv.get_venv(
        file_name=path,
        workers=config["trainer"]["workers"],
        base_worker_id=worker_id,
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
        project="optuna-sweep-fixed",
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
        use_uuid=True
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
        virtual_display=(1600, 900),
        no_graphics=False,
        worker_id=worker_id+5,
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
        # trajectory = du.read_trajectory(trajectory_path)
        #
        # plt.clf()
        # du.make_dashboard(trajectory, save_path=dashboard_path)

        # Upload to wandb
        # print("Uploading dashboard")
        # wandb.log(
        #     {
        #         "dashboard": wandb.Image(
        #             dashboard_path,
        #             caption=f"Dashboard {mode} {'det' if d else 'rng'} {i}",
        #         )
        #     }
        # )

        frame_size = renders.shape[1:3]

        print("Recording a video")
        video_path = os.path.join(
            trainer.path, "videos", f"video_{mode}_{'det' if d else 'rnd'}_{i}.webm"
        )
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"VP90"), 30, frame_size[::-1]
        )
        for frame in renders[..., ::-1]:
            out.write(frame)

        out.release()

        print(f"Video saved to {video_path}")

        wandb.log(
            {f"video_{mode}_{'det' if d else 'rnd'}_{idx}": wandb.Video(video_path)}
        )

        print("Video uploaded to wandb")

        trajectory_artifact = wandb.Artifact(
            name=f"trajectory_{mode}_{'det' if d else 'rnd'}_{idx}", type="json"
        )
        trajectory_artifact.add_file(trajectory_path)
        wandb.log_artifact(trajectory_artifact)

    env.close()

    wandb.finish()

    return mean_reward


if __name__ == "__main__":
    args = Parser()

    study = optuna.load_study(storage=f"sqlite:///{args.optuna_name}.db", study_name=args.optuna_name)

    study.optimize(
        lambda trial: objective(trial, args.worker_id, args.env), n_trials=args.n_trials
    )

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
