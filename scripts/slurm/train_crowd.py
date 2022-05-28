import itertools
import json
import os
from logging import ERROR
from typing import Optional, Type

import cv2
import numpy as np
import torch
import wandb
import yaml
from matplotlib import pyplot as plt
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.logging_util import get_logger, set_log_level
from typarse import BaseParser
import seaborn as sns

import coltra.utils
from coltra.agents import CAgent, Agent
from coltra.collectors import collect_renders
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.groups import HomogeneousGroup
from coltra.models import BaseModel
from coltra.models.mlp_models import MLPModel, RayMLPModel
from coltra.models.relational_models import RelationModel, RayRelationModel
from coltra.trainers import PPOCrowdTrainer
from coltra.models.raycast_models import LeeModel

import data_utils as du
from coltra.training_utils import evaluate
from coltra.utils import find_free_worker

set_log_level(ERROR)


class Parser(BaseParser):
    worker_id: int
    env: str

    observer: str
    dynamics: str
    model: str

    project: Optional[str] = None
    extra_config: Optional[str] = None

    _help = {
        "worker_id": "Worker id for the current worker.",
        "env": "Path to the Unity environment binary",
        "observer": "Name of the observations to use",
        "dynamics": "Name of the dynamics to use",
        "model": "Name of the model to use",
        "project": "Type of project to use",
        "extra_config": "Extra config items to override the config file. Should be passed in a json format.",
    }

    _abbrev = {
        "worker_id": "w",
        "env": "e",
        "observer": "o",
        "dynamics": "d",
        "model": "m",
        "project": "p",
        "extra_config": "ec",
    }


if __name__ == "__main__":
    try:
        CUDA = torch.cuda.is_available()

        args = Parser()

        config_path = f"top/{args.observer[:3]}_{args.dynamics[:3]}_{'Vel' if 'Velocity' in args.dynamics else 'Acc'}.yaml"

        with open(config_path, "r") as f:
            config = yaml.load(f.read(), yaml.Loader)

        assert args.model in (
            "blind",
            "relation",
            "ray",
            "rayrelation",
        ), ValueError(f"Wrong model type {args.model} in args.")

        config["model_type"] = args.model

        run_name = f"{args.observer[:3]}_{args.dynamics}_{args.model}"

        if args.extra_config is not None:
            extra_config = json.loads(args.extra_config)
            extra_config = coltra.utils.undot_dict(extra_config)
            coltra.utils.update_dict(target=config, source=extra_config)

        trainer_config = config["trainer"]
        model_config = config["model"]
        env_config = config["environment"]
        model_type = config["model_type"]

        trainer_config["tensorboard_name"] = run_name
        trainer_config["PPOConfig"]["use_gpu"] = CUDA

        workers = trainer_config["workers"]

        # Initialize the environment
        env = UnitySimpleCrowdEnv.get_venv(
            workers, base_worker_id=args.worker_id, file_name=args.env, no_graphics=True, extra_params=env_config
        )
        # env.reset(**env_config)


        # Initialize the agent

        wandb.init(
            project="crowdai" if args.project is None else args.project,
            entity="redtachyon",
            sync_tensorboard=True,
            config=config,
            name=run_name,
        )

        if model_type == "relation":
            model_cls = RelationModel
        elif model_type == "blind":
            model_cls = MLPModel
        elif model_type == "ray":
            model_cls = RayMLPModel
        elif model_type == "rayrelation":
            model_cls = RayRelationModel
        else:
            raise ValueError(
                "Wrong model type passed. This should have been caught sooner"
            )

        model = model_cls(model_config, env.observation_space, env.action_space)
        agent = CAgent(model)

        agents = HomogeneousGroup(agent)

        if CUDA:
            agents.cuda()

        trainer = PPOCrowdTrainer(agents, env, trainer_config)
        with open(os.path.join(trainer.path, "full_config.yaml"), "w") as f:
            yaml.dump(config, f)

        trainer.train(
            1000,
            disable_tqdm=False,
            save_path=trainer.path,
            collect_kwargs=env_config,
        )

        print("Evaluating...")
        performances, energies = evaluate(env, agents, 10, disable_tqdm=False)
        wandb.log(
            {
                "final/mean_episode_reward": np.mean(performances),
                "final/std_episode_reward": np.std(performances),
                "final/mean_episode_energy": np.mean(energies),
                "final/std_episode_energy": np.std(energies),
            },
            commit=False,
        )

        wandb.log({})

        env.close()

        print("Training complete. Evaluation starting.")

        env_config["evaluation_mode"] = 1.0

        worker_id = args.worker_id + 5
        env = UnitySimpleCrowdEnv(
            file_name=args.env,
            no_graphics=True,
            worker_id=worker_id,
            extra_params=env_config,
        )
        env.reset(**env_config)

        os.mkdir(os.path.join(trainer.path, "trajectories"))
        os.mkdir(os.path.join(trainer.path, "videos"))
        os.mkdir(os.path.join(trainer.path, "images"))

        sns.set()
        UNIT_SIZE = 3
        plt.rcParams["figure.figsize"] = (8 * UNIT_SIZE, 4 * UNIT_SIZE)

        # mode = "circle"
        mode = env_config["mode"]
        for i in range(3):
            idx = i % 3
            d = idx == 0

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

            env.reset(save_path=trajectory_path, **env_config)
            print(
                f"Collecting data for {'' if d else 'non'}deterministic {mode} video number {idx}"
            )

            renders, returns = collect_renders(
                agents,
                env,
                num_steps=trainer_config["steps"],
                disable_tqdm=False,
                env_kwargs=env_config,
                deterministic=d,
            )

            print(f"Mean return: {np.mean(returns)}")

            # Generate the dashboard
            print("Generating dashboard")
            trajectory = du.read_trajectory(trajectory_path)

            plt.clf()
            du.make_dashboard(trajectory, save_path=dashboard_path)

            # Upload to wandb
            print("Uploading dashboard")
            wandb.log(
                {
                    f"dashboard_{idx}": wandb.Image(
                        dashboard_path,
                        caption=f"Dashboard {mode} {'det' if d else 'rng'} {i}",
                    )
                },
                commit=False,
            )

            trajectory_artifact = wandb.Artifact(
                name=f"trajectory_{mode}_{'det' if d else 'rnd'}_{idx}", type="json"
            )
            trajectory_artifact.add_file(trajectory_path)
            wandb.log_artifact(trajectory_artifact)

        wandb.log({}, commit=True)
        env.close()
        wandb.finish()

    finally:
        print("Cleaning up")
        try:
            env.close()  # pytype: disable=name-error
            print("Env closed")
        except NameError:
            print("Env wasn't created. Exiting coltra")
        except UnityEnvironmentException:
            print("Env already closed. Exiting coltra")
        except Exception:
            print("Unknown error when closing the env. Exiting coltra")
