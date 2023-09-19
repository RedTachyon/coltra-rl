import itertools
import json
import os
import socket
from logging import ERROR
from sys import platform
from typing import Optional, Type

import cv2
import numpy as np
import torch
import wandb
import yaml
from matplotlib import pyplot as plt
from mlagents_envs.exception import UnityEnvironmentException, UnityWorkerInUseException
from mlagents_envs.logging_util import get_logger, set_log_level
from typarse import BaseParser
import seaborn as sns

import coltra.utils
from coltra.agents import CAgent, Agent
from coltra.collectors import collect_renders
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.groups import HomogeneousGroup, FamilyGroup
from coltra.models import BaseModel
from coltra.models.attention_models import AttentionModel
from coltra.models.mlp_models import MLPModel, RayMLPModel
from coltra.models.relational_models import RelationModel, RayRelationModel
from coltra.trainers import PPOCrowdTrainer, PPOFamilyTrainer

import coltra.data_utils as du
from coltra.training_utils import evaluate

set_log_level(ERROR)


def is_worker_free(worker_id: int, base_port: int = 5005):
    """
    Attempts to bind to the requested communicator port, checking if it is already in use.
    Returns whether the port is free.
    """
    port = base_port + worker_id
    status = True
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if platform == "linux" or platform == "linux2":
        # On linux, the port remains unusable for TIME_WAIT=60 seconds after closing
        # SO_REUSEADDR frees the port right after closing the environment
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("localhost", port))
    except OSError:
        status = False
    #         raise UnityWorkerInUseException(self.worker_id)
    finally:
        s.close()

    return status


def find_free_worker(max_value: int = 1000, step: int = 10) -> int:
    """
    Finds a free worker ID.
    """
    for worker_id in range(0, max_value, step):
        if is_worker_free(worker_id):
            return worker_id

    raise UnityWorkerInUseException("All workers are in use.")


class Parser(BaseParser):
    config: str = "v7-configs/crowd_config.yaml"
    iters: int = 500
    env: str
    name: str
    worker_id: Optional[int] = None
    project: Optional[str] = None
    extra_config: Optional[str] = None

    _help = {
        "config": "Config file for coltra. If preceded by 'wandb:', will use wandb to fetch config.",
        "iters": "Number of coltra iterations",
        "env": "Path to the Unity environment binary",
        "name": "Name of the tb directory to store the logs",
        "worker_id": "Worker id",
        "project": "Name of wandb project",
        "extra_config": "Extra config items to override the config file. Should be passed in a json format.",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env": "e",
        "name": "n",
        "worker_id": "w",
        "project": "p",
        "extra_config": "ec",
    }


if __name__ == "__main__":
    try:
        CUDA = torch.cuda.is_available()

        args = Parser()

        if args.config.startswith("wandb:"):
            api = wandb.Api()
            run = api.run(args.config[6:])
            config = run.config
        else:
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
        env_config = config["environment"]
        model_type = config["model_type"]

        assert model_type in (
            "blind",
            "relation",
            "ray",
            "rayrelation",
            "attention",
        ), ValueError(f"Wrong model type {model_type} in the config.")

        trainer_config["tensorboard_name"] = args.name
        trainer_config["PPOConfig"]["use_gpu"] = CUDA

        workers = trainer_config["workers"]

        # Initialize the environment
        if workers > 1:
            env = UnitySimpleCrowdEnv.get_venv(
                workers,
                base_worker_id=args.worker_id,
                file_name=args.env,
                no_graphics=True,
                extra_params=env_config,
            )
        else:
            env = UnitySimpleCrowdEnv.get_env_creator(
                file_name=args.env,
                no_graphics=True,
                extra_params=env_config,
                worker_id=args.worker_id,
            )()

        env.reset()

        # env.engine_channel.set_configuration_parameters(time_scale=100, width=100, height=100)

        # Initialize the agent

        wandb.init(
            project="crowdai" if args.project is None else args.project,
            sync_tensorboard=True,
            config=config,
            name=args.name,
        )

        [family_key, crowd_key] = env.behaviors.keys()

        crowd_model = RayRelationModel(
            model_config,
            env.observation_spaces[crowd_key],
            env.action_spaces[crowd_key],
        )
        crowd_agent = CAgent(crowd_model)

        family_model = MLPModel(
            model_config,
            env.observation_spaces[family_key],
            env.action_spaces[family_key],
        )
        family_agent = CAgent(family_model)

        family = FamilyGroup(crowd_agent, family_agent)

        if CUDA:
            family.cuda()

        trainer = PPOFamilyTrainer(family, env, trainer_config)
        with open(os.path.join(trainer.path, "full_config.yaml"), "w") as f:
            yaml.dump(config, f)

        trainer.train(
            args.iters,
            disable_tqdm=False,
            save_path=trainer.path,
            collect_kwargs=env_config,
        )

        print("Skipping regular final evaluation for now.")
        # print("Evaluating...")
        # performances, energies = evaluate(env, family, 10, disable_tqdm=False)
        # wandb.log(
        #     {
        #         "final/mean_episode_reward": np.mean(performances),
        #         "final/std_episode_reward": np.std(performances),
        #         "final/mean_episode_energy": np.mean(energies),
        #         "final/std_episode_energy": np.std(energies),
        #     },
        #     commit=False,
        # )
        #
        # wandb.log({})

        env.close()

        print("Training complete. Evaluation starting.")

        env_config["evaluation_mode"] = 1.0

        worker_id = find_free_worker(500)
        env = UnitySimpleCrowdEnv(
            file_name=args.env,
            virtual_display=(800, 800),
            no_graphics=False,
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
        mode = env_config["initializer"]
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
                family,
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

            frame_size = renders.shape[1:3]

            print("Recording a video")
            video_path = os.path.join(
                trainer.path, "videos", f"video_{mode}_{'det' if d else 'rnd'}_{i}.webm"
            )
            out = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"VP90"), 24, frame_size[::-1]
            )
            for frame in renders[..., ::-1]:
                out.write(frame)

            out.release()

            print(f"Video saved to {video_path}")

            wandb.log(
                {
                    f"video_{mode}_{'det' if d else 'rnd'}_{idx}": wandb.Video(
                        video_path
                    )
                },
                commit=False,
            )

            print("Video uploaded to wandb")

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
