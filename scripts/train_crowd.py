import itertools
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
from coltra.utils import find_free_worker

set_log_level(ERROR)


class Parser(BaseParser):
    config: str = "configs/nocollision.yaml"
    iters: int = 500
    env: str
    name: str
    model_type: str = "relation"
    dynamics: Optional[str] = None
    observer: Optional[str] = None
    project: Optional[str] = None
    start_dir: Optional[str]
    start_idx: Optional[int] = -1
    speed_coeff: Optional[float] = None

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env": "Path to the Unity environment binary",
        "name": "Name of the tb directory to store the logs",
        "model_type": "Type of the information that a model has access to",
        "dynamics": "Type of dynamics to use",
        "observer": "Type of observer to use",
        "project": "Type of project to use",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
        "speed_coeff": "Reward weighing parameter",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env": "e",
        "name": "n",
        "model_type": "mt",
        "dynamics": "d",
        "observer": "o",
        "project": "p",
        "start_dir": "sd",
        "start_idx": "si",
        "speed_coeff": "sc",
    }


if __name__ == "__main__":
    try:
        CUDA = torch.cuda.is_available()

        args = Parser()

        assert args.model_type in ("blind", "relation", "ray", "rayrelation"), ValueError(
            "Wrong model type passed."
        )

        with open(args.config, "r") as f:
            config = yaml.load(f.read(), yaml.Loader)

        if args.dynamics is not None:
            assert args.dynamics in (
                "CartesianVelocity",
                "CartesianAcceleration",
                "PolarVelocity",
                "PolarAcceleration",
            ), ValueError("Wrong dynamics type passed.")
            config["environment"]["dynamics"] = args.dynamics

        if args.observer is not None:
            assert args.observer in ("Absolute", "Relative", "Egocentric"), ValueError(
                "Wrong observer type passed."
            )
            config["environment"]["observer"] = args.observer

        if args.speed_coeff is not None:
            config["environment"]["comfort_speed_weight"] = args.speed_coeff
        trainer_config = config["trainer"]
        model_config = config["model"]
        env_config = config["environment"]

        trainer_config["tensorboard_name"] = args.name
        trainer_config["PPOConfig"]["use_gpu"] = CUDA

        workers = trainer_config["workers"]

        # Initialize the environment
        env = UnitySimpleCrowdEnv.get_venv(
            workers, file_name=args.env, no_graphics=True, extra_params=env_config
        )
        env.reset()

        # env.engine_channel.set_configuration_parameters(time_scale=100, width=100, height=100)

        # Initialize the agent

        wandb.init(
            project="crowdai" if args.project is None else args.project,
            entity="redtachyon",
            sync_tensorboard=True,
            config=config,
            name=args.name,
        )

        if args.model_type == "relation":
            model_cls = RelationModel
        elif args.model_type == "blind":
            model_cls = MLPModel
        elif args.model_type == "ray":
            model_cls = RayMLPModel
        elif args.model_type == "rayrelation":
            model_cls = RayRelationModel
        else:
            raise ValueError("Wrong model type passed. This should have been caught sooner")

        agent: Agent
        if args.start_dir:
            agent = CAgent.load(args.start_dir, weight_idx=args.start_idx)
        else:
            model = model_cls(model_config, env.observation_space, env.action_space)
            agent = CAgent(model)

        agents = HomogeneousGroup(agent)

        if CUDA:
            agents.cuda()

        trainer = PPOCrowdTrainer(agents, env, trainer_config)
        with open(os.path.join(trainer.path, "full_config.yaml"), "w") as f:
            yaml.dump(config, f)

        trainer.train(
            args.iters,
            disable_tqdm=False,
            save_path=trainer.path,
            collect_kwargs=env_config,
        )

        env.close()

        print("Training complete. Evaluation starting.")

        env_config["evaluation_mode"] = 1.0

        worker_id = find_free_worker(500)
        env = UnitySimpleCrowdEnv(
            file_name=args.env,
            virtual_display=(1600, 900),
            no_graphics=False,
            worker_id=worker_id,
            extra_params=env_config
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
                    "dashboard": wandb.Image(
                        dashboard_path,
                        caption=f"Dashboard {mode} {'det' if d else 'rng'} {i}",
                    )
                }
            )

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

    finally:
        print("Cleaning up")
        wandb.finish(0)
        try:
            env.close()  # pytype: disable=name-error
            print("Env closed")
        except NameError:
            print("Env wasn't created. Exiting coltra")
        except UnityEnvironmentException:
            print("Env already closed. Exiting coltra")
        except Exception:
            print("Unknown error when closing the env. Exiting coltra")
