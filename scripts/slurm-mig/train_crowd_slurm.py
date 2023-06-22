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
from coltra.models.attention_models import AttentionModel
from coltra.models.mlp_models import MLPModel, RayMLPModel
from coltra.models.relational_models import RelationModel, RayRelationModel
from coltra.trainers import PPOCrowdTrainer

import coltra.data_utils as du
from coltra.training_utils import evaluate
from coltra.utils import find_free_worker

set_log_level(ERROR)


class Parser(BaseParser):
    config: str = "v7-configs/crowd_config.yaml"
    iters: int = 500
    env: str
    name: Optional[str] = None
    worker_id: Optional[int] = None
    project: Optional[str] = None
    extra_config: Optional[str] = None
    run_eval: bool = False

    _help = {
        "config": "Config file for coltra. If preceded by 'wandb:', will use wandb to fetch config.",
        "iters": "Number of coltra iterations",
        "env": "Path to the Unity environment binary",
        "name": "Name of the tb directory to store the logs",
        "worker_id": "Worker id",
        "project": "Name of wandb project",
        "extra_config": "Extra config items to override the config file. Should be passed in a json format.",
        "run_eval": "Whether to run evaluation after training",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env": "e",
        "name": "n",
        "worker_id": "w",
        "project": "p",
        "extra_config": "ec",
        "run_eval": "re",
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
        curriculum = config.get("env_curriculum", None)

        assert model_type in (
            "blind",
            "relation",
            "ray",
            "rayrelation",
            "attention",
        ), ValueError(f"Wrong model type {model_type} in the config.")

        if args.name:
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

        env.reset(**env_config)

        # env.engine_channel.set_configuration_parameters(time_scale=100, width=100, height=100)

        # Initialize the agent

        wandb.init(
            project="crowdai" if args.project is None else args.project,
            sync_tensorboard=True,
            config=config,
            name=trainer_config["tensorboard_name"],
        )

        if model_type == "relation":
            model_cls = RelationModel
        elif model_type == "attention":
            model_cls = AttentionModel
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

        trainer = PPOCrowdTrainer(agents, env, trainer_config, use_uuid=True, save_path="/gpfswork/rech/axs/utu66tc")
        with open(os.path.join(trainer.path, "full_config.yaml"), "w") as f:
            yaml.dump(config, f)

        path_artifact = wandb.Artifact(
            "save_path", type="save_path", metadata={"save_path": trainer.path}
        )
        wandb.log_artifact(path_artifact)
        trainer.train(
            args.iters,
            disable_tqdm=False,
            save_path=trainer.path,
            collect_kwargs=env_config,
            curriculum=curriculum
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