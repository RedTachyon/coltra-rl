import os
from typing import Optional

import numpy as np
import yaml
import torch
import wandb
from mlagents_envs.exception import UnityEnvironmentException
from typarse import BaseParser
from coltra.agents import CAgent
from coltra.groups import HomogeneousGroup
from coltra.trainers import PPOCrowdTrainer
from coltra.training_utils import evaluate
from coltra.envs.unity_envs import UnitySimpleCrowdEnv


class Parser(BaseParser):
    agent_path: str
    env: str
    project: Optional[str] = None

    _help = {
        "agent_path": "Path to the trained agent",
        "env": "Path to the Unity environment binary",
        "project": "Name of wandb project",
    }

    _abbrev = {
        "agent_path": "a",
        "env": "e",
        "project": "p",
    }


if __name__ == "__main__":
    try:
        CUDA = torch.cuda.is_available()

        args = Parser()

        # We're assuming the config for the evaluation is at the same location as the trained agent
        with open(os.path.join(args.agent_path, "full_config.yaml"), "r") as f:
            config = yaml.load(f.read(), yaml.Loader)

        trainer_config = config["trainer"]
        model_config = config["model"]
        env_config = config["environment"]
        model_type = config["model_type"]

        # Initialize the environment
        env = UnitySimpleCrowdEnv(
            file_name=args.env,
            no_graphics=True,
            extra_params=env_config
        )

        env.reset()

        # Initialize the agent
        wandb.init(
            project="crowdai" if args.project is None else args.project,
            sync_tensorboard=True,
            config=config,
            name=trainer_config["tensorboard_name"],
        )

        agents = HomogeneousGroup.load(args.agent_path, weight_idx=-1)

        if CUDA:
            agents.cuda()

        print("Evaluating...")

        for num_agents in range(1, 51):
            env_config["num_agents"] = num_agents

            env.reset(**env_config)

            performances, energies = evaluate(env, agents, 10, disable_tqdm=False)

            wandb.log(
                {
                    f"eval/num_agents_{num_agents}/mean_episode_reward": np.mean(performances),
                    f"eval/num_agents_{num_agents}/std_episode_reward": np.std(performances),
                    f"eval/num_agents_{num_agents}/mean_episode_energy": np.mean(energies),
                    f"eval/num_agents_{num_agents}/std_episode_energy": np.std(energies),
                },
                commit=False,
            )

        wandb.log({})
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
