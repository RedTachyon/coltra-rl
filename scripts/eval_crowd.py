import os
from typing import Optional

import wandb
import yaml
import numpy as np
import torch
from mlagents_envs.exception import UnityEnvironmentException
from typarse import BaseParser
from coltra.groups import HomogeneousGroup
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from tqdm import trange


class Parser(BaseParser):
    agent_path: str
    env: str
    project: Optional[str] = None

    _help = {
        "env": "Path to the Unity environment binary",
        "project": "Name of wandb project",
        "agent_path": "Path to the trained agent",
    }

    _abbrev = {
        "env": "e",
        "project": "p",
        "agent_path": "a",
    }


def evaluate(
        env: UnitySimpleCrowdEnv,
        group: HomogeneousGroup,
        n_episodes: int = 5,
        n_steps: int = 200,
        disable_tqdm: bool = False,
        reset_kwargs: Optional[dict] = None,
) -> dict:
    if reset_kwargs is None:
        reset_kwargs = {}

    metrics = {
        "returns": [],
        "e_energy": [],
        "e_energy_complex": [],
        "e_energy_plus": [],
        "e_energy_complex_Plus": []
    }
    current_return = 0.0
    for ep in trange(n_episodes, disable=disable_tqdm):
        obs = env.reset(**reset_kwargs)
        for t in range(n_steps):
            action, _, _ = group.act(obs)
            obs, reward, done, info = env.step(action)

            mean_reward = np.mean(list(reward.values()))
            current_return += mean_reward
            if all(done.values()):
                metrics["returns"].append(current_return)
                for key in metrics.keys():
                    if key != "returns":
                        metrics[key].append(info[key])
                current_return = 0.0
                break

    for key, values in metrics.items():
        metrics[key] = np.array(values)

    return metrics


if __name__ == "__main__":
    try:
        CUDA = torch.cuda.is_available()

        args = Parser()

        with open(os.path.join(args.agent_path, "full_config.yaml"), "r") as f:
            config = yaml.load(f.read(), yaml.Loader)

        trainer_config = config["trainer"]
        model_config = config["model"]
        env_config = config["environment"]
        model_type = config["model_type"]

        wandb.init(
            project="crowdai" if args.project is None else args.project,
            sync_tensorboard=True,
            config=config,
            name="eval_" + trainer_config["tensorboard_name"],
        )

        agents = HomogeneousGroup.load(args.agent_path, weight_idx=-1)

        if CUDA:
            agents.cuda()

        print("Evaluating...")

        scenarios = {
            "Circle": [1, 2, 5, 10, 20, 40, 50, 70],
            "Crossway": [2, 8, 18, 32, 50, 72],
            "Corridor": [2, 8, 18, 32, 50, 72],
            "Choke": [1, 2, 5, 10, 20],
            "Car": [1, 2, 5, 10],
        }

        for scenario, nums_agents in scenarios.items():
            for num_agents in nums_agents:
                env_config["num_agents"] = num_agents
                env_config["initializer"] = scenario

                env = UnitySimpleCrowdEnv(
                    file_name=args.env,
                    no_graphics=True,
                    extra_params=env_config
                )

                env.reset()

                metrics = evaluate(env, agents, n_episodes=5, n_steps=200, disable_tqdm=False)

                for metric, values in metrics.items():
                    wandb.log(
                        {
                            f"eval/{scenario}/{metric}": np.mean(values),
                            f"eval/{scenario}/{metric}_std": np.std(values),
                        },
                        step=num_agents,
                    )

                env.close()

        wandb.finish()

    finally:
        print("Cleaning up")
        try:
            env.close()
            print("Env closed")
        except NameError:
            print("Env wasn't created. Exiting coltra")
        except UnityEnvironmentException:
            print("Env already closed. Exiting coltra")
        except Exception:
            print("Unknown error when closing the env. Exiting coltra")
