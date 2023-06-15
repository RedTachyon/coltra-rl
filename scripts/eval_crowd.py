import os
import sys
from typing import Optional

import cv2
import wandb
import yaml
import numpy as np
import torch
from matplotlib import pyplot as plt
from mlagents_envs.exception import UnityEnvironmentException
from typarse import BaseParser
from tqdm import trange
import seaborn as sns

from coltra import collect_renders
from coltra.groups import HomogeneousGroup
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
import coltra.data_utils as du
from coltra.utils import find_free_worker


class Parser(BaseParser):
    agent_path: str
    env: str
    project: Optional[str] = None
    record: bool = False

    _help = {
        "env": "Path to the Unity environment binary",
        "project": "Name of wandb project",
        "agent_path": "Path to the trained agent",
        "record": "Whether to record the evaluation to wandb",
    }

    _abbrev = {
        "env": "e",
        "project": "p",
        "agent_path": "a",
        "record": "r",
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
        "e_energy_complex_plus": []
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

        env = UnitySimpleCrowdEnv(
            file_name=args.env,
            no_graphics=True,
            extra_params=env_config
        )

        env.set_timescale(100.)

        for scenario, nums_agents in scenarios.items():
            print(f"Evaluating scenario {scenario}")
            for num_agents in nums_agents:
                print(f"Num agents: {num_agents}")
                env_config["num_agents"] = num_agents
                env_config["initializer"] = scenario

                env.reset(**env_config)

                metrics = evaluate(env, agents, n_episodes=5, n_steps=200, disable_tqdm=False, reset_kwargs=env_config)

                for metric, values in metrics.items():
                    wandb.log(
                        {
                            f"{scenario}/{metric}": np.mean(values),
                            # f"eval/{scenario}/{metric}_std": np.std(values),
                            "global_step": num_agents,
                        },
                    )

        env.close()

        # Record videos

        if not args.record:
            sys.exit(0)

        worker_id = find_free_worker(500)
        render_env = UnitySimpleCrowdEnv(
            file_name=args.env,
            virtual_display=(800, 800),
            no_graphics=False,
            worker_id=worker_id,
            extra_params=env_config,
        )

        for scenario, nums_agents in scenarios.items():
            print(f"Rendering scenario {scenario}")
            for num_agents in nums_agents:
                print(f"Num agents: {num_agents}")
                env_config["num_agents"] = num_agents
                env_config["initializer"] = scenario

                env.reset(**env_config)

                os.mkdir(os.path.join(args.agent_path, "tmp"))
                os.mkdir(os.path.join(args.agent_path, "tmp", "eval_trajectories"))
                os.mkdir(os.path.join(args.agent_path, "tmp", "eval_videos"))
                os.mkdir(os.path.join(args.agent_path, "tmp", "eval_images"))

                sns.set()
                UNIT_SIZE = 3
                plt.rcParams["figure.figsize"] = (8 * UNIT_SIZE, 4 * UNIT_SIZE)

                # mode = "circle"
                mode = env_config["initializer"]
                for i in range(3):
                    idx = i % 3
                    d = idx == 0

                    trajectory_path = os.path.join(
                        args.agent_path,
                        "trajectories",
                        f"trajectory_{mode}_{'det' if d else 'rnd'}_{idx}.json",
                    )

                    dashboard_path = os.path.join(
                        args.agent_path,
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
                            f"{scenario}{num_agents}/dashboard_{idx}": wandb.Image(
                                dashboard_path,
                                caption=f"Dashboard {mode} {'det' if d else 'rng'} {i}",
                            )
                        },
                        commit=False,
                    )

                    frame_size = renders.shape[1:3]

                    print("Recording a video")
                    video_path = os.path.join(
                        args.agent_path, "videos", f"video_{mode}_{'det' if d else 'rnd'}_{i}.webm"
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
                            f"{scenario}{num_agents}/video_{mode}_{'det' if d else 'rnd'}_{idx}": wandb.Video(
                                video_path
                            )
                        },
                        commit=False,
                    )

                    print("Video uploaded to wandb")

                    trajectory_artifact = wandb.Artifact(
                        name=f"{scenario}{num_agents}/trajectory_{mode}_{'det' if d else 'rnd'}_{idx}", type="json"
                    )
                    trajectory_artifact.add_file(trajectory_path)
                    wandb.log_artifact(trajectory_artifact)

                wandb.log({}, commit=True)



        wandb.finish()
        os.rmdir(os.path.join(args.agent_path, "tmp"))

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
