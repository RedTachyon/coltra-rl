import os
from logging import ERROR
from pprint import pprint

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mlagents_envs.logging_util import set_log_level
from typarse import BaseParser
import wandb
from wandb.apis.public import Run

from coltra import CAgent, HomogeneousGroup, PPOCrowdTrainer, collect_renders
from coltra.envs import UnitySimpleCrowdEnv
from coltra.models import RelationModel

import torch

import seaborn as sns
import data_utils_retrain as du

CUDA = torch.cuda.is_available()
set_log_level(ERROR)


class Parser(BaseParser):
    env: str
    entity: str = "redtachyon"
    project: str = "crowdai-top5"
    run_id: str = "rvpfws09"
    worker_id: int = 0

    new_project: str = "crowdai-retrain"

    _abbrev = {
        "env": "e",
        "entity": "en",
        "project": "p",
        "run_id": "r",
        "worker_id": "w",
        "new_project": "np",
    }

    _help = {
        "env": "Path to the environment to train on",
        "entity": "Wandb entity to use",
        "project": "Wandb project to use",
        "run_id": "Wandb run id to use",
        "worker_id": "Unity worker id to use",
        "new_project": "Wandb project to use for new runs",
    }


def train_crowd(config: dict, args: Parser):

    env = UnitySimpleCrowdEnv.get_venv(
        file_name=args.env,
        workers=config["trainer"]["workers"],
        base_worker_id=args.worker_id,
        no_graphics=True,
    )

    obs_size = env.observation_space.shape[0]
    buffer_size = env.get_attr("obs_buffer_size")[0]
    action_size = env.action_space.shape[0]

    config["model"]["input_size"] = obs_size
    config["model"]["rel_input_size"] = buffer_size
    config["model"]["num_actions"] = action_size

    wandb.init(
        project=args.new_project,
        entity=args.entity,
        name=args.run_id,
        config=config,
        sync_tensorboard=True,
    )

    model = RelationModel(config["model"], action_space=env.action_space)
    agent = CAgent(model)
    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    trainer = PPOCrowdTrainer(
        agents=agents, env=env, config=config["trainer"], use_uuid=True
    )

    final_metrics = trainer.train(
        num_iterations=1000,
        disable_tqdm=False,
        save_path=trainer.path,
        collect_kwargs=config["environment"],
    )

    env.close()

    # EVALUATION
    env = UnitySimpleCrowdEnv(
        file_name=args.env,
        no_graphics=False,
        worker_id=args.worker_id,
        virtual_display=(1000, 1000),
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
    for i in range(6):
        idx = i % 3
        d = idx == 0
        if i == 3:
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

        print("Skipping video")
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


if __name__ == "__main__":
    args = Parser()
    api = wandb.Api()

    wandb_path = f"{args.entity}/{args.project}/{args.run_id}"
    print(f"Getting run {wandb_path}")
    run: Run = api.run(wandb_path)

    config: dict = run.config
    print("Found config: ")
    pprint(config)

    train_crowd(config, args)
