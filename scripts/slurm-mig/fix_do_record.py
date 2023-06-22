import os
import re
import yaml
import subprocess

import numpy as np
import cv2
from typarse import BaseParser
import wandb

import matplotlib.pyplot as plt
import seaborn as sns

import coltra.data_utils as du
from coltra import collect_renders
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.groups import HomogeneousGroup
from coltra.utils import find_free_worker

class Parser(BaseParser):
    run_path: str
    save_path: str
    env_path: str
    force: bool = False

    _help = {
        "run_path": "ID of the wandb run",
        "save_path": "Path to the logs",
        "env_path": "Path to the Unity environment binary",
        "force": "Whether to force the recording even if the video already exists",
    }

    _abbrev = {
        "run_path": "r",
        "save_path": "p",
        "env_path": "e",
        "force": "f",
    }


def has_video(run: wandb.apis.public.Run):
    return any(key.startswith('video') for key in run.summary.keys())

def copy_logs(path: str):
    src = f"jeanzay:/gpfswork/rech/axs/utu66tc/tb_logs/{path}"
    dest = f"/home/ariel/remote_logs/tb_logs/{path}"
    result = subprocess.run(['scp', '-r', src, dest], check=True, text=True, capture_output=True)
    print(f'Successfully copied {src} to {dest}')

# Sketch of the code to be generated
#  Iterate over all runs in the project
#   For each run, check if any key in `run.summary` starts with `video`
#    If it does, ignore the run unless `force` is set to True
#    If it doesn't, pull the config and the model, and use it to record a new video


if __name__ == "__main__":
    args = Parser()

    api = wandb.Api()

    run = api.run(args.run_path)

    # missing: /home/ariel/titan_logs/tb_logs/xu_2023-05-19_20-00-58_QAywRTce89t87gZpuV5TME/
    # print(f"Processing run {i+1}/{num_runs}")
    # if not args.force and has_video(run):
    #     print(f"Skipping {run.name}")
    #     continue

    wandb.init(id=run.id, project=args.run_path.split('/')[1], resume="allow", reinit=True)
    print(f"Recording {run.name}")


    copy_logs(args.save_path)


    print(f"Found path: {args.save_path}")

    full_path = f"/home/ariel/remote_logs/tb_logs/{args.save_path}"

    config_path = os.path.join(full_path, "full_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.load(f.read(), yaml.Loader)

    env_config = config["environment"]
    trainer_config = config["trainer"]

    agents = HomogeneousGroup.load(full_path, -1)

    worker_id = find_free_worker(500)
    env = UnitySimpleCrowdEnv(
        file_name=args.env_path,
        virtual_display=(800, 800),
        no_graphics=False,
        worker_id=worker_id,
        extra_params=env_config,
    )
    env.reset(**env_config)

    for dirname in ["trajectories", "videos", "images"]:
        try:
            os.mkdir(os.path.join(full_path, dirname))
        except FileExistsError:
            pass

    sns.set()
    UNIT_SIZE = 3
    plt.rcParams["figure.figsize"] = (8 * UNIT_SIZE, 4 * UNIT_SIZE)

    # mode = "circle"
    mode = env_config["initializer"]
    for i in range(3):
        idx = i % 3
        d = idx == 0

        trajectory_path = os.path.join(
            full_path,
            "trajectories",
            f"trajectory_{mode}_{'det' if d else 'rnd'}_{idx}.json",
        )

        dashboard_path = os.path.join(
            full_path,
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

        frame_size = renders.shape[1:3]

        print("Recording a video")
        video_path = os.path.join(
            full_path, "videos", f"video_{mode}_{'det' if d else 'rnd'}_{i}.webm"
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

