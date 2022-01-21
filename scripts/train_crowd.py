import itertools
import os
from typing import Optional, Type

import cv2
import torch
import wandb
import yaml
from matplotlib import pyplot as plt
from typarse import BaseParser

from coltra.agents import CAgent, Agent
from coltra.collectors import collect_renders
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.groups import HomogeneousGroup
from coltra.models import BaseModel
from coltra.models.mlp_models import MLPModel
from coltra.models.relational_models import RelationModel
from coltra.trainers import PPOCrowdTrainer
from coltra.models.raycast_models import LeeModel

import data_utils as du


class Parser(BaseParser):
    config: str = "configs/nocollision.yaml"
    iters: int = 500
    env: str
    name: str
    model_type: str = "relation"
    start_dir: Optional[str]
    start_idx: Optional[int] = -1

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env": "Path to the Unity environment binary",
        "name": "Name of the tb directory to store the logs",
        "model_type": "Type of the information that a model has access to",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env": "e",
        "name": "n",
        "model_type": "mt",
        "start_dir": "sd",
        "start_idx": "si",
    }


if __name__ == "__main__":
    try:
        CUDA = torch.cuda.is_available()

        args = Parser()

        assert args.model_type in ("blind", "relation"), ValueError(
            "Wrong model type passed."
        )

        with open(args.config, "r") as f:
            config = yaml.load(f.read(), yaml.Loader)

        trainer_config = config["trainer"]
        model_config = config["model"]
        env_config = config["environment"]

        trainer_config["tensorboard_name"] = args.name
        trainer_config["PPOConfig"]["use_gpu"] = CUDA

        workers = trainer_config["workers"]

        # Initialize the environment
        env = UnitySimpleCrowdEnv.get_venv(
            workers, file_name=args.env, no_graphics=True
        )
        env.reset(save_trajectory=0.0)

        # env.engine_channel.set_configuration_parameters(time_scale=100, width=100, height=100)

        # Initialize the agent
        obs_size = env.observation_space.shape[0]
        buffer_size = env.get_attr("obs_buffer_size")[0]
        action_size = env.action_space.shape[0]

        # sample_obs = next(iter(env.reset().values()))
        # obs_size = sample_obs.vector.shape[0]
        # ray_size = sample_obs.rays.shape[0] if sample_obs.rays is not None else None

        model_config["input_size"] = obs_size
        model_config["rel_input_size"] = buffer_size
        model_config["num_actions"] = action_size

        wandb.init(
            project="crowdai",
            entity="redtachyon",
            sync_tensorboard=True,
            config=config,
            name=args.name,
        )

        if args.model_type == "relation":
            model_cls = RelationModel
        else:
            model_cls = MLPModel

        agent: Agent
        if args.start_dir:
            agent = CAgent.load(args.start_dir, weight_idx=args.start_idx)
        else:
            model = model_cls(model_config, env.action_space)
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

        print("Training complete. Collecting renders")

        env = UnitySimpleCrowdEnv(
            file_name=args.env,
            virtual_display=(1600, 900),
            no_graphics=False,
        )
        env_config["evaluation_mode"] = 1.0

        os.mkdir(os.path.join(trainer.path, "trajectories"))
        os.mkdir(os.path.join(trainer.path, "videos"))
        os.mkdir(os.path.join(trainer.path, "images"))

        for i in range(6):
            d = i > 0
            print(f"Recording {'' if d else 'non'}deterministic video number {i}")
            trajectory_path = os.path.join(
                trainer.path,
                "trajectories",
                f"trajectory_{'det' if d else 'rnd'}_{i}.json",
            )

            dashboard_path = os.path.join(
                trainer.path,
                "images",
                f"dashboard_{'det' if d else 'rnd'}_{i}.png",
            )

            env.reset(save_path=trajectory_path, **env_config)

            renders, _ = collect_renders(
                agents,
                env,
                num_steps=trainer_config["steps"],
                disable_tqdm=False,
                env_kwargs=env_config,
                deterministic=d,
            )

            # Generate the dashboard
            trajectory = du.read_trajectory(trajectory_path)

            plt.clf()
            du.make_dashboard(trajectory, save_path=dashboard_path)

            # Upload to wandb

            wandb.log({"dashboard": wandb.Image(dashboard_path, caption=f"Dashboard {'det' if d else 'rng'} {i}")})

            frame_size = renders.shape[1:3]

            print("Recording a video")
            video_path = os.path.join(
                trainer.path, "videos", f"video_{'det' if d else 'rnd'}_{i}.webm"
            )
            out = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"VP90"), 30, frame_size[::-1]
            )
            for frame in renders[..., ::-1]:
                out.write(frame)

            out.release()

            print(f"Video saved to {video_path}")

            wandb.log({f"video_{'det' if d else 'rnd'}_{i}": wandb.Video(video_path)})

            print("Video uploaded to wandb")

            trajectory_artifact = wandb.Artifact(
                name=f"trajectory_{'det' if d else 'rnd'}_{i}", type="json"
            )
            trajectory_artifact.add_file(trajectory_path)
            wandb.log_artifact(trajectory_artifact)

    finally:
        print("Cleaning up")
        try:
            env.close()  # pytype: disable=name-error
            print("Env closed")
        except NameError:
            print("Env wasn't created. Exiting coltra")
