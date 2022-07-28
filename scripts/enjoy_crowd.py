import json
import json
import os
from logging import ERROR
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.logging_util import set_log_level
from typarse import BaseParser

import coltra.utils
from coltra.collectors import collect_renders
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.groups import HomogeneousGroup
from coltra.utils import find_free_worker

set_log_level(ERROR)


class Parser(BaseParser):
    path: str = "../models/corridor"
    env: str
    base_video_dir: str = "temp"
    extra_config: Optional[str] = None
    deterministic: bool = False

    _help = {
        "path": "Path to the saved agent",
        "env": "Path to the Unity environment binary",
        "base_video_dir": "Base directory for the video files",
        "extra_config": "Extra config items to override the config file. Should be passed in a json format.",
        "deterministic": "Whether to use deterministic action sampling or not",
    }

    _abbrev = {
        "path": "p",
        "env": "e",
        "base_video_dir": "vd",
        "extra_config": "ec",
        "deterministic": "d",
    }


if __name__ == "__main__":
    try:
        CUDA = torch.cuda.is_available()

        args = Parser()

        with open(os.path.join(args.path, "full_config.yaml"), "r") as f:
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
        ), ValueError(f"Wrong model type {model_type} in the config.")

        trainer_config["PPOConfig"]["use_gpu"] = CUDA

        agents = HomogeneousGroup.load(args.path, weight_idx=-1)

        if CUDA:
            agents.cuda()

        # print("Evaluating...")
        # performances, energies = evaluate(env, agents, 10, disable_tqdm=False)

        # print("Training complete. Evaluation starting.")

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

        renders, returns = collect_renders(
            agents,
            env,
            num_steps=trainer_config["steps"],
            disable_tqdm=False,
            env_kwargs=env_config,
            deterministic=args.deterministic,
        )

        print(f"Mean return: {np.mean(returns)}")

        frame_size = renders.shape[1:3]

        print("Recording a video")
        video_path = os.path.join(args.base_video_dir, f"video.webm")
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"VP90"), 24, frame_size[::-1]
        )
        for frame in renders[..., ::-1]:
            out.write(frame)

        out.release()

        env.close()

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
