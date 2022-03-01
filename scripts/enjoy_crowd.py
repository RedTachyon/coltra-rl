from logging import ERROR
from typing import Optional

import torch
import wandb
import yaml
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.logging_util import set_log_level
from tqdm import trange
from typarse import BaseParser

from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.groups import HomogeneousGroup

set_log_level(ERROR)


class Parser(BaseParser):
    config: str = "configs/nocollision.yaml"
    iters: int = 500
    env: Optional[str] = None
    dynamics: Optional[str] = None
    observer: Optional[str] = None
    start_dir: Optional[str]
    start_idx: Optional[int] = -1

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env": "Path to the Unity environment binary",
        "dynamics": "Type of dynamics to use",
        "observer": "Type of observer to use",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env": "e",
        "dynamics": "d",
        "observer": "o",
        "start_dir": "sd",
        "start_idx": "si",
    }


if __name__ == "__main__":
    try:
        CUDA = torch.cuda.is_available()

        args = Parser()

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
            assert args.observer in ("Absolute", "Relative", "RotRelative"), ValueError(
                "Wrong observer type passed."
            )
            config["environment"]["observer"] = args.observer

        trainer_config = config["trainer"]
        model_config = config["model"]
        env_config = config["environment"]

        workers = trainer_config["workers"]

        # Initialize the environment
        env = UnitySimpleCrowdEnv(file_name=args.env, no_graphics=False, worker_id=0)
        env.reset(save_trajectory=0.0)

        group = HomogeneousGroup.load(args.start_dir, weight_idx=args.start_idx)

        if CUDA:
            group.cuda()

        env_config["evaluation_mode"] = 1.0

        obs = env.reset()
        for _ in trange(args.iters):
            action, _, _ = group.act(obs, deterministic=False)
            obs, _, _, _ = env.step(action)

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
