import subprocess
import json
from typing import Optional

from typarse import BaseParser


class Parser(BaseParser):
    dry: bool
    env_id: str
    observer: str
    dynamics: str
    model: str

    _help = {
        "dry": "Dry run, do not submit the job",
        "env_id": "Environment name to use, from the dictionary defined in the code",
        "observer": "Observer to use",
        "dynamics": "Dynamics to use",
        "model": "Model to use",
    }

    _abbrev = {"dry": "d", "env_id": "e", "observer": "o", "dynamics": "dy", "model": "m"}


def format_config(config: dict) -> str:
    base_json = json.dumps(config)
    return base_json.replace(" ", "").replace('"', '\\"')


if __name__ == "__main__":

    args = Parser()

    observers = {
        "abs": "Absolute",
        "rel": "Relative",
        "ego": "Egocentric",
    }
    dynamics = {
        "carvel": "CartesianVelocity",
        "caracc": "CartesianAcceleration",
        "polvel": "PolarVelocity",
        "polacc": "PolarAcceleration",
    }
    models = {
        "ray": {
            "model_type": "ray",
            "environment.destroy_raycasts": False,
            "environment.ray_agent_vision": True,
        },
        "relation": {
            "model_type": "relation",
            "environment.destroy_raycasts": True,
        },
        "rayrelation": {
            "model_type": "rayrelation",
            "environment.destroy_raycasts": False,
            "environment.ray_agent_vision": False,
        },
    }

    all_envs = {
        "circle30": {
            "environment.mode": "Circle",
            "environment.num_agents": 30,
            "environment.enable_obstacles": False,
            "environment.spawn_scale": 6,
            "trainer.workers": 1,
        },
        "circle12": {
            "environment.mode": "Circle",
            "environment.num_agents": 12,
            "environment.enable_obstacles": False,
            "environment.spawn_scale": 6,
            "trainer.workers": 2,
        },
        "crossway50": {
            "environment.mode": "Crossway",
            "environment.num_agents": 50,
            "environment.enable_obstacles": True,
            "trainer.workers": 1,
        },
        "corridor50": {
            "environment.mode": "Corridor",
            "environment.num_agents": 50,
            "environment.enable_obstacles": True,
            "trainer.workers": 1,
        },
        "random20": {
            "environment.mode": "Random",
            "environment.num_agents": 20,
            "environment.enable_obstacles": False,
            "trainer.workers": 2,
        },
    }

    # idx = args.env_id
    # if idx == "all":
    #     environment = all_envs
    # else:
    #     environment = {idx: all_envs[idx]}

    num_runs = 8

    observer = observers[args.observer]
    dynamics = dynamics[args.dynamics]
    model = models[args.model]
    env = all_envs[args.env_id]

    # total = len(observers) * len(dynamics) * len(models) * len(environment)
    i = 0
    total = 11

    for i in range(total):

        project_name = f"DCSRL-jz-exponent-{args.env_id}"
        # project_name = f"DCSRL-jz-timing"
        extra_config = {**model, **env}
        extra_config["environment.comfort_speed_exponent"] = 1 + i / 10
        cmd = [
            "sbatch",
            f"--export=ALL,NUM_RUNS={num_runs},OBSERVER={observer},DYNAMICS={dynamics},MODEL={args.model},PROJECTNAME={project_name},EXTRA_CONFIG=\"'{format_config(extra_config)}'\"",
            "crowd.sbatch",
            # "echo.sbatch"
        ]
        # print(" ".join(cmd))
        print(f"{i}/{total} Running {' '.join(cmd)}")
        cmd = " ".join(cmd)
        if args.dry:
            print(cmd)
        else:
            out = subprocess.run(cmd, shell=True, capture_output=True)
            print(out.stdout.decode("utf-8"))
        i += 1
