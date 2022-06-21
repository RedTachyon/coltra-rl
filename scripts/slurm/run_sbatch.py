import subprocess
import json
from typing import Optional

from typarse import BaseParser


class Parser(BaseParser):
    dry: bool
    env_id: str = "circle30"
    remove_globals: bool = False

    _help = {
        "dry": "Dry run, do not submit the job",
        "env_id": "Environment name to use, from the dictionary defined in the code",
        "remove_globals": "Whether to remove global observations from the environment"
    }

    _abbrev = {"dry": "d", "env_id": "e", "remove_globals": "rg"}


def format_config(config: dict) -> str:
    base_json = json.dumps(config)
    return base_json.replace(" ", "").replace('"', '\\"')


if __name__ == "__main__":

    args = Parser()

    observers = ["Absolute", "Relative", "Egocentric"]
    dynamics = [
        "CartesianVelocity",
        "CartesianAcceleration",
        "PolarVelocity",
        "PolarAcceleration",
    ]
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

    idx = args.env_id
    if idx == "all":
        environment = all_envs
    else:
        environment = {idx: all_envs[idx]}

    num_runs = 8

    total = len(observers) * len(dynamics) * len(models) * len(environment)
    i = 0

    for observer in observers:
        for dyn in dynamics:
            for model in models:
                for env in environment:
                    project_name = f"DCSRL-jz-{env}"
                    # project_name = f"DCSRL-jz-timing"
                    extra_config = {**models[model], **environment[env]}
                    if args.remove_globals:
                        extra_config["__remove_globals__"] = True
                    cmd = [
                        "sbatch",
                        f"--export=ALL,NUM_RUNS={num_runs},OBSERVER={observer},DYNAMICS={dyn},MODEL={model},PROJECTNAME={project_name},EXTRA_CONFIG=\"'{format_config(extra_config)}'\"",
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
