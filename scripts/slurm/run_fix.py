import subprocess
import json
from typarse import BaseParser


class Parser(BaseParser):
    dry: bool

    _help = {
        "dry": "Dry run, do not submit the job",
    }
    _abbrev = {
        "dry": "d",
    }


def format_config(config: dict) -> str:
    base_json = json.dumps(config)
    return base_json.replace(' ', '').replace('"', '\\"')


if __name__ == "__main__":

    args = Parser()

    observers = ["Absolute", "Relative", "Egocentric"]
    dynamics = ["CartesianVelocity", "CartesianAcceleration", "PolarVelocity", "PolarAcceleration"]
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

    environment = {
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


    fixes = [
        ('Egocentric', 'CartesianAcceleration', 'ray', 1),
        ('Relative', 'PolarAcceleration', 'relation', 2),
        ('Egocentric', 'CartesianVelocity', 'ray', 1),
        ('Egocentric', 'CartesianVelocity', 'relation', 5),
        ('Relative', 'PolarAcceleration', 'ray', 3),
        ('Egocentric', 'PolarAcceleration', 'ray', 1),
        ('Egocentric', 'PolarVelocity', 'relation', 3),
        ('Egocentric', 'PolarAcceleration', 'relation', 1),
        ('Absolute', 'PolarAcceleration', 'rayrelation', 1),
        ('Relative', 'CartesianVelocity', 'relation', 1),
        ('Relative', 'CartesianVelocity', 'ray', 1),
        ('Absolute', 'PolarVelocity', 'ray', 1),
        ('Absolute', 'PolarVelocity', 'relation', 2),
        ('Absolute', 'CartesianVelocity', 'rayrelation', 1),
        ('Relative', 'PolarVelocity', 'ray', 3),
        ('Relative', 'CartesianAcceleration', 'relation', 1),
        ('Absolute', 'PolarVelocity', 'rayrelation', 1),
        ('Relative', 'PolarVelocity', 'relation', 1),
        ('Absolute', 'PolarAcceleration', 'relation', 3),
        ('Absolute', 'CartesianAcceleration', 'ray', 1)
    ]

    env = "circle12"

    total = len(fixes)
    i = 0

    for (observer, dyn, model, num_runs) in fixes:

        project_name = f"DCSRL-jz-{env}"
        # project_name = f"DCSRL-jz-timing"
        extra_config = {**models[model], **environment[env]}
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
