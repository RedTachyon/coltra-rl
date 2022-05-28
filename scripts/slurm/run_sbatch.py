import subprocess
import json

def format_config(config: dict) -> str:
    base_json = json.dumps(config)
    return base_json.replace(' ', '').replace('"', '\\"')


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

FLAG = 0

if FLAG == 1:
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
else:
    environment = {
        "crossway50": {
            "environment.mode": "Crossway",
            "environment.num_agents": 50,
            "environment.enable_obstacles": True,
            "trainer.workers": 1,
        },
    }

total = len(observers) * len(dynamics) * len(models) * len(environment)
i = 0

for observer in observers:
    for dyn in dynamics:
        for model in models:
            for env in environment:
                project_name = f"DCSRL-jz-{env}"
                # project_name = f"DCSRL-jz-timing"
                extra_config = {**models[model], **environment[env]}
                cmd = [
                    "sbatch",
                    f"--export=ALL,OBSERVER={observer},DYNAMICS={dyn},MODEL={model},PROJECTNAME={project_name},EXTRA_CONFIG=\"'{format_config(extra_config)}'\"",
                    "crowd.sbatch",
                    # "echo.sbatch"
                ]
                # print(" ".join(cmd))
                print(f"{i}/{total} Running {' '.join(cmd)}")

                out = subprocess.run(cmd, capture_output=True)
                print(out.stdout.decode("utf-8"))
                i += 1

