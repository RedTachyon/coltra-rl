import subprocess
import json
from typing import Optional

from typarse import BaseParser


class Parser(BaseParser):
    dry: bool
    env_id: str

    _help = {
        "dry": "Dry run, do not submit the job",
        "env_id": "Gym environment name to use",
    }

    _abbrev = {
        "dry": "d",
        "env_id": "e",
    }


def format_config(config: dict) -> str:
    base_json = json.dumps(config)
    return base_json.replace(" ", "").replace('"', '\\"')


if __name__ == "__main__":

    args = Parser()

    num_runs = 8

    base_config = {}

    if args.env_id == "Humanoid-v4":
        gammas = [0.95, 0.99]
        etas = [0.0, 0.3, 0.5, 0.8, 1.0]
        lambdas = [0.0, 0.3, 0.5, 0.8, 0.9, 0.95, 1.0]
    else:
        gammas = [0.98, 0.99]
        etas = [0.0, 0.3, 0.5, 0.8, 1.0]
        lambdas = [0.0, 0.3, 0.5, 0.8, 0.9, 0.95, 1.0]

    configs = [(gamma, eta, lambda_) for gamma in gammas for eta in etas for lambda_ in lambdas]

    total = len(configs)

    for i, (gamma, eta, lam) in enumerate(configs):

        project_name = f"UGAE-jz-{args.env_id}-new"
        extra_config = {"trainer.PPOConfig.eta": eta, "trainer.PPOConfig.gae_lambda": lam, "trainer.PPOConfig.gamma": gamma}
        cmd = [
            "sbatch",
            f"--export=ALL,ENV_ID={args.env_id},NUM_RUNS={num_runs},PROJECTNAME={project_name},EXTRA_CONFIG=\"'{format_config(extra_config)}'\"",
            "ugae.sbatch",
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
        # i += 1
