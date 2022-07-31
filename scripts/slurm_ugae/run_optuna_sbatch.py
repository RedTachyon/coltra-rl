import subprocess
import json
from typing import Optional

from typarse import BaseParser


class Parser(BaseParser):
    dry: bool
    config: str = "optuna_config.yaml"
    optuna: str = "humanoid_standup"
    wandb_project: str = "UGAE-optuna-humanoid-standup"
    streams: int = 10
    jobs_per_stream: int = 10

    _help = {
        "dry": "Dry run, do not submit the job",
        "config": "Path to the config file",
        "optuna": "Name of the optuna study",
        "wandb_project": "Name of the wandb project",
        "streams": "Number of streams",
        "jobs_per_stream": "Number of jobs per stream",
    }

    _abbrev = {
        "dry": "d",
        "config": "c",
        "optuna": "o",
        "wandb_project": "wp",
        "streams": "s",
        "jobs_per_stream": "j",
    }


def format_config(config: dict) -> str:
    base_json = json.dumps(config)
    return base_json.replace(" ", "").replace('"', '\\"')


if __name__ == "__main__":

    args = Parser()


    for i in range(args.streams):
        cmd = f"sbatch --export=ALL,CONFIG={args.config},OPTUNA={args.optuna},WANDB={args.wandb_project} optuna_ugae.sbatch"
        print(f"Executing {cmd}")
        if not args.dry:
            out = subprocess.run(cmd, shell=True, capture_output=True)
            output = out.stdout.decode("utf-8")
            job_id = output.split(' ')[-1]
            print(f"Job {job_id} submitted")
            print(output)
        else:
            job_id = 0

        for j in range(1, args.jobs_per_stream):
            cmd = f"sbatch --dependency=afterok:{job_id} --export=ALL,CONFIG={args.config},OPTUNA={args.optuna},WANDB={args.wandb_project} optuna_ugae.sbatch"
            print(f"Executing {cmd}")
            if not args.dry:
                out = subprocess.run(cmd, shell=True, capture_output=True)
                output = out.stdout.decode("utf-8")
                job_id = output.split(' ')[-1]
                print(f"Job {job_id} submitted")
                print(output)
