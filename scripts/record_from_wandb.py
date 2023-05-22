import os
import sys
import subprocess

from typarse import BaseParser
import wandb


class Parser(BaseParser):
    project_name: str
    env_path: str
    skip: int = 0
    force: bool = False

    _help = {
        "project_name": "Name of the wandb project",
        "env_path": "Path to the Unity environment binary",
        "skip": "Number of runs to skip",
        "force": "Whether to force the recording even if the video already exists",
    }

    _abbrev = {
        "project_name": "p",
        "env_path": "e",
        "skip": "s",
        "force": "f",
    }


def has_video(run: wandb.apis.public.Run):
    return any(key.startswith('video') for key in run.summary.keys())


# Sketch of the code to be generated
#  Iterate over all runs in the project
#   For each run, check if any key in `run.summary` starts with `video`
#    If it does, ignore the run unless `force` is set to True
#    If it doesn't, pull the config and the model, and use it to record a new video


if __name__ == "__main__":
    args = Parser()

    api = wandb.Api()
    runs = list(api.runs(args.project_name))
    runs = runs[args.skip:]

    num_runs = len(runs)

    # missing: /home/ariel/titan_logs/tb_logs/xu_2023-05-19_20-00-58_QAywRTce89t87gZpuV5TME/
    for i, run in enumerate(runs):
        print(f"Processing run {i + 1}/{num_runs}")
        if not args.force and has_video(run):
            print(f"Skipping {run.name}")
            continue

        run_path = '/'.join(run.path)

        # Run `do_record_from_wandb.py` as a separate process

        process = subprocess.Popen(
            [sys.executable, "do_record_from_wandb.py", "--run_path", run_path, "--env_path", args.env_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        for line in iter(process.stdout.readline, b''):
            print(line.decode().strip())

        process.stdout.close()
        process.wait()

        stderr = process.stderr.read().decode()
        if stderr:
            print("Error: ", stderr)
        process.stderr.close()
