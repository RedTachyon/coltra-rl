import os
import subprocess
import json
from typing import Optional

from typarse import BaseParser


class Parser(BaseParser):
    dry: bool
    config_path: str
    project_name: str

    _help = {
        "dry": "Dry run, do not submit the job",
        "config_dir": "Path to the config file",
        "project_name": "Name of the project",
    }

    _abbrev = {"dry": "d", "config_path": "c", "project_name": "p"}


if __name__ == "__main__":

    args = Parser()

    i = 0

    for root, dirs, files in os.walk(args.config_path):
        files = sorted(files)
        total = len(files)
        for file in files:
            i += 1

            if not file.endswith(".yaml"): continue
            full_path = os.path.join(root, file)
            cmd = [
                "sbatch",
                f"--export=ALL,CONFIG={full_path},PROJECTNAME={args.project_name},NUM_RUNS=8",
                "crowd.sbatch",
                ]

            cmd = " ".join(cmd)
            if args.dry:
                print(cmd)
            else:
                out = subprocess.run(cmd, shell=True, capture_output=True)
                print(out.stdout.decode("utf-8"))
                if out.stderr: print(out.stderr.decode("utf-8"))

            print(f"{i}/{total} Running {cmd}")
            print()


    print(f"Submitted {i} jobs")
