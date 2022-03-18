import optuna
from typarse import BaseParser

from optuna_crowd import objective

"""
Plan: use prophet
Run 8 processes in tmux tiles, each will do 10 runs of 2 hparam settings

Trials to use:
fast:
47, 70, 99, 46, 52, 51, 50, 72
slow:
125, 107, 114, 132, 119, 22, 26, 36
"""
class Parser(BaseParser):
    env: str = "/home/ariel/projects/coltra-rl/builds/crowd-v5/crowd.x86_64"
    worker_id: int = 0
    n_trials: int = 10
    optuna_name: str = "egocentric"
    indices: list[str] = []

    _abbrev = {"env": "e", "worker_id": "w", "n_trials": "n", "optuna_name": "o", "indices": "i"}

    _help = {
        "env": "Path to the environment",
        "worker_id": "Worker ID to start from",
        "n_trials": "Number of trials",
        "optuna_name": "Name of the optuna study",
        "indices": "List of indices to train on",
    }


if __name__ == '__main__':
    args = Parser()

    print(args.indices)

    study = optuna.load_study(storage=f"sqlite:///{args.optuna_name}.db", study_name=args.optuna_name)
    trials = study.trials

    for idx in args.indices:
        trial = trials[int(idx)]
        print(f"Trial {idx}")
        for i in range(args.n_trials):
            print(f"Run {i}")
            objective(trial, args.worker_id, args.env)
