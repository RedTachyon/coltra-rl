import optuna
from typarse import BaseParser
import os
from os.path import exists


class Parser(BaseParser):
    name: str = "optuna"
    force: bool = False

    _abbrev = {"name": "n", "force": "f"}

    _help = {
        "name": "Name of the study",
        "force": "Force overwriting of existing study",
    }


if __name__ == "__main__":
    args = Parser()

    if exists(f"./{args.name}.db"):
        if args.force:
            os.remove(f"./{args.name}.db")
        else:
            raise Exception(
                f"Study {args.name} already exists. Use -f to force overwriting."
            )

    study = optuna.create_study(
        storage=f"sqlite:///{args.name}.db",
        study_name=args.name,
        pruner=None,
        direction="maximize",
    )
