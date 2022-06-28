from typarse import BaseParser
import subprocess


class Parser(BaseParser):
    model_name: str
    target_dir: str = "../models"

    _help = {
        "model_name": "Name of the model to download",
        "target_dir": "Path to the directory where to save the model",
    }

    _abbrev = {
        "model_name": "m",
        "target_dir": "t",
    }


if __name__ == "__main__":
    args = Parser()

    model_name = args.model_name

    out1 = subprocess.run(
        f"ssh -t ariel.geovic 'scp -r jeanzay:/gpfswork/rech/nbk/utu66tc/tb_logs/{model_name} temp/'",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(out1.stdout.decode("utf-8"))

    out2 = subprocess.run(
        f"scp -r ariel.geovic:temp/{model_name} {args.target_dir}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(out2.stdout.decode("utf-8"))
