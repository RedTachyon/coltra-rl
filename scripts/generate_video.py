from typarse import BaseParser
import subprocess


class Parser(BaseParser):
    model_name: str
    target_dir: str = "videos"
    use_cached_model: bool = False
    vid_name: str = "video"

    _help = {
        "model_name": "Name of the model to use",
        "target_dir": "Path to the directory where to save the video",
        "use_cached_model": "Whether to use the cached model or not. Only use if you know it exists",
        "vid_name": "Name of the video",
    }

    _abbrev = {
        "model_name": "m",
        "target_dir": "t",
        "use_cached_model": "uc",
        "vid_name": "v",
    }


if __name__ == "__main__":
    args = Parser()

    model_name = args.model_name

    if not args.use_cached_model:
        print("Downloading model on remote")
        out1 = subprocess.run(
            f"scp -r jeanzay:/gpfswork/rech/nbk/utu66tc/tb_logs/{model_name} /home/ariel/temp/",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(out1.stdout.decode("utf-8"))
        print(out1.stderr.decode("utf-8"))

    print("Generating video on remote")
    out2 = subprocess.run(
        f"/home/ariel/anaconda3/envs/coltra/bin/python /home/ariel/projects/coltra-rl/scripts/enjoy_crowd.py -p /home/ariel/temp/{model_name} -e /home/ariel/projects/coltra-rl/builds/crowd-v6a/crowd.x86_64",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(out2.stdout.decode("utf-8"))
    print(out2.stderr.decode("utf-8"))

    out3 = subprocess.run(
        f"cp /home/ariel/projects/coltra-rl/scripts/temp/video.webm {args.target_dir}/{args.vid_name}.webm",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(out3.stdout.decode("utf-8"))
    print(out3.stderr.decode("utf-8"))

    print("Done")
