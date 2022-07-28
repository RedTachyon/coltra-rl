from typarse import BaseParser
import subprocess


class Parser(BaseParser):
    model_name: str
    target_dir: str = "./videos"
    name: str = "video"
    use_cached_model: bool = False
    deterministic: bool = False

    _help = {
        "model_name": "Name of the model to use",
        "target_dir": "Path to the directory where to save the video",
        "name": "Name of the video",
        "use_cached_model": "Whether to use the cached model or not. Only use if you know it exists",
        "deterministic": "Whether to use deterministic action sampling or not",
    }

    _abbrev = {
        "model_name": "m",
        "target_dir": "t",
        "name": "n",
        "use_cached_model": "uc",
        "deterministic": "d",
    }


if __name__ == "__main__":
    args = Parser()

    model_name = args.model_name

    if not args.use_cached_model:
        print("Downloading model on remote")
        out1 = subprocess.run(
            f"ssh -t ariel.geovic 'scp -r jeanzay:/gpfswork/rech/nbk/utu66tc/tb_logs/{model_name} temp/'",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(out1.stdout.decode("utf-8"))

    print("Generating video on remote")
    out2 = subprocess.run(
        f"ssh -t ariel.geovic '/home/ariel/anaconda3/envs/coltra/bin/python /home/ariel/projects/coltra-rl/scripts/enjoy_crowd.py -p /home/ariel/temp/{model_name} -e /home/ariel/projects/coltra-rl/builds/crowd-v6a/crowd.x86_64'" + "-d" if args.deterministic else "",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(out2.stdout.decode("utf-8"))

    print("Downloading video on remote")
    out3 = subprocess.run(
        f"scp -r ariel.geovic:/home/ariel/temp/video.webm {args.target_dir}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(out3.stdout.decode("utf-8"))

    print("Converting video to mp4")
    out4 = subprocess.run(
        f"ffmpeg -f webm -i {args.target_dir}/video.webm {args.target_dir}/{args.name}.mp4",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(out4.stdout.decode("utf-8"))

    print("Cleaning up")
    out5 = subprocess.run(
        f"rm {args.target_dir}/video.webm",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(out5.stdout.decode("utf-8"))

    print("Done")
