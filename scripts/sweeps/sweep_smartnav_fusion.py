import os

import cv2
import torch
import wandb
import yaml
from wandb import Config

from coltra.agents import CAgent
from coltra.collectors import collect_renders
from coltra.envs import SmartNavEnv
from coltra.groups import HomogeneousGroup
from coltra.models.mlp_models import ImageMLPModel
from coltra.research import JointModel
from coltra.research.policy_fusion.fusion_trainer import FusionTrainer
from coltra.trainers import PPOCrowdTrainer


def fix_wandb_config(wandb_config: Config, main_config: dict):
    """
    !!! Mutates both wandb_config and main_config. !!!

    """
    keys = wandb_config.keys()

    for k in keys:
        names = k.split(".")
        sub_config = main_config
        for name in names[:-1]:
            sub_config = sub_config[name]

        sub_config[names[-1]] = wandb_config[k]

    # for k in keys:
    #     del wandb_config[k]

    wandb_config.update(config)
    return


if __name__ == "__main__":
    CUDA = torch.cuda.is_available()

    config_path = "smartnav_config.yaml"
    iters = 500
    env_path = "/home/akwiatkowski@ubisoft.org/projects/coltra-rl/builds/SmartHiderLinux-v1/smarthider.x86_64"
    tb_name = "smartnav-sweep"

    with open(config_path, "r") as f:
        config = yaml.load(f.read(), yaml.Loader)

    trainer_config = config["trainer"]
    model_config = config["model"]
    env_config = config["environment"]

    trainer_config["tensorboard_name"] = tb_name
    trainer_config["PPOConfig"]["use_gpu"] = CUDA

    workers = trainer_config["workers"]

    # Initialize the environment
    # env = UnitySimpleCrowdEnv.get_venv(workers, file_name=env_path, no_graphics=True)

    env = SmartNavEnv(
        file_name=env_path,
        metrics=[],
        env_params=env_config,
        no_graphics=True,
    )

    # env.engine_channel.set_configuration_parameters(time_scale=100, width=100, height=100)

    # Initialize the agent
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    model_config["input_size"] = obs_size
    model_config["num_actions"] = action_size

    wandb.init(
        project="debug",
        entity="redtachyon",
        sync_tensorboard=True,
        config={},
        # name=name,
    )

    fix_wandb_config(wandb.config, config)

    model_cls = ImageMLPModel

    model = model_cls(model_config, action_space=env.action_space)
    joint_model = JointModel(config={}, models=[model], action_space=env.action_space)
    agent = CAgent(model)
    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    try:
        trainer = FusionTrainer(agents, env, trainer_config)
        trainer.train(
            iters,
            disable_tqdm=False,
            save_path=trainer.path,
            collect_kwargs=env_config,
        )
    except Exception as err:
        print(f"Exception during training: {err}")
        env.close()
        quit()
    env.close()

    print("Training complete. Collecting renders")

    env = SmartNavEnv(
        file_name=env_path,
        virtual_display=(1600, 900),
        no_graphics=False,
        time_scale=1.0,
    )

    try:
        renders, _ = collect_renders(
            agents,
            env,
            num_steps=trainer_config["steps"],
            disable_tqdm=False,
            env_kwargs=env_config,
        )
    except Exception as err:
        print(f"Exception during visualization: {err}")
        env.close()
        quit()

    frame_size = renders.shape[1:3]

    print("Recording a video")
    video_path = os.path.join(trainer.path, "video.webm")
    out = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"VP90"), 5, frame_size[::-1]
    )
    for frame in renders[..., ::-1]:
        out.write(frame)

    out.release()

    print(f"Video saved to {video_path}")

    wandb.log({"video": wandb.Video(video_path)})

    print("Video uploaded to wandb")

    env.close()