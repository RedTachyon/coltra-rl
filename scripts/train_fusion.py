from typing import Optional

import torch
import wandb
import yaml
from typarse import BaseParser

from coltra.agents import CAgent, Agent
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.groups import HomogeneousGroup
from coltra.models.mlp_models import MLPModel
from coltra.models.relational_models import RelationModel
from coltra.research import JointModel
from coltra.research.policy_fusion.fusion_trainer import FusionTrainer
from coltra.trainers import PPOCrowdTrainer


class Parser(BaseParser):
    config: str = "configs/base_config.yaml"
    iters: int = 500
    env: str
    name: str
    model_type: str = "relation"
    start_dir: Optional[str]
    start_idx: Optional[int] = -1

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env": "Path to the Unity environment binary",
        "name": "Name of the tb directory to store the logs",
        "model_type": "Type of the information that a model has access to",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env": "e",
        "name": "n",
        "model_type": "mt",
        "start_dir": "sd",
        "start_idx": "si",
    }


if __name__ == "__main__":
    CUDA = torch.cuda.is_available()

    args = Parser()

    assert args.model_type in ("blind", "rays", "relation"), ValueError(
        "Wrong model type passed."
    )

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), yaml.Loader)

    trainer_config = config["trainer"]
    model_config = config["model"]
    env_config = config["environment"]

    trainer_config["tensorboard_name"] = args.name
    trainer_config["PPOConfig"]["use_gpu"] = CUDA

    if args.name:
        wandb.init(
            project="crowdai",
            entity="redtachyon",
            sync_tensorboard=True,
            config=config,
            name=args.name,
        )

    workers = trainer_config.get("workers") or 4  # default value

    # Initialize the environment
    env = UnitySimpleCrowdEnv.get_venv(workers, file_name=args.env)

    # env.engine_channel.set_configuration_parameters(time_scale=100, width=100, height=100)

    # Initialize the agent
    obs_size = env.observation_space.shape[0]
    buffer_size = 4  # TODO: Hardcoded, fix

    if args.model_type == "relation":
        model_cls = RelationModel
    else:
        model_cls = MLPModel

    agent: Agent
    if args.start_dir:
        agent = CAgent.load(args.start_dir, weight_idx=args.start_idx)
    else:
        model = model_cls(model_config)
        joint_model = JointModel(models=[model], num_actions=2, discrete=False)
        agent = CAgent(model)

    agents = HomogeneousGroup(agent)

    if CUDA:
        agents.cuda()

    trainer = FusionTrainer(agents, env, trainer_config)
    trainer.train(
        args.iters,
        disable_tqdm=False,
        save_path=trainer.path,
        # collect_kwargs=env_config,
    )
