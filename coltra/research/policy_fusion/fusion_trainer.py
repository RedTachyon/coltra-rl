import os
import datetime
from pathlib import Path
from typing import Optional, Any, Type, List, Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typarse import BaseConfig

from coltra.collectors import collect_crowd_data
from coltra.configs import PPOConfig
from coltra.envs import MultiAgentEnv
from coltra.groups import HomogeneousGroup
from coltra.policy_optimization import CrowdPPOptimizer
from coltra.research import JointModel
from coltra.trainers import Trainer
from coltra.utils import Timer, write_dict


class FusionConfig(BaseConfig):
    steps: int = 1000
    workers: int = 8

    tensorboard_name: Optional[str] = None
    save_freq: int = 50

    PPOConfig: Type[PPOConfig] = PPOConfig.clone()


class FusionTrainer(Trainer):
    def __init__(
        self,
        agents: HomogeneousGroup,
        env: MultiAgentEnv,
        config: dict[str, Any],
    ):
        super().__init__(agents, env, config)

        Config = FusionConfig.clone()

        Config.update(config)

        self.agents = agents

        # assert isinstance(
        #     self.agents.agent.model, JointModel
        # ), "FusionTrainer can only work with a JointModel"
        # self.model = self.agents.agent.model

        self.env = env
        self.config = Config

        # self.config = config
        # self.config = with_default_config(config["trainer"], default_config)

        self.path: Optional[str]

        self.ppo = CrowdPPOptimizer(self.agents, config=self.config.PPOConfig.to_dict())

        # Setup tensorboard
        self.writer: Optional[SummaryWriter]
        if self.config.tensorboard_name:
            dt_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = (
                Path.home() / "tb_logs" / f"{self.config.tensorboard_name}_{dt_string}"
            )

            self.writer = SummaryWriter(str(path))
            os.mkdir(str(path / "saved_weights"))

            # Log the configs
            with open(str(path / "trainer_config.yaml"), "w") as f:
                yaml.dump(self.config.to_dict(), f)
            with open(str(path / f"full_config.yaml"), "w") as f:
                yaml.dump(config, f)

            self.path = str(path)
        else:
            self.path = None
            self.writer = None

    def clone_model(self, copy_logstd: bool = False):
        self.agents.agent.model = JointModel.clone_model(
            self.agents.agent.model, copy_logstd=copy_logstd
        )
        # self.model = self.agents.agent.model
        self.agents.agent.model.freeze_models([True, False])
        self.reinitialize_ppo()

    def reinitialize_ppo(self):
        self.ppo = CrowdPPOptimizer(self.agents, config=self.config.PPOConfig.to_dict())

    def train(
        self,
        num_iterations: int,
        disable_tqdm: bool = False,
        save_path: Optional[str] = None,
        collect_kwargs: Optional[dict[str, Any]] = None,
    ):
        if save_path is None:
            save_path = self.path  # Can still be None

        print(f"Begin coltra, logged in {self.path}")
        timer = Timer()
        step_timer = Timer()

        if save_path:
            self.agents.save(save_path)

        params = {
            "visible_reward": -0.001,
        }

        for step in trange(1, num_iterations + 1, disable=disable_tqdm):
            ########################################### Collect the data ###############################################
            timer.checkpoint()

            if step == num_iterations // 5:  # 100-200, phase 2
                params["visible_reward"] = -0.01
                self.clone_model(copy_logstd=True)

            if step == 2 * num_iterations // 5:  # 200-500, phase 3
                params["visible_reward"] = -0.004
                assert isinstance(self.agents.agent.model, JointModel)
                self.agents.agent.model.freeze_models([False, False])
                self.agents.agent.model.reinitialize_head()
                self.reinitialize_ppo()

            full_batch, collector_metrics, shape = collect_crowd_data(
                agents=self.agents,
                env=self.env,
                num_steps=self.config.steps,
                env_kwargs=params,
            )

            data_time = timer.checkpoint()

            ############################################## Update policy ##############################################
            # Perform the PPO update
            metrics = self.ppo.train_on_data(
                full_batch, shape, step, writer=self.writer
            )

            end_time = step_timer.checkpoint()

            ########################################## Save the updated agent ##########################################

            # Save the agent to disk
            if save_path and (step % self.config.save_freq == 0):
                # torch.save(old_returns, os.path.join(save_path, "returns.pt"))
                torch.save(
                    self.agents.agent.model.state_dict(),
                    os.path.join(save_path, "saved_weights", f"weights_{step}"),
                )

            # Write remaining metrics to tensorboard
            extra_metric = {
                f"crowd/time_data_collection": data_time,
                f"crowd/total_time": end_time,
            }

            for key in collector_metrics:
                extra_metric[f"stats/{key}_mean"] = np.mean(collector_metrics[key])
                extra_metric[f"stats/{key}_min"] = np.min(collector_metrics[key])
                extra_metric[f"stats/{key}_max"] = np.max(collector_metrics[key])
                extra_metric[f"stats/{key}_std"] = np.std(collector_metrics[key])
                extra_metric[f"stats/{key}_median"] = np.median(collector_metrics[key])

                # extra_metric[f"stats/{key}_100"] = np.mean(collector_metrics[key][:100])
                # extra_metric[f"stats/{key}_l100"] = np.mean(
                #     collector_metrics[key][-100:]
                # )
                # extra_metric[f"stats/{key}_l1"] = np.mean(collector_metrics[key][-2])

            write_dict(extra_metric, step, self.writer)
