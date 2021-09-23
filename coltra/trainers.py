import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from typarse import BaseConfig

from coltra.agents import Agent
from coltra.collectors import collect_crowd_data
from coltra.envs import SubprocVecEnv, MultiAgentEnv
from coltra.policy_optimization import CrowdPPOptimizer
from coltra.utils import Timer, write_dict
from coltra.envs.unity_envs import Mode
from coltra.envs.base_env import VecEnv


class Trainer:
    def __init__(self, *args, **kwargs):
        pass

    def train(
        self,
        num_iterations: int,
        disable_tqdm: bool = False,
        save_path: Optional[str] = None,
        **collect_kwargs,
    ):
        raise NotImplementedError


class PPOCrowdTrainer(Trainer):
    """This performs coltra in a basic paradigm, with homogeneous agents"""

    def __init__(
        self, agent: Agent, env: Union[MultiAgentEnv, VecEnv], config: Dict[str, Any]
    ):
        super().__init__(agent, env, config)

        class Config(BaseConfig):
            steps: int = 500
            workers: int = 8

            mode: str = "random"
            num_agents: int = 20

            tensorboard_name: Optional[str] = None
            save_freq: int = 10

            class PPOConfig(BaseConfig):
                # Discounting and GAE - by default, exponential discounting at γ=0.99
                gamma: float = 0.99
                eta: float = 0.0
                gae_lambda: float = 1.0

                use_ugae: bool = False

                advantage_normalization: bool = False

                # PPO optimization parameters
                eps: float = 0.1
                target_kl: float = 0.03
                entropy_coeff: float = 0.001
                entropy_decay_time: float = 100.0
                min_entropy: float = 0.001
                value_coeff: float = 1.0  # Technically irrelevant

                # Number of gradient updates = ppo_epochs * ceil(batch_size / minibatch_size)
                ppo_epochs: int = 3
                minibatch_size: int = 8192

                use_gpu: bool = False

                optimizer: str = "adam"

                class OptimizerKwargs(BaseConfig):
                    lr: float = 1e-4
                    betas: Tuple[float, float] = (0.9, 0.999)
                    eps: float = 1e-7
                    weight_decay: float = 0.0
                    amsgrad: bool = False

        Config.update(config["trainer"])

        self.agent = agent

        self.env = env
        self.config = Config
        # self.config = config
        # self.config = with_default_config(config["trainer"], default_config)

        self.path: Optional[str]

        self.ppo = CrowdPPOptimizer(self.agent, config=self.config.PPOConfig.to_dict())

        # Setup tensorboard
        self.writer: Optional[SummaryWriter]
        if self.config.tensorboard_name:
            dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    def train(
        self,
        num_iterations: int,
        disable_tqdm: bool = False,
        save_path: Optional[str] = None,
        **collect_kwargs,
    ):

        if save_path is None:
            save_path = self.path  # Can still be None

        print(f"Begin coltra, logged in {self.path}")
        timer = Timer()
        step_timer = Timer()

        if save_path:
            torch.save(self.agent.model, os.path.join(save_path, "base_agent.pt"))

        for step in trange(1, num_iterations + 1, disable=disable_tqdm):
            ########################################### Collect the data ###############################################
            timer.checkpoint()

            full_batch, collector_metrics = collect_crowd_data(
                agent=self.agent,
                env=self.env,
                num_steps=self.config.steps,
                mode=Mode.from_string(self.config.mode),
                num_agents=self.config.num_agents,
            )

            data_time = timer.checkpoint()

            ############################################## Update policy ##############################################
            # Perform the PPO update
            metrics = self.ppo.train_on_data(full_batch, step, writer=self.writer)

            end_time = step_timer.checkpoint()

            ########################################## Save the updated agent ##########################################

            # Save the agent to disk
            if save_path and (step % self.config.save_freq == 0):
                # torch.save(old_returns, os.path.join(save_path, "returns.pt"))
                torch.save(
                    self.agent.model.state_dict(),
                    os.path.join(save_path, "saved_weights", f"weights_{step}"),
                )

            # Write remaining metrics to tensorboard
            extra_metric = {
                f"crowd/time_data_collection": data_time,
                f"crowd/total_time": end_time,
            }

            for key in collector_metrics:
                extra_metric[f"stats/{key}"] = np.mean(collector_metrics[key])
                extra_metric[f"stats/{key}_100"] = np.mean(collector_metrics[key][:100])
                extra_metric[f"stats/{key}_l100"] = np.mean(
                    collector_metrics[key][-100:]
                )
                extra_metric[f"stats/{key}_l1"] = np.mean(collector_metrics[key][-2])

            write_dict(extra_metric, step, self.writer)


# class PPOMultiPolicyTrainer(Trainer):
#     """WIP"""
#
#     def __init__(
#         self,
#         agents: Dict[str, Agent],
#         env: Union[MultiAgentEnv, SubprocVecEnv],
#         config: Dict[str, Any],
#     ):
#         super().__init__(agents, env, config)
#
#         class Config(BaseConfig):
#             steps: int = 500
#             workers: int = 8
#
#             mode: str = "random"
#             num_agents: int = 20
#
#             tensorboard_name: str = None
#             save_freq: int = 10
#
#             class PPOConfig(BaseConfig):
#                 # Discounting and GAE - by default, exponential discounting at γ=0.99
#                 gamma: float = 0.99
#                 eta: float = 0.0
#                 gae_lambda: float = 1.0
#
#                 # PPO optimization parameters
#                 eps: float = 0.1
#                 target_kl: float = 0.03
#                 entropy_coeff: float = 0.001
#                 entropy_decay_time: float = 100.0
#                 min_entropy: float = 0.001
#                 value_coeff: float = 1.0  # Technically irrelevant
#
#                 # Number of gradient updates = ppo_epochs * ceil(batch_size / minibatch_size)
#                 ppo_epochs: int = 3
#                 minibatch_size: int = 8192
#
#                 use_gpu: bool = False
#
#                 optimizer: str = "adam"
#
#                 class OptimizerKwargs(BaseConfig):
#                     lr: float = 1e-4
#                     betas: Tuple[float, float] = (0.9, 0.999)
#                     eps: float = 1e-7
#                     weight_decay: float = 0.0
#                     amsgrad: bool = False
#
#         Config.update(config["trainer"])
#
#         self.agents = agents
#
#         self.env = env
#         self.config = Config
#         # self.config = config
#         # self.config = with_default_config(config["trainer"], default_config)
#
#         self.optimizers = {
#             name: CrowdPPOptimizer(agent, config=self.config.PPOConfig.to_dict())
#             for name, agent in self.agents.items()
#         }
#
#         # self.ppo = CrowdPPOptimizer(self.agent, config=self.config.PPOConfig.to_dict())
#
#         # Setup tensorboard
#         self.writer: SummaryWriter
#         if self.config.tensorboard_name:
#             dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             self.path = (
#                 Path.home() / "tb_logs" / f"{self.config.tensorboard_name}_{dt_string}"
#             )
#
#             self.writer = SummaryWriter(str(self.path))
#             os.mkdir(str(self.path / "saved_weights"))
#
#             # Log the configs
#             with open(str(self.path / "trainer_config.yaml"), "w") as f:
#                 yaml.dump(self.config.to_dict(), f)
#             with open(str(self.path / f"full_config.yaml"), "w") as f:
#                 yaml.dump(config, f)
#
#             self.path = str(self.path)
#         else:
#             self.path = None
#             self.writer = None
#
#     def train(
#         self,
#         num_iterations: int,
#         save_path: Optional[str] = None,
#         disable_tqdm: bool = False,
#         **collect_kwargs,
#     ):
#
#         if save_path is None:
#             save_path = self.path  # Can still be None
#
#         print(f"Begin coltra, logged in {self.path}")
#         timer = Timer()
#         step_timer = Timer()
#
#         if save_path:
#             for name, agent in self.agents.items():
#                 torch.save(agent.model, os.path.join(save_path, f"{name}_agent.pt"))
#
#         for step in trange(1, num_iterations + 1, disable=disable_tqdm):
#             ########################################### Collect the data ###############################################
#             timer.checkpoint()
#
#             full_batch, collector_metrics = collect_crowd_data(
#                 agent=self.agent,
#                 env=self.env,
#                 num_steps=self.config.steps,
#                 mode=Mode.from_string(self.config.mode),
#                 num_agents=self.config.num_agents,
#             )
#             # breakpoint()
#             # full_batch = concat_subproc_batch(full_batch)
#
#             # full_batch, collector_metrics = collect_parallel_unity(num_workers=self.config["workers"],
#             #                                                        num_runs=self.config["workers"],
#             #                                                        agent=self.agent,
#             #                                                        env_path=self.env_path,
#             #                                                        num_steps=self.config["steps"],
#             #                                                        base_seed=step)
#
#             data_time = timer.checkpoint()
#
#             ############################################## Update policy ##############################################
#             # Perform the PPO update
#             metrics = self.ppo.train_on_data(full_batch, step, writer=self.writer)
#
#             end_time = step_timer.checkpoint()
#
#             ########################################## Save the updated agent ##########################################
#
#             # Save the agent to disk
#             if save_path and (step % self.config.save_freq == 0):
#                 # torch.save(old_returns, os.path.join(save_path, "returns.pt"))
#                 torch.save(
#                     self.agent.model.state_dict(),
#                     os.path.join(save_path, "saved_weights", f"weights_{step}"),
#                 )
#
#             # Write remaining metrics to tensorboard
#             extra_metric = {
#                 f"crowd/time_data_collection": data_time,
#                 f"crowd/total_time": end_time,
#             }
#
#             for key in collector_metrics:
#                 extra_metric[f"stats/{key}"] = np.mean(collector_metrics[key])
#                 extra_metric[f"stats/{key}_100"] = np.mean(collector_metrics[key][:100])
#                 extra_metric[f"stats/{key}_l100"] = np.mean(
#                     collector_metrics[key][-100:]
#                 )
#                 extra_metric[f"stats/{key}_l1"] = np.mean(collector_metrics[key][-2])
#
#             write_dict(extra_metric, step, self.writer)


if __name__ == "__main__":
    pass
    # from rollout import Collector

    # env_ = foraging_env_creator({})

    # agent_ids = ["Agent0", "Agent1"]
    # agents_: Dict[str, Agent] = {
    #     agent_id: Agent(LSTMModel({}), name=agent_id)
    #     for agent_id in agent_ids
    # }
    #
    # runner = Collector(agents_, env_)
    # data_batch = runner.rollout_steps(num_episodes=10, disable_tqdm=True)
    # obs_batch = data_batch['observations']['Agent0']
    # action_batch = data_batch['actions']['Agent0']
    # reward_batch = data_batch['rewards']['Agent0']
    # done_batch = data_batch['dones']['Agent0']
    #
    # logprob_batch, value_batch, entropy_batch = agents_['Agent0'].evaluate_actions(obs_batch,
    #                                                                                action_batch,
    #                                                                                done_batch)
