import os
import datetime
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import shortuuid
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from coltra.collectors import collect_crowd_data, collect_family_data
from coltra.configs import TrainerConfig
from coltra.envs import MultiAgentEnv
from coltra.groups import HomogeneousGroup, FamilyGroup
from coltra.policy_optimization import CrowdPPOptimizer
from coltra.schedule import SimpleCurriculum
from coltra.utils import Timer, write_dict


class Trainer:
    def __init__(self, *args, **kwargs):
        pass

    def train(
        self,
        num_iterations: int,
        disable_tqdm: bool = False,
        save_path: Optional[str] = None,
        **kwargs,
    ):
        raise NotImplementedError


class PPOCrowdTrainer(Trainer):
    """This performs coltra in a basic paradigm, with homogeneous agents"""

    def __init__(
        self,
        agents: HomogeneousGroup,
        env: MultiAgentEnv,
        config: dict[str, Any],
        use_uuid: bool = False,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        super().__init__(agents, env, config)

        Config = TrainerConfig.clone()
        Config.update(config)

        self.agents = agents

        self.env = env
        self.config = Config

        # self.config = config
        # self.config = with_default_config(config["trainer"], default_config)

        self.path: Optional[str]

        self.ppo = CrowdPPOptimizer(
            self.agents, config=self.config.PPOConfig.to_dict(), seed=seed
        )

        # Setup tensorboard
        self.writer: Optional[SummaryWriter]
        if self.config.tensorboard_name:
            dt_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if use_uuid:
                dt_string += "_" + shortuuid.uuid()
            if save_path is None:
                root = Path.home()
            else:
                root = Path(save_path)
            path = root / "tb_logs" / f"{self.config.tensorboard_name}_{dt_string}"

            self.writer = SummaryWriter(str(path))
            os.mkdir(str(path / "saved_weights"))

            # Log the configs
            with open(str(path / "trainer_config.yaml"), "w") as f:
                yaml.dump(self.config.to_dict(), f)
            # with open(str(path / f"full_config.yaml"), "w") as f:
            #     yaml.dump(config, f)

            self.path = str(path)
        else:
            self.path = None
            self.writer = None

    def train(
        self,
        num_iterations: int,
        disable_tqdm: bool = False,
        save_path: Optional[str] = None,
        collect_kwargs: Optional[dict[str, Any]] = None,
        trial: Optional[optuna.Trial] = None,
        curriculum: Optional[SimpleCurriculum] = None
    ) -> dict[str, float]:

        if save_path is None:
            save_path = self.path  # Can still be None

        print(f"Begin coltra, logged in {self.path}")
        timer = Timer()
        step_timer = Timer()
        metrics = {}

        best_so_far = -np.inf

        if save_path:
            self.agents.save(save_path)
            # torch.save(self.agent.model, os.path.join(save_path, "base_agent.pt"))

        for step in (pbar := trange(1, num_iterations + 1, disable=disable_tqdm)):
            ########################################### Collect the data ###############################################
            timer.checkpoint()

            if curriculum is not None and step in curriculum:
                update_kwargs = curriculum[step]
                collect_kwargs = {**collect_kwargs, **update_kwargs}


            full_batch, collector_metrics, shape = collect_crowd_data(
                agents=self.agents,
                env=self.env,
                num_steps=self.config.steps,
                env_kwargs=collect_kwargs,
            )

            data_time = timer.checkpoint()

            ############################################## Update policy ##############################################
            # Perform the PPO update
            metrics = self.ppo.train_on_data(
                full_batch, shape, step, writer=self.writer
            )

            end_time = step_timer.checkpoint()

            mean_reward = metrics["crowd/mean_episode_reward"]
            best_so_far = max(best_so_far, mean_reward)
            pbar.set_description(
                f"Reward: {mean_reward:8.3f}; Best: {best_so_far:8.3f}"
            )

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
                f"meta/time_data_collection": data_time,
                f"meta/total_time": end_time,
            }

            for key in collector_metrics:
                if key.startswith("m_"):
                    section = "stats"
                elif key.startswith("e_"):
                    section = "episode"
                else:
                    raise ValueError(f"Unknown metric type: {key}")
                metric_name = key[2:]
                extra_metric[f"{section}/mean_{metric_name}"] = np.mean(
                    collector_metrics[key]
                )
                extra_metric[f"{section}_extra/min_{metric_name}"] = np.min(
                    collector_metrics[key]
                )
                extra_metric[f"{section}_extra/max_{metric_name}"] = np.max(
                    collector_metrics[key]
                )
                extra_metric[f"{section}_extra/std_{metric_name}"] = np.std(
                    collector_metrics[key]
                )
                extra_metric[f"{section}_extra/median_{metric_name}"] = np.median(
                    collector_metrics[key]
                )

                # extra_metric[f"stats/{key}_100"] = np.mean(collector_metrics[key][:100])
                # extra_metric[f"stats/{key}_l100"] = np.mean(
                #     collector_metrics[key][-100:]
                # )
                # extra_metric[f"stats/{key}_l1"] = np.mean(collector_metrics[key][-2])

            write_dict(extra_metric, step, self.writer)

            if trial is not None:
                trial.report(mean_reward, step)
                if trial.should_prune():
                    print("Trial was pruned at step {}".format(step))
                    self.env.close()
                    raise optuna.TrialPruned()

        return metrics


class PPOFamilyTrainer(Trainer):
    """This performs coltra with two hierarchical agents - family and person"""

    def __init__(
        self,
        agents: FamilyGroup,
        env: MultiAgentEnv,
        config: dict[str, Any],
        use_uuid: bool = False,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        super().__init__(agents, env, config)

        Config = TrainerConfig.clone()
        Config.update(config)

        self.agents = agents

        self.env = env
        self.config = Config

        # self.config = config
        # self.config = with_default_config(config["trainer"], default_config)

        self.path: Optional[str]

        self.crowd_ppo = CrowdPPOptimizer(
            self.agents.agent, config=self.config.PPOConfig.to_dict(), seed=seed, policy_name="crowd"
        )

        self.family_ppo = CrowdPPOptimizer(
            self.agents.family_agent, config=self.config.PPOConfig.to_dict(), seed=seed, policy_name="family"
        )

        # Setup tensorboard
        self.writer: Optional[SummaryWriter]
        if self.config.tensorboard_name:
            dt_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if use_uuid:
                dt_string += "_" + shortuuid.uuid()
            if save_path is None:
                root = Path.home()
            else:
                root = Path(save_path)
            path = root / "tb_logs" / f"{self.config.tensorboard_name}_{dt_string}"

            self.writer = SummaryWriter(str(path))
            os.mkdir(str(path / "saved_weights"))

            # Log the configs
            with open(str(path / "trainer_config.yaml"), "w") as f:
                yaml.dump(self.config.to_dict(), f)
            # with open(str(path / f"full_config.yaml"), "w") as f:
            #     yaml.dump(config, f)

            self.path = str(path)
        else:
            self.path = None
            self.writer = None

    def train(
        self,
        num_iterations: int,
        disable_tqdm: bool = False,
        save_path: Optional[str] = None,
        collect_kwargs: Optional[dict[str, Any]] = None,
        trial: Optional[optuna.Trial] = None,
    ) -> dict[str, float]:

        if save_path is None:
            save_path = self.path  # Can still be None

        print(f"Begin coltra, logged in {self.path}")
        timer = Timer()
        step_timer = Timer()
        metrics = {}

        best_so_far = -np.inf
        best_so_far_family = -np.inf

        if save_path:
            self.agents.save(save_path)
            # torch.save(self.agent.model, os.path.join(save_path, "base_agent.pt"))

        for step in (pbar := trange(1, num_iterations + 1, disable=disable_tqdm)):
            ########################################### Collect the data ###############################################
            timer.checkpoint()

            full_batch, collector_metrics, shape = collect_family_data(
                agents=self.agents,
                env=self.env,
                num_steps=self.config.steps,
                env_kwargs=collect_kwargs,
            )

            data_time = timer.checkpoint()

            ############################################## Update policy ##############################################
            # Perform the PPO update
            crowd_metrics = self.crowd_ppo.train_on_data(
                {"crowd": full_batch["crowd"]}, shape["crowd"], step, writer=self.writer
            )

            family_metrics = self.family_ppo.train_on_data(
                {"family": full_batch["family"]}, shape["family"], step, writer=self.writer
            )

            end_time = step_timer.checkpoint()

            metrics = {**crowd_metrics, **family_metrics}

            mean_reward = crowd_metrics["crowd/mean_episode_reward"]
            family_mean_reward = family_metrics["family/mean_episode_reward"]

            best_so_far = max(best_so_far, mean_reward)
            best_so_far_family = max(best_so_far_family, family_mean_reward)

            pbar.set_description(
                f"Reward: {mean_reward:8.3f}/{family_mean_reward:8.3f}; Best: {best_so_far:8.3f}/{best_so_far_family:8.3f}"
            )

            ########################################## Save the updated agent ##########################################

            # Save the agent to disk
            if save_path and (step % self.config.save_freq == 0):
                self.agents.save_state(save_path, step)

            # Write remaining metrics to tensorboard
            extra_metric = {
                f"meta/time_data_collection": data_time,
                f"meta/total_time": end_time,
            }

            for key in collector_metrics:
                if key.startswith("m_"):
                    section = "stats"
                elif key.startswith("e_"):
                    section = "episode"
                else:
                    raise ValueError(f"Unknown metric type: {key}")
                metric_name = key[2:]
                extra_metric[f"{section}/mean_{metric_name}"] = np.mean(
                    collector_metrics[key]
                )
                extra_metric[f"{section}_extra/min_{metric_name}"] = np.min(
                    collector_metrics[key]
                )
                extra_metric[f"{section}_extra/max_{metric_name}"] = np.max(
                    collector_metrics[key]
                )
                extra_metric[f"{section}_extra/std_{metric_name}"] = np.std(
                    collector_metrics[key]
                )
                extra_metric[f"{section}_extra/median_{metric_name}"] = np.median(
                    collector_metrics[key]
                )

                # extra_metric[f"stats/{key}_100"] = np.mean(collector_metrics[key][:100])
                # extra_metric[f"stats/{key}_l100"] = np.mean(
                #     collector_metrics[key][-100:]
                # )
                # extra_metric[f"stats/{key}_l1"] = np.mean(collector_metrics[key][-2])

            write_dict(extra_metric, step, self.writer)

            if trial is not None:
                trial.report(mean_reward, step)
                if trial.should_prune():
                    print("Trial was pruned at step {}".format(step))
                    self.env.close()
                    raise optuna.TrialPruned()

        return metrics
