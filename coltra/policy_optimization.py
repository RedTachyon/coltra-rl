from __future__ import annotations

import copy
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
from coltra.groups import HomogeneousGroup, FamilyGroup
from numpy.random import Generator
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typarse import BaseConfig

from coltra.agents import Agent
from coltra.discounting import (
    discount_experience,
    get_episode_rewards,
    get_episode_lengths,
)
from coltra.utils import (
    get_optimizer,
    Timer,
    write_dict,
)
from coltra.configs import PPOConfig, DQNConfig

from coltra.buffers import (
    Reward,
    Value,
    Done,
    Multitype,
    get_batch_size,
    OnPolicyRecord,
    Observation,
    Action,
    DQNRecord,
    LSTMStateT,
)


def minibatches(
    *tensors: Union[Tensor, Multitype, LSTMStateT],
    batch_size: int = 32,
    shuffle: bool = True,
    rng: Optional[Generator] = None,
):
    full_size = get_batch_size(tensors[0])
    for tensor in tensors:
        assert get_batch_size(tensor) in (
            full_size,
            -1,
        ), "One of the tensors has a different batch size"

    if shuffle:
        _rng: Generator
        if rng is None:
            _rng = np.random.default_rng()
        else:
            _rng: Generator = rng
        indices = _rng.permutation(full_size)
    else:
        indices = np.arange(full_size)

    for i in range(0, full_size, batch_size):
        idx = indices[slice(i, i + batch_size)]

        yield [
            tensor.slice(idx)
            if isinstance(tensor, LSTMStateT)
            else ()
            if tensor == ()
            else tensor[idx, ...]
            for tensor in tensors
        ]


class CrowdPPOptimizer:
    """
    An optimizer for a single homogeneous crowd agent. Estimates the gradient from the whole batch (no SGD).
    """

    def __init__(
        self,
        agents: HomogeneousGroup | Agent,
        config: dict[str, Any],
        seed: Optional[int] = None,
        policy_name: str = "crowd",
    ):

        if isinstance(agents, Agent):
            agents = HomogeneousGroup(agents, policy_name=policy_name)

        self.agents = agents

        Config: PPOConfig = PPOConfig.clone()

        Config.update(config)
        self.config = Config

        self.policy_optimizer = get_optimizer(self.config.optimizer)(
            agents.parameters(), **self.config.OptimizerKwargs.to_dict()
        )

        self.gamma: float = self.config.gamma
        self.eps: float = self.config.eps
        self.gae_lambda: float = self.config.gae_lambda

        self.rng: Generator = (
            np.random.default_rng(seed=seed) if seed is not None else None
        )

    def train_on_data(
        self,
        data_dict: dict[str, OnPolicyRecord],
        shape: tuple[int, int],
        step: int = 0,
        writer: Optional[SummaryWriter] = None,
    ) -> dict[str, float]:
        """
        Performs a single update step with PPO on the given batch of data.

        Args:
            data_dict: DataBatch, dictionary
            shape: pre-flattening data shape of rewards
            step: which optimization step it is (for logging)
            writer: Tensorboard SummaryWriter

        Returns:
            metrics: a dict with the TB metrics of the update

        """
        metrics = {}
        timer = Timer()

        entropy_coeff = max(
            self.config.entropy_coeff * 0.1 ** (step / self.config.entropy_decay_time),
            self.config.min_entropy,
        )

        agents = self.agents
        agent_id = self.agents.policy_name
        ####################################### Unpack and prepare the data #######################################
        data = OnPolicyRecord.crowdify(data_dict)

        if self.config.use_gpu:
            data.cuda()
            agents.cuda()
        else:
            data.cpu()
            agents.cpu()

        # Unpacking the data for convenience

        # Compute discounted rewards to go
        # add the 'returns' and 'advantages' keys, and removes last position from other fields
        # advantages_batch, returns_batch = discount_td_rewards(agent_batch, gamma=self.gamma, lam=self.gae_lambda)

        obs: Observation = data.obs
        actions: Action = data.action
        rewards: Tensor = data.reward
        dones: Tensor = data.done
        last_values: Tensor = data.last_value
        states: Optional[LSTMStateT] = data.state if data.state is not None else ()

        # Evaluate actions to have values that require gradients
        with torch.no_grad():
            old_logprobs, old_values, old_entropies = agents.embed_evaluate(
                obs, actions, states
            )

        # breakpoint()
        # Compute the normalized advantage
        # advantages_batch = (discounted_batch - value_batch).detach()
        # advantages_batch = (advantages_batch - advantages_batch.mean())
        # advantages_batch = advantages_batch / (torch.sqrt(torch.mean(advantages_batch ** 2) + 1e-8))

        # Initialize metrics
        kl_divergence = 0.0
        ppo_step = -1
        gradient_updates = 0
        value_loss = torch.tensor(0)
        policy_loss = torch.tensor(0)

        # Start a timer
        timer.checkpoint()

        # Define variable to prevent line overflow later
        batch_size = self.config.minibatch_size

        saved_state_dict: Optional[dict] = None

        for ppo_step in range(self.config.ppo_epochs):
            returns, advantages = discount_experience(
                rewards,
                old_values,
                dones,
                last_values,
                use_ugae=self.config.use_ugae,
                γ=self.config.gamma,
                η=self.config.eta,
                λ=self.config.gae_lambda,
            )

            if hasattr(agents.agent, "update_ret_norm"):
                agents.agent.update_ret_norm(returns)
            if hasattr(agents.agent, "normalize_ret"):
                returns = agents.agent.normalize_ret(returns)

            if self.config.advantage_normalization:
                advantages = advantages - advantages.mean()
                advantages = advantages / (advantages.std() + 1e-8)

            broken = False

            for (
                m_obs,
                m_action,
                m_old_logprob,
                m_return,
                m_advantage,
                m_state,
            ) in minibatches(
                obs,
                actions,
                old_logprobs,
                returns,
                advantages,
                states,
                batch_size=batch_size,
                shuffle=True,
                rng=self.rng,
            ):
                # Evaluate again after the PPO step, for new values and gradients, aka forward pass
                m_logprob, m_value, m_entropy = agents.embed_evaluate(
                    m_obs, m_action, m_state
                )
                # Compute the KL divergence for early stopping
                kl_divergence = torch.mean(m_old_logprob - m_logprob).item()
                # log_ratio = m_logprob - m_old_logprob
                # kl_divergence = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()

                if np.isnan(kl_divergence):
                    raise ValueError("NaN detected in KL Divergence!")
                if kl_divergence > self.config.target_kl:
                    broken = True
                    if (
                        self.config.rewind
                        and gradient_updates > self.config.min_rewind_steps
                        and saved_state_dict is not None
                    ):
                        self.agents.agent.model.load_state_dict(saved_state_dict)
                    break

                ######################################### Compute the loss #############################################
                # Surrogate loss
                prob_ratio = torch.exp(m_logprob - m_old_logprob)
                surr1 = prob_ratio * m_advantage

                surr2 = torch.where(
                    torch.gt(m_advantage, 0),
                    (1 + self.eps) * m_advantage,
                    (1 - self.eps) * m_advantage,
                )

                policy_loss = -torch.min(surr1, surr2)

                # TODO: consider value clipping here (see: Implementation Matters, Engstrom et al. 2020)
                value_loss = (m_value - m_return) ** 2

                loss = (
                    policy_loss.mean()
                    + value_loss.mean()
                    - (entropy_coeff * m_entropy.mean())
                )

                if self.config.rewind:
                    saved_state_dict = copy.deepcopy(
                        self.agents.agent.model.state_dict()
                    )

                ############################################# Update step ##############################################
                self.policy_optimizer.zero_grad()
                loss.backward()

                self.policy_optimizer.step()
                gradient_updates += 1

            if broken:
                break

        # for value_step in range(self.config.value_steps):
        #     _, value_batch, _ = agent.evaluate(obs_batch, action_batch)
        #
        #     value_loss = (value_batch - discounted_batch) ** 2
        #
        #     loss = value_loss.mean()
        #
        #     self.value_optimizer.zero_grad()
        #     loss.backward()
        #     self.value_optimizer.step()

        ############################################## Collect metrics #############################################

        # Training-related metrics
        metrics[f"meta/{agent_id}/kl_divergence"] = kl_divergence
        metrics[f"meta/{agent_id}/ppo_steps_made"] = ppo_step + 1
        metrics[f"meta/{agent_id}/gradient_updates"] = gradient_updates
        metrics[f"meta/{agent_id}/policy_loss"] = policy_loss.mean().cpu().item()
        metrics[f"meta/{agent_id}/value_loss"] = (
            torch.sqrt(value_loss.mean()).cpu().item()
        )
        metrics[f"meta/{agent_id}/mean_value"] = old_values.mean().cpu().item()
        # metrics[f"{agent_id}/{agent_id}/total_loss"] = loss.detach().cpu().item()
        metrics[f"meta/{agent_id}/total_steps"] = rewards.numel()

        # ep_lens = ep_lens if self.config["pad_sequences"] else get_episode_lens(done_batch.cpu())
        # ep_lens = get_episode_lens(done_batch.cpu())

        # Group rewards by episode and sum them up to get full episode returns
        # if self.config["pad_sequences"]:
        #     ep_rewards = reward_batch.sum(0)
        # else:
        # ep_rewards = torch.tensor([torch.sum(rewards) for rewards in torch.split(reward_batch, ep_lens)])

        ep_rewards = get_episode_rewards(rewards.numpy(), dones.numpy(), shape)
        ep_lens = get_episode_lengths(dones.numpy(), shape)

        # Episode length metrics
        metrics[f"{agent_id}/mean_episode_len"] = np.mean(ep_lens)
        metrics[f"{agent_id}_extra/median_episode_len"] = np.median(ep_lens)
        metrics[f"{agent_id}_extra/min_episode_len"] = np.min(ep_lens, initial=0)
        metrics[f"{agent_id}_extra/max_episode_len"] = np.max(ep_lens, initial=0)
        metrics[f"{agent_id}_extra/std_episode_len"] = np.std(ep_lens)

        # Episode reward metrics
        metrics[f"{agent_id}/mean_episode_reward"] = np.mean(ep_rewards)
        metrics[f"{agent_id}_extra/median_episode_reward"] = np.median(ep_rewards)
        metrics[f"{agent_id}_extra/min_episode_reward"] = np.min(ep_rewards, initial=0)
        metrics[f"{agent_id}_extra/max_episode_reward"] = np.max(ep_rewards, initial=0)
        metrics[f"{agent_id}_extra/std_episode_reward"] = np.std(ep_rewards)

        # Other metrics
        metrics[f"meta/{agent_id}/episodes_this_iter"] = dones.sum().item()
        metrics[f"meta/{agent_id}/mean_entropy"] = torch.mean(old_entropies).item()

        metrics[f"meta/{agent_id}/entropy_bonus"] = entropy_coeff

        metrics[f"meta/{agent_id}/time_update"] = timer.checkpoint()

        # Write the metrics to tensorboard
        write_dict(metrics, step, writer)

        return metrics
