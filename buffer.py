"""Implementation of replay buffer in EPO-PMN algorithm, which is based on the omnisafe repository."""

import time
from typing import Dict, Tuple, Optional, Union
import torch
import numpy as np
import torch.nn.functional as F

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.utils import distributed
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import track
from collections import deque
from omnisafe.adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.typing import AdvatageEstimator, OmnisafeSpace
from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer

from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.models.actor.actor_builder import ActorBuilder
from omnisafe.utils.math import discount_cumsum
from torch import optim

from omnisafe.models.actor_critic.actor_critic import ActorCritic
from omnisafe.utils.config import ModelConfig
import torch.nn as nn
from scipy.stats import norm


class EPOOnPolicyBuffer(OnPolicyBuffer):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__(
            obs_space,
            act_space,
            size,
            gamma,
            lam,
            lam_c,
            advantage_estimator,
            penalty_coefficient,
            standardized_adv_r,
            standardized_adv_c,
            device,
        )
        self._standardized_adv_cs = standardized_adv_c
        self.data['value_cs'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['adv_cs'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_cs'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['discount_cost'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['time_step'] = torch.zeros((size,), dtype=torch.float32, device=device)

    def get(self) -> Dict[str, torch.Tensor]:
        """Get the data in the buffer."""
        self.ptr, self.path_start_idx = 0, 0

        data = {
            'obs': self.data['obs'],
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'adv_c': self.data['adv_c'],
            'target_value_c': self.data['target_value_c'],
            'cost': self.data['cost'],
            'discount_cost': self.data['discount_cost'],
            'value_c': self.data['value_c'],
            'value_cs': self.data['value_cs'],
            'adv_cs': self.data['adv_cs'],
            'target_value_cs': self.data['target_value_cs'],
            'time_step': self.data['time_step'],
        }

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        csadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_cs'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean
        if self._standardized_adv_cs:
            data['adv_cs'] = data['adv_cs'] - csadv_mean

        return data

    def finish_path(
        self,
        last_value_r: torch.Tensor = torch.zeros(1),
        last_value_c: torch.Tensor = torch.zeros(1),
        last_value_cs: torch.Tensor = torch.zeros(1),
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs."""
        path_slice = slice(self.path_start_idx, self.ptr)
        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)
        last_value_cs = last_value_cs.to(self._device)  #
        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])
        values_cs = torch.cat([self.data['value_cs'][path_slice], last_value_cs])

        discounted_ret = discount_cumsum(rewards, self._gamma)[:-1]
        self.data['discounted_ret'][path_slice] = discounted_ret
        rewards -= self._penalty_coefficient * costs
        discounted_cost = discount_cumsum(costs, self._gamma)[:-1]
        self.data['discount_cost'][path_slice] = discounted_cost

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r, rewards, lam=self._lam
        )
        adv_c, target_value_c = self._calculate_adv_and_value_targets(
            values_c, costs, lam=self._lam_c
        )
        adv_cs, target_value_cs = self._calculate_square_adv_and_value_targets(
            costs, values_c, values_cs, lam=self._lam_c
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c
        self.data['adv_cs'][path_slice] = adv_cs
        self.data['target_value_cs'][path_slice] = target_value_cs

        self.path_start_idx = self.ptr

    def _calculate_square_adv_and_value_targets(
        self,
        costs: torch.Tensor,
        values_c: torch.Tensor,
        values_cs: torch.Tensor,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        deltas = (
            torch.square(costs[:-1] + self._gamma * values_c[1:])
            - torch.square(values_c[:-1])
            + self._gamma**2 * values_cs[1:]
            - values_cs[:-1]
        )

        def square_discount_cumsum(x_vector: torch.Tensor, discount: float) -> torch.Tensor:
            """Compute the discounted cumulative sum of vectors."""
            length = x_vector.shape[0]
            x_vector = x_vector.type(torch.float64)
            for idx in reversed(range(length)):
                if idx == length - 1:
                    cumsum = x_vector[idx]
                else:
                    cumsum = x_vector[idx] + discount**2 * cumsum  # adv[t] + gamma**2*adv[t+1]
                x_vector[idx] = cumsum
            return x_vector

        adv_cs = square_discount_cumsum(deltas, self._gamma * lam)
        target_cs = torch.clamp(adv_cs + values_cs[:-1], 0.0, float('inf'))

        return adv_cs, target_cs


class EPOVectorOnPolicyBuffer(VectorOnPolicyBuffer):
    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float,
        standardized_adv_r: bool,
        standardized_adv_c: bool,
        num_envs: int = 1,
        device: torch.device = torch.device('cpu'),
    ):
        self._num_buffers = num_envs
        self._standardized_adv_r = standardized_adv_r
        self._standardized_adv_c = standardized_adv_c
        if num_envs < 1:
            raise ValueError('num_envs must be greater than 0.')
        self.buffers = [
            EPOOnPolicyBuffer(
                obs_space=obs_space,
                act_space=act_space,
                size=size,
                gamma=gamma,
                lam=lam,
                lam_c=lam_c,
                advantage_estimator=advantage_estimator,
                penalty_coefficient=penalty_coefficient,
                device=device,
            )
            for _ in range(num_envs)
        ]

    def get(self) -> Dict[str, torch.Tensor]:
        """Get the data from the buffer."""
        data_pre = {k: [v] for k, v in self.buffers[0].get().items()}
        for buffer in self.buffers[1:]:
            for k, v in buffer.get().items():
                data_pre[k].append(v)
        data = {k: torch.cat(v, dim=0) for k, v in data_pre.items()}

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        csadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_cs'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean
            data['adv_cs'] = data['adv_cs'] - csadv_mean
        return data

    def finish_path(
        self,
        last_value_r: torch.Tensor = torch.zeros(1),
        last_value_c: torch.Tensor = torch.zeros(1),
        last_value_cs: torch.Tensor = torch.zeros(1),
        idx: int = 0,
    ) -> None:
        """Finish the path."""
        self.buffers[idx].finish_path(last_value_r, last_value_c, last_value_cs)
