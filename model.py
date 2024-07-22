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


# model class
class MineConstraintActorCritic(ConstraintActorCritic):
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
    ) -> None:
        """Initialize ConstraintActorCritic."""
        super().__init__(obs_space, act_space, model_cfgs, epochs)  

        self.quadratic_cost_critic = CriticBuilder(  
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic('v')
        self.quadratic_cost_critic.add_module('softplus', nn.Softplus())
        self.add_module('quadratic_cost_critic', self.quadratic_cost_critic)

        if model_cfgs.critic.lr != 'None':
            self.quadratic_cost_critic_optimizer = optim.Adam(
                self.quadratic_cost_critic.parameters(), lr=model_cfgs.critic.lr
            )

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, ...]:
        """Choose action based on observation. """
        with torch.no_grad():
            value_r = self.reward_critic(obs)  
            value_c = self.cost_critic(obs)
            value_cs = self.quadratic_cost_critic(obs)

            action = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)

        return action, value_r[0], value_c[0], value_cs[0], log_prob

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, ...]:
        """Choose action based on observation."""
        return self.step(obs, deterministic=deterministic)