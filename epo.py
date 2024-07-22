"""Implementation of EPO-PMN algorithm, which is based on the omnisafe repository."""

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
from omnisafe.utils.math import discount_cumsum
import torch.nn as nn
from scipy.stats import norm
from EPOPMN.buffer import EPOVectorOnPolicyBuffer
from EPOPMN.rollout import EPOOnPolicyAdapter
from EPOPMN.model import MineConstraintActorCritic


@registry.register
class EPO_PMN(PPO):
    def _init_env(self) -> None:
        self._env = EPOOnPolicyAdapter(
            self._env_id, self._cfgs.train_cfgs.vector_env_nums, self._seed, self._cfgs
        )
        assert (self._cfgs.algo_cfgs.update_cycle) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, ('The number of steps per epoch is not divisible by the number of ' 'environments.')
        self._steps_per_epoch = (
            self._cfgs.algo_cfgs.update_cycle
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )  

    def _init_model(self) -> None:
        """Initialize the model."""
        self._actor_critic = MineConstraintActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init(self) -> None:
        self._buf = EPOVectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Train/Reward_obj')
        self._logger.register_key('Train/Penalty')
        self._logger.register_key('Value/Adv_c')
        self._logger.register_key('Value/Adv_cs')
        self._logger.register_key('Value/value_c')
        self._logger.register_key('Value/value_cs')
        self._logger.register_key('Train/factor')
        self._logger.register_key('Train/cost_limit')
        self._logger.register_key('Train/alpha')
        self._logger.register_key('Train/target_value_c')
        self._logger.register_key('Train/target_value_cs')
        self._logger.register_key('Loss/Loss_cost_square_critic', delta=True)

    def learn(self) -> Tuple[Union[int, float], ...]:  
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.train_cfgs.epochs):  
            epoch_time = time.time()

            roll_out_time = time.time()
            self._env.roll_out(             
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
            )
            self._logger.store(**{'Train/Epoch': epoch})

            update_time = time.time()
            self._logger.store(**{'Time/Rollout': time.time() - roll_out_time})
            self._update()  
            self._logger.store(**{'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)  

            if self._cfgs.model_cfgs.actor.lr != 'None':
                self._actor_critic.actor_scheduler.step() 

            self._logger.store(
                **{
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.update_cycle,
                    'Time/FPS': self._cfgs.algo_cfgs.update_cycle / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/LR': 0.0
                    if self._cfgs.model_cfgs.actor.lr == 'None'
                    else self._actor_critic.actor_scheduler.get_last_lr()[0],
                }
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None: 
        
        data = self._buf.get()  
        (
            obs,
            act,
            logp,
            target_value_r,
            target_value_c,
            target_value_cs,
            adv_r,
            adv_c,
            adv_cs,
            cost,
            discount_cost,
            time_step,
        ) = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['target_value_cs'],
            data['adv_r'],
            data['adv_c'],
            data['adv_cs'],
            data['cost'],  
            data['discount_cost'],  
            data['time_step'],
        )

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs) 

        dataloader = DataLoader(
            dataset=TensorDataset(
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                target_value_cs,
                adv_r,
                adv_c,
                adv_cs,
                cost,
                discount_cost,
                time_step,
            ),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )  

        for i in track(
            range(self._cfgs.algo_cfgs.update_iters), description='Updating...'
        ): 
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                target_value_cs,
                adv_r,
                adv_c,
                adv_cs,
                cost,  
                discount_cost, 
                time_step,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r) 
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(
                        obs, target_value_c, target_value_cs
                    )  
                self._update_actor(
                    obs,
                    act,
                    logp,
                    adv_r,
                    adv_c,
                    adv_cs,
                    cost,
                    discount_cost,
                    target_value_cs,
                    time_step,
                )  

            new_distribution = self._actor_critic.actor(original_obs)  

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            kl = distributed.dist_avg(kl) 

            if self._cfgs.algo_cfgs.kl_early_stop and kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            **{
                'Train/StopIter': i + 1,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': kl,
            }
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        adv_cs: torch.Tensor,
    ) -> None:
        
        loss, info = self._loss_safety_pi_via_cost(
            obs, act, logp, adv_r, adv_c, adv_cs
        )

        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.actor.parameters(), self._cfgs.algo_cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            **{
                'Train/Entropy': info['entropy'],
                'Train/PolicyRatio': info['ratio'],
                'Train/PolicyStd': info['std'],
                'Loss/Loss_pi': loss.mean().item(),
            }
        )

    def _loss_safety_pi_via_cost(
            self,
            obs: torch.Tensor,
            act: torch.Tensor,
            logp: torch.Tensor,
            adv: torch.Tensor,
            adv_c: torch.Tensor,
            adv_cs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        distribution = self._actor_critic.actor(obs) 
        logp_ = self._actor_critic.actor.log_prob(act)  
        std = self._actor_critic.actor.std  
        ratio = torch.exp(logp_ - logp)  
        ratio_cliped = torch.clamp(
            ratio, 1 - self._cfgs.algo_cfgs.clip, 1 + self._cfgs.algo_cfgs.clip
        )
        surr_adv = torch.min(ratio * adv, ratio_cliped * adv).mean()  
        pi_obj = (
                surr_adv + self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()
        )  
        self._logger.store(**{'Train/Reward_obj': pi_obj})

        surr_cadv = (ratio * adv_c).mean()  
        surr_csadv = (ratio * adv_cs).mean()  
        JcExp = (
                self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit
        ) 
        
        penalty_reg_fn = torch.exp(
            torch.clamp(torch.tensor(JcExp), max=self._cfgs.algo_cfgs.penalty_reg_max)
        )  

        cadv_penalty = self._cfgs.algo_cfgs.alpha * torch.log(
            1.0 + torch.exp(torch.clamp(surr_cadv, max=self._cfgs.algo_cfgs.clip_maximum))
        ) + (1 - self._cfgs.algo_cfgs.alpha) * torch.log(
            1.0 + torch.exp(torch.clamp(surr_csadv, max=self._cfgs.algo_cfgs.clip_maximum))
        )

        penalty = self._cfgs.algo_cfgs.penalty_factor * penalty_reg_fn * cadv_penalty

        loss = -pi_obj + F.relu(penalty)
        if self._cfgs.algo_cfgs.if_penalty_reg_fn:
            loss = loss / (1 + penalty_reg_fn)

        assert torch.isinf(loss).sum() == 0, print("loss", loss)

        # useful extra info
        self._logger.store(**{'Train/alpha': self._cfgs.algo_cfgs.alpha})
        self._logger.store(**{'Train/cost_limit': self._cfgs.algo_cfgs.cost_limit})
        self._logger.store(**{'Train/factor': self._cfgs.algo_cfgs.penalty_factor})
        self._logger.store(**{'Train/Penalty': penalty})
        self._logger.store(**{'Value/Adv_c': adv_c.mean().item()})
        self._logger.store(**{'Value/Adv_cs': adv_cs.mean().item()})
        entropy = distribution.entropy().mean().item()
        info = {'entropy': entropy, 'ratio': ratio.mean().item(), 'std': std}
        return loss, info

    def _update_cost_critic(
        self, obs: torch.Tensor, target_value_c: torch.Tensor, target_value_cs: torch.Tensor
    ) -> None:
       
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss_cost = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)
        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss_cost += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef
        loss_cost.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(), self._cfgs.algo_cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()
        self._logger.store(**{'Loss/Loss_cost_critic': loss_cost.mean().item()})
        self._logger.store(**{'Train/target_value_c': target_value_c.mean().item()})
        self._logger.store(
            **{'Value/value_c': self._actor_critic.cost_critic(obs)[0].mean().item()}
        )
      
        self._actor_critic.cost_square_critic_optimizer.zero_grad()
        loss_cost_square = nn.functional.mse_loss(
            self._actor_critic.cost_square_critic(obs)[0], target_value_cs
        )
    
        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_square_critic.parameters():
                loss_cost_square += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef
      
        loss_cost_square.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.cost_square_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_square_critic)
        self._actor_critic.cost_square_critic_optimizer.step()
        self._logger.store(**{'Loss/Loss_cost_square_critic': loss_cost_square.mean().item()})
        self._logger.store(**{'Train/target_value_cs': target_value_cs.mean().item()})
        self._logger.store(
            **{'Value/value_cs': self._actor_critic.cost_square_critic(obs)[0].mean().item()}
        )
       
    def normalization_data(self, data):
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        normalized_data = (data - mean) / std
        return normalized_data
