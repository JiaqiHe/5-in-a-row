#!/usr/bin/python
#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Modified by Jeremi Kaczmarczyk (jeremi.kaczmarczyk@gmail.com) 2018 
# Modified by Andrei Li (andreiliphd@gmail.com) 2019
# Modified by Immanuel Schwall (manuel.schwall@gmail.com) 2019

from collections import deque
from policy import ActorCritic
import numpy as np
import torch
import torch.onnx
import random
import torch
import torch.nn as nn
import torch.optim as optim
import random


SEED = 0                    # seed
LR = 5e-4                   # leanring rate for actor critic model
T_MAX_ROLLOUT = 1024        # maximum number of time steps per episode
GAMMA = 0.999               # discount factor for returns
TAU = 0.95					# gae (generalized advantage estimation) param
K_EPOCHS = 16				# optimize surrogate loss with K epochs
BATCH_SIZE = 64				# minibatch size ≤ T_MAX_ROLLOUT
EPSILON_PPO = 0.2           # clipping parameter for PPO surrogate
USE_ENTROPY = False         # apply entropy term y/n
ENTROPY_WEIGHT = 0.01       # coefficient for entropy term
GRADIENT_CLIPPING = 2       # gradient clipping norm
VALUE_LOSS_COEF = 0.5       # 
KL_TARGET = 0.01            # KL divergence target for early stopping

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class Agent():
    """ Actor Critic agent that implements the PPO algorithm 
    based on Schulman et al. (2017) https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, state_size, action_space_size, load_pretrained=False):
        """ Initialize the agent.
        Params
        ======
            state_size:             dimension of each state
            action_space_size:      dimension of each action
        """
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.ac_model = ActorCritic(state_size, action_space_size, seed=SEED)
        self.ac_model_optim = optim.Adam(self.ac_model.parameters(), lr=LR)
        random.seed(SEED)
        print('Number of trainable actor critic model parameters: ', \
        	self.count_parameters())

        if load_pretrained:
        	print('Loading pre-trained actor critic model from checkpoint.')
        	self.ac_model.load_state_dict(torch.load("checkpoints/ac_model.pth", \
        		map_location=torch.device(DEVICE)))


    def count_parameters(self):
        return sum(p.numel() for p in self.ac_model.parameters() if p.requires_grad)


    def get_states_from_obs_for_player(self, obs, i):
        return obs["obs"][i - 1, :]

    def act(self, env):
        obs = env.reset()
        assert(obs["obs"].shape == (2, 204))
        scores = np.zeros(2)
        self.ac_model.eval()
        
        while True:
            with torch.no_grad():
                # player1
                states1 = self.get_states_from_obs_for_player(obs, 1)
                action_to_take1 = self.ac_model(states1)[-1]
                obs, reward1, done1, _ = env.step(np.array([[action_to_take1], [0]]))
                scores[0] += reward1[0]
                if done1[0]:
                    break
                
                # player2
                states2 = self.get_states_from_obs_for_player(obs, 2)
                action_to_take2 = self.ac_model(states2)[-1]
                obs, reward2, done2, _ = env.step(np.array([[0], [action_to_take2]]))
                scores[1] += reward2[1]
                if done2[1]:
                    break

        self.ac_model.train()
        return scores


    def step(self, env):
        """收集轨迹并进行学习"""
        # 由于是 VecMonitor 包装的环境，我们应该使用 step 的返回值来获取观察
        obs = env.reset()  # 第一次需要重置以获取初始观察
        
        trajectory1 = deque()
        trajectory2 = deque()

        for k in range(T_MAX_ROLLOUT):
            with torch.no_grad():
                # player1
                states1 = self.get_states_from_obs_for_player(obs, 1)
                actions1, log_probs1, _, values1, _ = self.ac_model(states1)
                # step 返回的 obs 就包含了新的观察
                obs, rewards1, dones1, _ = env.step(np.array([[actions1], [0]]))
                trajectory1.append(
                    [states1, values1, actions1, log_probs1, rewards1[0], 1 - dones1[0]]
                )
                # handle game over
                if dones1[0]:
                    if rewards1[0] > 1e5:
                        # in this case, player1 secures a win to end the game. Penalize player2
                        states2 = self.get_states_from_obs_for_player(obs, 2)
                        pending_value2 = self.ac_model(states2)[-2]
                        trajectory2.append(
                            [states2, pending_value2, None, None, -1e7, 1 - dones1[0]]
                        )
                    break
                #
                # player2 takes action
                states2 = self.get_states_from_obs_for_player(obs, 2)
                actions2, log_probs2, _, values2, _ = self.ac_model(states2)
                obs, rewards2, dones2, _ = env.step(np.array([[0], [actions2]]))
                trajectory2.append(
                    [states2, values2, actions2, log_probs2, rewards2[1], 1 - dones2[1]])
                if dones2[1]:
                    if rewards2[1] > 1e5:
                        states1 = self.get_states_from_obs_for_player(obs, 1)
                        pending_value1 = self.ac_model(states1)[-2]
                        trajectory1.append([
                            states1, pending_value1, None, None, -1e7, 1 - dones2[1]
                        ])
                    break

        if len(trajectory1) > 2:
            self.learn(trajectory1)
        if len(trajectory2) > 2:
            self.learn(trajectory2)

        trajectory1.clear()
        trajectory2.clear()


    def learn(self, trajectory):
        """ Make PPO learning step. 
        Params
        ======
            trajectory:     trajectory/episode
        """
        storage = deque()
        advantages = torch.Tensor(np.zeros((1, 1)))
        returns = 0

        for i in reversed(range(len(trajectory) - 1)):
            states, value, actions, log_probs, rewards, dones = trajectory[i]
            states = torch.Tensor(states).resize_(1, 204) # 204 = observation space
            actions = torch.Tensor(actions)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            next_value = trajectory[i + 1][1]
            dones = torch.Tensor(dones).unsqueeze(1)
            returns = rewards + GAMMA * dones * returns

            # calculate generalized advantage estimation
            td_error = rewards + GAMMA * dones * next_value.detach() - value.detach()
            advantages = advantages * TAU * GAMMA * dones + td_error
            storage.append([states, actions, log_probs, returns, advantages])
            # print("============= storage: ", str(storage))

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*storage))
        advantages = (advantages - advantages.mean()) / advantages.std()

        storage.clear()
        dataset = torch.utils.data.TensorDataset(states, actions, log_probs_old, returns, advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        dataiter = iter(dataloader)

        # update the actor critic model K_EPOCHS times
        for _ in range(K_EPOCHS):
            # sample states, actions, log_probs_old, returns, advantages
            for batch in dataloader:
                sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages = batch

                _, log_probs, entropy, values, _ = self.ac_model(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                surrogate = ratio * sampled_advantages
                surrogate_clipped = torch.clamp(ratio, 1.0 - EPSILON_PPO, 1.0 + EPSILON_PPO) * sampled_advantages

                if USE_ENTROPY:
                    loss_policy = - torch.min(surrogate, surrogate_clipped).mean(0) - ENTROPY_WEIGHT * entropy.mean()
                else: 
                    loss_policy = - torch.min(surrogate, surrogate_clipped).mean(0)
                
                loss_value = 0.5 * (sampled_returns - values).pow(2).mean()
                
                loss_total = loss_policy + VALUE_LOSS_COEF * loss_value
                self.ac_model_optim.zero_grad()
                loss_total.backward()
                nn.utils.clip_grad_norm_(self.ac_model.parameters(), GRADIENT_CLIPPING)
                self.ac_model_optim.step()

                del loss_policy
                del loss_value     

                # 建议添加 KL 散度的早停机制
                kl = (sampled_log_probs_old - log_probs).mean()
                if kl > KL_TARGET * 1.5:
                    break  # 提前结束当前轮次的更新

                # 建议在每个epoch结束后清理中间变量
                del surrogate
                del surrogate_clipped
                torch.cuda.empty_cache()  # 如果使用GPU

    def save(self):
        dummy_input = torch.randn(1, 204)
        torch.onnx.export(self.ac_model, dummy_input, "ppo_selfplay.onnx")



