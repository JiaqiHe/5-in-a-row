# reference: https://github.com/ImmanuelXIV/ppo-self-play/blob/master/policy.py
#!/usr/bin/python
#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Modified by Immanuel Schwall (manuel.schwall@gmail.com) 2019

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random



class ActorCritic(nn.Module):
	""" Actor Critic neural network with shared body.
	The Actor maps states (observations) to action, log_probs, entropy.
	The Critic maps states to values.
	"""
	
	def __init__(self, state_size, action_space_size, seed=0):
		""" Initialize the neural net.
        
        Params
        ======
        	state_size: 	    dimension of each input state
        	action_space_size: 	dimension of each output
        	seed: 		    	random seed
        """
		super().__init__()
		self.seed = torch.manual_seed(seed)
		# fully connected body
		self.fc1_body = nn.Linear(state_size, 128)
		self.fc2_body = nn.Linear(128, 128)
		# actor head
		self.fc3_actor = nn.Linear(128, action_space_size)
		self.std = nn.Parameter(torch.ones(1, action_space_size))
		# critic head
		self.fc3_critic = nn.Linear(128, 1)


	def forward(self, state, action=None):
		x = torch.Tensor(state)
		x = F.relu(self.fc1_body(x))
		x = F.relu(self.fc2_body(x))

		# Actor head
		action_logits = self.fc3_actor(x)
		action_probs = F.softmax(action_logits, dim=-1)
		dist = torch.distributions.Categorical(action_probs)
		
		if action is None:
			action = dist.sample()
		
		log_prob = dist.log_prob(action)
		entropy = dist.entropy()
		
		# Critic head
		value = self.fc3_critic(x)
		
		# 如果需要获取确定性动作（例如在评估时）
		action_to_take = torch.argmax(action_probs, dim=-1)
		
		return action, log_prob, entropy, value, action_to_take
