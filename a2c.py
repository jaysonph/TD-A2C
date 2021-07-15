'''
    File name: a2c_train.py
    Author: Jayson Ng
    Email: iamjaysonph@gmail.com
    Date created: 15/7/2021
    Python Version: 3.7
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import gym
import matplotlib.pyplot as plt
import copy


if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	env = env.unwrapped

	HIDDEN_LAYER = 128  # NN hidden layer size

	# Hyper-parameters
	log_intv = 10
	capacity = 10000
	target_update_intv = 500  # in terms of iterations
	max_episodes = 5000
	max_steps = 500
	lr = 0.01
	discount_factor = 0.99
	batch_size = 256
	goal = 200

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	net = A2CNetwork(env.observation_space.high.shape[0], env.action_space.n, hid_size=HIDDEN_LAYER).to(device)
	agent = TDAgent(net, capacity, env.action_space.n, batch_size, discount_factor, lr, target_update_intv)

	losses = []
	reward_hist = []
	avg_reward_hist = []
	avg_reward = 8
	best_avg_reward = 0
	for episode_i in tqdm(range(max_episodes)):
	    s = env.reset()
	    if np.mean(reward_hist[-100:]) >= goal:  # benchmark of cartpole-v0 problem
	        print(f'Solved! Average Reward reaches {goal} over the past 100 runs')
	        break
	    ep_reward = 0
	    ep_loss = 0
	    for si in range(max_steps):
	    
	        a, a_prob = agent.select_action(s)
	        new_s, r, done, info = env.step(a)

	        if done:
	            r = -1

	        agent.store_transition(s, a, new_s, r, a_prob)

	        if done:
	            reward_hist.append(ep_reward)
	            break

	        s = new_s
	        ep_reward += 1

	        loss = agent.learn()
	        if loss is not None:
	            ep_loss += loss
	    
	    losses.append(ep_loss/(si+1))
	    avg_reward = int(0.95 * avg_reward + 0.05 * ep_reward)
	    avg_reward_hist.append(avg_reward)

	    if episode_i % log_intv == 0:
	        print(f'Episode {episode_i} | Reward: {ep_reward} | Avg Reward: {avg_reward} | Loss: {loss}')