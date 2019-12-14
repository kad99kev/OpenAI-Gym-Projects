#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:29:24 2019

@author: kad99kev
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

# Constants
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2_000
DISCRETE_OBS_SIZE = [20]*len(env.observation_space.high)
SHOW_EVERY = 5_00

# Exploration settings
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OBS_SIZE + [env.action_space.n]))

ep_rewards = []
aggre_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_obs_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):

    episode_reward = 0

    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) #Get action from Q Table
        else:
            action = np.random.randint(0, env.action_space.n) # Get random action

        new_state, reward, done, info = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state]) # Max value from q_table for new discrete state
            current_q = q_table[discrete_state + (action, )] # Max value from q_table for current state, using index slicing
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        avg_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards)
        aggre_ep_rewards['ep'].append(episode)
        aggre_ep_rewards['avg'].append(avg_reward)
        aggre_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggre_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Epsiode: {episode} | Avg: {avg_reward} | Min: {min(ep_rewards[-SHOW_EVERY:])} | Max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['avg'], label = 'avg')
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['min'], label = 'min')
plt.plot(aggre_ep_rewards['ep'], aggre_ep_rewards['max'], label = 'max')
plt.legend(loc = 4)
plt.show()
