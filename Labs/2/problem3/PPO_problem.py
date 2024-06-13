# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from PPO_agent import RandomAgent, PPOAgent

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 50               # Number of episodes to run for training
discount_factor = 0.99         # Value of gamma
n_ep_running_average = 50      # Running average of 20 episodes
m = len(env.action_space.high) # dimensionality of the action
eps = 0.2
M = 10
SAVE_MODEL = False
early_stop = False
COMPARE_RANDOM = True

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# Agent initialization
#agent = RandomAgent(m)
agent = PPOAgent(n_actions=m, dim_state=len(env.observation_space.high), lr_actor=1e-5, lr_critic=1e-3)
if COMPARE_RANDOM:
    agent.actor = torch.load('neural-network-3-actor.pth')
    agent.critic = torch.load('neural-network-3-critic.pth')

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data
    done = False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    agent.new_episode()
    while not done:
        # Take a random action
        action = agent.forward(state)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, terminated, _ = env.step(action)
        done = done or terminated

        agent.add_to_buffer(state, action, reward, next_state, done)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    agent.backward(discount_factor, eps, M)

    # Close environment
    env.close()

    average_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
    if average_reward >= 140 and early_stop:
        if SAVE_MODEL:
            agent.save()
        break
    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        average_reward,
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))
    
if COMPARE_RANDOM:
    rand_agent = RandomAgent(m)
    rand_episode_reward_list = []       # this list contains the total reward per episode
    rand_episode_number_of_steps = []
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state, _options = env.reset()
        total_episode_reward = 0.
        t = 0
        while not done:
            # Take a random action
            action = rand_agent.forward(state)
            next_state, reward, done, terminated, __ = env.step(action)

            done = done or terminated

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward and total number of steps
        rand_episode_reward_list.append(total_episode_reward)
        rand_episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        running_avg_reward = running_average(rand_episode_reward_list, n_ep_running_average)[-1]
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_avg_reward,
            running_average(rand_episode_number_of_steps, n_ep_running_average)[-1]))


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], episode_reward_list, label='Episode reward')
if COMPARE_RANDOM:
    ax[0].plot([i for i in range(1, len(rand_episode_reward_list)+1)], rand_episode_reward_list, label='Random agent reward')
else:
    ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], episode_number_of_steps, label='Steps per episode')
if COMPARE_RANDOM:
    ax[1].plot([i for i in range(1, len(rand_episode_number_of_steps)+1)], rand_episode_number_of_steps, label='Random agent reward')
else:
    ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
