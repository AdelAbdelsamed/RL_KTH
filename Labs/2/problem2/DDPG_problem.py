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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import DDPG_agent as DDPG_agents
import DDPG_utils as utils
import pdb

# Tasks
BASE = False
DISCOUNT_FACTOR_EFFECT = False
MEMORY_SIZE_EFFECT = False
Q_FCN_ANALYSIS = True
COMPARE_DDPG_RANDOM_AGENT = False

############################### Utility functions ###############################
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

def plot_rewards_and_no_steps(episode_reward_list, episode_number_of_steps):
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, len(episode_number_of_steps)+1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show(block = False)

def agent_simulate(N_episodes, agent):
    # Reward
    episode_reward_list = []  # Used to store episodes reward
    # Simulate episodes
    print('Simulating...')
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done = False
        state = env.reset()[0]
        total_episode_reward = 0.
        t = 0
        while t < max_t_steps and not done:
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
    return episode_reward_list

def ddpg_train(ddpg_agent, N_episodes, max_t_steps = 1000):
    ### Training process
    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode

    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()[0]

        total_episode_reward = 0.
        t = 0
        while t < max_t_steps and not done:
            # Take a random action
            #action = agent.forward(state)
            action = ddpg_agent.act(state, True)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _, _ = env.step(action)
            # Perform a step backward stepp through the network
            ddpg_agent.step(state, action, reward, next_state, done)

            # Update episode reward
            total_episode_reward += reward
            # Update state for next iteration
            state = next_state
            t+= 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        target_rewards = running_average(episode_reward_list, n_ep_running_average)[-1] > 150
        if target_rewards:
            break
                
        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))
        
    return episode_reward_list, episode_number_of_steps

#############################################################################

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 300                # Number of episodes to run for training
discount_factor = 0.99          # Value of gamma
buffer_size = 30000             # Size of the experience replay buffer
batch_size = 64                 # Size of the training batch 
target_networks_freq_update = 2 # Target network frequency update
n_ep_running_average = 50       # Running average of 50 episodes
soft_update_const = 1e-3        # Parameter for soft update of target networks
lr_actor = 5e-5                 # Learning rate for the actor network 
lr_critic = 5e-4                # Learning rate for the critic network 
max_t_steps = 1000              # Maximum no. steps
mu = 0.15                       # Noise Decay rate 
sigma = 0.2                     # Noise variance
m = len(env.action_space.high)  # dimensionality of the action
dim_state = len(env.observation_space.high)  # dimensionality of the state

if BASE:
    # Noise initialization
    noise = utils.OrnUHLnoise(m, sigma, mu)
    # Network initialization 
    actor_network = utils.ActorNetwork(dim_state, m)
    critic_network = utils.CriticNetwork(dim_state, m)
    # Agent initialization
    random_agent = DDPG_agents.RandomAgent(m)
    ddpg_agent = DDPG_agents.DDPGAgent(m, env, noise, critic_network, actor_network, discount_factor, buffer_size, batch_size, 
                    lr_actor, lr_critic, soft_update_const, target_networks_freq_update)
    episode_reward_list, episode_number_of_steps = ddpg_train(ddpg_agent, N_episodes)
    plot_rewards_and_no_steps(episode_reward_list, episode_number_of_steps)

    # Save the actor and critic networks
    torch.save(ddpg_agent.critic_main_network, 'neural-network-8-critic.pth')
    torch.save(ddpg_agent.actor_main_network, 'neural-network-8-actor.pth')
elif DISCOUNT_FACTOR_EFFECT:
    discount_factors_list = [1, 0.4]
    for i in range(len(discount_factors_list)):
        # Noise initialization
        noise = utils.OrnUHLnoise(m, sigma, mu)
        # Network initialization 
        actor_network = utils.ActorNetwork(dim_state, m)
        critic_network = utils.CriticNetwork(dim_state, m)
        ddpg_agent = DDPG_agents.DDPGAgent(m, env, noise, critic_network, actor_network, discount_factors_list[i], buffer_size, batch_size, 
                        lr_actor, lr_critic, soft_update_const, target_networks_freq_update)
        episode_reward_list, episode_number_of_steps = ddpg_train(ddpg_agent, N_episodes)
        # Plot the rewards and steps
        plot_rewards_and_no_steps(episode_reward_list, episode_number_of_steps)
elif MEMORY_SIZE_EFFECT:
    buffer_size_list = [10000, 40000]
    for i in range(len(buffer_size_list)):
        # Noise initialization
        noise = utils.OrnUHLnoise(m, sigma, mu)
        # Network initialization 
        actor_network = utils.ActorNetwork(dim_state, m)
        critic_network = utils.CriticNetwork(dim_state, m)
        ddpg_agent = DDPG_agents.DDPGAgent(m, env, noise, critic_network, actor_network, discount_factor, buffer_size_list[i], batch_size, 
                        lr_actor, lr_critic, soft_update_const, target_networks_freq_update)
        episode_reward_list, episode_number_of_steps = ddpg_train(ddpg_agent, N_episodes)
        # Plot the rewards and steps
        plot_rewards_and_no_steps(episode_reward_list, episode_number_of_steps)
elif Q_FCN_ANALYSIS:
    # Load model
    try:
        actor = torch.load('neural-network-2-actor.pth')
        print('Network model: {}'.format(actor))
        critic = torch.load('neural-network-2-critic.pth')
        print('Network model: {}'.format(critic))
    except:
        print('File neural-network-2-actor.pth or neural-network-2-critic.pth not found!')
        exit(-1)

    # Generate state values for y and w
    y_values = np.linspace(0, 1.5, 100)
    w_values = np.linspace(-np.pi, np.pi, 100)
    y_grid, w_grid = np.meshgrid(y_values, w_values)

    # Evaluate Q(s, a) for each combination of y and w
    z_values = np.zeros_like(y_grid)
    engine_direction = np.zeros_like(y_grid)
    for i in range(y_grid.shape[0]):
        for j in range(y_grid.shape[1]):
            state = np.array([0, y_grid[i, j], 0, 0, w_grid[i, j], 0, 0, 0])
            z_values[i, j] = critic(torch.tensor(np.array([state]), requires_grad=False, 
                                    dtype= torch.float32), actor(torch.tensor(np.array([state]), 
                                    requires_grad = False, dtype= torch.float32))).detach().numpy()
            # Dertermine Direction
            a = actor(torch.tensor(np.array([state]), requires_grad=False, 
                                    dtype= torch.float32))[0].detach().numpy()
            if -1 <= a[-1] < 0.5:
                engine_direction[i,j] = -1.0 # Fire left
            elif -0.5 <= a[-1] < 0.5: 
                engine_direction[i,j] = 0.0  # Do nothing
            else:
                engine_direction[i,j] = 1.0  # Fire right 

    # Create a figure with two subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    # Plot the first subplot (z_values)
    ax1.plot_surface(y_grid, w_grid, z_values, cmap='viridis')
    ax1.set_xlabel('Height of the lander y')
    ax1.set_ylabel('Angle of the lander omega')
    ax1.set_zlabel('Q(s,pi(s))')
    # Plot the second subplot (arg_z_values)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(y_grid, w_grid, engine_direction, cmap='viridis')
    ax2.set_xlabel('Height of the lander y')
    ax2.set_ylabel('Angle of the lander omega')
    ax2.set_zticks([-1.0, 0.0, 1.0], [' Left En',' Nothing',' Right En'])
    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show(block=False)
elif COMPARE_DDPG_RANDOM_AGENT:
    # Load model
    try:
        actor = torch.load('neural-network-2-actor.pth')
        print('Network model: {}'.format(actor))
        critic = torch.load('neural-network-2-critic.pth')
        print('Network model: {}'.format(critic))
    except:
        print('File neural-network-2-actor.pth or neural-network-2-critic.pth not found!')
        exit(-1)
    # Noise initialization
    noise = utils.OrnUHLnoise(m, sigma, mu)
    # Random agent initialization
    random_agent = DDPG_agents.RandomAgent(m)
    # DDPG Agent initilization
    ddpg_agent = DDPG_agents.DDPGAgent(m, env, noise, critic, actor)
    # Simulate
    random_agent_rewards_list = agent_simulate(50, random_agent)
    ddpg_agent_rewards_list = agent_simulate(50, ddpg_agent)
    # Plotting
    plt.plot(random_agent_rewards_list,'g', label='Random Agent')
    plt.plot(np.arange(0,50), np.ones((50))*np.mean(np.array(random_agent_rewards_list)), 'g--')
    plt.plot(ddpg_agent_rewards_list,'b', label='DDPG Agent')
    plt.plot(np.arange(0,50), np.ones((50))*np.mean(ddpg_agent_rewards_list), 'b--')
    # Adding labels and title
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Comparison of Rewards between Agents')
    # Adding legend
    plt.legend()
    # Show the plot
    plt.show(block = False)





env.close()
plt.show()