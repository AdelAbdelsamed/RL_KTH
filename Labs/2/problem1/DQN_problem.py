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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import DQN_utils as util
import pdb
from tqdm import trange
import DQN_agent as DQN

# Tasks
BASE = True
DISCOUNT_FACTOR_EFFECT = False
NO_EPISODES_EFFECT = False
MEMORY_SIZE_EFFECT = False
Q_FCN_ANALYSIS = False
COMPARE_DQN_RANDOM_AGENT = False



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

def decay_epsilon_linearly(episode_k):
    return np.max([epsilon_min, epsilon_max - (epsilon_max-epsilon_min)*episode_k/(Z - 1)])

def decay_epsilon_exponentially(episode_k):
    return np.max([epsilon_min, epsilon_max*np.power(epsilon_min/epsilon_max, episode_k/(Z-1))])

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

def dqn_train(dqn_agent, N_episodes, max_t_steps = 1000):
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

        # Decay ecploration rate
        #epsilon_k = decay_epsilon_linearly(i)
        epsilon_k = decay_epsilon_exponentially(i)


        total_episode_reward = 0.
        t = 0
        while t < max_t_steps and not done:
            # Take a random action
            #action = agent.forward(state)
            action = dqn_agent.act(state, epsilon_k)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _, _ = env.step(action)
            # Perform a step backward stepp through the network
            dqn_agent.step(state, action, reward, next_state, done)

            # Update episode reward
            total_episode_reward += reward
            # Update state for next iteration
            state = next_state
            t+= 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        target_rewards = running_average(episode_reward_list, n_ep_running_average)[-1] > 100
        target_episodes = running_average(episode_number_of_steps, n_ep_running_average)[-1] < 200
        if target_rewards and target_episodes:
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


############################### DQN ###############################
# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()
# For reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Parameters
N_episodes = 400                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
buffer_size = 10000                          # Size of the experience replay buffer
batch_size = 96                              # Size of the training samples batch 
learning_rate = 1e-4                         # Learning Rate of the NN
cer = False                                  # Combined experience replay
max_t_steps = 1000                           # Maximum no. of steps
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# Exploration parameters
epsilon_max = 0.95                           # Maximum of exploration factor
epsilon_min = 0.05                           # Minimum of exploration factor
Z = np.ceil(0.9*N_episodes)   

if BASE:
    # Initialize Main Network
    main_network = util.NeuralNetwork(input_size= dim_state, output_size= n_actions, hidden_layer_size =  100)
    dqn_agent = DQN.DQNAgent(n_actions, env, main_network, discount_factor, buffer_size, batch_size, learning_rate)    
    # Train the DQN agent
    episode_reward_list, episode_number_of_steps = dqn_train(dqn_agent, N_episodes)
    plot_rewards_and_no_steps(episode_reward_list, episode_number_of_steps)
    # Save the neural network
    torch.save(dqn_agent.main_network, 'neural-network-2.pth')  
elif DISCOUNT_FACTOR_EFFECT:
    discount_factors_list = [1, 0.99, 0.4]
    for i in range(len(discount_factors_list)):
        # Initialize Main Network
        main_network = util.NeuralNetwork(input_size= dim_state, output_size= n_actions, hidden_layer_size =  100)
        dqn_agent = DQN.DQNAgent(n_actions, env, main_network, discount_factors_list[i], buffer_size, batch_size, learning_rate)    
        # Train the DQN agent
        episode_reward_list, episode_number_of_steps = dqn_train(dqn_agent, N_episodes)
        # Plot the rewards and steps
        plot_rewards_and_no_steps(episode_reward_list, episode_number_of_steps)
elif NO_EPISODES_EFFECT:
    no_episodes_list = [800]
    # Initialize Main Network
    for i in range(len(no_episodes_list)):
        main_network = util.NeuralNetwork(input_size= dim_state, output_size= n_actions, hidden_layer_size =  100)
        dqn_agent = DQN.DQNAgent(n_actions, env, main_network, discount_factor, buffer_size, batch_size, learning_rate)    
        # Train the DQN agent
        episode_reward_list, episode_number_of_steps = dqn_train(dqn_agent, no_episodes_list[i])
        # Plot the rewards and steps
        plot_rewards_and_no_steps(episode_reward_list, episode_number_of_steps)
elif MEMORY_SIZE_EFFECT:
    buffer_size_list = [20000, 30000]
    for i in range(len(buffer_size_list)):
        # Initialize Main Network
        main_network = util.NeuralNetwork(input_size= dim_state, output_size= n_actions, hidden_layer_size =  100)
        dqn_agent = DQN.DQNAgent(n_actions, env, main_network, discount_factor, buffer_size_list[i], batch_size, learning_rate)    
        # Train the DQN agent
        episode_reward_list, episode_number_of_steps = dqn_train(dqn_agent, N_episodes)
        # Plot the rewards and steps
        plot_rewards_and_no_steps(episode_reward_list, episode_number_of_steps)
elif Q_FCN_ANALYSIS:
    # Load model
    try:
        Q_network = torch.load('neural-network-1.pth')
        print('Network model: {}'.format(Q_network))
    except:
        print('File neural-network-1.pth not found!')
        exit(-1)

    # Generate state values for y and w
    y_values = np.linspace(0, 1.5, 100)
    w_values = np.linspace(-np.pi, np.pi, 100)
    y_grid, w_grid = np.meshgrid(y_values, w_values)

    # Evaluate Q(s, a) for each combination of y and w
    z_values = np.zeros_like(y_grid)
    arg_z_values = np.zeros_like(y_grid)
    for i in range(y_grid.shape[0]):
        for j in range(y_grid.shape[1]):
            state = np.array([0, y_grid[i, j], 0, 0, w_grid[i, j], 0, 0, 0])
            Q_values = Q_network(torch.tensor([state], requires_grad = False, dtype = torch.float32))
            z_values[i, j] = Q_values.max(1)[0].item()
            arg_z_values[i,j] = Q_values.max(1)[1].item()

    # Create a figure with two subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    # Plot the first subplot (z_values)
    ax1.plot_surface(y_grid, w_grid, z_values, cmap='viridis')
    ax1.set_xlabel('Height of the lander y')
    ax1.set_ylabel('Angle of the lander omega')
    ax1.set_zlabel('max_a Q(s,a)')
    ax1.set_title('Q*(s)')
    # Plot the second subplot (arg_z_values)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(y_grid, w_grid, arg_z_values, cmap='viridis')
    ax2.set_xlabel('Height of the lander y')
    ax2.set_ylabel('Angle of the lander omega')
    ax2.set_title('Best actions: argmax_a Q(s,a)')
    ax2.set_zticks([0, 1, 2, 3], ['Nothing','Left En', 'Main En','Right En'])
    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show(block=False)
elif COMPARE_DQN_RANDOM_AGENT:
    # Load model
    try:
        Q_network = torch.load('neural-network-1.pth')
        print('Network model: {}'.format(Q_network))
    except:
        print('File neural-network-1.pth not found!')
        exit(-1)

    # Random agent initialization
    random_agent = DQN.RandomAgent(n_actions)
    # Dueling DQN initialization
    dueling_dqn_agent = DQN.DQNAgent(n_actions, env, Q_network)
    # Simulate
    random_agent_rewards_list = agent_simulate(50, random_agent)
    dueling_dqn_agent_rewards_list = agent_simulate(50, dueling_dqn_agent)

    # Plotting
    plt.plot(random_agent_rewards_list,'g', label='Random Agent')
    plt.plot(np.arange(0,50), np.ones((50))*np.mean(np.array(random_agent_rewards_list)), 'g--')
    plt.plot(dueling_dqn_agent_rewards_list,'b', label='Dueling DQN Agent')
    plt.plot(np.arange(0,50), np.ones((50))*np.mean(dueling_dqn_agent_rewards_list), 'b--')
    # Adding labels and title
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Comparison of Rewards between Agents')
    # Adding legend
    plt.legend()
    # Show the plot
    plt.show(block = False)



# Close environment
env.close()

plt.show()