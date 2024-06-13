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
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pickle
from tqdm import tqdm

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 1000        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma


m = 8 # Number of features
state_dim = 2 # Dimension of the state space
eta = np.array([
    #[0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 0],
    [2, 1],
    [0, 2],
    [1, 2],
    [2, 2]
])
lmbda = 0.9
epsilon = 0.001
lr = 0.0015

PLOT_STATE_SPACE = False
SAVE_WEIGHTS = False
VALIDATE_SOLUTION = True
# choices: None, 'learning rate', 'lambda', 'weight'
PLOT_AVERAGE_REWARD_AS_FUNCTION_OF = 'temperature'

exploration_strategy = 1
temperature = 0.5

def fourier(s):
    result = np.cos(np.pi * np.dot(eta, s))
    return result

class SarsaAgent:
    def __init__(self, action_space, num_features, discount, lmbda, epsilon, lr, exploration_strategy=0, temperature=0.5):
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.discount = discount
        self.action_space = action_space
        self.num_features = num_features
        self.lr = lr
        self.exploration_strategy = exploration_strategy
        self.temperature = temperature

        # Eligibility trace
        self.z = np.zeros((action_space, num_features))

        # Weights
        #self.w = np.random.random((action_space, num_features))
        #self.w = np.zeros((action_space, num_features))
        #self.w = np.random.normal(0, 1, (action_space, num_features))
        
        self.w = np.zeros((action_space, num_features))
        for i in range(action_space):
            self.w[i] = np.random.normal(0, 1, (num_features,)) -1 + i

    
    def get_action(self, state):
        if self.exploration_strategy == 0:
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.action_space)
            else:
                return np.argmax(np.dot(self.w, fourier(state)))
        else:
            if np.random.rand() < self.epsilon:
                soft_P = np.zeros((self.action_space))
                for i in range(self.action_space):
                    soft_P[i] = np.exp(np.dot(self.w[i, :], fourier(state))/self.temperature)
                soft_P = np.exp(soft_P - np.max(soft_P))
                #print(soft_P)
                soft_P = soft_P / np.sum(soft_P)
                return np.random.choice(self.action_space, p=soft_P)
            else:
                return np.argmax(np.dot(self.w, fourier(state)))
    
    def update(self, state, action, reward, next_state):
        next_action = self.get_action(next_state)
        self.z *= self.discount * self.lmbda
        # z[action] += grad_w Q(s, a)
        self.z[action, :] += fourier(state)
        self.z = np.clip(self.z, -5, 5)
        delta = reward + self.discount * np.dot(self.w[next_action], fourier(next_state)) - np.dot(self.w[action], fourier(state))
        self.w += self.lr * delta * self.z
    
    def reset(self):
        self.z = np.zeros((self.action_space, self.num_features))
    
    def plot_state_space(self, n_points=100):
        s_1 = np.linspace(low[0], high[0], n_points)
        s_2 = np.linspace(low[1], high[1], n_points)

        s_1_n = scale_state_variables(s_1, low[0], high[0])
        s_2_n = scale_state_variables(s_2, low[1], high[1])
        
        zs = np.zeros((n_points, n_points))
        action = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                value = np.dot(self.w, fourier([s_1_n[i], s_2_n[j]]))
                zs[i, j] = np.max(value)
                action[i, j] = np.argmax(value)
        
        S_1, S_2 = np.meshgrid(s_1, s_2)
        norm = Normalize(action.min(), action.max())

        # Map the normalized action values to colors
        colors = cm.coolwarm(norm(action))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S_1.reshape(n_points, n_points), S_2.reshape(n_points, n_points), zs, facecolors=colors)
        #surf.set_facecolor((0,0,0,0))

        m = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
        m.set_array(action)
        fig.colorbar(m, ax=ax)

        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Value')

        plt.show()
    
    def save_weights(self):
        data = {
            'W': self.w,
            'N': eta
        }
        with open('weights.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x


def train(stop_at_goal=True, verbose=True):# Reward
    episode_reward_list = []  # Used to save episodes reward
    agent = SarsaAgent(k, m, discount_factor, lmbda, epsilon, lr, exploration_strategy, temperature)
    if PLOT_STATE_SPACE: agent.plot_state_space()

    # Training process
    for epoch_nr in range(N_episodes):
        # Reset enviroment data
        done = False
        state = scale_state_variables(env.reset()[0])
        total_episode_reward = 0.

        t = 0
        agent.reset()
        while not done and t < 200:
            # Get action
            action = agent.get_action(state)
                
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _, __ = env.step(action)
            #print(next_state, action)
            next_state = scale_state_variables(next_state)

            # Update agent
            agent.update(state, action, reward, next_state)

            # Update episode reward
            total_episode_reward += reward
                
            # Update state for next iteration
            state = next_state
            
            t += 1
        
        if verbose: print(f'Episode {epoch_nr} reward: {total_episode_reward}')
        #if i % 10 == 0:
        #    print(agent.w)

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Check for solved condition
        if stop_at_goal and epoch_nr >= 50 and running_average(episode_reward_list, 50)[-1] > -135:
            if verbose: print(f'Solved in {epoch_nr} episodes!')
            break

        # Close environment
        env.reset()

    if SAVE_WEIGHTS: agent.save_weights()
    return agent, episode_reward_list

def Qvalues(s, w):
    ''' Q Value computation '''
    return np.dot(w, s)

def scale_state_varibles(s, eta, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2
        and features transformation
    '''
    x = (s-low) / (high-low)
    return np.cos(np.pi * np.dot(eta, x))

# Plot Rewards
def plot_rewards(episode_reward_list):
    plt.plot([i for i in range(1, len(episode_reward_list)+1)], episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, len(episode_reward_list)+1)], running_average(episode_reward_list, 10), label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if PLOT_AVERAGE_REWARD_AS_FUNCTION_OF == None:
    agent, rs = train()
    plot_rewards(rs)
elif PLOT_AVERAGE_REWARD_AS_FUNCTION_OF == 'learning rate':
    N_episodes = 500
    # distribute the learning rate on a log scale
    learning_rates = np.logspace(-4, -1, 10)
    avg_rewards = []
    conf_intervals = []
    for lr_new in learning_rates:
        lr = lr_new
        reward_for_lr = []
        for i in tqdm(range(50), desc=f'Learning rate {lr_new}'):
            agent, rs = train(verbose=False)
            reward_for_lr.append(np.mean(rs))
        avg_rewards.append(np.mean(reward_for_lr))
        conf_intervals.append(np.std(reward_for_lr) * 1.96 / np.sqrt(50))
    plt.errorbar(learning_rates, avg_rewards, yerr=conf_intervals, fmt='o')
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('Average reward')
    plt.title('Average reward as a function of the learning rate')
    plt.show()
elif PLOT_AVERAGE_REWARD_AS_FUNCTION_OF == 'lambda':
    N_episodes = 500
    lambdas = np.linspace(0, 1, 11)
    avg_rewards = []
    conf_intervals = []
    for lmbda_new in lambdas:
        lmbda = lmbda_new
        reward_for_lmbda = []
        for i in tqdm(range(50), desc=f'Lambda {lmbda_new}'):
            agent, rs = train(verbose=False)
            reward_for_lmbda.append(np.mean(rs))
        avg_rewards.append(np.mean(reward_for_lmbda))
        conf_intervals.append(np.std(reward_for_lmbda) * 1.96 / np.sqrt(50))
    plt.errorbar(lambdas, avg_rewards, yerr=conf_intervals, fmt='o')
    plt.xlabel('Lambda')
    plt.ylabel('Average reward')
    plt.title('Average reward as a function of the egibility trace')
    plt.show()
elif PLOT_AVERAGE_REWARD_AS_FUNCTION_OF == 'weight':
    N_episodes = 500
    rewards = []
    for i in tqdm(range(50)):
        agent, rs = train(stop_at_goal=False, verbose=False)
        rewards.append(rs)
    print(f'Average reward: {np.mean(rewards)}')
    print(f'Confidence interval: {np.std(rewards) * 1.96 / np.sqrt(50)}')
elif PLOT_AVERAGE_REWARD_AS_FUNCTION_OF == 'temperature':
    N_episodes = 500
    temp = np.array([1, 10, 100, 1000, 10000])
    avg_rewards = []
    conf_intervals = []
    for lmbda_new in temp:
        temperature = lmbda_new
        reward_for_lmbda = []
        for i in tqdm(range(50), desc=f'Temperature {lmbda_new}'):
            agent, rs = train(verbose=False)
            reward_for_lmbda.append(np.mean(rs))
        avg_rewards.append(np.mean(reward_for_lmbda))
        conf_intervals.append(np.std(reward_for_lmbda) * 1.96 / np.sqrt(50))
    plt.errorbar(temp, avg_rewards, yerr=conf_intervals, fmt='o')
    plt.xlabel('Temperature')
    plt.xscale('log')
    plt.ylabel('Average reward')
    plt.title('Average reward as a function of the temperature')
    plt.show()
    





# validate
if VALIDATE_SOLUTION:
    episode_reward_list_v = []
    for i in range(50):
        # Reset enviroment data
        done = False
        state = scale_state_varibles(env.reset()[0], eta, low, high)
        total_episode_reward = 0.

        qvalues = Qvalues(state, agent.w)
        action = np.argmax(qvalues)

        t = 0
        while not done and t < 10000:
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _, _ = env.step(action)
            next_state = scale_state_varibles(next_state, eta, low, high)
            qvalues_next = Qvalues(next_state, agent.w)
            next_action = np.argmax(qvalues_next)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            qvalues = qvalues_next
            action = next_action

            t += 1

        # Append episode reward
        episode_reward_list_v.append(total_episode_reward)

        # Close environment
        env.close()

    avg_reward = np.mean(episode_reward_list_v)
    confidence = np.std(episode_reward_list_v) * 1.96 / np.sqrt(50)

    print(f'Validation: Policy achieves an average total reward of {avg_reward:.1f} +/- {confidence:.1f} with confidence 95%.')

if PLOT_STATE_SPACE: agent.plot_state_space()


env.close()