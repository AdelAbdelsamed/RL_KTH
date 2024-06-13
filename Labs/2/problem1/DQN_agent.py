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
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch.optim as optim
import DQN_utils as utils
import torch 
import torch.nn as nn 
import pdb

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def act(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action
    

class DQNAgent(Agent):
    ''' Agent taking actions uniformly at , random, child of the class Agent'''
    def __init__(self, n_actions: int, env, main_network, discount_factor = 0.99, buffer_size = 30000, batch_size = 64, learning_rate = 1e-4, cer = False):
        super(DQNAgent, self).__init__(n_actions)
        self.dim_state = len(env.observation_space.high)  # State dimensionality
        self.discount_factor = discount_factor

        # Training Parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency = 104
        #int(buffer_size/self.batch_size)
        self.cer = cer # Combined experience replay
        
        # Network Architecture
        self.main_network = main_network
        self.target_network = main_network
        self.optimizer = optim.Adam(self.main_network.parameters(), lr = self.learning_rate)

        # Replay Buffer
        self.buffer = utils.ExperienceReplayBuffer(buffer_size)
        self.buffer.fill_buffer_randomly(buffer_size) 

        # Time step
        self.t = 0 

    def act(self, state, epsilon = 0.):
        ''' Function used to compute the next action according to an
            epsilon-greedy policy w.r.t to the main network
        '''
        values = self.main_network(torch.tensor(np.array([state]), 
                                            requires_grad=False,
                                            dtype=torch.float32))
        if np.random.uniform() <= epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = values.max(1)[1].item()
        return a
    
    def step(self, state, action, reward, next_state, done):
        # Add experience in buffer
        exp = utils.Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

        # Perform backward pass on the main network
        self.backward()

        # Update target network
        if np.mod(self.t, self.update_frequency) == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        if done:
            self.t = 0
        else:
            self.t += 1    

    def backward(self):
        # Sample from the batch 
        states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size, self.cer)

        # Compute target values y
        dones[dones == True] = 1
        dones[dones != True] = 0
        next_s_tensor = torch.tensor(np.array(next_states), requires_grad=False,
                                dtype=torch.float32)
        max_next_s_value = self.target_network(next_s_tensor).max(1)[0].detach().numpy()
        y = np.array(rewards) + (1-dones)*self.discount_factor*max_next_s_value

        # Compute output of the network given the states batch
        pred_values = self.forward(np.vstack(states), actions)
        # Compute loss function
        loss = nn.functional.mse_loss(torch.tensor(y, 
                                                    requires_grad=False, dtype= torch.float32),
                                                    pred_values)
        # Update theta 
        self.optimizer.zero_grad()                                             # Reset the gradient
        loss.backward()                                                        # Compute gradient
        nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.)  # Clip gradient norm to 1
        self.optimizer.step()  

    def forward(self, state, action):
        value = self.main_network(torch.tensor(state, requires_grad = False, dtype = torch.float32))
        return value[np.arange(len(action)), action]
    