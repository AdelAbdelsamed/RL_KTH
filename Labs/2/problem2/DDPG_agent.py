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
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch.optim as optim
import DDPG_utils as utils
import torch 
import torch.nn as nn 
import pdb


class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

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

    def act(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
    

class DDPGAgent(Agent):
    ''' Agent taking actions uniformly at , random, child of the class Agent'''
    def __init__(self, dim_action: int, env, noise, critic_network, actor_network, discount_factor = 0.99, buffer_size = 30000, batch_size = 64, 
                 lr_actor = 5e-5, lr_critic = 5e-4, soft_update_const = 1e-3, update_freq = 2, cer = False):
        super(DDPGAgent, self).__init__(dim_action)
        self.dim_state = len(env.observation_space.high)  # State dimensionality
        self.dim_action = dim_action      # Action dimensionality
        self.discount_factor = discount_factor

        # Training Parameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.update_frequency = update_freq
        self.soft_update_const = soft_update_const
        self.cer = cer # Combined experience replay
        
        # Networks
        self.actor_main_network = actor_network
        self.actor_target_network = actor_network
        self.critic_main_network = critic_network
        self.critic_target_network = critic_network
        self.optimizer_actor = optim.Adam(self.actor_main_network.parameters(), lr = self.lr_actor)
        self.optimizer_critic = optim.Adam(self.critic_main_network.parameters(), lr = self.lr_critic)

        # Noise
        self.noise = noise

        # Replay Buffer
        self.buffer = utils.ExperienceReplayBuffer(buffer_size, self.dim_action)
        self.buffer.fill_buffer_randomly(buffer_size) 

        # Time step
        self.t = 0 

    def act(self, state, explore = False):
        ''' Function used to compute the next action according to the current
            actor network with added noise
        '''
        action = self.actor_main_network(torch.tensor(np.array(state), 
                                            requires_grad=False,
                                            dtype=torch.float32))
        action = action.detach().numpy()
        if explore:
            action += self.noise.forward()
        return np.clip(action, -1, 1)
    
    def step(self, state, action, reward, next_state, done):
        # Add experience in buffer
        exp = utils.Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

        # Perform backward pass on the networks
        self.backward()

        if done:
            self.t = 0
            self.noise.reset() # Reset the noise filter
        else:
            self.t += 1    

    def backward(self):
        # Sample from the batch 
        states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size, self.cer)

        # Compute target values y
        dones = np.array(dones)
        dones[dones == True] = 1
        dones[dones != True] = 0        
        next_s_tensor = torch.tensor(np.array(next_states), requires_grad=False,
                                dtype=torch.float32)
        next_action_tensor = self.actor_target_network(next_s_tensor)
        next_value = self.critic_target_network(next_s_tensor, next_action_tensor).detach().numpy()
        y = np.array(rewards) + (1-dones)*self.discount_factor*next_value.flatten()

        # Compute output of the network given the states batch
        pred_values = self.forward_critic(np.array(states), np.array(actions))
        # Compute loss function
        q_loss = nn.functional.mse_loss(torch.tensor(y, 
                                                    requires_grad=False, dtype= torch.float32),
                                                    pred_values.flatten())
        # Update critc network 
        self.optimizer_critic.zero_grad()                                             # Reset the gradient
        q_loss.backward()                                                               # Compute gradient
        nn.utils.clip_grad_norm_(self.critic_main_network.parameters(), max_norm=1.)  # Clip gradient norm to 1
        self.optimizer_critic.step()

        # Delayed actor network uodate and target networks update
        if np.mod(self.t, self.update_frequency) == 0:
            # Update actor network
            states_tensor = torch.tensor(np.array(states), requires_grad=False, dtype=torch.float32)
            policy_loss = -self.critic_main_network.forward(states_tensor, self.actor_main_network.forward(states_tensor)).mean()
            self.optimizer_actor.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_main_network.parameters(), max_norm=1.)  # Clip gradient norm to 1
            self.optimizer_actor.step()

            # Soft update target networks
            self.actor_target_network = utils.soft_updates(self.actor_main_network,
                 self.actor_target_network, self.soft_update_const)
            self.critic_target_network = utils.soft_updates(self.critic_main_network,
                 self.critic_target_network, self.soft_update_const)  

    def forward_critic(self, state, action):
        state_tensor = torch.tensor(state, requires_grad = False, dtype = torch.float32)
        action_tensor = torch.tensor(action, requires_grad = False, dtype = torch.float32)
        value = self.critic_main_network(state_tensor, action_tensor)
        return value
    
