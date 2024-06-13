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
# Last update: 29th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch

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

class ActorModel(torch.nn.Module):
    def __init__(self, dim_state, n_actions):
        super(ActorModel, self).__init__()
        self.actor_shared = torch.nn.Sequential(
            torch.nn.Linear(dim_state, 400),
            torch.nn.ReLU()
        )
        self.actor_mean = torch.nn.Sequential(
            torch.nn.Linear(400, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, n_actions),
            torch.nn.Tanh()
        )
        self.actor_var = torch.nn.Sequential(
            torch.nn.Linear(400, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, n_actions),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        shared_output = self.actor_shared(x)
        mean = self.actor_mean(shared_output)
        var = self.actor_var(shared_output)
        return mean, var

class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)

class PPOAgent(Agent):
    def __init__(self, n_actions: int, dim_state: int, lr_actor: float = 3e-4, lr_critic: float = 3e-4):
        super(PPOAgent, self).__init__(n_actions)
        self.dim_state = dim_state

        self.actor = ActorModel(dim_state, n_actions)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(dim_state, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 1)
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def new_episode(self):
        self.buffer = []
        self.log_probs = []
    
    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            mean, var = self.actor(state)
        
        dist = torch.distributions.MultivariateNormal(mean, torch.diag(var))
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action).detach().numpy())

        return action.numpy()
    
    def backward(self, discount_factor, eps, epochs=10):
        states = torch.tensor(np.array([x[0] for x in self.buffer]), dtype=torch.float32)
        actions = torch.tensor(np.array([x[1] for x in self.buffer]), dtype=torch.float32)
        rewards = torch.tensor(np.array([x[2] for x in self.buffer]), dtype=torch.float32)
        #next_states = torch.tensor([x[3] for x in self.buffer], dtype=torch.float32)
        dones = torch.tensor([x[4] for x in self.buffer], dtype=torch.float32)

        # compute returns
        returns = torch.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = rewards[t] + discount_factor * running_add * (1 - dones[t])
            returns[t] = running_add

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(epochs):
            # compute advantages
            values = self.critic(states).flatten()
            advantages = returns - values.detach()

            # normalize advantages
            #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            old_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32)
            # Get the policy distribution based on the current policy
            mean, var = self.actor(states)
            dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(var))

            # Get the log probabilities for the actions taken
            new_log_probs = dist.log_prob(actions)

            # Calculate the ratio between new and old probabilities
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Calculate surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps, 1 + eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Update actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic network
            critic_loss = torch.nn.functional.mse_loss(values, returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
    
    def save(self):
        torch.save(self.actor, 'neural-network-3-actor.pth')
        torch.save(self.critic, 'neural-network-3-critic.pth')