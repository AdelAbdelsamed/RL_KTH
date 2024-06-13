import numpy as np
import gym
import torch
import torch.nn as nn
import DDPG_agent as DDPG_agent
import pdb
from collections import deque, namedtuple



############################### Experience class (from Lab 0) ###############################

# namedtuple is used to create a special type of tuple object. Namedtuples
# always have a specific name (like a class) and specific fields.
# In this case I will create a namedtuple 'Experience',
# with fields: state, action, reward,  next_state, done.
# Usage: for some given variables s, a, r, s, d you can write for example
# exp = Experience(s, a, r, s, d). Then you can access the reward
# field by  typing exp.reward
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length, n_actions):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)
        self.n_actions = n_actions # Dimensionality of the actions

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def fill_buffer_randomly(self, num_exps):
        # Fill buffer with random experiences
        env = gym.make('LunarLanderContinuous-v2')
        state = env.reset()[0]
        random_agent = DDPG_agent.RandomAgent(self.n_actions)

        act_no_exp = 0
        t0 = 0
        while act_no_exp <  num_exps:
            # Take a random action from the random agent 
            action = random_agent.act(state)
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _, _ = env.step(action)
            # Create experience tuple
            exp = Experience(state, action, reward, next_state, done)
            # Append experience in buffer
            self.append(exp)

            act_no_exp += 1

            if done or t0 == 1000:
                state = env.reset()[0]
                t0 = 0
            else:
                state = next_state
                t0 += 1

        env.close()

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n, cer = False):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # Combined experience replay
        if cer:
            batch[0] = self.buffer[-1]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)
    

############################### Neural Network classes ###############################
class ActorNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, dim_s, dim_a):
        super().__init__()
        # Attributes
        self.dim_s = dim_s
        self.dim_a = dim_a
        # Create NN architecture
        self.input_layer = nn.Linear(dim_s, 400)
        self.act_input_layer = nn.ReLU()
        self.hidden_layer1 = nn.Linear(400, 200)
        self.act_hidden_layer1 = nn.ReLU()
        self.out = nn.Linear(200, dim_a)
        self.act_out = nn.Tanh()
       
    def forward(self, s):
        # Function used to compute the forward pass
        s = self.act_input_layer(self.input_layer(s))
        s = self.act_hidden_layer1(self.hidden_layer1(s)) 
        a = self.act_out(self.out(s))
        return a # Return pi(s)
    
class CriticNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, dim_s, dim_a): 
        super().__init__()
        # Attributes
        self.dim_s = dim_s
        self.dim_a = dim_a
        # Create NN architecture
        self.input_layer = nn.Linear(dim_s + dim_a, 400)
        self.act_input_layer = nn.ReLU()
        self.hidden_layer1 = nn.Linear(400, 200)
        self.act_hidden_layer1 = nn.ReLU()
        self.out = nn.Linear(200, 1)
       
    def forward(self, s, a):
        # Function used to compute the forward pass
        s_cat = torch.cat([s, a], dim = 1)
        s_cat = self.act_input_layer(self.input_layer(s_cat))
        x = self.act_hidden_layer1(self.hidden_layer1(s_cat)) 
        x = self.out(x)
        return x # Return computed Q(s,a)
    

def soft_updates(network: nn.Module,
                 target_network: nn.Module,
                 tau: float) -> nn.Module:
    """ Performs a soft copy of the network's parameters to the target
        network's parameter

        Args:
            network (nn.Module): neural network from which we want to copy the
                parameters
            target_network (nn.Module): network that is being updated
            tau (float): time constant that defines the update speed in (0,1)

        Returns:
            target_network (nn.Module): the target network

    """
    tgt_state = target_network.state_dict()
    for k, v in network.state_dict().items():
        tgt_state[k] = (1 - tau)  * tgt_state[k]  + tau * v
    target_network.load_state_dict(tgt_state)
    return target_network


class OrnUHLnoise():
    """ Class simulating a noise generated from the Ornstein-Uhlenbeck
        stochastic process
    """
    def __init__(self, dim_a: int, sigma, mu) -> None:
        self.sigma  = sigma # Noise variance
        self.mu     = mu    # Decay
        self.dim_a  = dim_a # Dimensionality of the noise
        self.last_n = np.zeros(self.dim_a)     # Noise at time n(t-1)

    def forward(self):
        self.last_n =  -self.mu*self.last_n + self.sigma*np.random.randn(self.dim_a)
        return self.last_n

    def reset(self):
        self.last_n = np.zeros(self.dim_a) 
        