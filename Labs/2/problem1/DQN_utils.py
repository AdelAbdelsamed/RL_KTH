import numpy as np
import gym
import torch
import torch.nn as nn
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
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def fill_buffer_randomly(self, num_exps):
        # Fill buffer with random experiences
        env = gym.make('LunarLander-v2')
        state = env.reset()[0]

        act_no_exp = 0
        while act_no_exp <  num_exps:
            # Take a random action
            action = np.random.randint(0, env.action_space.n)
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _, _ = env.step(action)
            # Create experience tuple
            exp = Experience(state, action, reward, next_state, done)
            # Append experience in buffer
            self.append(exp)

            act_no_exp += 1

            if done:
                state = env.reset()[0]
            else:
                state = next_state


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
    
############################### Neural Network class (from Lab 0) ###############################
class NeuralNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()

        # Create NN architecture
        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.act_input_layer = nn.ReLU()
        self.hidden_layer1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.act_hidden_layer1 = nn.ReLU()
        # V(s) stream
        self.v_out = nn.Linear(hidden_layer_size, 1)
        # A(s,a) stream
        self.a_out = nn.Linear(hidden_layer_size, output_size)
       
    def forward(self, x):
        # Function used to compute the forward pass
        x = self.act_input_layer(self.input_layer(x))
        x = self.act_hidden_layer1(self.hidden_layer1(x)) 
        #value = self.act_v_layer1(self.v_layer1(x))
        value = self.v_out(x)
        #advantage = self.act_a_layer1(self.a_layer1(x))
        advantage = self.a_out(x)
        advAvg = torch.mean(advantage, dim=1, keepdim=True) 
        return value + advantage - advAvg # Return Q_value
    
    
    