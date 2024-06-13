import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
BLUE         = '#0000FF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100


    def __init__(self, maze, mini_stay = False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.mini_stay                = mini_stay;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards();

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    # Creates the mapping from state position to state identifier and vice versa 
    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        # Initialize the states moving in rows (States are represented using numbers)
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                    for k in range(self.maze.shape[0]):
                        for l in range(self.maze.shape[1]):
                            if self.maze[i,j] != 1:
                                # Mapping from State identifier to state position 
                                states[s] = (i,j,k,l);
                                # Mapping from State position to state identifier
                                map[(i,j,k,l)] = s;
                                s += 1;
        
        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.
            Minitaur takes a step following a random walk. The probavility is uniform for all
            admissible directions

            :return tuple next_cell: Position on the maze that agent and minitaur transition to. !!In form of the identifier
        """
        # Next state is a random variable hence we determine the possible next states and sample from the returned list
        pos_next_states = [];

        # If the game is already won or lost, then there is no more moves!
        if self.states[state][:2] == self.states[state][2:] or self.maze[self.states[state][:2]] == 2:
            pos_next_states.append(state)
        else: # Still game in play!
            # Determine Player's next state
            # Compute the future position given current (state, action) for the player
            row_p = self.states[state][0] + self.actions[action][0];
            col_p = self.states[state][1] + self.actions[action][1];
            # Is the future position an impossible one ?
            hitting_maze_walls =  (row_p == -1) or (row_p == self.maze.shape[0]) or \
                                (col_p == -1) or (col_p == self.maze.shape[1]) or \
                                (self.maze[row_p,col_p] == 1); 
            
            # Based on the impossiblity check return the next position of the player.
            if hitting_maze_walls:
                p_new = self.states[state]; # Return same state
            else:
                p_new  = (row_p, col_p); # Return next state (Admissible case)
            
            # If the minitaur can stay then consider all five actions
            if self.mini_stay:
                mini_action_start = 0;
            else:
                mini_action_start = 1;
            
            # Determine Minotaur's next state
            for minitaur_move in range(mini_action_start, self.n_actions):
                row_m = self.states[state][2] + self.actions[minitaur_move][0]
                col_m = self.states[state][3] + self.actions[minitaur_move][1]

                hitting_maze_walls = (row_m == -1) or (row_m == self.maze.shape[0]) or \
                                    (col_m == -1) or (col_m == self.maze.shape[1])

                if not hitting_maze_walls:
                    pos_next_states.append(self.map[(p_new[0], p_new[1], row_m, col_m)])

        return pos_next_states


    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Due to the Minotaur randomness, the transition probabilities are stochastic
        # Compute the transition probabilities.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # Get the list of possible next states (states with non-zero probabilities)
                pos_next_states = self.__move(s,a);
                # Assign the probabilities
                for next_s in pos_next_states:
                    transition_probabilities[next_s, s, a] = 1/len(pos_next_states);
        return transition_probabilities;

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            for a in range(self.n_actions):
                prob_next_s = self.__move(s,a);
                # Sice the rewards are a random variable we replace the rewards by the average of the rewards
                for next_s in prob_next_s:
                    # Reward for hitting a wall (here it does not matter if we average over them since they should still be suff. high)
                    if self.states[s][0:2] == self.states[next_s][0:2] and a != self.STAY:
                        rewards[s,a] += self.IMPOSSIBLE_REWARD*self.transition_probabilities[next_s,s,a];
                    # Reward for getting eaten by the minotaur (here it does not matter if we average over them since they should still be suff. high)
                    elif self.states[next_s][0:2] == self.states[next_s][2:4]:
                        rewards[s,a] += self.IMPOSSIBLE_REWARD*self.transition_probabilities[next_s,s,a];
                    # Reward for reaching the exit
                    elif self.states[s][0:2] == self.states[next_s][0:2] and self.maze[self.states[next_s][0:2]] == 2:
                        rewards[s,a] += self.GOAL_REWARD*self.transition_probabilities[next_s,s,a];
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] += self.STEP_REWARD*self.transition_probabilities[next_s,s,a];
        return rewards;

    # Dictate if minitaur can stay in the same position
    def set_mini_stay(self, bool):
        self.mini_stay = bool;
    
    # Return state identifier
    def get_state_id(self, state):
        return self.map[state];

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Next state is chosen at random
                next_s = random.choice(self.__move(s,policy[s,t]));
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Next state is chosen randomly
            next_s = random.choice(self.__move(s,policy[s]));
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Update state
            s = next_s;
            # Loop while state is not the goal state
            while self.maze[self.states[s][:2]] != 2 and self.states[s][:2] != self.states[s][2:]:
                # Move to next state given the policy and the current state
                next_s = random.choice(self.__move(s,policy[s]));
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization u_T = max_pi r(s,a)
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def compute_POS(env, start, n_iterations, horizon, method, gamma = 0.9, epsilon = 0.00001):
    """ Computes the Probability of Success (POS)
        :input Maze env              : The maze environment in which we seek to
                                       find the shortest path.
        :input tuple start           : The start position of the player.
        :input integer n_iterations  : accuracy of the value iteration procedure.
        :input integer horizon       : Horizon for DP
        :input string method         : 'DynProg' for Dynamic Programming
        :return exit_prob            : Probability of exiting the Maze given the time horizon
                                    
    """

    if method == 'DynProg':
        exit_prob = np.zeros((n_iterations, len(horizon)), dtype= float)
        # Simulate the shortest path starting from position A n_iterations times
        for n_horizon in range(len(horizon)):
            # Solve the MDP problem with dynamic programming 
            V, policy= dynamic_programming(env,horizon[n_horizon]);
            for n_sim in range(n_iterations):
                path = env.simulate(start, policy, method);
                if path[-1][:2] != path[-1][2:] and env.maze[path[-1][:2]] == 2:
                    exit_prob[n_sim,n_horizon] = 1

        return np.sum(exit_prob, 0)/n_iterations;
    elif method == 'ValIter':
        exit_prob = np.zeros((n_iterations, 1), dtype= float)
        # Simulate the shortest path starting from position A n_iterations times
        # Solve the MDP problem with dynamic programming 
        V, policy= value_iteration(env, gamma, epsilon);
        for n_sim in range(n_iterations):
            path = env.simulate(start, policy, method);
            if path[-1][:2] != path[-1][2:] and env.maze[path[-1][:2]] == 2:
                exit_prob[n_sim] = 1 - np.power(gamma, 1 + np.floor(len(path)))

        return np.mean(exit_prob);


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        if i > 0:
            # Updates: Remove previous blocks
            grid.get_celld()[(path[i-1][:2])].set_facecolor(col_map[maze[path[i-1][:2]]])
            grid.get_celld()[(path[i-1][2:])].set_facecolor(col_map[maze[path[i-1][2:]]])
            grid.get_celld()[(path[i-1][:2])].get_text().set_text('')
            grid.get_celld()[(path[i-1][2:])].get_text().set_text('')

            # Minotaur eats player?
            if path[i][2:] == path[i][:2]: 
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i][:2])].get_text().set_text('Game over!')
                break;
            # Player wins
            elif maze[path[i][:2]] == 2:
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][2:])].set_facecolor(BLUE)
                grid.get_celld()[(path[i][:2])].get_text().set_text('Player wins!')
                break;
            else:
            # New positions
                grid.get_celld()[(path[i][:2])].get_text().set_text('Player')
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(path[i][2:])].get_text().set_text('Minotaur')
                grid.get_celld()[(path[i][2:])].set_facecolor(BLUE)
        else:
            grid.get_celld()[(path[i][:2])].get_text().set_text('Player')
            grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path[i][2:])].get_text().set_text('Minotaur')
            grid.get_celld()[(path[i][2:])].set_facecolor(BLUE)

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)


def draw_policy(maze, env, policy, mini_state):
    """ Draw the policy for a fixed minitaur position 
        :input Maze maze             :                                     
    """
    action_strings = {
        0.0: "stay",
        1.0: "left",
        2.0: "right",
        3.0: "up",
        4.0: "down"
    }
    
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Optimal Policy illustration');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);
    
    # Color the fixed minitaur's position
    grid.get_celld()[mini_state].get_text().set_text('Minotaur')
    grid.get_celld()[mini_state].set_facecolor(BLUE)

    # Draw the optimal policy
    for row in range(rows):
        for col in range(cols):
            if (row,col) == mini_state or maze[row,col] == 1: # If obstacle or minitaur position skip
                continue;
            else:
                state_id = env.get_state_id((row, col, mini_state[0], mini_state[1]))
                grid.get_celld()[(row,col)].get_text().set_text(action_strings[policy[state_id,0]])

    #display.display(fig)




        

        
         