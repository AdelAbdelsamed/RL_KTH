import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
# Implemented methods
methods = ['DynProg', 'ValIter', 'QLearning', 'SARSA'];

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
        for m in range(2):
            for i in range(self.maze.shape[0]):
                for j in range(self.maze.shape[1]):
                        for k in range(self.maze.shape[0]):
                            for l in range(self.maze.shape[1]):
                                if self.maze[i,j] != 1:
                                    # Mapping from State identifier to state position 
                                    states[s] = (i,j,k,l,m);
                                    # Mapping from State position to state identifier
                                    map[(i,j,k,l,m)] = s;
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
        # List of next states that would make the minotaur closer
        closer_next_states = []; 

        # If the game is already won or lost, then there is no more moves!
        if self.states[state][:2] == self.states[state][2:4] or (self.maze[self.states[state][:2]] == 2 and self.states[state][-1] == 1):
            pos_next_states.append(state)
            closer_next_states.append(state) # to ensure probability of staying will be one!
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
                p_new = self.states[state][0:2]; # Return same state
            else:
                p_new  = (row_p, col_p); # Return next state (Admissible case)
            
            # If next state at C, let key = 1
            if p_new == (0,7):
                key = 1;
            else:
                key = self.states[state][-1]
            
            # If the minitaur can stay then consider all five actions
            if self.mini_stay:
                mini_action_start = 0;
            else:
                mini_action_start = 1;
            
            # Determine Minotaur's possible next states
            for minitaur_move in range(mini_action_start, self.n_actions):
                row_m = self.states[state][2] + self.actions[minitaur_move][0]
                col_m = self.states[state][3] + self.actions[minitaur_move][1]                

                hitting_maze_walls = (row_m == -1) or (row_m == self.maze.shape[0]) or \
                                    (col_m == -1) or (col_m == self.maze.shape[1])

                if not hitting_maze_walls:
                    pos_next_states.append(self.map[(p_new[0], p_new[1], row_m, col_m, key)])
                    # Is next minitaur position closer? Compute manhattan distance
                    ref_mh_dist = np.abs(p_new[0] - self.states[state][2]) + np.abs(p_new[1] - self.states[state][3])
                    curr_mh_dist = np.abs(p_new[0] - row_m) + np.abs(p_new[1] - col_m)

                    if curr_mh_dist <= ref_mh_dist or ref_mh_dist == 0:
                        closer_next_states.append(self.map[(p_new[0], p_new[1], row_m, col_m, key)])

        return pos_next_states, closer_next_states


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
                pos_next_states, closer_next_states = self.__move(s,a);
                # Assign the probabilities
                for next_s in pos_next_states:
                    if next_s in closer_next_states:
                        transition_probabilities[next_s, s, a] = 0.35*(1/len(closer_next_states)) + 0.65*(1/len(pos_next_states));
                    else: 
                        transition_probabilities[next_s, s, a] = 0.65*(1/len(pos_next_states));
        return transition_probabilities;

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            for a in range(self.n_actions):
                prob_next_s, closer_next_states = self.__move(s,a);
                # Sice the rewards are a random variable we replace the rewards by the average of the rewards
                for next_s in prob_next_s:
                    # Reward for hitting a wall (here it does not matter if we average over them since they should still be suff. high)
                    if self.states[s][0:2] == self.states[next_s][0:2] and a != self.STAY:
                        rewards[s,a] += self.IMPOSSIBLE_REWARD*self.transition_probabilities[next_s,s,a];
                    # Reward for getting eaten by the minotaur (here it does not matter if we average over them since they should still be suff. high)
                    elif self.states[next_s][0:2] == self.states[next_s][2:4]:
                        rewards[s,a] += self.IMPOSSIBLE_REWARD*self.transition_probabilities[next_s,s,a];
                    # # Reward for reaching the exit
                    elif (self.states[s][0:2] == self.states[next_s][0:2]) and (self.maze[self.states[next_s][0:2]] == 2) \
                            and (self.states[s][-1] == 1):
                        rewards[s,a] += self.GOAL_REWARD*self.transition_probabilities[next_s,s,a]
                    else:
                        rewards[s,a] += self.STEP_REWARD*self.transition_probabilities[next_s,s,a];
        return rewards;

    # Dictate if minitaur can stay in the same position
    def set_mini_stay(self, bool):
        self.mini_stay = bool;
    
    # Return state identifier
    def get_state_id(self, state):
        return self.map[state];

    # Sample next state
    def get_next_state(self, state, action):
        poss_next_states, closer_next_states = self.__move(state, action);
        if random.random() > 0.35: # Sample from the random actions
            next_s = random.choices(poss_next_states)[0]
        else: # Sample from the actions that moves the minotaur closer
            next_s = random.choices(closer_next_states)[0]
        return next_s

    def generate_random_states(self):
        done = False
        while not done:
            px = random.randint(0,6)
            py = random.randint(0,7)

            if self.maze[(px,py)] != 1 and self.maze[(px,py)] != 2 and (px, py) != (5,6):
                break
        return (px, py)
    
    def e_greedy(self, state, Q,  epsilon):
        # Returns a sample action using e-greedy
        A = np.ones(5,dtype=float) * epsilon/5
        best_action =  np.argmax(Q[state,:])
        A[best_action] += (1.0-epsilon)
        sample = np.random.choice(5,p=A)
        return sample
    
    def episode_terminated(self, state):
        # Returns true if episode has terminated
        win  = self.maze[self.states[state][:2]] == 2 and self.states[state][-1] == 1
        lose = self.states[state][:2] == self.states[state][2:4]
        return win or lose

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
                next_s = self.get_next_state(s, policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter' or method == 'QLearning' or method == 'SARSA':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Next state is chosen randomly
            next_s = self.get_next_state(s, policy[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while (self.states[next_s][:2] != self.states[next_s][2:4]) \
                   and not (self.maze[self.states[next_s][:2]] == 2 and self.states[next_s][-1] == 1):
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.get_next_state(s, policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
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

def Q_learning(env, start, gamma, stepsize_exp, epsilon, no_episodes, cvg_plot = False):
        """ Solves the RL problem using Q-learning
            :input tuple start              : initial state 
            :input float gamma              : The discount factor.
            :input float stepsize_exp       : Exponent of the stepsize
            :input float epsilon            : Parameter for epsilon greedy algorithm.
            :input float no_episodes        : Number of episodes
            :input bool cvg_plot            : If true, convergence plot will generate
            :return numpy.array Q           : Optimal values for every state at every
                                            time, dimension S,A
            :return numpy.array policy      :Optimal time-varying policy at every state,
                                            dimension S
        """

        # The Q learning requires the knowledge of :
        # - Rewards
        # - State space
        # - Action space
        r         = env.rewards;
        n_states  = env.n_states;
        n_actions = env.n_actions;

        # Initialization:
        # The variables involved in the Q learning
        Q    = np.zeros((n_states, n_actions));
        n_sc = np.zeros((n_states, n_actions));

        # Lists to store episode and state values
        plt_state_values = []
        episode_rewards = []
        
        # Iterate through episodes:
        for ep in range(no_episodes):
            # Reset environment and start state
            s = env.map[start];
            done = False 
            tot_rew = 0
            # Iterate until episode has terminated 
            while not done:    
                # Select action a_t based on an epsilon greedy-policy
                a = env.e_greedy(s, Q,  epsilon)
                # Increase counter of visiting (s, a)
                n_sc[s][a] +=  1 
                # Observe next state s_(t+1):
                next_s = env.get_next_state(s, a)
                # Check if episode has terminated
                done = env.episode_terminated(s)

                # Compute the step size
                alpha_t = 1/np.power(n_sc[s][a], stepsize_exp)
                # Update of the Q values
                Q[s][a] = Q[s][a] + alpha_t*(r[s][a] + gamma*np.max(Q[next_s]) - Q[s][a])
                tot_rew += r[s][a]

                # Update the state
                s = next_s;
                
                # Check if episode has ended?
                if done:
                    episode_rewards.append(tot_rew)
                    plt_state_values.append(np.max(Q[env.map[start]]))
                
        # Update plot
        if cvg_plot:
            fig = plt.figure
            plt.plot(np.arange(no_episodes), plt_state_values)
            # Set plot labels and title
            plt.xlabel('Episode')
            plt.ylabel('Initial State Value')
            plt.title('Initial State Value over Episodes')
            # Display the final plot
            plt.show()

        # From Q construct the policy
        policy = np.argmax(Q,1)
        return Q, policy, plt_state_values, episode_rewards;

def SARSA_learning(env, start, gamma, stepsize_exp, epsilon, no_episodes, exploration_exp, variable_stepsize = False , cvg_plot = False):
    """ Solves the RL problem using SARSA-learning
        :input tuple start              : initial state 
        :input float gamma              : The discount factor.
        :input float stepsize_exp       : Exponent of the stepsize
        :input float epsilon            : Parameter for epsilon greedy algorithm.
        :input float no_episodes        : Number of episodes
        :input bool cvg_plot            : If true, convergence plot will generate
        :return numpy.array Q           : Optimal values for every state at every
                                        time, dimension S,A
        :return numpy.array policy      :Optimal time-varying policy at every state,
                                        dimension S
    """

    # The SARSA learning requires the knowledge of:
    # - Rewards
    # - State space
    # - Action space
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Initialization:
    # The variables involved in the SARSA learning
    Q    = np.zeros((n_states, n_actions));
    n_sc = np.zeros((n_states, n_actions));

    # For plotting:
    init_s_value = []
    episode_rewards = []

    for ep in range(no_episodes):
        # Define decaying epsilon
        if variable_stepsize:
            epsilon_k = 1/np.power((1+ep), exploration_exp)
        else:
            epsilon_k = epsilon 
        # Reset the state to initial state
        s = env.map[start]
        # Variable endicating end of episode
        done = False
        tot_rew = 0 
        
        # Loop until episode terminates
        while not done:
            # Pick action epsilon-greedy w.r.t to Q
            a = env.e_greedy(s, Q, epsilon_k)
            n_sc[s][a] += 1 # Increase Counter of visiting the (s,a) pair
            done = env.episode_terminated(s) # Check if episode terminated
            next_s = env.get_next_state(s, a) # Observe next state
            next_a = env.e_greedy(next_s, Q, epsilon_k) # Pick next action e-greedy w.r.t Q

            # Compute learning rate
            alpha = 1/np.power(n_sc[s][a] + 1, stepsize_exp)
            # Sarsa Update
            Q[s][a] = Q[s][a] + alpha*(r[s][a] + gamma*Q[next_s][next_a] - Q[s][a]) 

            tot_rew += r[s][a]
            s = next_s

            if done:
                episode_rewards.append(tot_rew)
                init_s_value.append(np.max(Q[env.map[start],:]))

    # Construct the policy
    policy = np.argmax(Q,1)

    if cvg_plot:
        fig = plt.figure
        plt.plot(np.arange(no_episodes), init_s_value)
        # Set plot labels and title
        plt.xlabel('Episode')
        plt.ylabel('Initial State Value')
        plt.title('Initial State Value over Episodes')
        # Display the final plot
        plt.show()

    return Q, policy, init_s_value, episode_rewards

def compute_POS(env, start, n_iterations, horizon, method, policy, gamma = 49/50):
    """ Computes the Probability of Success (POS)
        :input Maze env              : The maze environment in which we seek to
                                       find the shortest path.
        :input tuple start           : The start position of the player.
        :input integer n_iterations  : accuracy of the value iteration procedure.
        :input integer horizon       : Horizon for DP
        :input string method         : 'DynProg' for Dynamic Programming, ...
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
                if path[-1][:2] != path[-1][2:] and (env.maze[path[-1][:2]] == 2 and path[-1][-1] == 1):
                    exit_prob[n_sim,n_horizon] = 1

        return np.sum(exit_prob, 0)/n_iterations;
    elif method == 'ValIter' or method == 'QLearning' or method == 'SARSA':
        exit_prob = np.zeros((n_iterations, 1), dtype= float)
        # Simulate the shortest path starting from position A n_iterations times
        # Solve the MDP problem with dynamic programming 
        for n_sim in range(n_iterations):
            path = env.simulate(start, policy, method);
            if path[-1][:2] != path[-1][2:] and (env.maze[path[-1][:2]] == 2 and path[-1][-1] == 1):
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
            grid.get_celld()[(path[i-1][2:4])].set_facecolor(col_map[maze[path[i-1][2:4]]])
            grid.get_celld()[(path[i-1][:2])].get_text().set_text('')
            grid.get_celld()[(path[i-1][2:4])].get_text().set_text('')

            # Minotaur eats player?
            if path[i][2:] == path[i][:2]: 
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i][:2])].get_text().set_text('Game over!')
                break;
            # Player wins
            elif maze[path[i][:2]] == 2:
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][2:4])].set_facecolor(BLUE)
                grid.get_celld()[(path[i][:2])].get_text().set_text('Player wins!')
                break;
            else:
            # New positions
                grid.get_celld()[(path[i][:2])].get_text().set_text('Player')
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(path[i][2:4])].get_text().set_text('Minotaur')
                grid.get_celld()[(path[i][2:4])].set_facecolor(BLUE)
        else:
            grid.get_celld()[(path[i][:2])].get_text().set_text('Player')
            grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path[i][2:4])].get_text().set_text('Minotaur')
            grid.get_celld()[(path[i][2:4])].set_facecolor(BLUE)

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)


def draw_policy(maze, env, policy, mini_state, key):
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
                state_id = env.get_state_id((row, col, mini_state[0], mini_state[1], key))
                grid.get_celld()[(row,col)].get_text().set_text(action_strings[policy[state_id]])

    #display.display(fig)




        

        
         