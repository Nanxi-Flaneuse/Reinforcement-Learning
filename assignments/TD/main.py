import numpy as np
import random
import matplotlib.pyplot as plt


# target policy: keeps moving right
# behavior policy: arbitrarily chooses between going right and left
# moving right incurrs a reward of 1, and left -1

# @type can either be of values: simple, control, or on
class TD():
    def __init__(self,step, n, seed, maze, discount, target_prob, epochs = 1000, type = 'simple') -> None:
        self.rng = np.random.default_rng(seed) # setting the random seed
        self.state_num = len(maze)
        values = list(self.rng.normal(1, 0.01, self.state_num)) # initialize v
        self.v = dict(zip(maze, values)) # stores the values for each state
        self.epochs = epochs # number of epochs
        self.state_storage = {}# stores the state in each episode for update purposes. Cleared after each episode
        self.reward_storage = {} # stores the reward in each episode for update purposes. Cleared after each episode
        self.start = 0  # starting point
        self.step = step # step factor alpha when updating value functions
        self.n = n # how large of a step the agent is taking to update its values in each episode
        self.maze = maze # the environment/states
        self.type = type # indicates which type of algorithm will be implemented - simple or control
        self.discount = discount # lambda
        self.target_prob = target_prob # probablity for the target policy to take action 'R'. P('L') = 1 - target_prob
        self.true_value = self.get_true_values() # find the true value for each state under the target policy
        self.value_storage = [] # retain the values of each episode

    def train(self):
        self.state_storage.clear()
        self.reward_storage.clear()
        for _ in range(self.epochs):
            T = float('inf')
            t = 0
            tau = 0
            curr_pos = 0
            self.state_storage[0] = 1
            self.reward_storage[0] = 0  # No reward at initial position
            while tau != T - 1:
                if t < T:
                    if self.type != 'on':
                        curr_pos += 1 
                        self.state_storage[t+1] = self.maze[curr_pos]
                        self.reward_storage[t+1] = 1
                    else:
                        action = self.rng.choice([1,-1],p=[self.target_prob, 1 - self.target_prob])
                        if action == -1 and curr_pos > 0:
                            curr_pos -= 1
                            self.reward_storage[t+1] = action
                        elif action == 1 and curr_pos < self.state_num - 1:
                            curr_pos += 1
                            self.reward_storage[t+1] = action
                        else:
                            self.reward_storage[t+1] = 0
                        self.state_storage[t+1] = self.maze[curr_pos]
                    try:
                        if self.maze[curr_pos] == self.maze[-1]:
                            T = t + 1
                    except:
                        pass
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += self.discount ** (i - tau - 1) * self.reward_storage[i]

                    # calculate rho
                    rho = self.target_prob/1 ** (min(tau + self.n, T) - (tau + 1) + 1) # pie/b^(n or number of steps taken before reaching terminal)

                    if tau + self.n < T:
                        G += self.discount ** self.n * self.v[self.state_storage[tau + self.n]]
                    if self.type =='control':
                        G = self.target_prob/1 * G + (1 - self.target_prob/1) * self.v[self.state_storage[tau]]
                    
                    # update V
                    if self.type != 'on':
                        self.v[self.state_storage[tau]] += self.step * rho * (G - self.v[self.state_storage[tau]])
                    else:
                        self.v[self.state_storage[tau]] += self.step * (G - self.v[self.state_storage[tau]])

                t += 1 # update t
            # storing the values of this episode
            self.value_storage.append(list(self.v.values()))
    def get_values(self):
        return self.v
    
    # finding the true value of each state using the bellman equation the maze, the target policy, and the reward to get the true values of each state
    # this function is written by chatGPT.
    def get_true_values(self):
        num_states = self.state_num - 1  # Non-terminal states (1 to 4)
        A = np.zeros((num_states, num_states))
        b = np.zeros(num_states)
        
        # Loop over each state (1-based indexing for clarity)
        for s in range(1, self.state_num):
            row = s - 1
            A[row, row] = -1  # Coefficient of V(s)
            
            # Right action contribution (prob p)
            if s < self.state_num - 1:
                A[row, row + 1] += self.target_prob  # V(s+1)
            else:
                b[row] += self.target_prob * 1  # Reaching terminal state (V(5) = 0), reward = +1
            
            # Left action contribution (prob 1-p)
            if s > 1:
                A[row, row - 1] += (1 - self.target_prob)  # V(s-1)
            else:
                b[row] += (1 - self.target_prob) * (-1)  # Hitting left boundary (V(0) = 0), reward = -1
            
            # Immediate rewards:
            b[row] += self.target_prob * 1 + (1 - self.target_prob) * (-1)
        
        # Solve linear system
        V = np.linalg.solve(A, b)
        
        # Add terminal state value
        V = np.append(V, 0)  # V(5) = 0 terminal
        
        return V

    # the following function is written by chatGPT: plot the mean square error or learned values vs true value
    def plot(self, error_type = 'mse', plot = False, window = 10):
        true_values = np.array(self.true_value)
        error_per_episode = []

        for episode_values in self.value_storage:
            episode_values = np.array(episode_values)
            if error_type == 'mse':
                err = np.mean((episode_values - true_values) ** 2)
            else:
                err = np.sqrt(np.mean((episode_values - true_values) ** 2))
            error_per_episode.append(err)
        
        if self.type == 'on':
            error_per_episode = np.convolve(error_per_episode, np.ones(window)/window, mode='valid')

        if plot:
            # Plotting
            plt.figure(figsize=(8, 5))
            plt.plot(error_per_episode)
            plt.xlabel('Episode')
            plt.ylabel('Mean Squared Error (MSE)')
            plt.title('MSE of Learned Values vs True Values Over Episodes')
            plt.grid(True)
            file = 'plots/on-policy/'+self.type + '_'+error_type+'_'+str(self.n)+'steps_'+str(self.state_num)+'walk_'+str(self.epochs)+'.jpg'
            plt.savefig(file) 
            # plt.show()
        return error_per_episode
    
# see documentation on the parameters in the TD class for reference
class Experiment():
    def __init__(self,seeds, step, n, maze, discount, target_prob, epochs = 100, error_type='mse') -> None:
        self.seeds = seeds
        self.step = step
        self.n = n
        self.maze = maze
        self.discount = discount
        self.target_prob = target_prob
        self.epochs = epochs
        self.error_type = error_type
        self.simple_errors = []
        self.control_errors = []
        self.on_errors = []
    # train the agents
    def train(self):
        for seed in self.seeds:
            simple = TD(self.step, self.n, seed, self.maze, self.discount, self.target_prob, self.epochs)
            control = TD(self.step, self.n, seed, self.maze, self.discount, self.target_prob, self.epochs, 'control')
            on = TD(self.step, self.n, seed, self.maze, self.discount, self.target_prob, self.epochs, 'on')

            simple.train()
            self.simple_errors.append(simple.plot(self.error_type))
            control.train()
            self.control_errors.append(control.plot(self.error_type))
            on.train()
            self.on_errors.append(on.plot(self.error_type))
    # plotting mse error
    def plot(self):
        average_simple = np.mean(self.simple_errors, axis=0)
        average_control = np.mean(self.control_errors, axis=0)
        average_on = np.mean(self.on_errors, axis=0)
        plt.plot(average_simple, label="off-policy simple method", color='b')
        plt.plot(average_control, label="off-policy control method", color='r')
        plt.plot(average_on, label="on-policy method", color='g')
        plt.xlabel("episodes")
        plt.ylabel("error of each episode")
        plt.title("changes of "+self.error_type+' error over '+str(self.epochs)+' episodes')

        plt.legend()
        plt.grid()

        # Save the plots
        plt.tight_layout()
        file = 'plots/'+str(self.target_prob)+'/'+self.error_type+'_'+str(self.n) +'steps_'+str(len(self.maze))+'walk_'+str(self.epochs)+'.jpg'
        plt.savefig(file) 


if __name__ == '__main__':
    maze = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # 1 marks beginning, and 2 marks the terminal state
    
    # INSTRUCTION: uncomment the following three blocks to train an agent on a single algorithm
    # simple algorithm
    # simple = TD(0.2, 2, 46288, maze, 0.8, 0.5, 20)
    # simple.train()
    # simple.plot('rms')

    # control variate algorithm
    # control = TD(0.2, 2, 46288, maze, 0.8, 0.5, 20, 'control')
    # control.train()
    # control.plot('rms')
    
    # on-policy algorithm
    # on = TD(0.2, 6, 46288, maze, 0.8, 0.5, 60, 'on')
    # on.train()
    # on.plot('rms', True,30)

    # generating seeds for the experiment
    random.seed(5436)
    seeds = random.sample(range(1, 10**6), 20) 
   
    
    # INSTRUCTION: run one snippet at the time to execute one experiment. Do no uncomment and run all of them at the same time because if so, all the error curves will be drawn in the same plot
    
    # exp = Experiment(seeds, 0.2, 2, maze, 0.8, 0.5, 20)
    # exp.train()
    # exp.plot()

    # exp1 = Experiment(seeds, 0.2, 4, maze, 0.8, 0.5, 20)
    # exp1.train()
    # exp1.plot()

    # exp2 = Experiment(seeds, 0.2, 6, maze, 0.8, 0.5, 20)
    # exp2.train()
    # exp2.plot()

    # exp3 = Experiment(seeds, 0.2, 8, maze, 0.8, 0.5, 20)
    # exp3.train()
    # exp3.plot()

    # exp4 = Experiment(seeds, 0.2, 2, maze, 0.8, 0.8, 200)
    # exp4.train()
    # exp4.plot()

    # exp5 = Experiment(seeds, 0.2, 4, maze, 0.8, 0.8, 200)
    # exp5.train()
    # exp5.plot()

    # exp6 = Experiment(seeds, 0.2, 6, maze, 0.8, 0.8, 200)
    # exp6.train()
    # exp6.plot()

    # exp7 = Experiment(seeds, 0.2, 8, maze, 0.8, 0.2, 10)
    # exp7.train()
    # exp7.plot()



'''references
1. https://towardsdatascience.com/introducing-n-step-temporal-difference-methods-7f7878b3441c/
'''