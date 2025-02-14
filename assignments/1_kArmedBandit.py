import numpy as np
import matplotlib.pyplot as plt
import random

# still stuck on how optimal action is calcualted over time.
# question: how does having too many arms affect the learning results?
# play around with parameters - explore vs exploit
# should run around 100 experiments for each set of parameter
# calculating proportion of optimal action - compile an array of the optimal action result of each experiment. calculate the mean in each column of each timestep. eg. five '1' nad ten '0', mean = 1/3 
# increasing bandit arm number can be problem 2
# make sure your paper is clear so that someone can replicate your results
'''tasks
1. for each type, try with 1 set of parameters. Each set of parameters should have 100 experiments
2. the averages for the 2 types should be in the same graph, so are the optimal actions
3. experiment with 3 different arm width for both the step and averages, do a similar plotting like number 2.
'''

# set up a step-size QA agent class
class k_armed_non_stationary():
    def __init__(self, arm_num, epsilon, seed, epoch, step, method = 'step') -> None:
        self.arm = arm_num # the number of arms/choices
        self.ep = epsilon # the epsilon that determines the degree of exploration
        np.random.seed(seed)
        self.method = method
        self.frequency = [1] * self.arm
        self.rewards = list(np.random.normal(1, 0.01, arm_num)) # generating random reward for the arms
        self.optimal = self.rewards.index(max(self.rewards))
        self.values = [0] * self.arm # tracking the reward exploration for the arms
        self.epoch = epoch # the number of epochs for training
        self.step = step # the step-size parameter used in updating the value function
        self.reward_overtime = [] # keeping track of reward gained overtime
        self.optimal_action = [] # keeping track of proportions of optimal actions overtime

    # return the learned rewards
    def get_reward(self):
        return self.reward_overtime
    
    # return the optimal actions
    def get_optimal(self):
        return self.optimal_action
    
    # training the agent
    def train(self):
        for _ in range(self.epoch):
            # select action
            action = random.choices([0,1],weights = (self.ep, 1-self.ep), k = 1)[0]
            # check the current optimal action for recording purpose
            curr_optimal = self.values.index(max(self.values))

            # take the actual action
            if action == 0:
                # randomly choose an action
                act_ind = random.randint(0,self.arm - 1)
            else:
                act_ind = self.values.index(max(self.values))
            # perform action and update value function
            reward = self.rewards[act_ind]
            if self.method == 'step':
                self.values[act_ind] += self.step * (reward - self.values[act_ind]) + np.random.normal(0, 0.01, 1)[0]
            else:
                self.values[act_ind] += 1/self.frequency[act_ind] * (reward - self.values[act_ind]) + np.random.normal(0, 0.01, 1)[0]
                self.frequency[act_ind] += 1 # updates the frequency
            # record reward
            self.reward_overtime.append(reward)
            # record action
            
            if act_ind == curr_optimal:
                self.optimal_action.append(1)
            else:
                self.optimal_action.append(0)
            # update rewards and optimal_action

    # # plot the reward gained over time
    # def draw_reward(self, window_size = 100):

    #     """Plots the average reward over a specified window size."""

    #     # Calculate the running average of rewards
    #     running_avg = np.convolve(self.reward_overtime, np.ones(window_size)/window_size, mode='valid')

    #     # Plot the running average
    #     plt.plot(running_avg)
    #     plt.xlabel("Episode")
    #     plt.ylabel(f"Average Reward (over {window_size} steps)")
    #     plt.title("Running Average Reward")
    #     # plt.show()
    #     if self.method == 'step':
    #         file = 'assignments/plots/average_reward_step.png'
    #     else:
    #         file = 'assignments/plots/average_reward_average.png'
    #     plt.savefig(file)

    # # plot the proportion of optimal action over time
    # def draw_action(self, step = 10000):
    #     # Compute the running proportion of optimal actions
    #     actions = np.array(self.optimal_action)
    #     #################reference to chatGPT instructions start ###############
    #     cumulative_optimal = np.cumsum(actions)
    #     proportion_optimal = cumulative_optimal / np.arange(1, step + 1)

    #     # Plot the proportion of optimal actions over time
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(proportion_optimal, label="Proportion of Optimal Actions", color='b')
    #     plt.xlabel("Timesteps")
    #     plt.ylabel("Proportion of Optimal Actions")
    #     plt.title("Proportion of Optimal Actions Over Time")
    #     if self.method == 'step':
    #         file = 'assignments/plots/optimal_action_step.png'
    #     else:
    #         file = 'assignments/plots/optimal_action_average.png'
    #     plt.savefig(file)
        #################reference to chatGPT instructions end ###############

class experiments():
    def __init__(self, seeds, epsilon = 0.1, epochs = 10000, arm = 10, method = 'step') -> None:
        self.seeds = seeds
        self.method = method # select the type of value function. Same as in the other class.
        self.epochs = epochs
        self.ep = epsilon
        self.arm = arm
        self.reward = []
        self.optimal = []

    def get_results(self):
        for seed in self.seeds:
            arm = k_armed_non_stationary(self.arm, self.ep, seed, self.epochs, 0.1, self.method)
            arm.train()
            self.reward.append(arm.get_reward())
            self.optimal.append(arm.get_optimal())

    def plot(self, window_size = 100):
        # first calculate the averages of each
        # then plot
        average_rewards = np.mean(self.reward, axis=0)
        proportion_optimal_actions = np.mean(self.optimal, axis=0)
        # Plotting Results
        plt.figure(figsize=(12, 5))

        # Plot Average Reward
        plt.subplot(1, 2, 1)
        running_avg = np.convolve(average_rewards, np.ones(window_size)/window_size, mode='valid')

        # Plot the running average
        plt.plot(running_avg, label="Average Reward", color='b')
        plt.xlabel("Episode")
        plt.ylabel(f"Average Reward (over {window_size} steps)")
        plt.title("Running Average Reward")
        plt.legend()
        plt.grid()

        # Plot Proportion of Optimal Actions
        plt.subplot(1, 2, 2)


        # actions = np.array(self.optimal_action)
        #################reference to chatGPT instructions start ###############
        # cumulative_optimal = np.cumsum(proportion_optimal_actions)
        # proportion_optimal = cumulative_optimal / np.arange(1, self.epochs + 1)

        # Plot the proportion of optimal actions over time
        # plt.figure(figsize=(10, 5))
        plt.plot(proportion_optimal_actions, label="Proportion of Optimal Actions", color='g')
        plt.xlabel("Timesteps")
        plt.ylabel("Proportion of Optimal Actions")
        plt.title("Proportion of Optimal Actions Over Time")

        plt.legend()
        plt.grid()

        # Show the plots
        plt.tight_layout()
        # plt.show()
        if self.method == 'step':
            file = 'assignments/plots/step_'+str(self.ep)+'_'+str(self.epochs)+'_'+str(self.arm)+'_'+self.method+'.png'
        else:
            file = 'assignments/plots/average_'+str(self.ep)+'_'+str(self.epochs)+'_'+str(self.arm)+'_'+self.method+'.png'
        plt.savefig(file)



if __name__ == '__main__':
    # arm = k_armed_non_stationary(10, 0.1, 7685, 10000, 0.1, 'average')
    # print(arm.get_reward())
    # arm.train()
    # arm.draw_reward()
    # arm.draw_action()

    # 108 seeds
    seeds = [
    1,2,3,4,5,6,7,8,9,
    11,22,33,44,55,66,77,88,99,
    112,223,334,445,556,667,778,889,999,
    1152,2253,3354,4455,5556,6657,7758,8859,9959,
    1212,2223,3234,4245,5256,6267,7278,8289,9299,
    161,262,363,464,565,666,767,868,969,
    12127,22237,32347,42457,52567,62677,72787,82897,92997,
    12182,22283,32384,42485,52586,62687,72788,82889,92989,
    812182,822283,832384,842485,852586,862687,872788,882889,892989,
    121820,222830,323840,424850,525860,626870,727880,828890,929890,
    120182,220283,320384,420485,520586,620687,720788,820889,920989,
    121802,222803,323804,424805,525806,62607,727808,828809,929809,
]

    # seeds = [1, 10, 21]

    # testing with different arms ################
    # arms = [5, 10, 50, 100, 500]
    # # for a in arms:
    # #     # exp = experiments(seeds, 0.1, 10000, a)
    # #     exp = experiments(seeds, 0.1, 10000, a, 'average')
    # #     exp.get_results()
    # #     exp.plot()
    ##############################################

    exp = experiments(seeds, 0.1, 10000, 10, 'step')
    exp.get_results()
    exp.plot()

    # action = random.choices([0,1],weights = (0.1, 0.05), k = 1)
    # print(action)


'''refernces
1. https://www.geeksforgeeks.org/multi-armed-bandit-problem-in-reinforcement-learning/
2. https://stackoverflow.com/questions/30558087/is-from-matplotlib-import-pyplot-as-plt-import-matplotlib-pyplot-as-plt
3. https://www.geeksforgeeks.org/python-find-index-of-maximum-item-in-list/
4. https://www.w3schools.com/python/ref_random_randint.asp
5. https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
5. chatGPT consultation on how to plot the proportion of optimal actions. See appendix.
6. https://numpy.org/doc/2.1/reference/random/generated/numpy.random.normal.html
'''