import numpy as np
import matplotlib.pyplot as plt
import random


# set up a step-size QA agent class
class k_armed_non_stationary():
    def __init__(self, arm_num, epsilon, seed, epoch, step, method = 'step') -> None:
        self.arm = arm_num # the number of arms/choices
        self.ep = epsilon # the epsilon that determines the degree of exploration
        np.random.seed(seed)
        self.method = method
        self.frequency = [1] * self.arm
        self.rewards = list(np.random.normal(1, 0.01, arm_num)) # generating random reward for the arms
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

            # check the current optimal action for recording purpose
            curr_optimal = np.argmax(self.rewards)

            # select and take the actual action
            if np.random.rand() < self.ep:
                # randomly choose an action
                act_ind = np.random.choice(self.arm)
            else:
                act_ind = np.argmax(self.values)
            # perform action and update value function
            reward = self.rewards[act_ind]
            if self.method == 'step':
                self.values[act_ind] += self.step * (reward - self.values[act_ind])
            else:
                self.values[act_ind] += (reward - self.values[act_ind])/self.frequency[act_ind] 
                self.frequency[act_ind] += 1 # updates the frequency
            ### CHATGPT consultation on how to take random walk ############
            self.rewards += np.random.normal(0, 0.01, self.arm)
            ###############################################################
            # record reward
            self.reward_overtime.append(reward)

            # record action          
            if act_ind == curr_optimal:
                self.optimal_action.append(1)
            else:
                self.optimal_action.append(0)


class experiments():
    def __init__(self, seeds, epsilon = 0.1, epochs = 10000, arm = 10) -> None:
        self.seeds = seeds
        self.epochs = epochs # number of epochs each agent needs to train
        self.ep = epsilon # epsilon variable for exploration
        self.arm = arm # nubmer of arms
        self.reward_step = [] # keeping track of the rewards for the step-method
        self.optimal_step = [] # keeping track of the optimal actions for the step-method
        self.reward_average = [] # keeping track of the rewards for the average-method
        self.optimal_average = [] # keeping track of the optimal actions for the average-method

    def get_results(self):
        for seed in self.seeds:
            # set up the two methods
            arm_step = k_armed_non_stationary(self.arm, self.ep, seed, self.epochs, 0.1, 'step')
            arm_average = k_armed_non_stationary(self.arm, self.ep, seed, self.epochs, 0.1, 'average')

            # train and record the outputs
            arm_step.train()
            self.reward_step.append(arm_step.get_reward())
            self.optimal_step.append(arm_step.get_optimal())

            arm_average.train()
            self.reward_average.append(arm_average.get_reward())
            self.optimal_average.append(arm_average.get_optimal())

    def plot(self, window_size = 100):
        # first calculate the averages of each
        # then plot
        #################reference to chatGPT instructions start ###############
        average_rewards_step = np.mean(self.reward_step, axis=0)
        average_rewards_average = np.mean(self.reward_average, axis=0)

        proportion_optimal_actions_step = np.mean(self.optimal_step, axis=0)
        proportion_optimal_actions_average = np.mean(self.optimal_average, axis=0)

        # Plotting Results
        plt.figure(figsize=(12, 5))

        # Plot Average Reward
        plt.subplot(1, 2, 1)
        running_avg_step = np.convolve(average_rewards_step, np.ones(window_size)/window_size, mode='valid')
        running_avg_average = np.convolve(average_rewards_average, np.ones(window_size)/window_size, mode='valid')

        # Plot the running average
        plt.plot(running_avg_step, label="Reward for step method", color='b')
        plt.plot(running_avg_average, label="Reward for moving averages", color='r')
        plt.xlabel("Episode")
        plt.ylabel(f"Average Reward (over {window_size} steps)")
        plt.title("Running Average Reward")
        plt.legend()
        plt.grid()

        # Plot Proportion of Optimal Actions
        plt.subplot(1, 2, 2)

        step = np.convolve(proportion_optimal_actions_step, np.ones(window_size)/window_size, mode='valid')
        average = np.convolve(proportion_optimal_actions_average, np.ones(window_size)/window_size, mode='valid')
        plt.plot(step, label="Proportion of Optimal Actions for Step method", color='b')
        plt.plot(average, label="Proportion of Optimal Actions for average method", color='r')

        plt.xlabel("Timesteps")
        plt.ylabel("Proportion of Optimal Actions")
        plt.title("Proportion of Optimal Actions Over Time")

        plt.legend()
        plt.grid()

        # Save the plots
        plt.tight_layout()
        file = 'plots/average_'+str(self.ep)+'_'+str(self.epochs)+'_'+str(self.arm)+'.png'
        plt.savefig(file)


# execution
if __name__ == '__main__':

    # generatingi random seeds for experiments. Change the number of experiments using the last parameter of random.sample
    random.seed(5436)
    seeds = random.sample(range(1, 10**6), 900)

    # setting up agent 1 for 10-armed bandit problem
    exp = experiments(seeds, 0.1, 10000, 10)
    exp.get_results()
    exp.plot()

    # setting up agent 2 for 50-armed bandit problem
    exp1 = experiments(seeds, 0.1, 10000, 50)
    exp1.get_results()
    exp1.plot()

    # setting up agent 3 for 100-armed bandit problem
    exp2 = experiments(seeds, 0.1, 20000, 100)
    exp2.get_results()
    exp2.plot()



'''refernces
1. https://www.geeksforgeeks.org/multi-armed-bandit-problem-in-reinforcement-learning/
2. https://stackoverflow.com/questions/30558087/is-from-matplotlib-import-pyplot-as-plt-import-matplotlib-pyplot-as-plt
3. https://www.geeksforgeeks.org/python-find-index-of-maximum-item-in-list/
4. https://www.w3schools.com/python/ref_random_randint.asp
5. https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
5. chatGPT consultation on how to plot the graphs. See appendix.
6. https://numpy.org/doc/2.1/reference/random/generated/numpy.random.normal.html
'''