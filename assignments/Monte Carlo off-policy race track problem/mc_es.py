from racetracks import tracks
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


class raceCar():
    def __init__(self, track_num, seed, epochs, epsilon, discount, algorithm = 'ES') -> None:
        np.random.seed(seed)
        random.seed = seed
        self.seed = seed
        self.algorithm = algorithm
        self.track_num = track_num
        self.track = tracks[track_num] # randomly select a track to start with
        self.actions = [(0,0),(0,1),(0,-1),(1,0),(1,1),(1,-1),(-1,0),(-1,1),(-1,-1)]
        self.start = np.argwhere(self.track == 'S') # start line
        self.end = np.argwhere(self.track == 'F') # end line
        self.epochs = epochs
        self.ep = epsilon
        self.discount = discount # discount factor for updating G
        self.sa_value = {} # each key is a state, the value is a list consisting of values corresponding to each action. The index of a value is the index of its action
        self.returns = {} # keeps track of the returns of each action in a given state. key is (s,a_ind), value is a deque of length 200
        self.road = {} # the coordinates of grids where the car is allowed to drive in
        self.policy = {} # the target policy
        self.g_history = [] # keeping track of returns
        self.c = {}
        #### building self.road ################
        char_to_exclude = '|'
        coor = np.argwhere(self.track != char_to_exclude)
        # making coor into tuples so that we can construct the hashmap
        coor_tuples = list(map(tuple, coor))
        self.road = dict.fromkeys(coor_tuples)
        ########################################
        
    def get_track(self):
        return self.track
    
    def get_g_hisoty(self):
        return self.g_history

    # selects an action for a given state using the behavior policy
    def select_action(self, state):
        speed = state[2:]
        x = speed[0]
        y = speed[1]
        actions = [(0,0),(0,1),(0,-1),(1,0),(1,1),(1,-1),(-1,0),(-1,1),(-1,-1)]
        filtered_actions = list(filter(lambda t: (not (x + t[0] == 0 and y+t[1] == 0)) and 4 >= x+t[0] >= 0 and 4 >= y+t[1] >= 0, actions))
        # removes 0 acceleration from starting positions
        if state[0] == len(self.track) - 1:
            try:
                filtered_actions.remove((0,0))
            except:
                pass
        # implements the epsilon greedy behavior policy
        if random.random() > self.ep and self.sa_value.get(state) is not None:
            # find indices of filtered actions in self.actions
            indices =  list(map(lambda x: self.actions.index(x), filtered_actions))
            # get values of all filtered actions from sa_value[state]
            values = list(map(lambda x: self.sa_value[state][x], indices))
            # find index through argmax value
            temp_ind = np.argmax(values)
            # use the temp_ind, find the index of the action in self.actions
            act_ind = indices[temp_ind]
        else:
            action = random.choices(filtered_actions, k=1)[0]
            act_ind = self.actions.index(action)
        return act_ind

    # iterate through episodes and update the Q table
    def learn(self):

        # execution for the off policy control method
        if self.algorithm == 'OffControl':
            i = 0
            for _ in range(self.epochs):
                start = random.choices(self.start, k = 1)[0]
                if self.epochs - i < self.epochs//2:
                    self.ep = max(0.1, self.ep * 0.999)
                episodes = self.get_episode(start)
                g = 0
                w = 1
                for sap in reversed(episodes):
                    # behavior policy b: epsilon greedy
                    s = sap[0]
                    a_ind = sap[1]
                    g = self.discount * g - 1
                    # update c
                    if self.c.get(s) is None:
                        self.c[s] = [0]*9
                        self.c[s][a_ind] = w
                    else:
                        self.c[s][a_ind] += w

                    # update Q
                    if self.sa_value.get(s) is None:
                        self.sa_value[s] = [-10]*9
                    self.sa_value[s][a_ind] += w/self.c[s][a_ind] * (g - self.sa_value[s][a_ind])
                    
                    # use target policy to get the optimal action
                    pol_val = max(self.sa_value[s])
                    if self.sa_value[s][a_ind] != pol_val:
                        break
                    w = w/(1-self.ep + self.ep/9)
                    i += 1
                self.g_history.append(g)

        else: # execution for the ES algorithm
            i = 0
            for _ in range(self.epochs):
                start = random.choices(self.start, k = 1)[0]
                if self.epochs - i < self.epochs//2:
                    self.ep = max(0.1, self.ep * 0.999)
                episodes = self.get_episode(start)
                g = 0
                for sap in reversed(episodes):
                    # behavior policy b: epsilon greedy
                    s = sap[0]
                    a_ind = sap[1]
                    g = self.discount * g - 1
                    ep_ind = episodes.index(sap)
                    # update the average returns for a given state action pair
                    if sap not in episodes[:ep_ind]:
                        if self.returns.get(sap) is not None:
                            self.returns[sap][1] += 1
                            self.returns[sap][0] += g
                        else:
                            self.returns[sap] = [g, 1]

                        if self.sa_value.get(s) is None:
                            self.sa_value[s] = [-10]*9
                        self.sa_value[s][a_ind] = self.returns[sap][0]/self.returns[sap][1]
                self.g_history.append(g)

        for k in self.sa_value.keys():
            self.policy[k] = np.argmax(self.sa_value[k])
        return 'policy:', self.policy

    # return the state action pairs of an episode given a start state and epsilon greedy policy
    def get_episode(self, start):
        s = (start[0], start[1], 0, 0) # start state
        sa_pairs = [] # stores all state and action pairs of the episode

        # helper function - check if car has finished the race. Returns 'crashed' if the car has crashed, 'finished' if the car has crossed the finish line, 'destination reached' if the car has reached the next state without crashing or finishing       
        # the check_finish function is generated by chatGPT
        def check_finish(old_state, new_state):
            """
            Checks whether the agent crashes into a boundary ('|') or crosses the finish line ('F').
            Uses Bresenham’s algorithm to check for obstacles along the trajectory.
            """
            x1, y1 = old_state[:2]  # Start position
            x2, y2 = new_state[:2]  # End position

            # Bresenham’s Line Algorithm (for accurate trajectory checking)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1  # Step direction in x
            sy = 1 if y1 < y2 else -1  # Step direction in y
            err = dx - dy  # Bresenham error term
            while True:
                # Check if the car hits a boundary
                if self.track[x1, y1] == '|':
                    return 'crashed'
                elif self.track[x1, y1] == 'F':
                    return 'finished'

                # Stop if we reach the destination
                if (x1, y1) == (x2, y2):
                    return 'destination reached'

                # Bresenham step calculation
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx  # Move in x direction
                if e2 < dx:
                    err += dx
                    y1 += sy  # Move in y direction

        while True:
            # check if that state is a key in sa_value
            act_ind = self.select_action(s)
            action = self.actions[act_ind]
            sa_pairs.append((s, act_ind))
            
            new_speed_x = s[2] + action[0]
            new_speed_y = s[3] + action[1]
            # 0.1 chance the speed doesn't change if state is not start date
            if random.random() < 0.1 and not (s[0]==start[0] and s[1] ==start[1]):
                new_speed_x = s[2]
                new_speed_y = s[3]
            new_coor = (max(0,s[0] - new_speed_x), min(len(self.track[0]), s[1] + new_speed_y))
            # establishing the next state
            new_s = (new_coor[0],new_coor[1],new_speed_x,new_speed_y)
            # check if action hits endline, if so, set end to True and end while loop
            if check_finish(s,new_s) == 'finished':
                return sa_pairs
            
            # check if action hits edge, if so, go back to a randomly selected start state
            elif check_finish(s,new_s) == 'crashed': 
                start = random.choices(self.start, k = 1)[0]
                s = (start[0], start[1], 0, 0)
                
            else:
                s = new_s

    # returns the learned policy
    def get_policy(self):
        return self.policy
    
    # saves the training data to pickle files for future use
    def save_data(self, file):
        with open("trained_racetrack_"+file+".pkl", "wb") as f:
            pickle.dump(self.track, f)

        with open("trained_policy_"+file+".pkl", "wb") as f:
            pickle.dump(self.policy, f)

        print("Track and policy saved successfully!")


# the class that help us run multiple experiments
class experiments():
    def __init__(self, track_num, seeds, epochs, epsilon, discount) -> None:
        self.seeds = seeds
        self.epochs = epochs # number of epochs each agent needs to train
        self.ep = epsilon # epsilon variable for exploration
        self.track_num = track_num # keeping track of the rewards for the step-method
        self.discount = discount # discount factor for return calculation
        self.returns_ES = [] 
        self.returns_OffC = []

    def get_results(self):
        # for seed in self.seeds:
        for seed in tqdm(self.seeds, desc="experiment progress"):
            # set up the two methods
            car = raceCar(self.track_num, seed, self.epochs, self.ep, self.discount)

            # train and record the outputs
            car.learn()
            self.returns_ES.append(car.get_g_hisoty())

            car1 = raceCar(self.track_num, seed, self.epochs, self.ep, self.discount, 'OffControl')

            # train and record the outputs
            car1.learn()
            self.returns_OffC.append(car1.get_g_hisoty())

            # save trajectory
    def plot(self, window_size = 500):
        # plotting rewards
        average_rewards = np.mean(self.returns_ES, axis=0)
        average = np.convolve(average_rewards, np.ones(window_size)/window_size, mode='valid')

        average_rewards_Off = np.mean(self.returns_OffC, axis=0)
        average_Offc = np.convolve(average_rewards_Off, np.ones(window_size)/window_size, mode='valid')
        plt.plot(average, label="average of ES returns", color='b')
        plt.plot(average_Offc, label="average of MC Off Policy Control returns", color='r')
        plt.xlabel("Timesteps")
        plt.ylabel("average of the returns of each episode")
        plt.title("Proportion of Optimal Actions Over Time")

        plt.legend()
        plt.grid()

        # Save the plots
        plt.tight_layout()
        file = 'plots/returns_'+str(self.track_num)+'_'+str(self.epochs)+'.png'
        plt.savefig(file)      

# this function visualizes the trajectory of a car and is generated using chatGPT.
def visualize_race(track, policy, start, filename):
    """
    Visualizes the car's movement in the racetrack following the learned policy.

    Args:
        track (numpy array): The racetrack representation.
        policy (dict): The learned policy mapping state -> best action index.
    """

    # Convert track into a visualizable grid
    track_grid = np.zeros(track.shape)  # 0 = open space
    start_positions = []
    finish_positions = []

    for i in range(track.shape[0]):
        for j in range(track.shape[1]):
            if track[i, j] == '|':
                track_grid[i, j] = -1  # Walls (black)
            elif track[i, j] == 'S':
                track_grid[i, j] = 0.5  # Start positions (green)
                start_positions.append((i, j))
            elif track[i, j] == 'F':
                track_grid[i, j] = 1  # Finish line (blue)
                finish_positions.append((i, j))

    # Pick a random start position
    start = start#random.choice(start_positions)
    state = (start[0], start[1], 0, 0)  # (x, y, v_x, v_y)
    
    # Define movement actions corresponding to action indices
    actions = [(0,0),(0,1),(0,-1),(1,0),(1,1),(1,-1),(-1,0),(-1,1),(-1,-1)]
    
    # Store trajectory
    trajectory = [state[:2]]

    def check_finish(old_state, new_state):
        """
        Checks whether the agent crashes into a boundary ('|') or crosses the finish line ('F').
        Uses Bresenham’s algorithm to check for obstacles along the trajectory.
        """
        x1, y1 = old_state[:2]  # Start position
        x2, y2 = new_state[:2]  # End position

        # Bresenham’s Line Algorithm (for accurate trajectory checking)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1  # Step direction in x
        sy = 1 if y1 < y2 else -1  # Step direction in y
        err = dx - dy  # Bresenham error term
        while True:
            # Check if the car hits a boundary
            if track_grid[x1, y1] == -1:
                return 'crashed'
            elif track_grid[x1, y1] == 1:
                return 'finished'

            # Stop if we reach the destination
            if (x1, y1) == (x2, y2):
                return 'destination reached'

            # Bresenham step calculation
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx  # Move in x direction
            if e2 < dx:
                err += dx
                y1 += sy  # Move in y direction

    # Simulate the car movement
    for step in range(1000):  # Max steps to prevent infinite loops
        if policy.get(state) is None:
            ### FIX BUG ##########
            # policy[state] = random.choice()
            action_index = random.randint(0,8)
            policy[state] = action_index
            # print(f"Terminating early: state {state} not in policy.")  # Debugging info
            # break  # Stop if no policy available for this state
        else:
            action_index = policy[state]  # Get best action from policy
        action = actions[action_index]  # Convert action index to (dx, dy)

        # Update speed (ensuring non-negative speed and not exceeding limits)
        v_x = min(4, max(0, state[2] + action[0]))
        v_y = min(4, max(0, state[3] + action[1]))

        # Update position
        new_x = max(0, state[0] - v_x)
        new_y = min(track.shape[1] - 1, max(0, state[1] + v_y))
        new_state = (new_x, new_y, v_x, v_y)

        # Debugging print statement
        print(f"Step {step}: Moving from {state[:2]} to {new_state[:2]} with speed ({v_x}, {v_y})")

        if check_finish(state, new_state) == 'finished':
            trajectory.append((new_x, new_y))
            print("Reached finish line!")
            break
        elif check_finish(state, new_state) == 'crashed':
            print("Crashed into a wall!")
            break  # Stop simulation on crash

        # Append the new state to trajectory
        trajectory.append((new_x, new_y))
        state = new_state  # Move to the new state

    # Visualization
    trajectory = np.array(trajectory)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(track_grid, cmap="gray", origin="upper")

    # Plot trajectory
    if len(trajectory) > 1:
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 1], trajectory[:, 0], marker="o", color="red", markersize=3, linestyle="-", label="Car Path")

    # Plot start and finish markers
    for pos in start_positions:
        ax.scatter(pos[1], pos[0], color="green", marker="s", s=100, label="Start" if pos == start_positions[0] else "")
    for pos in finish_positions:
        ax.scatter(pos[1], pos[0], color="blue", marker="s", s=100, label="Finish" if pos == finish_positions[0] else "")

    # Fix: Convert trajectory[-1] to tuple before checking in finish_positions
    if tuple(trajectory[-1]) not in finish_positions:
        ax.scatter(trajectory[-1, 1], trajectory[-1, 0], color="black", marker="X", s=100, label="Crash")

    ax.legend()
    ax.set_title("Race Car Simulation")
    plt.savefig(filename)

# this functino runs a single experiment and saves the data to pickle files for generating trajectories
def run(track_num, seed, epochs, epsilon, discount, file, algo):
    car = raceCar(track_num, seed, epochs, epsilon, discount, algo)
    car.learn()
    policy = car.get_policy()
    print(policy)
    car.save_data(file)

# this function uses the pickle files to generate the trajectories
def visualize(file, start, filename):
    ### visualization
    with open("trained_racetrack_"+file+".pkl", "rb") as f:
        track = pickle.load(f)

    with open("trained_policy_"+file+".pkl", "rb") as f:
        policy = pickle.load(f)
        print(len(policy))
    visualize_race(track, policy, start, filename)


if __name__ == '__main__':
    
    # run the following to generate trajectories
    run(0, 2345, 30000, 0.5,0.9,'1','OffControl')
    visualize('1',(29,12),'plots/0_off_30000.png')

    run(0, 2345, 30000, 0.5,0.9,'1','ES')
    visualize('1',(29,12),'plots/0_ES_30000.png')

    run(1, 2345, 30000, 0.5,0.9,'2','OffControl')
    visualize('2',(56,3),'plots/1_off_30000.png')

    run(1, 2345, 30000, 0.5,0.9,'2','ES')
    visualize('2',(56,3),'plots/1_ES_30000.png')

    # run the following to generate average returns plots
    random.seed(5436)
    seeds = random.sample(range(1, 10**6), 20) # adjust the last parameter to change the number of experiments

    exp = experiments(0, seeds, 30000, 0.5, 0.9)
    exp.get_results()
    exp.plot()

    exp = experiments(1, seeds, 30000, 0.5, 0.9)
    exp.get_results()
    exp.plot()
    


'''references
1. https://www.geeksforgeeks.org/how-to-randomly-select-elements-of-an-array-with-numpy-in-python/
2. https://medium.com/towards-data-science/solving-racetrack-in-reinforcement-learning-using-monte-carlo-control-bdee2aa4f04e
3. https://kngwyu.github.io/rlog/en/2022/03/07/sandb-exercise-racetrack.html

'''

