from racetracks import tracks, speed
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

'''bugs and problems
2.some policy updates not reflected in the policy dictionary
3. need 2 graphs - one for performance measure and one for trajectory.
    performance measure - can be reward function.
    4. additional question: can explore different reward functions
5. if you don't have the exact same files as the ones in textbook, it might not work
6. need to fix action selection
7. can compare yours with the medium article implementation
'''
# converging criteria
    # calculate the changes in the Q table in the past x episodes if you have a Q function
    # if you don't have a Q function: learning progress checks the level of performance over a short window and compare it to a long window. If the results are similar, you can stop learning. The question if how long your windows should be.
    # or just set a fixed number of episodes - this is what most people do
# racetrack type numbers
# when starting at the starting line, you can't have (0,0) accelration
# detection of the finish line: 
# randomly selecting reward functions - measuring crash rate, time to destination,etc
# visulizing matrices: matplotlib?
# additional experiment: can try different reward functions or MC methods.
# diagram: can use dots to mark the trajectory


# initialize environment - 3 different n x m matrices with boundaries marked with '|'. Each episode will randomly select a matrix.

# each state would be the coordinate and velocity of the agent

# action will be -1, 0, 1 in x and y directions

# for all state-action pairs, randomly initialize state-action values, and pick the best policy from the generated values for each state-action

class raceCar():
    def __init__(self, track_num, seed, epochs, epsilon, discount) -> None:
        np.random.seed(seed)
        random.seed = seed
        self.track = tracks[track_num] # randomly select a track to start with
        self.actions = [(0,0),(0,1),(0,-1),(1,0),(1,1),(1,-1),(-1,0),(-1,1),(-1,-1)]
        self.speed = speed
        # self.road = {} # a hashmap that keeps track of coordinates of open road. Excluding edges.
        # self.states = [] # a list of tuples
        # self.start_states = []
        self.start = np.argwhere(self.track == 'S') # start line
        self.end = np.argwhere(self.track == 'F') # end line
        self.epochs = epochs
        self.ep = epsilon
        self.discount = discount # discount factor for updating G
        self.sa_value = {} # each key is a state, the value is a list consisting of values corresponding to each action. The index of a value is the index of its action
        self.c = {}
        self.road = {} # the coordinates of grids where the car is allowed to drive in
        self.policy = {} # the target policy
        #### building self.road ################
        char_to_exclude = '|'
        coor = np.argwhere(self.track != char_to_exclude)
        # print(len(coor))
        # making coor into tuples so that we can construct the hashmap
        coor_tuples = list(map(tuple, coor))
        self.road = dict.fromkeys(coor_tuples)
        ########################################
        # self.initialize_states() # constructing the states using track and actions
        # self.sa_value = dict.fromkeys(self.states, [0])
        # self.c = dict.fromkeys(self.states, [0])
        
    def get_track(self):
        return self.track

    # selects an action for a given state using the behavior policy
    def select_action(self, state):
        speed = state[2:]
        x = speed[0]
        y = speed[1]
        actions = [(0,0),(0,1),(0,-1),(1,0),(1,1),(1,-1),(-1,0),(-1,1),(-1,-1)]
        filtered_actions = list(filter(lambda t: (not (x + t[0] == 0 and y+t[1] == 0)) and 4 >= x+t[0] >= 0 and 4 >= y+t[1] >= 0, actions))
        if state[0] == len(self.track) - 1:
            try:
                filtered_actions.remove((0,0))
            except:
                pass
        if self.sa_value.get(state) is not None:
            prob = 1-self.ep + self.ep/len(filtered_actions)

        else:
            prob = 1/len(filtered_actions)
        action = random.choices(filtered_actions, k=1)[0]
        act_ind = self.actions.index(action)
        return act_ind, prob


    def learn(self):
        # while not converging
        # for _ in range(self.epochs):
        i = 0
        for _ in tqdm(range(self.epochs), desc="Epoch"):
            # pass
            # start an episode
            # choose a random start state
            # print(self.start)
            # print('running epoch _______________________________________________________')
            start = random.choices(self.start, k = 1)[0]
            # print(start)
            if self.epochs - i < self.epochs//2:
                self.ep = max(0.1, self.ep * 0.999)
            episodes = self.get_episode(start)
            # print('episodes of this epoch _______________________________________________________')
            # print(episodes)
            g = 0
            w = 1
            for sap in reversed(episodes):
                # print('processing episode data')
                # print(sa)
                # behavior policy b: epsilon greedy
                s = sap[0]
                a_ind = sap[1]
                prob = sap[2]
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
                # pol = int(np.argmax(self.sa_value.get(s)))
                pol_val = max(self.sa_value[s])
                #If A(t) != PIE(S(t)) then exit inner Loop
                if self.sa_value[s][a_ind] != pol_val:
                    # print('unequal')
                    break
                w = w/(1-self.ep + self.ep/9)
                i += 1
        
        # return the target policy
        # pass

    # return the state action pairs of an episode given a start state and epsilon greedy policy
    def get_episode(self, start):
        s = (start[0], start[1], 0, 0) # start state
        # check self.sa_value for all available action values for a certain state
        # end = False
        sa_pairs = []

        # helper function check if car has finished the race. Only returns true if a car crossed the finish line without hitting an edge first       
        # instructed by chatGPT on the crash detection algorithm
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

        # generate an action to change speed. Speed have to be <= 5 in both directions and can't both be 0 at the same time
        while True:
            # print('generating a timestep for an episode')
            # at each state, choose action
            # check if that state is a key in sa_value
            act_ind, prob = self.select_action(s)
            action = self.actions[act_ind]
            sa_pairs.append((s, act_ind, prob))
            
            new_speed_x = s[2] + action[0]
            new_speed_y = s[3] + action[1]
            # 0.1 chance the speed doesn't change if state is not start date
            if random.random() < 0.1 and not (s[0]==start[0] and s[1] ==start[1]):
                new_speed_x = s[2]
                new_speed_y = s[3]
            new_coor = (max(0,s[0] - new_speed_x), min(len(self.track[0]), s[1] + new_speed_y))
            # establishing the next state
            new_s = (new_coor[0],new_coor[1],new_speed_x,new_speed_y)
            if new_speed_x == 0 and new_speed_y == 0:
                print('invalid speed',action,'detected for',s,"resulting in",new_s,'-----------------------------------')
            # print(s)
            # print(new_s)
            # check if action hits endline, if so, set end to True and end while loop
            if check_finish(s,new_s) == 'finished':
                return sa_pairs
            
            # check if action hits edge, if so, go back to a randomly selected start state
            elif check_finish(s,new_s) == 'crashed': #self.road.get(new_coor) is None or check_finish(s,new_s) == 'crashed':
                # print(s,new_s)
                # print('crashed')
                start = random.choices(self.start, k = 1)[0]
                # print(start)
                s = (start[0], start[1], 0, 0)
            else:
                # print('ongoing')
                s = new_s


    def get_policy(self):
        for k in self.sa_value.keys():
            self.policy[k] = np.argmax(self.sa_value[k])
        return 'policy:', self.policy
    
    def save_data(self, file):
        with open("trained_racetrack_"+file+".pkl", "wb") as f:
            pickle.dump(self.track, f)

        with open("trained_policy_"+file+".pkl", "wb") as f:
            pickle.dump(self.policy, f)

        print("Track and policy saved successfully!")


def visualize_race(track, policy, start):
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

    # Simulate the car movement
    for step in range(500):  # Max steps to prevent infinite loops
        if policy.get(state) is None:
            policy[state] = [-10] * 9
            # print(f"Terminating early: state {state} not in policy.")  # Debugging info
            # break  # Stop if no policy available for this state
        
        action_index = np.argmax(policy[state])  # Get best action from policy
        action = actions[action_index]  # Convert action index to (dx, dy)

        # Update speed (ensuring non-negative speed and not exceeding limits)
        v_x = min(5, max(0, state[2] + action[0]))
        v_y = min(5, max(0, state[3] + action[1]))

        # Update position
        new_x = min(track.shape[0] - 1, max(0, state[0] + v_x))
        new_y = min(track.shape[1] - 1, max(0, state[1] + v_y))
        new_state = (new_x, new_y, v_x, v_y)

        # Debugging print statement
        print(f"Step {step}: Moving from {state[:2]} to {new_state[:2]} with speed ({v_x}, {v_y})")

        # Check for finish line or crash
        if (new_x, new_y) in finish_positions:
            trajectory.append((new_x, new_y))
            print("Reached finish line!")
            break  # Stop when reaching finish
        elif track[new_x, new_y] == '|':  # Crash
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
    plt.savefig('trajectory.png')

def run(track_num, seed, epochs, epsilon, discount, file):
    car = raceCar(track_num, seed, epochs, epsilon, discount)
    car.learn()
    policy = car.get_policy()
    print(policy)
    car.save_data(file)

def visualize(file, start):
    ### visualization
    with open("trained_racetrack_"+file+".pkl", "rb") as f:
        track = pickle.load(f)

    with open("trained_policy_"+file+".pkl", "rb") as f:
        policy = pickle.load(f)
        print(policy)
    visualize_race(track, policy, start)


if __name__ == '__main__':
    # run(2, 2345, 10000, 0.5,0.8,'0')
    visualize('0',(4,1))


'''references
1. https://www.geeksforgeeks.org/how-to-randomly-select-elements-of-an-array-with-numpy-in-python/

'''

