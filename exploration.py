'''
**Figure 4 from paper**

Test our abstracion with different exploration schemes
'''

from utils import ReplayBuffer
import numpy as np
import random

class State:
    def __init__(self, state, goal, env_grid, dist_from_start=0):
        self.state = state
        self.goal = goal
        self.env_grid = env_grid
        
        # in each state we store the distance from the start
        self.dist_from_start = dist_from_start
        
        self.min_x = self.min_y = 0
        self.max_x = len(env_grid[0])
        self.max_y = len(env_grid)
    
    # helper function to determine if the state is within the boundary
    def in_boundary(self):
        if self.state[0] < self.min_y or \
            self.state[0] >= self.max_y or \
            self.state[1] < self.min_x or \
            self.state[1] >= self.max_x:
            return False
        return True
    
    def is_free(self):
        return self.in_boundary() and self.env_grid[self.state[0], self.state[1]] == 0
        
    # return a list of neighboring States
    def get_neighbors(self):
        # the 4 neighbors are right, left, down, up
        directions = [[0,1],[0,-1],[1,0],[-1,0]]
        random.shuffle(directions)
        neighboring_positions = [(self.state[0] + direction[0],
                                  self.state[1] + direction[1])
                                 for direction in directions]
        # the neighbor is one further from the start in terms of steps taken and has the same goal
        neighboring_states = [State(state = position,
                                    goal = self.goal,
                                    env_grid = self.env_grid,
                                    dist_from_start = self.dist_from_start + 1)
                             for position in neighboring_positions]
        # remove the neighbor if it is outside the grid or in collision with obstacle
        return [s for s in neighboring_states if s.is_free()]
    
    # checks if the current state is the goal
    def is_goal(self):
        if type(self.goal) == int:
            return self.is_room_goal()

        return self.state == self.goal
    def is_room_goal(self):
        if self.state[0] < self.max_y // 2 and self.state[1] < self.max_x // 2 and self.goal == 0:
            return True
        if self.state[0] < self.max_y // 2 and self.state[1] > self.max_x // 2 and self.goal == 1:
            return True
        if self.state[0] > self.max_y // 2 and self.state[1] < self.max_x // 2 and self.goal == 2:
            return True
        if self.state[0] > self.max_y // 2 and self.state[1] > self.max_x // 2 and self.goal == 3:
            return True
        return False
        
    # a state is defined by it's position which is already hashable
    def __hash__(self):
        return hash(self.state)
    def __eq__(self, other):
        return self.state == other.state
    def __str__(self):
        return str(self.state)
    def __repr__(self):
        return str(self.state)

def simple_search(starting_state, do_bfs = True):
    # starting from current state we look backwards at the parent of each state until we reach the start
    def backtrack(visited_states, current_state):
        path = []
        # the parent of start is None
        while current_state is not None:
            # add the state to the path
            path.append(current_state)
            # set the current_state to its parent
            current_state = visited_states[current_state][0]
        # we return the path starting from the start state
        path.reverse()
        return path
    # we can put states into dictionaries because we implement the __hash__ method of State
    visited_states = {starting_state: (None, starting_state)}
    # the frontier is a queue of states yet to visit in the search
    frontier = [starting_state]
    
    # while the frontier is not empty we keep searching
    while frontier:
        if do_bfs:
            current_state = frontier.pop(0) # BFS we remove the first element from the frontier
        else:
            current_state = frontier.pop(-1) # DFS we remove the last element from the frontier

        # if we remove a goal from the frontier we're done
        if current_state.is_goal():
            # backtrack is a function that gives the full path starting from start_state until current_state
            return backtrack(visited_states, current_state), visited_states
        
        # get_neighbors returns the neighboring states
        neighbors = current_state.get_neighbors()
        for neighbor in neighbors:
            if neighbor not in visited_states:
                frontier.append(neighbor)
                visited_states[neighbor] = (current_state, neighbor)
                
    # if our loop ends (the frontier is empty) and we have not found the goal, then we return an empty list
    return []

def leave_room_option(grid, goal=(4,10), max_steps=100):
    diff_to_action = {(-1, 0): 0, (0,1): 1, (1,0):2, (0,-1):3}
    option_policy = np.zeros(grid.shape)
    option_termination = np.ones(grid.shape)
    for i in range(10):
        for j in range(10):
            if grid[i,j] == 1:
                continue
            # print(i,j)
            option_termination[i,j] = 0.0
            start = (i,j)
            starting_state = State(start, goal, grid)
            path, _ = simple_search(starting_state)
            option_policy[i,j] = diff_to_action[(path[1].state[0] - i, path[1].state[1] - j)]

    max_steps_option = np.zeros((1, *grid.shape)) + max_steps
    return np.expand_dims(option_policy, 0), np.expand_dims(option_termination, 0), max_steps_option

# for each primitive action we have an option that takes that action everywhere and terminates immediately
# NOTE: no termination set because initiation set is all_states / termination_set
def get_action_options(grid, repeat=1):
    final_shape = (4, *grid.shape)
    option_policies = np.zeros(final_shape)
    for i in range(final_shape[0]):
        option_policies[i,:] = i
    option_termination = np.zeros(final_shape)
    option_max_steps = np.ones(final_shape) * repeat
    return option_policies, option_termination, option_max_steps

def wall_hugging_option(grid):
    dir_to_vec = {0: [-1, 0], 1: [0,1], 2: [1,0], 3: [0,-1]}
    option_policy = np.zeros(grid.shape)
    option_termination = np.zeros(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i,j] == 1:
                continue
            dir_to_wall = []
            for action in dir_to_vec:
                # 1 if wall, else 0
                dir_to_wall.append(grid[i + dir_to_vec[action][0]][j + dir_to_vec[action][1]] > 0)
            
            if sum(dir_to_wall) == 0:
                option_termination[i,j] = 1
                option_policy[i,j] = -1 # should never happen
                continue
            
            action = random.randrange(4)
            # wall above --> go left
            if dir_to_wall[0]:
                action = 3
            # wall left --> go down
            if dir_to_wall[3]:
                action = 2
            # wall down --> go right
            if dir_to_wall[2]:
                action = 1
            # wall right and not above --> go up
            if dir_to_wall[1] and not dir_to_wall[0]:
                action = 0
            
            # horizontal hallway, go random direction
            if dir_to_wall[0] and dir_to_wall[2]:
                action = 1
                if random.random() < 0.5:
                    action = 3
            # vertical hallway, go random direction
            if dir_to_wall[1] and dir_to_wall[3]:
                action = 0
                if random.random() < 0.5:
                    action = 2
            
            option_policy[i,j] = action
            # if i == 1 and j == 2:
            #     print(action, dir_to_wall)
            # option_termination[i,j] = 0 # default...
    max_steps = np.zeros((1, *grid.shape)) + 100
    return np.expand_dims(option_policy, 0), np.expand_dims(option_termination, 0), max_steps
            
def expert_data(grid, num_steps):
    paths = []
    total = 0
    grid = grid == 1
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            # first sample random start (in free space)
            #start = (np.random.randint(0, len(grid)), np.random.randint(0, len(grid[0])))
            if grid[row, col] == 1:
                continue

            start = (row, col)
            if True:
                # all_goals = np.random.randint(0,19*19)
                # for _ in range(32):
                #     goal = np.random.randint(0,19*19)
                #     goal_r = goal // 19
                #     goal_c = goal % 19
                all_goals = [[17,17]]
                for goal_r, goal_c in all_goals:
                    # for goal_r in range(len(grid)):
                    #     for goal_c in range(len(grid[0])):
                    if grid[goal_r, goal_c] == 1:
                        continue
                    goal = (goal_r, goal_c)
                    starting_state = State(start, goal, grid)
                    # print(start, goal)
                    path, _ = simple_search(starting_state)
                    paths.append([path, start, goal])
                    total += len(path) - 1        
            else:
                for goal in [0,1,2,3]:
                    starting_state = State(start, goal, grid)
                    if not starting_state.is_free() or starting_state.is_goal():
                        continue
                    path, _ = simple_search(starting_state)
                    paths.append([path, start, goal])
                    total += len(path) - 1
    
    print("total steps expert", total, "max", num_steps)
    replay_buffer = ReplayBuffer(1000000)
    for path in paths:
        for i in range(len(path[0])-1):
            replay_buffer.add((path[0][i].state, path[0][i+1].state, None, 0, 0))
    return replay_buffer


def option_exploration(env, num_steps, option_policies, option_termination, option_max_steps):
    print(f"Performing option exploration with {len(option_policies)} options for {num_steps} steps**")
    data = []
    replay_buffer = ReplayBuffer(num_steps)
    state = env.reset()
    done = True
    for _ in range(num_steps):
        while done or \
                option_termination[current_option][state[0], state[1]] or \
                    num_steps_in_option >= option_max_steps[current_option][state[0], state[1]]:
            if False:#not option_termination[4][state[0], state[1]]:
                current_option = 4
            else:
                current_option = np.random.choice(range(len(option_policies)))
            done = False
            num_steps_in_option = 0
        
        action = option_policies[current_option][state[0], state[1]]
        # print(current_option, action, state)
        num_steps_in_option += 1
        # action = env.action_space.sample()
        next_state , _, done, _ = env.step(action)
        
        data.append([state, next_state])
        replay_buffer.add((state, next_state, action, 0, 0))
        
        if done:
            print("state reset")
            state = env.reset()
        else:
            state = next_state
    return replay_buffer, data

def random_exploration(env, num_steps, add_noise=False):
    print(f"Performing random exploration for {num_steps} steps**")
    data = []
    replay_buffer = ReplayBuffer(num_steps)
    state = env.reset()

    if add_noise:
        random_bit = np.random.randint(0,19)
        state += [random_bit]
        
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_state , _, done, _ = env.step(action)

        if add_noise:
            random_bit = np.random.randint(0,19)
            next_state += [random_bit]

        data.append([state, next_state])
        replay_buffer.add((state, next_state, action, 0, 0))

        if done:
            state = env.reset()
            if add_noise:
                random_bit = np.random.randint(0,19)
                state += [random_bit]
        else:
            state = next_state
    return replay_buffer, data

import torch
import matplotlib.pyplot as plt

from transfer_experiments.transfer_utils import train_dsaa, FourRoomsNoReward

if __name__=="__main__":
    env_config = {
        "max_steps": 1000000 # no reset
    }
    env = FourRoomsNoReward(env_config)
    env_grid = env.example_obs[0] == 1

    print("**Getting Options**")
    num_states = 19*19
    option_policies, option_termination, option_max_steps = get_action_options(env_grid, repeat=1)    
    option_funcs =[
        leave_room_option(env_grid, goal=(4,10), max_steps=1),
        leave_room_option(env_grid, goal=(10,4), max_steps=1),
        # wall_hugging_option(env_grid),
        # get_action_options(env_grid, repeat=4)
    ]
    for pol, term, max_steps in option_funcs:
        option_policies = np.concatenate([option_policies, pol], axis=0)
        option_termination = np.concatenate([option_termination, term], axis=0)
        option_max_steps = np.concatenate([option_max_steps, max_steps], axis=0)

    print("**Exploring**")
    num_steps = 1000000
    replay_buffer, data = option_exploration(env, num_steps, option_policies, option_termination, option_max_steps)
    # replay_buffer = expert_data(env.example_obs[0], num_steps)
    # replay_buffer.buffer.extend(replay_buffer_.buffer)
    # replay_buffer, data = random_exploration(env, num_steps, add_noise=True)

    count_grid = np.zeros((19,19))
    for exp in replay_buffer.buffer:
        s = exp[0]
        count_grid[s[0], s[1]] += 1

    plt.imshow(count_grid / np.sum(count_grid))
    plt.colorbar()
    plt.savefig("tmp_data/00_counts.png")
    plt.clf()

    print("**Training Abstraction**")
    phi = train_dsaa(replay_buffer, config={
        "num_abstract_states": 16,
        "num_abstraction_updates": 10000,
        "abstraction_batch_size": 1024,
        "use_gumbel": True,
        "gumbel_tau": 0.8,
        "sr_gamma": 0.9,
        "abstraction_entropy_coef": 10.0,
        "hard": False,
        "learning_rate": 0.001
    })

    if True:
        env_grid = (env.example_obs == 1)*1.0
        # print(env_grid)
        with torch.no_grad():
            all_phis = torch.zeros((19, 19))
            for i in range(env_grid.shape[1]):
                for j in range(env_grid.shape[2]):
                    if env_grid[0,i,j] > 0:
                        all_phis[i,j] = -1
                        continue
                    
                    # tmp_grid = torch.FloatTensor([i,j, np.random.randint(0,19)])
                    tmp_grid = torch.FloatTensor([i,j])
                    tmp_enc = phi(tmp_grid.unsqueeze(0))
                    all_phis[i,j] = torch.argmax(tmp_enc[0])
        
        plt.imshow(all_phis)
        plt.xticks([])
        plt.yticks([])
        # plt.colorbar()

        name = "first_room_abstraction_16"
        # name = "random_abstraction_4"
        # name = "expert_to_corner_8"
        # name = "noise_16"
        plt.savefig(f"tmp_data/{name}.png", bbox_inches='tight')
        plt.savefig(f"tmp_data/{name}.svg", bbox_inches='tight', format="svg")