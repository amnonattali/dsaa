import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random

import numpy as np

TILE_SIZE = 40
WALL_IMG = np.zeros(shape=(TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8) # black
FREE_IMG = np.zeros(shape=(TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8) + 255 # white
AGENT_IMG_0 = np.zeros(shape=(TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8) + [255,0,0] # red
AGENT_IMG_1 = np.zeros(shape=(TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8) + [0,255,0] # green
AGENT_IMG_2 = np.zeros(shape=(TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8) + [0,0,255] # blue
AGENT_IMG_3 = np.zeros(shape=(TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8) + [255,255,0] # yellow

class FourRoomsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=19, max_steps=100, goal=None, no_env_reward=False):
        # print(size, max_steps, goal)
        self.max_steps = max_steps
        self.loc_goal = goal
        self.no_env_reward = no_env_reward
        if self.no_env_reward:
            print("Unsupervised environment")
        elif self.loc_goal is None:
            print("Goal is fourth room")
        else:
            print("Goal is:", self.loc_goal)
        print("Max steps:", self.max_steps)

        self.width = size
        self.height = size
        
        # make the grid
        self._make_grid()
        
        # turn left, turn right, forward
        self.action_space = spaces.Discrete(4)
        # free/wall/agent
        '''
        self.grid_space = spaces.Box(
            low=0,
            high=2,
            shape=(self.height, self.width, 1),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.grid_space,
            #'direction': spaces.Discrete(4)
        })
        '''
        
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, self.height, self.width),
            dtype='uint8'
        )
        
        #self.observation_space = spaces.Box(low=0, high=19, shape=(2,))
    
        # Initialize the state
        self.reset()
        
        self.viewer = None
        
    def _make_grid(self):
        # zero means free space
        self.grid = np.zeros(shape=(self.height, self.width))

        # surrounding walls
        self.grid[0, :] = 1
        self.grid[self.height-1, :] = 1
        self.grid[:,0] = 1
        self.grid[:,self.width-1] = 1
        
        # middle walls
        self.grid[self.height // 2, :] = 1
        self.grid[:, self.width // 2] = 1

        # entry-ways
        self.grid[self.height // 2, self.width // 4] = 0
        self.grid[self.height // 2, 3*self.width // 4] = 0
        self.grid[self.height // 4, self.width // 2] = 0
        self.grid[3*self.height // 4, self.width // 2] = 0
    
    def _is_free(self, pos):
        return self.grid[pos[0], pos[1]] == 0
    def is_free(self, pos):
        return self.grid[pos[0], pos[1]] == 0
    
    def _is_goal(self, pos):
        if not self.loc_goal is None: 
            if pos[0] == self.loc_goal[0] and pos[1] == self.loc_goal[1]:
                return True
            return False
        
        if pos[0] < self.height // 2 and pos[1] < self.width // 2 and self.goal == 0:
            return True
        if pos[0] < self.height // 2 and pos[1] > self.width // 2 and self.goal == 1:
            return True
        if pos[0] > self.height // 2 and pos[1] < self.width // 2 and self.goal == 2:
            return True
        if pos[0] > self.height // 2 and pos[1] > self.width // 2 and self.goal == 3:
            return True
        return False
    
    def step(self, action):
        # never eat sour wheat
        dir_to_vec = {0: [-1, 0], 1: [0,1], 2: [1,0], 3: [0,-1]}

        self.step_count += 1

        reward = 0
        done = False
        
        tmp_pos = self.agent_pos[:2] + dir_to_vec[action]
        # this while loop means always move
        # while not self._is_free(tmp_pos):
        #     rand_act = random.randint(0,3)
        #     tmp_pos = self.agent_pos[:2] + dir_to_vec[rand_act]

        if self._is_free(tmp_pos):
            self.agent_pos[:2] = np.copy(tmp_pos)
            reward -= 0.0
        else:
            done = False
            #reward -= 1
            
        
        if self._is_goal(self.agent_pos):
            done = True
            reward += 1
        
        obs = self._make_obs()
        
        #reward += 0.1*(self.agent_pos[0] + self.agent_pos[1])
        if self.no_env_reward:
            done = False
            reward = 0

        if self.step_count >= self.max_steps:
            done = True
        
        return obs, reward, done, {}
    
    def _make_obs_2(self):
        grid_copy = np.copy(self.grid).reshape(self.height, self.width, 1)
        grid_copy[self.agent_pos[0], self.agent_pos[1]] = 2
        return {'image': grid_copy} #, 'direction': self.agent_pos[2]}
    def _make_obs(self):
        grid_copy = np.copy(self.grid).reshape(1, self.height, self.width)
        grid_copy[0, self.agent_pos[0], self.agent_pos[1]] = 2
        return grid_copy
    def _make_obs_3(self):
        return self.agent_pos[:2]
        
    # set a goal
    # set the agent position and orientation randomly in the grid, not in goal
    def reset(self):
        
        #self.goal = np.random.randint(4)
        # hard code goal for now
        self.goal = 3
        
        pos = np.random.randint(low=(0,0,0), high=(self.height, self.width, 4), size=3)
        while not self._is_free(pos) or self._is_goal(pos):
            pos = np.random.randint(low=(0,0,0), high=(self.height, self.width, 4), size=3)
        pos = np.array([2,2,0])

        self.agent_pos = pos
        self.step_count = 0
        
        return self._make_obs()
    
    def render(self, mode='human'):
        # Compute the total grid size
        width_px = self.width * TILE_SIZE
        height_px = self.height * TILE_SIZE

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                ymin = j * TILE_SIZE
                ymax = (j+1) * TILE_SIZE
                xmin = i * TILE_SIZE
                xmax = (i+1) * TILE_SIZE
                
                if self.grid[j, i] == 0: # free:
                    img[ymin:ymax, xmin:xmax, :] = FREE_IMG
                elif self.grid[j, i] == 1: # wall:
                    img[ymin:ymax, xmin:xmax, :] = WALL_IMG
                else:
                    print("ERROR, grid should only have 0 and 1")
        # agent
        ymin = self.agent_pos[0] * TILE_SIZE
        ymax = (self.agent_pos[0]+1) * TILE_SIZE
        xmin = self.agent_pos[1] * TILE_SIZE
        xmax = (self.agent_pos[1]+1) * TILE_SIZE
        
        if self.agent_pos[2] == 0:
            img[ymin:ymax, xmin:xmax, :] = AGENT_IMG_0
        elif self.agent_pos[2] == 1:
            img[ymin:ymax, xmin:xmax, :] = AGENT_IMG_1
        elif self.agent_pos[2] == 2:
            img[ymin:ymax, xmin:xmax, :] = AGENT_IMG_2
        elif self.agent_pos[2] == 3:
            img[ymin:ymax, xmin:xmax, :] = AGENT_IMG_3
        else:
            print("ERROR AGENT DIR")
                        
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
    
    def get_keys_to_action(self):
        keys_to_action = {}
        keys_to_action[ord('a')] = 0
        keys_to_action[ord('d')] = 1
        keys_to_action[ord('w')] = 2

        return keys_to_action
    
    def get_action_meanings(self):
        return ["LEFT", "RIGHT", "FORWARD"]
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None