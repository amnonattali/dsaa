import random, gym, torch
from gym import Wrapper
import numpy as np

#-------------------
class MontezumaNoReward(Wrapper):
    def __init__(self, config):
        env = gym.make("MontezumaRevenge-ram-v4")
        super(MontezumaNoReward, self).__init__(env)
        self.example_obs = env.reset()

        self.observation_size = 128
        self.action_size = 18
        self.preprocessors = []
        self.name = "MZ_ram"

    def get_image(self):
        return self.env.render(mode="rgb_array")

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return obs.tolist()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.tolist(), 0, done, info

    def close(self):
        return self.env.close()
#-------------------


# the environment fourrooms-v0 returns a grid with 2 for agent location, 1 for obstacle, 0 for free
def obs_to_loc(obs):
    obs = torch.tensor(obs, dtype=torch.float).view(1,1,obs.shape[-2], obs.shape[-1])
    idx = obs.view(len(obs), -1).argmax(dim=1)
    return [idx.item() // obs.shape[2], idx.item() % obs.shape[3]]

# Basic FourRooms environment
#   state is the [x,y] coordinate of the agent
#   actions are in [0-3], to move the agent in each of 4 directions
class BaseFourRooms(Wrapper):
    def __init__(self, config):
        max_steps = config["max_steps"]
        env = gym.make('dsaa_envs:fourrooms-v0', max_steps=max_steps)
        super(BaseFourRooms, self).__init__(env)
        self.example_obs = env.make_obs()

        self.observation_size = 2
        self.action_size = 4
        self.preprocessors = [obs_to_loc]
        self.name = "four_rooms"

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return obs_to_loc(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs_to_loc(obs), reward, term, trunc, info

    def close(self):
        return self.env.close()

# Basic FourRooms environment
#   state is the [x,y] coordinate of the agent
#   actions are in [0-3], to move the agent in each of 4 directions
#   reward is always zero
# TODO: consolidate this into BaseFourRooms with no_env_reward as a parameter
class FourRoomsNoReward(Wrapper):
    def __init__(self, config):
        max_steps = config["max_steps"]
        env = gym.make('dsaa_envs:fourrooms-v0', max_steps=max_steps, no_env_reward=True)
        super(FourRoomsNoReward, self).__init__(env)
        self.example_obs = env.make_obs()

        self.observation_size = 2
        self.action_size = 4
        self.preprocessors = [obs_to_loc]
        self.name = "four_rooms"

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return obs_to_loc(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs_to_loc(obs), 0, term, trunc, info

    def close(self):
        return self.env.close()

# Here we append to the state a random number in [0-100]
class FourRoomsRandomNoise(Wrapper):
    def __init__(self, config):
        max_steps = config["max_steps"]
        env = gym.make('dsaa_envs:fourrooms-v0', max_steps=max_steps)
        super(FourRoomsRandomNoise, self).__init__(env)
        self.example_obs = env.make_obs()

        self.observation_size = 3
        self.action_size = 4
        self.preprocessors = [self.add_bit]
        self.name = "four_rooms_random_bit"

    def add_bit(self, x):
        return obs_to_loc(x) + [100*random.random()]

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return self.add_bit(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self.add_bit(obs), reward, term, trunc, info

    def close(self):
        return self.env.close()

class TwoRoomsViz(Wrapper):
    def __init__(self, make_two_rooms=True):
        self.env = gym.make('dsaa_envs:fourrooms-v0', max_steps=10000)
        super(TwoRoomsViz, self).__init__(self.env)
        
        self.observation_size = 2
        self.action_size = 4
        self.name = "make_vis"
        self.make_two_rooms = make_two_rooms

    def make_state(self, obs):
        # Make one_hot
        obs = torch.tensor(obs, dtype=torch.float).view(obs.shape[-2], obs.shape[-1]).flatten()
        return list(1.0*(obs == 2).numpy())
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        # block off the bottom two rooms
        if self.make_two_rooms:
            self.env.grid[9,4] = 1
            self.env.grid[9,14] = 1
        return self.make_state(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        state = self.make_state(obs)
        return state, reward, term, trunc, info

# The basic manipulator environment in which 
#   we preprocess the reward based on desired task
#   we preprocess the state to append the movable object
# TODO: need to update this to new gym etc.
class Manipulator2D(Wrapper):
    def __init__(self, config):
        self.num_joints = config["num_arm_joints"]
        self.arm_lengths = np.zeros(self.num_joints) + config["arm_joint_lengths"]
        self.task_height = config["ball_goal_height"]
        env = gym.make('dsaa_envs:manipulator2d-v0', num_joints = self.num_joints, 
            arm_lengths = self.arm_lengths, max_steps=config["max_steps"])
        
        super(Manipulator2D, self).__init__(env)
        
        self.example_obs = self.make_obs(env.config)

        # there is one movable object
        self.observation_size = self.num_joints + len(self.env.movable_objects[0]) - 1
        self.action_size = self.num_joints * 2 # for each joint you can move angle +-1
        self.name = "manipulator_2d_yes_obj"

    def make_obs(self, obs):
        return list(np.concatenate((obs, self.env.movable_objects[0][:2]), axis=0))

    def reset(self, seed=None, options=None):
        return self.make_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        env_reward = 1 if self.env.movable_objects[0][1] < self.task_height else 0
        return self.make_obs(obs), env_reward, done, info

    def close(self):
        return self.env.close()

# Here we do not append the object to the state
# TODO: need to update this to new gym etc.
class Manipulator2DNoOBJ(Wrapper):
    def __init__(self, config):
        self.num_joints = config["num_arm_joints"]
        self.arm_lengths = np.zeros(self.num_joints) + config["arm_joint_lengths"]
        self.task_height = config["ball_goal_height"]
        env = gym.make('dsaa_envs:manipulator2d-v0', num_joints = self.num_joints, 
            arm_lengths = self.arm_lengths, max_steps=config["max_steps"])

        super(Manipulator2DNoOBJ, self).__init__(env)
        
        self.example_obs = self.make_obs(env.config)

        self.observation_size = self.num_joints
        self.action_size = self.num_joints * 2
        self.name = "manipulator_2d_no_obj"

    def make_obs(self, obs):
        return list(obs)

    def reset(self, seed=None, options=None):
        return self.make_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        env_reward = 1 if self.env.movable_objects[0][1] < self.task_height else 0
        return self.make_obs(obs), env_reward, done, info

    def close(self):
        return self.env.close()