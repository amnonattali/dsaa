'''
**Figure 11 from paper**

Here we visualize the abstraction when the input is an image (FourRooms)
TODO: this needs to be cleaned up and incorporated with other experiments
'''

import gym
from gym import Wrapper
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from update_models import update_abstraction
from torch_models import SuccessorRepresentation, Abstraction
from utils import ReplayBuffer
from environments.env_wrappers import obs_to_loc

class FourRoomsImg(Wrapper):
    def __init__(self, config):
        max_steps = config["max_steps"]
        env = gym.make('dsaa_envs:fourrooms-v0', max_steps=max_steps)
        super(FourRoomsImg, self).__init__(env)
        self.example_obs = env._make_obs()

        self.observation_size = 19*19
        self.action_size = 4
        self.name = "four_rooms_img"

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def close(self):
        return self.env.close()

class TransitionsDataset(Dataset):
    # A transition is a pair of consecutive states (x,x')
    # If we want images we can do the img transformation here
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.transitions[idx])

# The abstraction is a simple feedforward neural network with K inputs and N outputs
class ConvAbstraction(nn.Module):
    def __init__(self, num_abstract_states):
        super(ConvAbstraction, self).__init__()
        self.num_abstract_states = num_abstract_states
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            # TODO: currently hardcoded...
            nn.Linear(32*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, num_abstract_states)
        )
    
    # Get the abstract state index
    def to_num(self, abstract_state):
        return torch.argmax(abstract_state, dim=1, keepdim=True)

    def phi(self, obs):
        return self.features(obs) 
        
    # tau is gumbel softmax temperature, smaller the value the closer the output is to one_hot
    def sample(self, obs, tau=1.0, hard=False):
        abstract_state = self.phi(obs)
        return torch.nn.functional.gumbel_softmax(abstract_state, tau=tau, hard=hard, dim=-1) + 0.1**16
    
    # Use forward (ie., softmax) when we want a deterministic encoding    
    def forward(self, obs):
        abstract_state = self.phi(obs)
        return torch.nn.functional.softmax(abstract_state, dim=-1) + 0.1**16

def transform_state(s):
    return (s / 2)*255

def fourrooms_img_dsaa():
    config = {
        "num_abstract_states": 8,
        "num_abstraction_updates": 15000,
        "abstraction_batch_size": 256,
        "use_gumbel": True,
        "gumbel_tau": 0.75,
        "sr_gamma": 0.95,
        "abstraction_entropy_coef": 10.0,
        "option_replay_buffer_size": 1000000,
        "hard": False,
        "do_loc": False,
    }

    env = FourRoomsImg({"max_steps": 500})

    # initialize replay buffer
    replay_buffer_size = config["option_replay_buffer_size"]
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # explore randomly to gather data
    print("**Exploring**")
    state = env.reset()
    
    if config["do_loc"]: # do_loc means use (x,y) instead for debug purposes
        state = obs_to_loc(state)
    else:
        state = transform_state(state)
    
    for _ in range(100000):
        action = env.action_space.sample()
        next_state , _, done, _ = env.step(action)
        if config["do_loc"]:
            next_state = obs_to_loc(next_state)
        else:
            next_state = transform_state(next_state)
        
        replay_buffer.add((state, next_state, action, 0, 0))
        
        # Don't reset environment
        if False:#done:
            state = env.reset()
        else:
            state = next_state
            
    learning_rate = 0.001
    # initialize abstraction model
    if config["do_loc"]:
        phi = Abstraction(obs_size=2, num_abstract_states=config["num_abstract_states"])
    else:
        phi = ConvAbstraction(config["num_abstract_states"])

    phi_optimizer = torch.optim.Adam(phi.parameters(), lr=learning_rate)
    # initialize successor representation
    psi = SuccessorRepresentation(config["num_abstract_states"])
    psi_optimizer = torch.optim.Adam(psi.parameters(), lr=learning_rate)

    update_abstraction(phi, phi_optimizer, psi, psi_optimizer, replay_buffer, config)

    env_grid = (env.example_obs == 1)*1.0
    with torch.no_grad():
        all_phis = torch.zeros((19, 19))
        for i in range(env_grid.shape[1]):
            for j in range(env_grid.shape[2]):
                if env_grid[0,i,j] > 0:
                    all_phis[i,j] = -1
                    continue
                
                if config["do_loc"]:
                    tmp_grid = torch.FloatTensor([i,j])
                else:
                    tmp_grid = torch.FloatTensor(np.copy(env_grid))
                    tmp_grid[0,i,j] = 2
                    
                    tmp_grid = transform_state(tmp_grid)
                
                tmp_enc = phi(tmp_grid.unsqueeze(0))
                all_phis[i,j] = torch.argmax(tmp_enc[0])
    print(all_phis)
    plt.imshow(all_phis)
    plt.savefig(f"tmp_data/fourrooms_img_dsaa_{config['num_abstract_states']}.png")
                
if __name__=="__main__":
    fourrooms_img_dsaa()