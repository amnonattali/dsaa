'''
**Figure 11 from paper**

Here we visualize the abstraction when the input is an image (Arm2D)

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

from skimage.draw import line, disk

def cspace_video(save_path, phi, num_abstract_states, env):
    import matplotlib.animation as animation

    frame_skip = 3
    def plot_stuff(frame):
        print(f"\r{frame}", end="")
        plt.clf()
        xvalues = np.arange(180) - 90
        yvalues = np.arange(180) - 90
        dim2, dim3 = np.meshgrid(xvalues, yvalues)
        dim1 = np.ones(180*180)*(frame*frame_skip-90)
        pts = np.concatenate((dim1.reshape(-1,1), dim2.reshape(-1,1), dim3.reshape(-1,1)), axis=1)
        grid = np.zeros(180*180)
        all_imgs = np.zeros((180*180,1,env.canvas_size,env.canvas_size))

        for i,pt in enumerate(pts):
            env.env.config = pt
            env.env.update_workspace_config()
            img = env.make_obs(None)
            all_imgs[i] = img

            # with torch.no_grad():
            #     abstract_state = phi(torch.FloatTensor(img).unsqueeze(0)).argmax(dim=-1)
            # grid[i] = abstract_state
        with torch.no_grad():
            abstract_state = phi(torch.FloatTensor(all_imgs)).argmax(dim=-1)
        grid = abstract_state.numpy()
        

        plt.imshow(grid.reshape(180,180), vmin=0, vmax=num_abstract_states-1.0)
        plt.colorbar()
        return []

    ani = animation.FuncAnimation(plt.figure(), plot_stuff, frames=180//frame_skip, interval=100, blit=True)
    ani.save(save_path)

# Here we do not append the object to the state
class Arm2DImage(Wrapper):
    def __init__(self, config):
        self.num_joints = config["num_arm_joints"]
        self.arm_lengths = np.zeros(self.num_joints) + config["arm_joint_lengths"]
        self.task_height = config["ball_goal_height"]
        self.canvas_size = config["canvas_size"]

        env = gym.make('dsaa_envs:manipulator2d-v0', num_joints = self.num_joints, 
            arm_lengths = self.arm_lengths, max_steps=config["max_steps"])

        super(Arm2DImage, self).__init__(env)
        
        self.example_obs = self.make_obs(env.config)

        self.observation_size = None
        self.action_size = self.num_joints * 2
        self.name = "arm2d_img"
        

    def make_obs(self, obs):
        # img = np.zeros((canvas_size,canvas_size,3), dtype=np.float)
        img = np.zeros((1, self.canvas_size,self.canvas_size), dtype=np.float)
        # img = np.random.rand(1,self.canvas_size, self.canvas_size)
        for i in range(len(self.env.workspace_config)-1):
            p1 = np.array(self.env.workspace_config[i], dtype=int) + (self.canvas_size // 2)
            p2 = np.array(self.env.workspace_config[i+1], dtype=int) + (self.canvas_size // 2)
            rr, cc = line(-p1[1], p1[0], -p2[1], p2[0])
            # img[rr, cc, :] = (255, 0, 0)
            img[0, rr, cc] = 255
        

        # for obs in self.env.movable_objects:
        #     new_center = obs[:2] + (canvas_size // 2)
        #     rr, cc = disk((-new_center[1], new_center[0]), obs[2])
        #     img[rr, cc, :] = (0, 255, 0)
        #     # print(rr, cc)

        # print(img)
        # return np.swapaxes(np.swapaxes(img, 0, 1), 0, 2)
        return img #/ 255.0

    def reset(self):
        return self.make_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # env_reward = 1 if self.env.movable_objects[0][1] < self.task_height else 0
        return self.make_obs(obs), reward, done, info

    def close(self):
        return self.env.close()

# The abstraction is a simple feedforward neural network with K inputs and N outputs
class ConvAbstraction(nn.Module):
    def __init__(self, num_abstract_states):
        super(ConvAbstraction, self).__init__()
        self.num_abstract_states = num_abstract_states
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, 128),
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

def arm2d_img():
    config = {
        "num_abstract_states": 16,
        "num_abstraction_updates": 20000,
        "abstraction_batch_size": 256,
        "use_gumbel": True,
        "gumbel_tau": 0.5,
        "sr_gamma": 0.95,
        "abstraction_entropy_coef": 10.0,
        "option_replay_buffer_size": 1000000,
        "hard": False,
        "do_loc": False,
        "load_save": True
    }

    env = Arm2DImage(
        {
            "num_arm_joints": 3,
            "arm_joint_lengths": 10,
            "ball_goal_height": 10,
            "max_steps": 2000,
            "canvas_size": 3 * 10 * 2 + 2
        }
    )

    # initialize replay buffer
    replay_buffer_size = config["option_replay_buffer_size"]
    replay_buffer = ReplayBuffer(replay_buffer_size)

    learning_rate = 0.001
    # initialize abstraction model
    phi = ConvAbstraction(config["num_abstract_states"])
    phi_optimizer = torch.optim.Adam(phi.parameters(), lr=learning_rate)
    # initialize successor representation
    psi = SuccessorRepresentation(config["num_abstract_states"])
    psi_optimizer = torch.optim.Adam(psi.parameters(), lr=learning_rate)

    # explore randomly to gather data
    print("**Exploring**")
    state = env.reset()
    for i in range(100000):
        if i%10000 == 0:
            print("explored", i)
        action = env.action_space.sample()
        next_state , _, done, _ = env.step(action)
        replay_buffer.add((state, next_state, action, 0, 0))
        
        if done:
            state = env.reset()
        else:
            state = next_state

    if config["load_save"]:
        phi.load_state_dict(torch.load(f"tmp_data/arm2d_phi_{env.env.num_joints}.torch"))
        psi.load_state_dict(torch.load(f"tmp_data/arm2d_psi_{env.env.num_joints}.torch"))

        cspace_video(f"tmp_data/cspace_{env.env.num_joints}_joints.mp4", phi, config["num_abstract_states"], env)

        plt.clf()
        fig, axes = plt.subplots(4, 4)

        with torch.no_grad():
            batch = replay_buffer.sample(50000)
            batch_state, batch_next_state, _, _, _ = zip(*batch)

            batch_state = torch.FloatTensor(batch_state)
            abstract_state = phi.sample(batch_state, hard=True)
            rate = abstract_state.sum(dim=0) / len(batch_state)
            print(rate)
            
            for a_num in range(16):
                img_to_show = torch.zeros((env.canvas_size, env.canvas_size))# - 1
                for i in range(len(abstract_state)):
                    cur_a_num = torch.argmax(abstract_state[i])
                    if cur_a_num == a_num:
                        img_to_show[batch_state[i][0].numpy().nonzero()] = 1.0#1.0*a_num
                axes[a_num%4,a_num//4].imshow(img_to_show)#, cmap=cmap, norm=norm)
        plt.savefig("tmp_data/many_arm_img_2.png")
    else:
        print("**Training Abstraction**")
        update_abstraction(phi, phi_optimizer, psi, psi_optimizer, replay_buffer, config)

        torch.save(phi.state_dict(), f"tmp_data/arm2d_phi_{env.env.num_joints}.torch")
        torch.save(psi.state_dict(), f"tmp_data/arm2d_psi_{env.env.num_joints}.torch")
                
if __name__=="__main__":
    arm2d_img()