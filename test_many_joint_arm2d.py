'''
Here we test the abstraction when the arm has more than 3 joints
We use the joint angles as the input
We visualize by making one image per abstract state and drawing samples of arm positions at those states
'''

import gym
from gym import Wrapper
import numpy as np
import torch
import matplotlib.pyplot as plt

from update_models import update_abstraction
from torch_models import SuccessorRepresentation, Abstraction
from utils import ReplayBuffer
from environments.env_wrappers import Manipulator2DNoOBJ

from skimage.draw import line, disk
    
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

# TODO: currently the canvas size is configured for 10 joints.... need to fix
def train():
    config = {
        "num_abstract_states": 16,
        "num_abstraction_updates": 20000,
        "abstraction_batch_size": 512,
        "use_gumbel": True,
        "gumbel_tau": 1.0,
        "sr_gamma": 0.99,
        "abstraction_entropy_coef": 100.0,
        "option_replay_buffer_size": 1000000,
        "hard": False,
        "do_loc": False,
        "load_save": True,
        "num_joints": 10
    }
    
    learning_rate = 0.001
    # initialize abstraction model
    phi = Abstraction(obs_size=config["num_joints"], num_abstract_states=config["num_abstract_states"])
    phi_optimizer = torch.optim.Adam(phi.parameters(), lr=learning_rate)
    # initialize successor representation
    psi = SuccessorRepresentation(config["num_abstract_states"])
    psi_optimizer = torch.optim.Adam(psi.parameters(), lr=learning_rate)

    if config["load_save"]:
        env = Arm2DImage(
            {
                "num_arm_joints": config['num_joints'],
                "arm_joint_lengths": 10,
                "ball_goal_height": -10,
                "max_steps": 5000,
                "canvas_size": 10 * 10 * 2 + 2
            }
        )
        phi.load_state_dict(torch.load(f"tmp_data/arm2d_phi_{config['num_joints']}_{config['sr_gamma']}.torch"))
        psi.load_state_dict(torch.load(f"tmp_data/arm2d_psi_{config['num_joints']}_{config['sr_gamma']}.torch"))
        img_to_show = torch.zeros((16, env.canvas_size, env.canvas_size))
        time_since_last = np.zeros(16)

        # explore randomly to gather data
        print("**Exploring**")
        state = env.reset()
        for i in range(100000):
            if i%10000 == 0:
                print("explored", i)
                cur_a_num = torch.argmax(phi(torch.FloatTensor(env.config).unsqueeze(0))[0]).item()
                if time_since_last[cur_a_num] < 20:
                    time_since_last[cur_a_num] += 1
                else:
                    img_to_show[cur_a_num][state[0].nonzero()] = 1.0#1.0*a_num
                    time_since_last[cur_a_num] = 0
            
            action = env.action_space.sample()
            next_state , _, done, _ = env.step(action)
                
            if done:
                state = env.reset()
            else:
                state = next_state
        
        plt.clf()
        fig, axes = plt.subplots(4, 4)

        for a_num in range(16):
            axes[a_num%4,a_num//4].imshow(img_to_show[a_num])
            axes[a_num%4,a_num//4].set_axis_off()
            axes[a_num%4,a_num//4].set_title(f"{a_num}", pad=0, fontsize=10.0)
        fig.suptitle("Samples of arm positions at each of 16 abstract states")
        plt.savefig(f"tmp_data/vis_arms_{env.env.num_joints}_99.png")
    else:
        env = Manipulator2DNoOBJ(
            {
                "num_arm_joints": config['num_joints'],
                "arm_joint_lengths": 10,
                "max_steps": 5000,
                "ball_goal_height": -10
            }
        )
        # initialize replay buffer
        replay_buffer_size = config["option_replay_buffer_size"]
        replay_buffer = ReplayBuffer(replay_buffer_size)
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

        print("**Training Abstraction**")
        update_abstraction(phi, phi_optimizer, psi, psi_optimizer, replay_buffer, config)

        torch.save(phi.state_dict(), f"tmp_data/arm2d_phi_{env.env.num_joints}_{config['sr_gamma']}.torch")
        torch.save(psi.state_dict(), f"tmp_data/arm2d_psi_{env.env.num_joints}_{config['sr_gamma']}.torch")
            
if __name__=="__main__":
    train()