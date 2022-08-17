import random, time
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

from update_models import update_abstraction, train_option_policies
from torch_models import Abstraction, SuccessorRepresentation, MonolithicPolicy, SoftQNetwork
from utils import ReplayBuffer, plan_abstract_path, get_nbrs

import gym
from gym import Wrapper
from environments.env_wrappers import obs_to_loc

import matplotlib.pyplot as plt

'''
Here we compile utilities for the following works

Contrastive:    Erraqabi, A., Zhao, M., Machado, M. C., Bengio, Y., Sukhbaatar, S., Denoyer, L., & Lazaric, A. (2021, June). 
                Exploration-Driven Representation Learning in Reinforcement Learning. 
                In ICML 2021 Workshop on Unsupervised Reinforcement Learning.

Eigenoption:    Machado, M. C., Rosenbaum, C., Guo, X., Liu, M., Tesauro, G., & Campbell, M. (2017). 
                Eigenoption discovery through the deep successor representation. 
                arXiv preprint arXiv:1710.11089. (In ICLR 2018)

'''

# Basic FourRooms environment
#   state is the [x,y] coordinate of the agent
#   actions are in [0-3], to move the agent in each of 4 directions
#   reward is always zero
class FourRoomsNoReward(Wrapper):
    def __init__(self, config):
        max_steps = config["max_steps"]
        env = gym.make('dsaa_envs:fourrooms-v0', max_steps=max_steps, no_env_reward=True)
        super(FourRoomsNoReward, self).__init__(env)
        self.example_obs = env._make_obs()

        self.observation_size = 2
        self.action_size = 4
        self.preprocessors = [obs_to_loc]
        self.name = "four_rooms"

    def reset(self):
        obs = self.env.reset()
        return obs_to_loc(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs_to_loc(obs), 0, done, info

    def close(self):
        return self.env.close()

# Torch dataset... not actually used
class NormalizedTransitionsDataset(Dataset):
    # A transition is a pair of consecutive states (x,x')
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        # Normalize
        return torch.FloatTensor(self.transitions[idx]) / 18.0

# Encoder for Contrastive
#   NOTE: the output is currently passed through sigmoid activation as it seems to yield more stable results
class ContrastiveEncoder(nn.Module):
    def __init__(self, num_input_features, z_dim):
        super(ContrastiveEncoder, self).__init__()
        
        self.num_input_features = num_input_features
        self.z_dim = z_dim

        self.phi = nn.Sequential(
            nn.Linear(self.num_input_features, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.z_dim),
            nn.Sigmoid()
        )
    
    def forward(self, obs):
        enc = self.phi(obs)
        return enc

# Linear Variational Autoencoder
#   NOTE: the output is passed through sigmoid activation for binary cross entropy loss
class LinearVAE(nn.Module):
    def __init__(self, input_features=2, h_dim=1024, z_dim=32):
        super(LinearVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, h_dim),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(h_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_features),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

# vae loss: reconstruction (binary cross entropy) + prior (KL)
def vae_loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

# TODO: rewrite this to simply use the environment reward
# TODO: extend this to continuous environments (for example option termination must be a classifier...?)
def solve_task_with_options(env, task, option_policies, option_termination):
    """Trains a soft Q learning agent using options to reach a specific state in the environment

    Parameters
    ----------
    env : discrete OpenAI Gym environment
        a gym environment without any reward
        done=True iff max_steps is reached
    task : numpy array (or list??)
        the goal state from env
    option_policies : (k,n,n) numpy array
        first dimension corresponds to k option policies
        n = number of states in the environment
        each entry for each option is the action to take at the corresponding environment state
    option_termination : (k,n,n) boolean array
        each entry corresponds to whether that state is in the termination set of the option

    Returns
    -------
    torch.nn.Module 
        online_policy : the trained soft q learning policy
    float
        running avg_success over episodes
    list
        successes for each episode 
    
    """

    # basically we need to explore the environment like any baseline but reward with intrinsic
    input_size = env.observation_size
    num_actions = env.action_size + len(option_policies)
    
    # Params
    # TODO: move these into config parameter
    batch_size = 256
    learn_steps = 0
    target_update_steps = 20
    gamma = 0.95
    num_epochs = 100
    learning_rate = 0.001
    softQ_entropy_coeff = 0.01
    # NOTE: technically soft q learning does not need to be forced to take random actions but this seems to help
    prob_force_random = 0.1
    
    online_policy = SoftQNetwork(inputs=input_size, 
                                            outputs=num_actions, 
                                            entropy_coef = softQ_entropy_coeff)

    target_policy = SoftQNetwork(inputs=input_size, 
                                            outputs=num_actions, 
                                            entropy_coef = softQ_entropy_coeff)
    target_policy.load_state_dict(online_policy.state_dict())
    online_optimizer = torch.optim.Adam(online_policy.parameters(), lr=learning_rate)

    replay_buffer = ReplayBuffer(1000000)

    # the first epoch is skipped...
    env_done = True
    avg_success = 0
    all_successes = []
    for epoch in range(num_epochs):
        current_option = -1
        steps_in_option = 0
        while not env_done:
            if current_option > -1 and (option_termination[current_option, int(state[0]), int(state[1])] > 0):
                # or steps_in_option > 30):
                current_option = -1
                steps_in_option = 0
    
            if current_option > -1:
                action = option_policies[current_option, int(state[0]), int(state[1])]
                steps_in_option += 1
            else:
                if random.random() < prob_force_random:
                    action = random.randrange(num_actions)
                else:
                    with torch.no_grad():
                        action = online_policy.choose_action(state)

                if action >= env.action_size:
                    action = option_policies[int(action - env.action_size), int(state[0]), int(state[1])]
                    current_option = action - env.action_size

            # Step in the environment
            next_state, env_reward, env_done, info = env.step(action)
            env_reward = (next_state == task)*10
            if env_reward > 0:
                env_done = True
            if env_done:
                avg_success = 0.95 * avg_success + 0.05 * (env_reward>0)
                all_successes.append(env_reward > 0)

            replay_buffer.add((state, next_state, action, env_reward, env_reward > 0))

            state = next_state

            # Finally we update the policy
            if len(replay_buffer) > batch_size:
                learn_steps += 1
                if learn_steps % target_update_steps == 0:
                    target_policy.load_state_dict(online_policy.state_dict())

                Q_learning_update(online_policy, target_policy, online_optimizer, gamma, replay_buffer, batch_size)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg Success: {avg_success:1.3f}", end="\r")

        # Here env_done = True, reset everything
        env_done = False
        state = env.reset()
        
    return online_policy, avg_success, all_successes

# TODO: extend this to work for arbitrary discrete environments
# TODO: add an implementation which uses the deep successor representation
#       and add a test to make sure these two are compatible (it will be the largest e_vec of SR)
def get_eigen_reward(env_grid):
    '''Compute the reward function for Eigenoptions for the FourRooms environment
    
    Parameters
    ----------
    env_grid : 2d numpy array
        1 corresponds to obstacles
        otherwise free
    
    Returns
    -------
        reward_func(s1, s2, e_vec, dir) gives the reward for transitioning from s1 to s2 
            under eigenvector index e_vec, with dir = +-1 determining the direction of the vector
            e_vec should be an integer (0 being the first eigenvector)
            s1, s2 are [x,y] coordinates in the grid
    '''

    # The first step is to compute the adjacency matrix for the MDP
    num_states = (8*8)* 4 + 4 # TODO: should not be hardcoded
    adj = np.zeros((num_states, num_states))
    # each state in gridworld has 4 possible actions, note that hitting wall is still an action
    degrees = np.zeros(num_states) + 4
    loc_to_node = {}
    cur_state = 0
    for i in range(len(env_grid)):
        for j in range(len(env_grid[0])):
            if env_grid[i,j] == 1:
                continue
            # mark this node
            loc_to_node[(i,j)] = cur_state
            # now mark it's neighbors
            # only look backwards... later double up
            for t_i, t_j in [[i-1, j], [i, j-1]]:
                if t_i >= 0 and t_j >= 0 and t_i < 19 and t_j < 19:
                    if env_grid[t_i,t_j] == 1:
                        adj[cur_state][cur_state] += 1
                    else:
                        nbr_node = loc_to_node[(t_i, t_j)]
                        adj[cur_state][nbr_node] += 1
            for t_i, t_j in [[i+1, j], [i, j+1]]:
                if env_grid[t_i,t_j] == 1:
                    adj[cur_state][cur_state] += 1
            cur_state += 1
    
    # Now that we have the adjacency matrix we can compute the graph laplacian
    adj = adj + adj.T - np.diag(np.diag(adj))
    D_neg_inv = np.diag(-(degrees ** (0.5)))
    D = np.diag(degrees)
    L = np.matmul(D_neg_inv, np.matmul((D - adj), D_neg_inv)) # normalized laplacian
    # Our intrinsic reward function comes from the (*smallest) eigenvectors of this laplacian
    _,_, Vh = np.linalg.svd(L, full_matrices=True)
    reward_func = lambda s1, s2, e_vec, dir: dir*(Vh[-(e_vec+1)][loc_to_node[tuple(s2)]] - Vh[-(e_vec+1)][loc_to_node[tuple(s1)]])
    return reward_func

def get_eigen_options(env_grid, reward_func, num_options, num_epochs=5000, gamma = 0.99, display=False):
    '''Given an eigenoption reward function return the corresponding options in FourRooms
     
    Parameters
    ----------
    env_grid : 2d numpy array
        1 corresponds to obstacles
        otherwise free
    reward_func : the reward function returned by "get_eigen_reward(...)"
    num_options : the number of eigenoptions to return
        NOTE: this will return the first (num_options // 2) eigenoptions and their negatives 
    display : boolean, default=False
        if True will save visualization for each eigenoption

    Returns
    -------
        option_policies
            for each option for each grid cell what action to take
        option_termination
            for each option for each grid cell boolean whether to terminate 

    '''
    plt.clf()
    # Now that we have our eigenvectors (i.e., the reward function), we can compute our options
    xx,yy = np.meshgrid(np.linspace(0,18,19), np.linspace(0,18,19))
    all_states = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1).astype(int)
    # for each option for each state we have an action
    option_policies = np.zeros((num_options, 19, 19))
    # (up, right, down, left): dir_to_vec = {0: [-1, 0], 1: [0,1], 2: [1,0], 3: [0,-1]}
    actions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    # for each option for each state we have a 1 or 0
    option_termination = np.zeros((num_options, 19, 19))
    for e_vec in range(num_options // 2):
        for dir in [-1.0,1.0]: # we take each option and its negative (as per the paper)
            # First step is to compute the state value function
            cur_option_num = int(e_vec*2 + (dir+1)//2)
            print("e_vec", cur_option_num, e_vec, dir)
            state_value = np.zeros((19,19))
            avg_error = 0
            for epoch in range(num_epochs): # arbitrary large enough constant number of TD update steps
                for i,j in np.random.permutation(all_states):
                    if env_grid[i,j] == 1:
                        state_value[i, j] = -1
                        continue
                    max_dir = None
                    max_val = -100000
                    actions = [[i-1, j], [i, j+1], [i+1, j], [i, j-1]]
                    for t_i, t_j in actions:
                        if t_i >= 0 and t_j >= 0 and t_i < 19 and t_j < 19:
                            if env_grid[t_i,t_j] == 1:
                                reward = reward_func([i,j], [i, j], e_vec, dir)
                                tmp_v = reward + gamma * state_value[i, j]
                            else:
                                reward = reward_func([i,j], [t_i, t_j], e_vec, dir)
                                tmp_v = reward + gamma*state_value[t_i, t_j]
                            if tmp_v > max_val:
                                max_val = tmp_v
                                max_dir = [t_i, t_j]

                    target_v = max_val
                    td_error = (target_v - state_value[i, j])
                    state_value[i, j] += 0.01 * td_error
                    avg_error = avg_error*0.99 + 0.01*td_error
                if epoch % 100 == 0:
                    print(f"{avg_error.item():2.4f}", end="\r")
            # Now that we have the state value function we can compute the actual option, 
            #   which is the argmax of state_value of neighbor over each actions at each state
            XY = np.zeros((19*19, 2))
            UV = np.zeros((19*19, 2)) # UV is the direction of the action, necessary for plotting results
            for i in range(state_value.shape[0]):
                for j in range(state_value.shape[1]):
                    if env_grid[i,j] == 1:
                        continue
                    max_dir = None
                    max_val = -100000
                    max_action = None
                    actions = [[i-1, j], [i, j+1], [i+1, j], [i, j-1]]
                    for action_index, (t_i, t_j) in enumerate(actions):
                        if t_i >= 0 and t_j >= 0 and t_i < 19 and t_j < 19:
                            if env_grid[t_i,t_j] == 1:
                                if state_value[i, j] > max_val:
                                    max_val = state_value[i, j]
                                    max_dir = [t_i, t_j]
                                    max_action = action_index
                            else:
                                if state_value[t_i, t_j] > max_val:
                                    max_val = state_value[t_i, t_j]
                                    max_dir = [t_i, t_j]
                                    max_action = action_index
                                
                    XY[i*19 + j] = np.array([j,i])
                    UV[i*19 + j] = np.array([max_dir[1]-j, i - max_dir[0]])
                    # XY[i*19 + j] = np.array([i,j])
                    # UV[i*19 + j] = np.array([max_dir[0]-i, j - max_dir[1]])
                    option_policies[cur_option_num, i, j] = max_action
            # Now that we have the options computed, we can compute their termination set T
            #   and the intitiation set is S \ T
            #   We mark each state as terminal if we don't make progress from it
            #       NOTE: this is not as in the original paper. This is a more generous definition that avoids dead ends.
            
            for i in range(state_value.shape[0]):
                for j in range(state_value.shape[1]):
                    if env_grid[i,j] == 1:
                        continue
                    cur_dir = UV[i*19 + j]
                    nbr = [i-cur_dir[1], j + cur_dir[0]] #t_i, t_j
                    nbr_dir = UV[int(nbr[0]*19 + nbr[1])]
                    nbr_nbr = [nbr[0]-nbr_dir[1], nbr[1] + nbr_dir[0]]
                    if env_grid[int(nbr[0]), int(nbr[1])] == 1 or (int(nbr_nbr[0]) == i and int(nbr_nbr[1]) == j):
                        #np.abs(cur_dir + nbr_dir).sum() < 0.01:
                        option_termination[cur_option_num, i,j] = 1.0
            # We can also visualize the option along with its termination set
            if display:
                plt.quiver(XY[:,0], XY[:,1], UV[:,0], UV[:,1])
                state_value = state_value - option_termination[cur_option_num]
                plt.imshow(state_value)
                plt.colorbar()
                dir_ = "neg" if dir < 0 else "pos"
                plt.savefig(f"tmp_data/arrows_{e_vec}_{dir_}.png")
                plt.clf()
    
    return option_policies, option_termination

# TODO: UNTESTED
# implementation of eigenoptions using the deep successor representation 
#   train VAE and use encoding to compute the empirical successor matrix
#   then find it's right eigenvectors...
def deep_successor_eigenoptions():
    # ------------- INIT -------------
    env = FourRoomsNoReward({"max_steps": 500})

    # explore randomly to gather data
    print("**Exploring**")
    data = []
    state = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        next_state , _, done, _ = env.step(action)
        data.append([state, next_state])
        if False:#done:
            state = env.reset()
        else:
            state = next_state
    
    # prepare data for training VAE
    # TODO: MAKE SURE TO NORMALIZE FEATURES TO 0-1 SINCE WE HAVE A SIGMOID AT THE END OF RECONSTRUCTION
    print("**Preparing Data**")
    dataset = NormalizedTransitionsDataset(transitions=data)
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # train VAE with data
    print("**Training VAE**")
    start_time = time.time()
    z_dim = 8
    # vae = ConvVAE(image_channels=1, z_dim=z_dim)
    vae = LinearVAE(input_features=2, z_dim=z_dim)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, image_pairs in enumerate(dataloader):
            # print(image_pairs.shape)
            # print(image_pairs)
            # return
            images = image_pairs[:,0] # we don't need the transitions for training the VAE, just single samples
            recon_images, mu, logvar = vae(images)
            loss, bce, kld = vae_loss_fn(recon_images, images, mu, logvar) #TODO: should be next_images not images
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()
            running_loss = running_loss*0.99 + 0.01*loss.item()
        to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, running_loss)
        print(to_print, end="\r")
    print(f"**Finished VAE Training: total time: {time.time()-start_time:3.2f}**")

    # print(images[:10]*18, recon_images[:10]*18)

    # now using the trained VAE we can train the SR
    print("**Training Successor Representation**")
    psi = SuccessorRepresentation(num_input_features = z_dim)
    psi_optimizer = torch.optim.Adam(psi.parameters(), lr=1e-3)
    gamma = 0.99
    for epoch in range(epochs*3):
        running_loss = 0.0
        for idx, image_pairs in enumerate(dataloader):
            img1 = image_pairs[:,0]
            img2 = image_pairs[:,1]
            # first get feature encoding
            _, phi1, _ = vae.encode(img1) # TODO: should we use z or mu? 
            _, phi2, _ = vae.encode(img2)
            # now compute dsr for each
            psi1 = psi(phi1)
            psi2 = psi(phi2)
            # loss is TD
            target = (phi1 + gamma*psi2).detach()
            loss = ((psi1 - target)**2).sum(dim=1).mean()

            psi_optimizer.zero_grad()
            loss.backward()
            psi_optimizer.step()
            running_loss = running_loss*0.99 + 0.01*loss.item()

        to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, running_loss)
        print(to_print, end="\r")
    
    # how do we debug the SR? we can draw the fourrooms and compute distances
    env_grid = env.example_obs[0]
    # print(env_grid)
    with torch.no_grad():
        grid = np.zeros((19,19))
        all_phis = torch.zeros((19, 19, z_dim))
        ref_loc = torch.FloatTensor([[4,4]]) / 18.0
        ref_psi = psi(vae.encode(ref_loc)[1])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                tmp_tensor = torch.FloatTensor([[i,j]]) / 18.0 # TODO: magic normalization
                _, tmp_enc, _ = vae.encode(tmp_tensor)
                tmp_psi = psi(tmp_enc)
                grid[i,j] = ((tmp_psi - ref_psi)**2).sum().item()
                all_phis[i,j] = tmp_enc[0]
                if env_grid[i,j] == 1:
                    grid[i,j] = -1
        
        plt.imshow(grid)
        plt.colorbar()
        plt.savefig("tmp_data/grid.png")
        plt.clf()

        # now compute an SR matrix from some number of samples and find right eigenvectors
        num_samples = batch_size * 20
        psi_matrix = torch.zeros((num_samples, z_dim))
        filled_entries = 0
        for image_pairs in dataloader:
            img1 = image_pairs[:,0]
            _, phi1, _ = vae.encode(img1) 
            psi1 = psi(phi1)
            psi_matrix[filled_entries:filled_entries + batch_size] = psi1
            filled_entries += batch_size
            if filled_entries >= num_samples:
                break
        
        # now compute right eigenvectors
        U, S, Vh = torch.linalg.svd(psi_matrix, full_matrices=False)
        reward_func = lambda s1, s2, e_vec: torch.dot(Vh[e_vec], all_phis[s2[0], s2[1]] - all_phis[s1[0],s1[1]])

        # Vh, loc_to_node = discrete_eigen()
        # reward_func = lambda s1, s2, e_vec: Vh[-e_vec][loc_to_node[tuple(s2)]] - Vh[-e_vec][loc_to_node[tuple(s1)]]
        
        # now check for each cell in what direction would each e_vector take you
        xx,yy = np.meshgrid(np.linspace(0,18,19), np.linspace(0,18,19))
        all_states = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1).astype(int)
        
        for e_vec in range(4):
            # max_loc = None
            # max_loc_reward = -100000
            # for i in range(state_value.shape[0]):
            #     for j in range(state_value.shape[1]):
            #         if env_grid[i,j] == 1:
            #             reward_grid[i, j] = -1
            #         else:
            #             reward_grid[i, j] = torch.dot(Vh[e_vec], all_phis[i, j])
            #             if reward_grid[i, j] > max_loc_reward:
            #                 max_loc_reward = reward_grid[i, j]
            #                 max_loc = [i, j]

            # TODO: THIS NEEDS TO BE DONE IN BATCHES?
            state_value = np.zeros((19,19))
            avg_error = 0
            for epoch in range(500):
                # update_lag = 128
                for i,j in np.random.permutation(all_states):
                    if env_grid[i,j] == 1:
                        state_value[i, j] = -1
                        continue
                    # for i in range(state_value.shape[0]):
                    #     for j in range(state_value.shape[1]):
                    max_dir = None
                    max_val = -10000
                    for t_i, t_j in [[i+1, j], [i-1, j], [i, j+1], [i, j-1]]:
                        if t_i >= 0 and t_j >= 0 and t_i < 19 and t_j < 19 and env_grid[t_i,t_j] != 1:
                            # reward = torch.dot(Vh[e_vec], all_phis[t_i, t_j] - all_phis[i,j])
                            reward = reward_func([i,j], [t_i, t_j], e_vec)
                            tmp_v = reward + 0.9*state_value[t_i, t_j]
                            if tmp_v > max_val:
                                max_val = tmp_v
                                max_dir = [t_i, t_j]

                    # Q function eval: Q(s,a) = r + gamma * max_{a'} [Q(s'~p(s,a), a')]
                    # V function eval: V(s) = max_a (r(s,a) + gamma * V(s'~p(s,a)))
                    # target_v = max_val + 0.99 * state_value[max_dir[0], max_dir[1]]
                    target_v = max_val
                    td_error = (target_v - state_value[i, j])
                    state_value[i, j] += 0.01 * td_error
                    avg_error = avg_error*0.99 + 0.01*td_error
                if epoch % 20 == 0:
                    print(avg_error.item())

            XY = np.zeros((19*19, 2))
            UV = np.zeros((19*19, 2))
            reward_grid = np.zeros((19,19)) - 5
            for i in range(state_value.shape[0]):
                for j in range(state_value.shape[1]):
                    if env_grid[i,j] == 1:
                        continue
                    max_dir = None
                    max_val = -10000
                    for t_i, t_j in [[i+1, j], [i-1, j], [i, j+1], [i, j-1]]:
                        if t_i >= 0 and t_j >= 0 and t_i < 19 and t_j < 19 and env_grid[t_i,t_j] != 1:
                            if state_value[t_i, t_j] > max_val:
                                max_val = state_value[t_i, t_j]
                                max_dir = [t_i, t_j]
                    XY[i*19 + j] = np.array([i,j])
                    UV[i*19 + j] = np.array([max_dir[1] - j, i - max_dir[0]])
                    # reward_grid[i, j] = torch.dot(Vh[e_vec], all_phis[i, j])

            plt.quiver(XY[:,0], XY[:,1], UV[:,0], UV[:,1])
            plt.imshow(state_value)
            # plt.imshow(reward_grid)
            
            plt.colorbar()
            plt.savefig(f"tmp_data/arrows_{e_vec}.png")
            plt.clf()

# Given two tensors of consecutive states compute the contrastive loss:
#   consecutive pairs should be similar (phi1-phi2)
#   random pairs should be disimilar (assuming random batch we can simply take )
def contrastive_loss(phi1, phi2):
    consec = ((phi1 - phi2)**2).sum(dim=1).mean()
    # TODO: magic constant 2... the range of the values in these vectors matters... 
    rand_pairs = 2*torch.exp(-(torch.abs(phi1[:len(phi1)//2] - phi1[len(phi1)//2:])).sum(dim=1)).mean()
    return rand_pairs + consec

def train_contrastive_encoder(dataloader, z_dim=3, epochs = 40):
    '''Given a dataloader of transitions in the environment train an encoder using a contrastive'''
    encoder = ContrastiveEncoder(num_input_features=2, z_dim=z_dim)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, pairs in enumerate(dataloader):
            state1 = pairs[:,0]
            state2 = pairs[:,1]
            phi1 = encoder(state1) 
            phi2 = encoder(state2)
            
            loss = contrastive_loss(phi1, phi2)
            encoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            running_loss = running_loss*0.99 + 0.01*loss.item()
        
        to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, running_loss)
        print(to_print, end="\r")
    print("**Finished Encoder Training**")
    return encoder

# Essentially normal Q learning where the reward is r = env_reward + intrinsic_reward
#   in this case intrinsic reward is simply the distance between the embedding (defined 
#   by contrastive_encoder) of the task (i.e., goal state) and the current state
def solve_contrastive_task(env, task, contrastive_encoder):
    # basically we need to explore the environment like any baseline but reward with intrinsic
    input_size = env.observation_size
    num_actions = env.action_size
    
    # TODO: remove hard coded normalization, which also occurs below
    abstract_goal = contrastive_encoder(torch.FloatTensor(task).unsqueeze(0)/18.0)[0]

    # Params
    batch_size = 256
    learn_steps = 0
    target_update_steps = 20
    gamma = 0.95
    num_epochs = 100
    learning_rate = 0.001
    softQ_entropy_coeff = 0.01
    prob_force_random = 0.1
    
    online_policy = SoftQNetwork(inputs=input_size, 
                                            outputs=num_actions, 
                                            entropy_coef = softQ_entropy_coeff)

    target_policy = SoftQNetwork(inputs=input_size, 
                                            outputs=num_actions, 
                                            entropy_coef = softQ_entropy_coeff)
    target_policy.load_state_dict(online_policy.state_dict())
    online_optimizer = torch.optim.Adam(online_policy.parameters(), lr=learning_rate)

    replay_buffer = ReplayBuffer(1000000)

    # the first epoch is skipped...
    env_done = True
    avg_success = 0
    all_successes = []
    for epoch in range(num_epochs):
        while not env_done:        
            with torch.no_grad():
                # Get the primitive action from the option_policy for the current abstract_state and skill
                if random.random() < prob_force_random:
                    action = random.randrange(num_actions)
                else:
                    action = online_policy.choose_action(state)

            # Step in the environment
            next_state, env_reward, env_done, info = env.step(action)
            env_reward = (next_state == task)*10
            if env_reward > 0:
                env_done = True
            if env_done:
                avg_success = 0.95 * avg_success + 0.05 * (env_reward>0)
                all_successes.append(env_reward > 0)

            with torch.no_grad():
                next_abstract_state = contrastive_encoder(torch.FloatTensor(next_state).unsqueeze(0)/18.0)[0]            

            intrinsic_reward = -((next_abstract_state - abstract_goal)**2).sum()
            
            replay_buffer.add((state, next_state, action, env_reward + intrinsic_reward, env_reward > 0))

            state = next_state

            # Finally we update the policy
            if len(replay_buffer) > batch_size:
                learn_steps += 1
                if learn_steps % target_update_steps == 0:
                    target_policy.load_state_dict(online_policy.state_dict())

                Q_learning_update(online_policy, target_policy, online_optimizer, gamma, replay_buffer, batch_size)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg Success: {avg_success:1.3f}", end="\r")

        # Here env_done = True, reset everything
        env_done = False
        state = env.reset()
        
    return online_policy, avg_success, all_successes

# Given an abstraction phi and an abstract graph in a discrete state space compute options
#   corresponding to the edges of the abstract graph
# TODO: extend this to work for arbitrary discrete spaces instead of just FourRooms
def get_dsaa_indiv_options(env_grid, phi, abstract_adjacency):
    with torch.no_grad():
        xx,yy = np.meshgrid(np.linspace(0,18,19), np.linspace(0,18,19))
        all_states = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1).astype(int)
        
        actions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        
        # fig, axes = plt.subplots(2,4)
        gamma = 0.9
        option_policies = {}
        for a_num in range(4):
            skill_idx = 0
            for skill in get_nbrs(abstract_adjacency, a_num):
                if skill == a_num:
                    continue
                
                option_policies[(a_num, skill)] = np.zeros((19, 19))
                # print(option_policies.keys())

                cur_option = a_num * 2 + skill_idx
                skill_idx += 1
                print("Training option", cur_option, a_num, skill)
                state_value = np.zeros((19,19))
                avg_error = 0
                for epoch in range(300):
                    for i,j in np.random.permutation(all_states):
                        if env_grid[i,j] == 1:
                            state_value[i, j] = -1
                            continue
                        tmp_tensor = torch.FloatTensor([[i,j]])
                        tmp_enc = phi(tmp_tensor)[0]
                        if torch.argmax(tmp_enc).item() != a_num:
                            state_value[i, j] = -1
                            continue
                        
                        max_dir = None
                        max_val = -100000
                        actions = [[i-1, j], [i, j+1], [i+1, j], [i, j-1]]
                        for t_i, t_j in actions:
                            if t_i >= 0 and t_j >= 0 and t_i < 19 and t_j < 19:
                                if env_grid[t_i,t_j] == 1:
                                    reward = -1
                                    tmp_v = reward + gamma * state_value[i, j]
                                else:
                                    nbr_tensor = torch.FloatTensor([[t_i,t_j]])
                                    nbr_state = torch.argmax(phi(nbr_tensor)[0]).item()
                                    reward = (nbr_state == skill)*10.0
                                    if reward > 0:
                                        tmp_v = reward
                                    else:
                                        tmp_v = gamma*state_value[t_i, t_j]
                                if tmp_v > max_val:
                                    max_val = tmp_v
                                    max_dir = [t_i, t_j]
                        target_v = max_val
                        td_error = (target_v - state_value[i, j])
                        state_value[i, j] += 0.05 * td_error
                        avg_error = avg_error*0.99 + 0.01*td_error
                    if epoch % 100 == 0:
                        print(f"{avg_error.item():2.4f}", end="\r")
                
                # Now that we have the state value function we can compute the actual option, 
                #   which is the argmax of state_value of neighbor over each actions at each state
                XY = []
                UV = []    
                # XY = np.zeros((19*19, 2))
                # UV = np.zeros((19*19, 2)) # UV is the direction of the action, necessary for plotting results
                for i in range(state_value.shape[0]):
                    for j in range(state_value.shape[1]):
                        if env_grid[i,j] == 1:
                            continue
                        tmp_tensor = torch.FloatTensor([[i,j]])
                        tmp_enc = phi(tmp_tensor)[0]
                        if torch.argmax(tmp_enc).item() != a_num:
                            continue

                        max_dir = None
                        max_val = -100000
                        max_action = None
                        actions = [[i-1, j], [i, j+1], [i+1, j], [i, j-1]]
                        for action_index, (t_i, t_j) in enumerate(actions):
                            if t_i >= 0 and t_j >= 0 and t_i < 19 and t_j < 19:
                                if env_grid[t_i,t_j] == 1:
                                    continue
                                nbr_tensor = torch.FloatTensor([[t_i,t_j]])
                                nbr_state = torch.argmax(phi(nbr_tensor)[0]).item()
                                reward = (nbr_state == skill)*10.0
                                if reward > 0:
                                    max_val = reward
                                    max_dir = [t_i, t_j]
                                    max_action = action_index
                                else:
                                    if state_value[t_i, t_j] > max_val:
                                        max_val = state_value[t_i, t_j]
                                        max_dir = [t_i, t_j]
                                        max_action = action_index
                                    
                        # XY[i*19 + j] = np.array([j,i])
                        # UV[i*19 + j] = np.array([max_dir[1]-j, i - max_dir[0]])
                        XY.append([j,i])
                        UV.append([max_dir[1]-j, i - max_dir[0]])
                        option_policies[(a_num, skill)][i, j] = max_action
                
                #     state = [i,j] + tmp_enc.tolist()
                #     action = options.choose_action(state, skill)
                #     XY.append([j,i])
                #     UV.append(actions[action])
                #     vis[i,j] = action
                XY = np.array(XY)
                UV = np.array(UV)
                # # axes[s_idx, a_num].quiver(XY[:,0], XY[:,1], UV[:,0], UV[:,1])
                # # axes[s_idx, a_num].imshow(env_grid + 2)
                plt.quiver(XY[:,0], XY[:,1], UV[:,0], UV[:,1])
                # plt.imshow(env_grid * -1)
                # plt.imshow(vis)
                plt.imshow(state_value)
                plt.colorbar()
                plt.savefig(f"tmp_data/dsaa_options_{a_num}_{skill}.png")
                plt.clf()
    return option_policies

# DSAA: 
#   Given a replay buffer of transitions in the environment train our encoder(phi) decoder(psi) model
#   return the abstraction phi
#   NOTE: config needs to contain a whole bunch of things...
def train_dsaa(replay_buffer, config):
    dsaa_config = {
        "num_abstract_states": 4,
        "num_abstraction_updates": 10000,
        "abstraction_batch_size": 512,
        "use_gumbel": True,
        "gumbel_tau": 1.0,
        "sr_gamma": 0.95,
        "abstraction_entropy_coef": 10.0,
        "hard": False,
        "learning_rate": 0.001
    }
    dsaa_config.update(config)

    # initialize abstraction model
    phi = Abstraction(obs_size=2, num_abstract_states=dsaa_config["num_abstract_states"])
    phi_optimizer = torch.optim.Adam(phi.parameters(), lr=dsaa_config["learning_rate"])
    # initialize successor representation
    psi = SuccessorRepresentation(dsaa_config["num_abstract_states"])
    psi_optimizer = torch.optim.Adam(psi.parameters(), lr=dsaa_config["learning_rate"])
    update_abstraction(phi, phi_optimizer, psi, psi_optimizer, replay_buffer, dsaa_config)
    
    return phi

# NOTE: this is old code for training a monolithic DSAA option policy...
# ...does not work... 
def train_dsaa_options(phi, replay_buffer, config):
    config = {
        "num_abstract_states": 4,
        "learning_rate": 0.001,
        "option_entropy_coef": 0.001,
        "num_option_updates": 20000,
        "option_batch_size": 512,
        "ddqn_target_update_steps": 4,
        "option_success_reward": 100,
        "reward_self": True,
        "soft_Q_update": True,
        "option_gamma": 0.9,

    }
    # dsaa_config.update(config)

    num_abstract_states = config["num_abstract_states"]
    num_skills = num_abstract_states
    obs_size = 2
    num_actions = 4

    online_Q = MonolithicPolicy(num_abstract_states, num_skills, obs_size, num_actions, config)
    target_Q = MonolithicPolicy(num_abstract_states, num_skills, obs_size, num_actions, config)
    option_optimizer = online_Q.option_optimizer    
    target_Q.network.load_state_dict(online_Q.network.state_dict())

    if False:
        train_option_policies(online_Q, target_Q, option_optimizer, 
                    phi, replay_buffer, config, num_updates=config["num_option_updates"])

    # compute adjacency graph
    batch = replay_buffer.sample(len(replay_buffer)) # NOTE: needs to fit in memory...
    batch_state, batch_next_state, _, _, _ = zip(*batch)
    batch_state = torch.FloatTensor(batch_state)
    batch_next_state = torch.FloatTensor(batch_next_state)
    with torch.no_grad():
        abstract_state = phi(batch_state)
        next_abstract_state = phi(batch_next_state)
        
        abstract_state_nums = phi.to_num(abstract_state).flatten()
        next_abstract_state_nums = phi.to_num(next_abstract_state).flatten()
    abstract_adjacency = torch.eye(num_abstract_states)
    abstract_adjacency[abstract_state_nums, next_abstract_state_nums] = 1

    return online_Q, abstract_adjacency
    
# Given a discrete abstraction, an abstract graph, and option policies for each edge in the graph
#   Train a soft q learning agent to navigate to solve the task (reach a certain state)
# NOTE: current implementation is to follow shortest abstract path to the encoding of the goal, then randomly explore
def solve_dsaa_task(env, task, phi, option_policies, abstract_adjacency):
    # model data
    batch_size = 256
    replay_buffer = ReplayBuffer(20000)
    # model definitisons
    learn_steps = 0
    gamma = 0.95
    target_update_steps = 20
    online_q = SoftQNetwork(inputs=2, outputs=4, entropy_coef=0.01)
    target_q = SoftQNetwork(inputs=2, outputs=4, entropy_coef=0.01)
    online_optimizer = torch.optim.Adam(online_q.parameters(), lr=0.001)
    target_q.load_state_dict(online_q.state_dict())
    # get abstract goal
    with torch.no_grad():
        goal_abstract_state = torch.argmax(phi(torch.FloatTensor(task).unsqueeze(0))[0]).item()
    # print(task, goal_abstract_state)
    # begin training loop
    num_epochs = 100
    env_done = True
    avg_success = 0
    all_successes = []
    for epoch in range(num_epochs):
        if env_done:
            env_done = False
            option_done = False
            
            state = env.reset()
            with torch.no_grad():
                abstract_state = phi(torch.FloatTensor(state).view(1,-1))
                a_num = phi.to_num(abstract_state)[0].item()
                abstract_state = abstract_state[0]

            # we follow the path to the goal (once at goal we follow self)
            max_reward_path = plan_abstract_path(a_num, goal_abstract_state, abstract_adjacency)
            skill = max_reward_path[0][1]

            # augment the primitive state with the abstract state
            state += abstract_state.tolist()
            # if epoch == 0:
            #     print(a_num, goal_abstract_state, max_reward_path)

        while not env_done:
            # if the option has terminated (but the episode has not) we need a new skill
            if option_done:
                option_done = False
                max_reward_path = plan_abstract_path(a_num, goal_abstract_state, abstract_adjacency)
                skill = max_reward_path[0][1]
                # print(a_num, skill, max_reward_path)

            if random.random() < 0.1:
                action = random.randrange(4)
            elif skill == a_num:
                if avg_success > 0.1:
                    action = online_q.choose_action(state[:2])
                else:
                    action = random.randrange(4)
                    # action = options.choose_action(state, skill)
                # action = random.randrange(4)
            else:
                # action = options.choose_action(state, skill)
                action = option_policies[(a_num, skill)][int(state[0]), int(state[1])]
                
            # step in environment... env takes care of structuring state correctly
            next_state, env_reward, env_done, _ = env.step(action)
            # print(state[:2], next_state)
            env_reward = (next_state == task)*10
            if env_reward > 0:
                env_done = True
            if env_done:
                avg_success = 0.95 * avg_success + 0.05 * (env_reward>0)
                all_successes.append(env_reward > 0)

            # get the next abstract state
            with torch.no_grad():
                next_abstract_state = phi(torch.FloatTensor(next_state).view(1,-1))
                next_a_num = phi.to_num(next_abstract_state)[0].item()
                next_abstract_state = next_abstract_state[0]
            # again, we augment the state with the abstract state
            next_state += next_abstract_state.tolist()     
            
            option_done = (a_num != next_a_num)

            # add experience to replay buffers
            replay_buffer.add((state[:2], next_state[:2], action, env_reward, env_reward > 0))
            
            # prepare next iteration
            state = next_state
            abstract_state = next_abstract_state
            a_num = next_a_num

            # update policy
            if len(replay_buffer) > batch_size:
                learn_steps += 1
                if learn_steps % target_update_steps == 0:
                    target_q.load_state_dict(online_q.state_dict())
                Q_learning_update(online_q, target_q, online_optimizer, gamma, replay_buffer, batch_size)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg success {avg_success:1.3f}", end="\r")

    return online_q, avg_success, all_successes

# Baseline soft q learning update
def Q_learning_update(online_policy, target_policy, online_optimizer, gamma, replay_buffer, batch_size):
    batch = replay_buffer.sample(batch_size)
    batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

    batch_state = torch.FloatTensor(batch_state)
    batch_next_state = torch.FloatTensor(batch_next_state)
    batch_action = torch.FloatTensor(batch_action).unsqueeze(1)
    batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
    batch_done = torch.FloatTensor(batch_done).unsqueeze(1)

    with torch.no_grad():
        next_q = target_policy(batch_next_state)
        next_v = target_policy.getV(next_q)
        y = batch_reward + (1 - batch_done) * gamma * next_v

    loss = F.mse_loss(online_policy(batch_state).gather(1, batch_action.long()), y)
    
    online_optimizer.zero_grad()
    loss.backward()
    online_optimizer.step()
    return online_policy

# make plots of episode successes for DSAA, Eigenoption, and Contrastive
def process_results():
    import pickle
    contrastive = pickle.load(open("tmp_data/episode_success_contrastive.pickle", "rb"))
    eigenoptions = pickle.load(open("tmp_data/episode_success_eigenoptions.pickle", "rb"))
    dsaa = pickle.load(open("tmp_data/episode_success_dsaa.pickle", "rb"))
    c = np.array([t[1] for t in contrastive])
    e = np.array([t[1] for t in eigenoptions])
    d = np.array([t[1] for t in dsaa])[:,:-1]

    print(c.shape, d.shape, e.shape)

    first_c = np.argmax(c, axis=1)
    first_d = np.argmax(d, axis=1)
    first_e = np.argmax(e, axis=1)
    # if it failed then the argmax will be 0, make it 100
    first_c += 50*(c.sum(axis=1) == 0) + 1
    first_d += 50*(d.sum(axis=1) == 0) + 1
    first_e += 50*(e.sum(axis=1) == 0) + 1
    
    failed_c = (c.sum(axis=1) == 0).sum()
    failed_d = (d.sum(axis=1) == 0).sum()
    failed_e = (e.sum(axis=1) == 0).sum()
    print(failed_c, failed_d, failed_e)

    # print(first_c, first_d, first_e)
    print("Average first occurence of sparse reward:")
    print(f"\tContrastive mean {np.mean(first_c):2.2f}, std {np.std(first_c):2.2f}")
    print(f"\tDSAA mean {np.mean(first_d):2.2f}, std {np.std(first_d):2.2f}")
    print(f"\tEigenoptions mean {np.mean(first_e):2.2f}, std {np.std(first_e):2.2f}")
    
    gamma = 0.9
    for col in range(1, c.shape[1]):
        c[:,col] = c[:,col-1]*gamma + c[:,col]*(1-gamma)
        d[:,col] = d[:,col-1]*gamma + d[:,col]*(1-gamma)
        e[:,col] = e[:,col-1]*gamma + e[:,col]*(1-gamma)

    max_len = 50
    c = c[:,:max_len]
    d = d[:,:max_len]
    e = e[:,:max_len]

    c[:,0] = 0
    d[:,0] = 0
    e[:,0] = 0

    mean_c = np.mean(c, axis=0)
    stds_c = np.std(c, axis=0)
    mean_d = np.mean(d, axis=0)
    stds_d = np.std(c, axis=0)
    mean_e = np.mean(e, axis=0)
    stds_e = np.std(c, axis=0)
    
    x = np.arange(len(mean_c))
    plt.plot(x, mean_c, label="contrastive", color="blue")
    plt.plot(x, mean_d, label="dsaa", color="red")
    plt.plot(x, mean_e, label="eigenoptions", color="green")
    plt.xlabel("Number of Episodes", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=14)
    plt.ylabel("Average Return", fontsize=13)
    plt.fill_between(x, (mean_c-stds_c).clip(0,1), (mean_c+stds_c).clip(0,1), color="blue", alpha=0.2)
    plt.fill_between(x, (mean_d-stds_d).clip(0,1), (mean_d+stds_d).clip(0,1), color="red", alpha=0.2)
    plt.fill_between(x, (mean_e-stds_e).clip(0,1), (mean_e+stds_e).clip(0,1), color="green", alpha=0.2)
    plt.savefig("tmp_data/returns.png")
    

if __name__=="__main__":
    # from environments.env_wrappers import BaseFourRooms 
    # env = BaseFourRooms({"max_steps": 500})
    # env_grid = env.example_obs[0]
    # option_policies, option_termination = get_eigen_options(env_grid, 4, True)
    # print(option_termination[3])
    # print(option_policies[3])

    process_results()