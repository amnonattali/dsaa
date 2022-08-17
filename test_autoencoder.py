'''
Straightforward implementation of training a variational autoencoder in FourRooms 
and then clustering states based on their encoding to form discrete abstract states
--> demonstrates that the resulting abstract states are not conducive to planning

TODO: extend this implementation for arbitrary environments?

'''

from environments.env_wrappers import BaseFourRooms

import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans

import gym
from gym import Wrapper

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

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class ConvVAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
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

class TransitionsDataset(Dataset):
    # A transition is a pair of consecutive states (x,x')
    # If we want images we can do the img transformation here
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        # normalize
        return torch.FloatTensor(self.transitions[idx]) / 18.0

def vae_loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

def test():
    do_conv = False

    if do_conv:
        env = FourRoomsImg()
    else:
        env = BaseFourRooms({"max_steps": 500})

    # explore randomly to gather data
    print("**Exploring**")
    data = []
    state = env.reset()    
    for _ in range(100000):
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
    batch_size = 256
    dataset = TransitionsDataset(transitions=data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # train VAE with data
    print("**Training VAE**")
    start_time = time.time()
    z_dim = 4

    if do_conv:
        vae = ConvVAE(image_channels=1, z_dim=z_dim)
    else:
        vae = LinearVAE(input_features=2, z_dim=z_dim)
    
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, image_pairs in enumerate(dataloader):
            images = image_pairs[:,0]/18.0 # we don't need the transitions for training the VAE, just single samples
            recon_images, mu, logvar = vae(images)
            loss, bce, kld = vae_loss_fn(recon_images, images, mu, logvar) #TODO: should be next_images not images
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()
            running_loss = running_loss*0.99 + 0.01*loss.item()
        to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, running_loss)
        print(to_print, end="\r")
    print(f"**Finished VAE Training: total time: {time.time()-start_time:3.2f}**")

    # Now let's cluster the states based on their encodings
    xx,yy = np.meshgrid(np.linspace(0,18,19), np.linspace(0,18,19))
    all_states = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1).astype(int)
    
    if do_conv:
        env_grid = (env.example_obs == 1)*1.0
        phi = []
        with torch.no_grad():
            for i,j in all_states:
                if env_grid[0,i,j] > 0:
                    continue
                
                tmp_grid = torch.FloatTensor(np.copy(env_grid))
                tmp_grid[0,i,j] = 2
                _, cur_phi, _ = vae.encode(tmp_grid.unsqueeze(0))
                phi.append(cur_phi)

    else:
        torch_states = torch.FloatTensor(all_states) / 18.0
        with torch.no_grad():
            _, phi, _ = vae.encode(torch_states)

    env_grid = env.example_obs[0]
    true_states = []
    loc_to_idx = {}
    for idx, (i,j) in enumerate(all_states):
        if env_grid[i,j] == 1:
            continue
        loc_to_idx[(i,j)] = len(true_states)
        true_states.append(phi[idx].tolist())
        
    # print(phi)
    print(phi.shape)
    print(len(true_states))

    X = np.array(true_states)
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    print(kmeans.labels_)
    grid = np.zeros((19,19)) - 1
    for idx, (i,j) in enumerate(all_states):
        if env_grid[i,j] == 1:
            continue
        grid[i,j] = kmeans.labels_[loc_to_idx[(i,j)]]
    plt.imshow(grid)
    plt.colorbar()
    plt.savefig(f"tmp_data/autoencoder_abstraction_{num_clusters}.png")

if __name__=="__main__":
    test()