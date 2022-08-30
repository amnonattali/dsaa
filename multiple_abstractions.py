'''
- we want to learn multiple abstractions which are different
- there are multiple ways two abstractions can be different
- a really nice way to ensure two abstractions are different is by having them use different input features
    - or alternatively, to make sure they are independent?
        - independence is not the same thing as simply using different features, for example using two different patches of an 
        input image does not mean those patches are independent

    - How do we ensure independence between abstractions?
        - independence means that knowing one does not make it easier to know the other
        - given phi_1 we would like to learn phi_2 such that:
            - given x --> s1 = phi_1(x)
            - s2 = phi_2(x)
            - D(s2, s1) = real
                - given x' --> s_fake = phi_2(x')
                - D(s_fake, s1) = fake
            - phi_2 is trained as a generator

            - Discriminator loss: *maximize* log(D(s_fake, s1)) + log(1-D(s2, s1))
                - first term large --> D(s_fake) = 1
                - second term large --> D(s2) = 0
            - phi_2 loss: *minimize* log(1-D(s2, s1))
                - meaning make s2 fool the discriminator

Implement:
- First explore the environment to get dataset
- Now train an abstraction phi, SR psi, discriminator D
    - phi   : X             --> \Delta^{K}^2
        - Loss1 = max entropy
        - Loss2 = "GAN generator" loss
    - psi   : \Delta^{K}    --> \R^k
        - Loss3 = SR TD loss
    - D     : \Delta^{K}^2  --> [0,1]
        - Loss4 = "GAN discriminator" loss

**
- How to deal with the symmetry of it? We want to apply the loss in both directions...
    - phi1(x) vs phi2(x) (real)
    - phi1(x) vs phi2(x') (fake)
    - phi1(x') vs phi2(x) (fake)
    
- Something is flipped here - normally the generator is creating the "fake" data, 
    here our fake data simply something unrelated to the real
**

'''

from transfer_utils import NormalizedTransitionsDataset, FourRoomsNoReward

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# The abstraction is a simple feedforward neural network with K inputs and N outputs
class MultiAbstraction(nn.Module):
    def __init__(self, obs_size, num_abstract_states, num_abstractions):
        super(MultiAbstraction, self).__init__()
        
        self.obs_size = obs_size
        self.num_abstract_states = num_abstract_states
        self.num_abstractions = num_abstractions

        self.features = nn.Sequential(
            nn.Linear(self.obs_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.num_abstractions*self.num_abstract_states),
        )
    
    # Get the abstract state index
    def to_num(self, abstract_state):
        return torch.argmax(abstract_state, dim=-1, keepdim=True)

    def phi(self, obs):
        return self.features(obs).view(obs.size(0), self.num_abstractions, self.num_abstract_states) 
        
    # tau is gumbel softmax temperature, smaller the value the closer the output is to one_hot
    def sample(self, obs, tau=1.0, hard=False):
        abstract_state = self.phi(obs)
        return torch.nn.functional.gumbel_softmax(abstract_state, tau=tau, hard=hard, dim=-1) + 0.1**16
    
    # Use forward (ie., softmax) when we want a deterministic encoding    
    def forward(self, obs):
        abstract_state = self.phi(obs)
        return torch.nn.functional.softmax(abstract_state, dim=-1) + 0.1**16

class Discriminator(nn.Module):
    def __init__(self, num_inputs):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(num_inputs*2, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        # inp = torch.cat([s1, s2], dim=1)
        return self.main(inp)

class MultiSuccessorRepresentation(nn.Module):
    def __init__(self, num_abstract_states, num_abstractions):
        super(MultiSuccessorRepresentation, self).__init__()
        
        self.num_abstract_states = num_abstract_states
        self.num_abstractions = num_abstractions
        
        self.psi = nn.Sequential(
            nn.Linear(num_abstract_states*num_abstractions, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.num_abstractions*self.num_abstract_states),
        )
    
    def forward(self, abstract_state):
        successor_representation = self.psi(abstract_state.view(abstract_state.size(0), -1))
        return successor_representation.view(successor_representation.size(0), self.num_abstractions, self.num_abstract_states)

'''
class Discriminator_OLD(nn.Module):
    def __init__(self):
        super(Discriminator_OLD).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
'''


def train_abstraction():
    # ------------- 1. Explore Environment -------------
    env_config = {
        "max_steps": 50000 # no reset
    }
    env = FourRoomsNoReward(env_config)

    print("**Exploring Environment**")
    data = []
    state = env.reset()
    for _ in range(env_config["max_steps"]):
        action = env.action_space.sample()
        next_state , _, done, _ = env.step(action)
        data.append([state, next_state])
        
        # NOTE: this will never happen since we explore for exactly max_steps and have no reward
        if done:
            state = env.reset()
        else:
            state = next_state
    
    print("Num samples:", len(data))
    batch_size = 256
    transition_dataset = NormalizedTransitionsDataset(data)
    transition_dataloader = DataLoader(transition_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ------------- 2. Define Models -------------

    # Abstraction phi
    num_abstract_states = 4
    entropy_coef = 5
    num_abstractions = 2
    sr_gamma = 0.95
    tau = 0.5

    phi = MultiAbstraction(obs_size=env.observation_size, num_abstract_states=num_abstract_states, num_abstractions=num_abstractions)
    phi_optimizer = torch.optim.Adam(phi.parameters(), lr=0.001)
    # Successor Representation psi
    psi = MultiSuccessorRepresentation(num_abstract_states, num_abstractions=num_abstractions)
    psi_optimizer = torch.optim.Adam(psi.parameters(), lr=0.001)
    # Discriminator netD
    discriminator = Discriminator(num_abstract_states)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # ------------- 3. Train Models -------------
    num_batches = 40
    real_label = 1.
    fake_label = 0.
    do_gan = True
    bce = nn.BCELoss()
    previous_batch = None
    for batch_idx in range(num_batches):
        total_sr_loss = 0
        total_entropy_loss = 0
        for batch in transition_dataloader:
            state = batch[:,0]
            next_state = batch[:,1]

            if do_gan:
                if previous_batch is None:
                    previous_batch = state
                    continue

                with torch.no_grad():
                    abstract_state = phi.sample(state, tau=1.0, hard=False)
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch --> our "real" batch is just s2
                discriminator_optimizer.zero_grad()
                # Format batch
                b_size = batch.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float)
                # Forward pass real batch through D
                output = discriminator(abstract_state.view(b_size, -1)).view(-1)
                # Calculate loss on all-real batch
                errD_real = bce(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                
                ## Train with all-fake batch --> our "fake" batch is phi_2(x_new)
                # We get a batch of new states by just looking at the previous batch
                #   TODO: make sure that using the *previous* batch isn't introducing dependencies in the learning...
                previous_abstract_state = phi.sample(previous_batch, tau=1.0, hard=False)
                label.fill_(fake_label)
                # Classify all fake batch with D
                # fake_1: phi1(x) vs phi2(x')
                fake_1 = torch.cat([abstract_state[:,0:1,:], previous_abstract_state[:,1:2,:]], dim=1)
                # fake_2: phi1(x') vs phi2(x)
                fake_2 = torch.cat([previous_abstract_state[:,0:1,:], abstract_state[:,1:2,:]], dim=1)
                
                output_1 = discriminator(fake_1.view(b_size, -1).detach()).view(-1)
                output_2 = discriminator(fake_2.view(b_size, -1).detach()).view(-1)
                
                # Calculate D's loss on the all-fake batch
                errD_fake = 0.5* (bce(output_1, label) + bce(output_2, label))
                
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                discriminator_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            phi_optimizer.zero_grad()
            psi_optimizer.zero_grad()
            phi.zero_grad()
            abstract_state = phi.sample(state, tau=tau, hard=False)
            with torch.no_grad():
                next_abstract_state = phi.sample(next_state, tau=tau, hard=False)

            # Loss 1 : max entropy of encoding
            mean_abstract_state_probs = abstract_state.mean(dim=0) # --> num_abstractions x num_abstract_states
            avg_entropy = (- mean_abstract_state_probs * torch.log(mean_abstract_state_probs)).sum() / num_abstractions
            ent_loss = - entropy_coef * avg_entropy # TODO: should we divide by the number of abstractions?

            # Loss 3: SR TD loss
            successor_representation = psi(abstract_state)
            with torch.no_grad():
                next_successor_representation = psi(next_abstract_state)
            
            sr_td_loss = ((successor_representation -
                (abstract_state.detach() + sr_gamma * next_successor_representation))**2).sum() / (num_abstractions * batch_size)

            if do_gan:
                label.fill_(fake_label)  # IN OUR CASE we want the discriminator to think we are fake
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = discriminator(abstract_state.view(b_size, -1)).view(-1)
                # Calculate G's loss based on this output
                errG = bce(output, label)
                abstraction_loss = errG + ent_loss + sr_td_loss
                # previous batch is used for "fake" data
                previous_batch = state
            else:
                abstraction_loss = ent_loss + sr_td_loss
            
            abstraction_loss.backward()
            phi_optimizer.step()
            psi_optimizer.step()

            total_sr_loss += sr_td_loss.item()
            total_entropy_loss += ent_loss.item()

        print(f"sr_loss {total_sr_loss / (len(data) // batch_size):3.3f}, "+
                f"entropy_loss {total_entropy_loss / (len(data) // batch_size):3.3f}")

        # TODO: 
        #   need to debug the discriminator loss as well as the generator loss

        # print(errG.item(), ent_loss.item(), sr_td_loss.item(), errD_fake.item(), errD_real.item())
    
    env_grid = (env.example_obs == 1)*1.0
    with torch.no_grad():
        all_phis = torch.zeros((num_abstractions, 19, 19))
        for i in range(env_grid.shape[1]):
            for j in range(env_grid.shape[2]):
                if env_grid[0,i,j] > 0:
                    all_phis[:,i,j] = -1
                    continue
                
                tmp_state = torch.FloatTensor([i,j]) / 18.0
                tmp_enc = phi(tmp_state.unsqueeze(0))
                a_nums = torch.argmax(tmp_enc[0], dim=-1)
                all_phis[:, i,j] = a_nums
                
    # print(all_phis)
    if num_abstractions == 1:
        plt.imshow(all_phis[0])
    else:
        fig, axes = plt.subplots(1, num_abstractions)

        for abstraction_num in range(num_abstractions):
            axes[abstraction_num].imshow(all_phis[abstraction_num])
    # plt.colorbar()
    plt.savefig(f"tmp_data/fourrooms_multiple_abstractions.png")
    
    


if __name__=="__main__":
    train_abstraction()