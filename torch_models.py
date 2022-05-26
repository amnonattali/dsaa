import torch
import torch.nn as nn
from torch.distributions import Categorical

# DSAA consists of 3 models: 
#   phi - the abstraction
#   psi - the successor representation
#   pi  - the option policies Q network

# Variable names are defined as follows:
#   A/num_actions           = number of discrete actions
#   N/num_abstract_states   = number of discrete abstract states
#   K/obs_size              = number of observation features
#   B                       = batch size


# The abstraction is a simple feedforward neural network with K inputs and N outputs
class Abstraction(nn.Module):
    def __init__(self, obs_size, num_abstract_states):
        super(Abstraction, self).__init__()
        
        self.obs_size = obs_size
        self.num_abstract_states = num_abstract_states
        
        self.features = nn.Sequential(
            nn.Linear(self.obs_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.num_abstract_states),
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

# The successor representation module is a simple feedforward neural network is N inputs and N outputs
class SuccessorRepresentation(nn.Module):
    def __init__(self, num_abstract_states):
        super(SuccessorRepresentation, self).__init__()
        
        self.num_abstract_states = num_abstract_states
        
        self.psi = nn.Sequential(
            nn.Linear(num_abstract_states, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.num_abstract_states),
        )
    
    def forward(self, abstract_state):
        successor_representation = self.psi(abstract_state)
        return successor_representation

# MonolithicPolicy is a single model that encapsulate all our option policies
# Input: (state=[x,s], skill=z) 
#   - x is the primitive state, shape=[B,K], dtype=float
#   - s is the abstract state, shape=[B,N], dtype=float
#   - state has shape=[B,K+N]
#   - z is the skill to execute, meaning the next abstract state number, shape=[B,1], dtype=int
# Output: model(state, skill) has shape=[B,A], dtype=float
#   - It models the Q value for each discrete action
#   - Note that the underlying neural network has A*N output features
class MonolithicPolicy(object):
    def __init__(self, num_abstract_states, num_skills, obs_size, num_actions, config):
        self.num_abstract_states = num_abstract_states
        self.num_skills = num_skills
        self.obs_size = obs_size
        self.num_actions = num_actions
        
        self.config = config
        self.model_input_size = num_abstract_states + obs_size
        self.model_output_size = num_actions * num_skills

        self.reset()

    def reset(self):
        self.network = SoftQNetwork(self.model_input_size, self.model_output_size, self.config["option_entropy_coef"])
        self.learn_steps = 0
        self.option_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"])

    def __call__(self, state, skill):
        return self.network(state)[:, skill*self.num_actions : (skill+1)*self.num_actions]

    # choose_action gets the action for a single (state, skill) pair, instead of a batch
    def choose_action(self, state, skill):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.network(state)[:,skill*self.num_actions : (skill+1)*self.num_actions]
            dist = torch.softmax(q, dim=1)
            
            try:
                c = Categorical(dist)
            except ValueError:
                print("ERROR", dist)

            a = c.sample()
        return a.item()

class SoftQNetwork(nn.Module):
    def __init__(self, inputs, outputs, entropy_coef = 0.01):
        super(SoftQNetwork, self).__init__()
        self.alpha = 4
        self.entropy_coef = entropy_coef
        self.features = nn.Sequential(
            nn.Linear(inputs, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, outputs)
        )
        
    def forward(self, x):
        return self.features(x)

    # In soft Q learning our policy samples actions according to the softmax of the Q values (probs*qvalue)
    # When computing the target value of the next state we add some reward based on the entropy (-probs*log(probs))
    def getV(self, q_value):
        probs = torch.softmax(q_value, dim=1) + 0.1**16
        v = (probs * (q_value - self.entropy_coef*torch.clip(torch.log(probs), min=-5, max=0))).sum(dim=1, keepdim=True)
        return v
        
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.forward(state)
            dist = torch.softmax(q, dim=1)
            c = Categorical(dist)
            a = c.sample()
            # a = torch.argmax(dist)
        return a.item()

