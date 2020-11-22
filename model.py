import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,hidden=[400,300],init_w=3e-3):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        param: hidden: 
        """
        super(Actor, self).__init__()
        assert len(hidden)==2,"Two hidden layers"

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], self.action_dim)

        self.init_weights(init_w)
        
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init____init__(self, state_dim, action_dim,hidden=[400,300],init_w=3e-3):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()
        assert len(hidden)==2,"Two hidden layers"
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0] + self.action_dim, hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)

        self.init_weights(init_w)
        
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """
        Define the forward pass of the critic
        """
        
        x = F.relu(self.fc1(state)) 
        x = F.relu(self.fc2(torch.cat([x,action],1)))
        x = self.fc3(x)
        return x