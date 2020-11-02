from structures import Trajectory
import torch.nn as nn
import torch 
import collections
import torch.nn.functional as F
import gym 
import random
from buffer import Buffer
import torch.optim as optim 
from utils import plot_grad_flow
from os.path import join
from pathlib import Path
from structures import Params

class Agent:

    def update(self, batch):
        # add update function to weights of the agent.
        raise NotImplementedError

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        # plot_grad_flow(self.q_net.named_parameters())
        self.optimizer.step()

# test_dqn()

