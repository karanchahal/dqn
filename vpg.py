import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from structures import Params
from agent import Agent
import matplotlib.pyplot as plt


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class VPG(Agent):

    def __init__(self, env, params):
        self.env = env
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.params = params

        self.rewards_per_eps = []
        self.losses = []


    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()
    

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.params.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()

        self.losses.append(policy_loss.item())

        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def algorithm(self, render=False):
        running_reward = 10
        for i_episode in count(1):
            state, ep_reward = env.reset(), 0
            for t in range(1, 10000):  # Don't infinite loop while learning
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()

                self.policy.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self.rewards_per_eps.append(ep_reward)

            self.finish_episode()
            if i_episode % self.params.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break
    
    def plot_curves(self):
        fig, axs = plt.subplots(1,2)
        axs[0].plot(np.arange(len(self.rewards_per_eps)), self.rewards_per_eps)
        axs[0].set_xlabel('Number of episodes')
        axs[0].set_ylabel('Mean Rewards')
        axs[1].plot(np.arange(len(self.losses)), self.losses)
        axs[1].set_xlabel('Number of episodes')
        axs[1].set_ylabel('Losses')
        plt.show()
        # the mean rew plot


if __name__ == '__main__':
    params = {
        'gamma': 0.99,
        'render': False,
        'log_interval': 10,
    }
    params = Params(params)
    env = gym.make('CartPole-v1')
    vpg = VPG(env, params)
    vpg.algorithm()
    vpg.plot_curves()