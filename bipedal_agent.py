from agent import Agent
import gym 
from bipedal_buffer import ReplayBuffer
from model_factory import get_model
from structures import Params 
import torch
import numpy as np
import sac_utils as utils 
import torch.nn.functional as F

class BipedalAgent(Agent):

    def __init__(self, env, params: Params):

        super().__init__()
        self.device = params.device
        self.buffer = ReplayBuffer(obs_shape=params.state_dim, action_shape=params.action_dim, capacity=params.buf_size, device=self.device)
        self.env = env
        self.params = params


        self.action_range = params.action_range
        self.device = torch.device(self.device)
        self.discount = params.gamma
        self.critic_tau = params.critic_tau
        self.actor_update_frequency = params.actor_update_frequency
        self.critic_target_update_frequency = params.critic_target_update_frequency
        self.batch_size = params.batch_sz
        self.learnable_temperature = params.learnable_temperature

        self.actor, self.critic = get_model(params)
        _, self.critic_target = get_model(params)

        self.critic_target.load_state_dict(self.critic.state_dict())


        self.log_alpha = torch.tensor(np.log(params.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -params.action_dim[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=params.actor_lr,
                                                betas=params.actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=params.critic_lr,
                                                 betas=params.critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=params.alpha_lr,
                                                    betas=params.alpha_betas)

        # self.critic_target.train()

        self.steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, obs, sample=False):
        # sample policy
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])
    

    def update_critic(self, obs, action, reward, next_obs, not_done,
                      step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.params.gamma * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        # print('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(step)

    def update_actor_and_alpha(self, obs, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # print('train_actor/loss', actor_loss, step)
        # print('train_actor/target_entropy', self.target_entropy, step)
        # print('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                        (-log_prob - self.target_entropy).detach()).mean()
            # print('train_alpha/loss', alpha_loss, step)
            # print('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update_agent(self, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = self.buffer.sample(
            self.batch_size)

        # print('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, step)

        if step % self.params.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step)

        if step % self.params.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
    


    def algorithm(self):
        for i in range(self.params.total_eps): # 5000
            obs = self.env.reset()
            ep_step = 0
            ep_rew = 0
            while True:
                if self.steps < self.params.min_exploration_steps:
                    a = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        self.actor.eval()
                        self.critic.eval()
                        a = self.get_action(obs, sample=True)

                new_obs, rew, done, _ = self.env.step(a)
                ep_rew += rew
                # add to replay buffer
                if self.steps > self.params.min_exploration_steps:
                    # TODO
                    self.actor.train()
                    self.critic.train()
                    self.update_agent(step=self.steps)

                self.steps += 1

                done_no_max = 0 if ep_step + 1 == self.env._max_episode_steps else done

                if done:
                    break
                # TODO
                self.buffer.add(obs, a, rew, new_obs, done, done_no_max)
                ep_step += 1
            
            print(f"Iteration {i} | Episode Reward {ep_rew}")





def test_agent():
    # env = gym.make("BipedalWalkerHardcore-v3")
    env = gym.make("Pendulum-v0")
    print(env.action_space.shape)
    params = {
        "total_eps": 5000,
        "device": "cpu",
        "state_dim": (3,), # 24
        "action_dim": (1,), # 4
        "min_exploration_steps" : 200,
        "buf_size" : 10e5,
        "action_range" : [float(env.action_space.low.min()), float(env.action_space.high.max())],
        "gamma" : 0.99,
        "actor_update_frequency": 1,
        "learnable_temperature": True,
        "batch_sz": 128, 
        "actor_betas": [0.9, 0.999],
        "critic_betas": [0.9, 0.999],
        "alpha_betas": [0.9, 0.999],
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "alpha_lr": 1e-4,
        "critic_tau": 0.005,
        "critic_target_update_frequency": 2,
        "network_type": "bipedal",
        "init_temperature": 0.1,
    }

    params = Params(params)


    agent = BipedalAgent(env, params)
    agent.algorithm()

test_agent()