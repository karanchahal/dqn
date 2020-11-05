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
from agent import Agent
from model_factory import get_model
from utils import average_gradients
import tracemalloc
from utils import display_top

class ParallelDQN(Agent):

    def __init__(self, env, params):
        super().__init__()

        self.buffer = Buffer(params)
        self.device = params.device
        self.env = env
        self.params = params
        self.gamma = params.gamma

        self.epsilon = params.epsilon
        self.total_eps = params.total_eps
        self.update_steps = params.update_steps
        self.num_rollouts = params.num_rollouts
        self.rollout_len = params.rollout_len
        self.batch_size = params.batch_sz
        self.target_update_step = params.target_update_step
        self.steps_done = 0
        self.best_metric_till_yet = 0


        # networks
        self.q_net = get_model(params).to(self.device)
        self.q_target_net = get_model(params).to(self.device)

        # optims
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.0001)
        
        # copying q_net to q_target
        self.copy_q_target_net()

        self.losses = []

    def copy_q_target_net(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())

    def get_value_of(self, states):
        q_vals = self.q_target_net(states)
        vals, _ = torch.max(q_vals, dim=1)
        return vals.unsqueeze(1)

    def compute_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def get_target(self, states, rews):
        with torch.no_grad():
            # get values of next state
            vals = self.get_value_of(states)

            # construct target
            target = rews + self.gamma*vals

            target[rews == 0] = 0
        
        return target
    
    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)

        # average gradients over all processes
        if self.params.parallel:
            average_gradients(self.q_net)


        # plot_grad_flow(self.q_net.named_parameters())
        self.optimizer.step()

    def get_pred(self, states, actions):
        """
        States : N x state_dim
        Actions: N x action_dim (here action_dim is generally 1)
        """
        # q vals for all actions of curr states
        q_vals_curr_state = self.q_net(states)

        # find q vals for actions taken
        actions = actions.long()
        pred = q_vals_curr_state.gather(1, actions)
        return pred
        

    def update(self, batch: Trajectory):
        """
        Batch has
        N actions -> N X A
        N states -> N x S
        N next states N x S
        N rewards N x R
        """

        actions = batch.actions
        rewards = batch.rewards
        states = batch.states
        next_states = batch.next_states

        target = self.get_target(next_states, rewards)
        pred = self.get_pred(states, actions)

        loss = self.compute_loss(pred, target)

        self.step(loss)

        self.losses.append(loss.item())

        # print(loss.item())


    def sample_actions(self, state, stochastic=True) -> int:
        """
        State is a pytorch tensor 1 x state_dim
        """
        state = state.unsqueeze(0)
        vals, indxs = torch.max(self.q_net(state), dim=1)

        if not stochastic or random.random() < self.epsilon:
            return indxs[0].item()

        return self.env.action_space.sample()

    def reset_env(self):
        """
        Returns the first state spit out by the env
        Return pytorch tensor
        """
        s = self.env.reset()
        s = torch.tensor(s).to(self.device).to(torch.float32)
        return s

    def step_env(self, a):
        """
        This function steps through the env and returns the next state
        a is assumed to be a discrete value / float
        """
        new_s, r, done, _ = self.env.step(a)
        new_s = torch.tensor(new_s).to(self.device).to(torch.float32)
        r = torch.tensor(r).to(self.device).to(torch.float32).view(1)

        return new_s, r, done, _


    def sample(self):

        for _ in range(self.num_rollouts):

            s = self.reset_env()
            traj = Trajectory()
            # print(s.shape)
            for _ in range(self.rollout_len):
                a = self.sample_actions(s)

                new_s, r, done, _ = self.step_env(a)
                # print(new_s.shape, r.shape, a)

                # turn a into tensor
                a = torch.tensor(a).to(self.device).to(torch.float32).view(1)

                if done:
                    # TODO: only for cartpole, this will be a signal that q target should be 0 for next state
                    # r[0] = 0, comment out for cartpole
                    r = self.change_reward_at_rollout_end(r)

                traj.add(s, new_s, r, a)

                if done:
                    break
                
                s = new_s
            
            # add to buffer
            self.buffer.add(traj) 

            self.decay_epsilon()
    
    def change_reward_at_rollout_end(self, r):
        # for cartpole return 0 ! r[0] = 0
        return r

    def decay_epsilon(self):

        self.steps_done += 1

        max_epsilon = 0.8
        min_epsilon = 0

        num_total_steps = self.total_eps*self.num_rollouts

        new_eps = max(min_epsilon, max_epsilon*self.steps_done / num_total_steps)
        self.epsilon = new_eps
        # print(f"Epsilon {self.epsilon}")
        


    def eval_agent(self, render=False):
        with torch.no_grad():
            self.q_net.eval()
            total_rew = 0
            for _ in range(self.num_rollouts):
                s = self.reset_env()
                for _ in range(self.rollout_len):
                    if render:
                        self.env.render()

                    a = self.sample_actions(s, stochastic=False)

                    new_s, r, done, _ = self.step_env(a)

                    total_rew += r.item()

                    if done:
                        break

                    # clear memory
                    del s

                    s = new_s
                del s
            del s
            mean_rew = total_rew/self.num_rollouts
            print("Mean reward is {}".format(mean_rew))

            # save best model
            self.save_best_model(metric=mean_rew)
        self.q_net.train() # set back to train

    def save_best_model(self, metric):
        if self.best_metric_till_yet < metric:
            Path(self.params.save_path).mkdir(parents=True, exist_ok=True)
            self.best_metric_till_yet = metric
            model_path = torch.save({
                'q_net_state_dict': self.q_net.state_dict(),
                'optim_state_dict': self.optimizer.state_dict(),
                'params': self.params
            }, join(self.params.save_path, 'model.pt'))

    def algorithm(self):
        for i in range(self.total_eps):
            self.env = gym.make('CarRacing-v0', verbose=0).unwrapped
            print(f"Sampling {i}..")
            self.sample()
            print(f"Updating {i}..")
            for _ in range(self.update_steps):
                batch = self.buffer.get(self.batch_size)
                self.update(batch)

            if i % self.target_update_step == 0:
                self.copy_q_target_net()
            print(f"Evaluating {i}..")
            self.eval_agent()

            self.env.close()
            # snapshot = tracemalloc.take_snapshot()
            # display_top(snapshot)
                
    def load_saved(self, model_path):
        obj = torch.load(join(model_path, 'model.pt'))
        self.q_net.load_state_dict(obj['q_net_state_dict'])
        self.optimizer.load_state_dict(obj['optim_state_dict'])
        self.params = obj['params']

