from dqn_parallel import ParallelDQN
from utils import rgb2gray
from structures import Trajectory
import torch 
import numpy as np
import random
import torch.distributed as dist
from pathlib import Path
from os.path import join
from memory_profiler import profile
import gc
import tracemalloc
import gym

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if a == [-1.0, 0.0, 0.0]: return LEFT               # LEFT: 1
    elif a == [1.0, 0.0, 0.0]: return RIGHT             # RIGHT: 2
    elif a == [0.0, 1.0, 0.0]: return ACCELERATE        # ACCELERATE: 3
    elif a == [0.0, 0.0, 0.2]: return BRAKE             # BRAKE: 4
    else:       
        return STRAIGHT                                      # STRAIGHT = 0


class CarDQN(ParallelDQN):


    def __init__(self, env, params):
        super().__init__(env, params)
        self.history_queue = []
        self.id2action = {
            0: [0.0, 0.0, 0.2], # STRAIGHT
            1: [1.0, 0.0, 0.0], # LEFT
            2: [1.0, 0.0, 0.0], # RIGHT
            3: [0.0, 1.0, 0.0], # ACC,
            4: [0.0, 0.0, 0.2] # BRAKE
        }
        # tracemalloc.start()

    def reset_env(self):
        s = self.env.reset()
        s = rgb2gray(s)
        # delete old stuff
        for i in self.history_queue:
            del i

        self.history_queue = [s, s, s]
        with torch.no_grad():
            state = torch.tensor(np.stack(self.history_queue)).to(torch.float32).to(self.device)

        return state
    
    def step_env(self, a):
        with torch.no_grad():
            next_state, r, done, info = self.env.step(a)

            next_state = rgb2gray(next_state)
            self.history_queue.append(next_state)
            to_be_del =  self.history_queue[0]
            del to_be_del
            self.history_queue = self.history_queue[1:]
            next_state = torch.tensor(np.stack(self.history_queue)).to(torch.float32).to(self.device) # 3, 96, 96

        return next_state, torch.tensor(r).to(self.device).to(torch.float32).view(1), done, info

    def sample_actions(self, state, stochastic=True) -> int:
        """
        State is a pytorch tensor of state_dim
        """
        state = state.unsqueeze(0) # to get 1 batch size
        if not stochastic or random.random() < self.epsilon: # 1 means no exploration, 0 means lots of exploration
            _, indxs = torch.max(self.q_net(state), dim=1)
            action = self.id2action[indxs[0].item()]
            del indxs
            return action

        # random actions
        act_id = random.randint(0,4)
        action = self.id2action[act_id]
        return action

    
    def sample(self):
        print(f"Exploration epsilon is {self.epsilon} (1 means no exploration)")
        for _ in range(self.num_rollouts):

            s = self.reset_env()
            traj = Trajectory()
            # print(s.shape)

            model_gone_wild = 0
            for _ in range(self.rollout_len):

                a = self.sample_actions(s)
                new_s, r, done, _ = self.step_env(a)
                # print(new_s.shape, r.shape, a)

                # keeps a count of num steps the model has gone into the wild
                if r.item() < 0:
                    model_gone_wild += 1
                else:
                    model_gone_wild = 0

                # turn a into tensor
                with torch.no_grad():
                    a = torch.tensor(action_to_id(a)).to(self.device).to(torch.float32).view(1)

                if done:
                    r = self.change_reward_at_rollout_end(r)

                traj.add(s, new_s, r, a)

                if done or model_gone_wild > 100: # either done or in the green for more than 50 time steps, then 
                    break
                
                s = new_s
            
            # add to buffer
            self.buffer.add(traj)

            self.decay_epsilon()

    def eval_agent(self, render=False):
        with torch.no_grad():
            self.q_net.eval()
            total_rew = 0
            num_steps = 0
            for _ in range(self.num_rollouts):
                s = self.reset_env()
                model_gone_wild = 0
                for _ in range(self.rollout_len):
                    # if render:
                    self.env.render()

                    a = self.sample_actions(s, stochastic=False)
                    new_s, r, done, _ = self.step_env(a)

                    # keeps a count of num steps the model has gone into the wild
                    if r.item() < 0:
                        model_gone_wild += 1
                    else:
                        model_gone_wild = 0


                    total_rew += r.item()
                    num_steps += 1

                    if done or model_gone_wild > 100: # either done or in the green for more than 50 time steps, then 
                        break

                    s = new_s
            
            mean_rew = total_rew/num_steps
            print("Mean reward is {}".format(mean_rew))
            # save best model
            self.save_best_model(metric=mean_rew)
            self.q_net.train() # set back to train
    
    def save_best_model(self, metric):
        if not self.params.parallel or dist.get_rank() == 0:
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

            # required for memory leaks
            self.env = gym.make('CarRacing-v0', verbose=0).unwrapped

            # print(f"Sampling {i}..")
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


