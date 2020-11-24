import torch
import attr
from utils import discount_cumsum, get_mean_std_across_processes
from typing import List

@attr.s(auto_attribs=True)
class EnvParams(object):
    action_dim : int
    rew_dim : int
    state_dim : int


@attr.s(auto_attribs=True)
class Trajectory(object):
    actions : torch.Tensor = None
    rewards : torch.Tensor = None
    states : torch.Tensor = None 
    next_states: torch.Tensor = None


    # add method
    def add(self, s, next_s, r, a):


        s = s.unsqueeze(0)
        next_s = next_s.unsqueeze(0)
        r = r.unsqueeze(0)
        a = a.unsqueeze(0)

        if self.actions is None:
            self.actions = a
            self.states = s
            self.next_states = next_s
            self.rewards = r
            return

        self.actions = torch.cat((self.actions, a),dim=0)
        self.states = torch.cat([self.states, s],dim=0)
        self.next_states = torch.cat([self.next_states, next_s],dim=0)
        self.rewards = torch.cat([self.rewards, r],dim=0)

@attr.s(auto_attribs=True)
class OnPolicyBatch:
    states : torch.Tensor = None
    next_states : torch.Tensor = None
    actions : torch.Tensor = None
    returns_to_go : torch.Tensor = None
    advantages : torch.Tensor = None
    log_action_probs : torch.Tensor = None



@attr.s(auto_attribs=True)
class OnPolicyTrajectory(object):
    actions : torch.Tensor = None
    rewards : torch.Tensor = None
    states : torch.Tensor = None 
    next_states: torch.Tensor = None
    log_action_probs: torch.Tensor = None
    values: torch.Tensor = None
    gamma: float = None

    # add method
    def add(self, s, next_s, r, a, log_prob_a, values):


        s = s.unsqueeze(0)
        next_s = next_s.unsqueeze(0)
        r = r.unsqueeze(0)
        a = a.unsqueeze(0)
        log_prob_a = log_prob_a.unsqueeze(0)
        # don't do it for values as they are output from the value function


        if self.actions is None:
            self.actions = a
            self.states = s
            self.next_states = next_s
            self.rewards = r
            self.log_action_probs = log_prob_a
            self.values = values
            return

        self.actions = torch.cat((self.actions, a),dim=0)
        self.states = torch.cat([self.states, s],dim=0)
        self.next_states = torch.cat([self.next_states, next_s],dim=0)
        self.rewards = torch.cat([self.rewards, r],dim=0)
        self.log_action_probs = torch.cat([self.log_action_probs, log_prob_a], dim=0)
        self.values = torch.cat([self.values, values], dim=0)


    def get_batch(self, last_val, parallel, gamma=0.99, lam=0.95):
        # last val is 0 if simulator has returned done, else if contains value of last state if we've reached max episode cut off
        self.rewards = torch.cat([self.rewards, last_val], dim=0)
        self.values = torch.cat([self.values, last_val], dim=0)

        # curr rewards + gamma*next_values - curr_values: Bellman update ?
        deltas = self.rewards[:-1] + gamma * self.values[1:] - self.values[:-1]
        advantages = discount_cumsum(deltas, gamma*lam)
    
        # finish episode
        returns_to_go = discount_cumsum(self.rewards, gamma)[:-1]

        # the next two lines implement the advantage normalization trick
        if parallel:
            adv_mean, adv_std = get_mean_std_across_processes(advantages)
        else:
            adv_mean, adv_std = advantages.mean(), advantages.std()

        advantages = (advantages - adv_mean) / adv_std
        batch = OnPolicyBatch(states=self.states, next_states=self.next_states, actions=self.actions, returns_to_go=returns_to_go,  advantages=advantages, log_action_probs=self.log_action_probs)

        return batch






class Params:

    def __init__(self, d):
        self.state_dim: int = None
        self.action_dim: int = None
        self.action_label_dim: int = None
        self.gamma: float = None
        self.rew_dim: int = None
        self.epsilon: float = None
        self.update_steps: int = None
        self.total_eps: int = None
        self.num_rollouts: int = None
        self.test_num_rollouts:int = None
        self.test_rollout_len: int = None
        self.rollout_len: int = None
        self.buf_size: int = None
        self.batch_sz: int = None
        self.target_update_step: int = None
        self.device: str = None
        self.save_path: str =  None
        self.env_name: str = None
        self.network_type: str = None
        self.parallel: bool = None
        self.log_interval: int = None 
        self.render: bool = None
        self.clip_ratio: float = None
        self.value_func_itrs: int = None
        self.policy_itrs: int = None
        self.load_saved: bool = None
        self.min_exploration_steps: int = None
        self.action_range: float = None
        self.critic_tau: float = None
        self.actor_update_frequency: float = None
        self.learnable_temperature: float = None
        self.actor_betas: List[float] = None
        self.critic_betas: List[float] = None
        self.alpha_betas: List[float] = None
        self.actor_lr: float = None 
        self.critic_lr: float = None
        self.alpha_lr: float = None
        self.critic_tau: float = None
        self.critic_target_update_frequency: float = None
        self.init_temperature: float = None
        self.__dict__ = d

    def __repr__(self):
        print("Loading config..")
        for k, v in self.__dict__.items():
            if k is not None and v is not None:
                print(f"{k} : {v}")
        return ""
