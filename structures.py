import torch
import attr


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
        self.__dict__ = d
