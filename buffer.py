import torch
import attr 
from utils import get_random_indxs
from structures import Trajectory, EnvParams, Params


class Buffer:

    def __init__(self, env_params: Params): 
        size = env_params.buf_size
        self.size = size
        self.actions = torch.empty((round(size), env_params.action_label_dim))
        self.rewards = torch.empty((round(size), env_params.rew_dim))
        self.states = torch.empty((round(size), *env_params.state_dim))
        self.next_states = torch.empty((round(size), *env_params.state_dim))
        self.ptr = 0
        self.full = False

    def add(self, trajectory: Trajectory):
        sz = trajectory.actions.shape[0]

        if self.ptr + sz >= self.size:
            self.ptr = 0

        a_ = self.actions[self.ptr:self.ptr+sz]
        r_ = self.rewards[self.ptr:self.ptr+sz]
        s_ =self.states[self.ptr:self.ptr+sz]
        next_s = self.next_states[self.ptr:self.ptr+sz] 

        del a_
        del r_
        del s_
        del next_s

        self.actions[self.ptr:self.ptr+sz] = trajectory.actions
        self.rewards[self.ptr:self.ptr+sz] = trajectory.rewards
        self.states[self.ptr:self.ptr+sz] = trajectory.states
        self.next_states[self.ptr:self.ptr+sz] = trajectory.next_states

        self.ptr += sz

    def get(self, sz: int):
        start = 0
        end = self.size - 1 if self.full else self.ptr

        rand_indxs = get_random_indxs(sz, start=start, end=end)
        actions = self.actions[rand_indxs]
        states = self.states[rand_indxs]
        next_states = self.next_states[rand_indxs]
        rewards = self.rewards[rand_indxs]

        return Trajectory(actions=actions, states=states, next_states=next_states, rewards=rewards)


def test_buffer():
    bt_sz = 10
    action_dim = 4
    rew_dim = 1
    state_dim = 10

    actions = torch.randn(bt_sz, action_dim)
    rewards = torch.randn(bt_sz, rew_dim)
    states = torch.randn(bt_sz, state_dim)
    next_states = torch.randn(bt_sz, state_dim)


    traj = Trajectory(actions=actions, rewards=rewards, states=states, next_states=next_states)

    env_params = EnvParams(action_dim=action_dim, rew_dim=rew_dim, state_dim=state_dim)
    buf = Buffer(10e6, env_params)
    buf.add(traj)

    assert(buf.ptr != 0)
    assert(buf.ptr == bt_sz)

    act = buf.get(2)
    assert(act.shape[0] == 2)
