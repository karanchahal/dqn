import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from dqn_parallel import ParallelDQN
import gym
from structures import Params


def train_dqn():

    # env = gym.make('CartPole-v1')
    env = gym.make('LunarLander-v2')
    params = {
        'env_name': 'CartPole-v1',
        'state_dim': 8,
        'action_dim': env.action_space.n,
        'gamma': 0.99,
        'rew_dim': 1,
        'epsilon': 0,
        "update_steps" : 10,
        "total_eps": 300,
        "num_rollouts": 10,
        "rollout_len": 300,
        "buf_size": 10000,
        "batch_sz": 128,
        "target_update_step": 10,
        "device": "cpu",
        "save_path": "./model/dqn/",
        "network_type": "mlp",
        "action_label_dim": 1,
    }

    params = Params(params)
    dqn = ParallelDQN(env=env, params=params)

    dqn.algorithm()


def run(rank, size):
    """ Distributed function to be implemented later. """
    train_dqn()

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 6
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()