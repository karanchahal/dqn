import gym
import matplotlib.pyplot as plt
import numpy as np
from dqn_car_racing import CarDQN
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import gym
from structures import Params


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def run_car_racing(parallel=True):
    env = gym.make('CarRacing-v0', verbose=0).unwrapped

    params = {
        'env_name': 'CarRacing-v0',
        'state_dim': (3,96,96), # history of past 3 frames
        'action_dim': 5, # action space of brake, acc, left, right, straight
        'gamma': 0.99,
        'rew_dim': 1,
        'epsilon': 0,
        "update_steps" : 10,
        "total_eps": 500,
        "num_rollouts": 1,
        "rollout_len": 1000,
        "test_num_rollouts": 2,
        "test_rollout_len": 100,
        "buf_size": 1000,
        "batch_sz": 128,
        "target_update_step": 10,
        "device": "cpu",
        "save_path": "./model/dqn/",
        "network_type": "cnn",
        "action_label_dim": 1,
        "parallel": parallel
    }
    params = Params(params)
    dqn = CarDQN(env, params)
    dqn.algorithm()
    # dqn.eval_agent()


def run(rank, size):
    """ Distributed function to be implemented later. """
    print(f"Rank is {dist.get_rank()}")
    run_car_racing()

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def parallel_run(size=1):
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    # parallel_run(size=4)
    run_car_racing(parallel=False)