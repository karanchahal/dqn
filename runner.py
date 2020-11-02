import torch.nn as nn
import torch 
import gym 
import random
from structures import Params
from dqn import DQN


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
    dqn = DQN(env=env, params=params)

    dqn.algorithm()


def test_dqn():
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
        "rollout_len": 1000, # higher rollout len
        "buf_size": 10000,
        "batch_sz": 128,
        "target_update_step": 10,
        "device": "cpu",
        "save_path": "./model/dqn/",
        "network_type": "mlp",
        "action_label_dim": 1,
    }

    params = Params(params)

    dqn = DQN(env=env, params=params)

    dqn.load_saved(params.save_path)
    dqn.eval_agent(render=True)

# train_dqn()
test_dqn()
