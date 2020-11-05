from structures import Params
from dqn_car_racing import CarDQN
import gym

def eval_car_racing(parallel=True):
    env = gym.make('CarRacing-v0', verbose=0).unwrapped

    params = {
        'env_name': 'CarRacing-v0',
        'state_dim': (3,96,96), # history of past 3 frames
        'action_dim': 5, # action space of brake, acc, left, right, straight
        'gamma': 0.99,
        'rew_dim': 1,
        'epsilon': 0,
        "update_steps" : 10,
        "total_eps": 1000,
        "num_rollouts": 3,
        "rollout_len": 1000,
        "test_num_rollouts": 2,
        "test_rollout_len": 500,
        "buf_size": 10000,
        "batch_sz": 512,
        "target_update_step": 10,
        "device": "cpu",
        "save_path": "./model/dqn/",
        "network_type": "cnn",
        "action_label_dim": 1,
        "parallel": False,
        "load_model": True
    }

    params = Params(params)
    dqn = CarDQN(env, params)
    dqn.eval_agent(save=False, render=True)


eval_car_racing()