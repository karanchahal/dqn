import torch 
from model_factory import get_model, MLPParam
from torch.distributions import Categorical
from structures import Params, OnPolicyBatch, OnPolicyTrajectory
from agent import Agent
from torch.optim import Adam
from utils import get_mean_std_across_processes
import gym
import torch.nn.functional as F

def tensorize(a, dtype=torch.float32, device="cpu"):
    return torch.tensor(a).to(dtype).to(device)


class PPO(Agent):

    def __init__(self, env, params: Params):

        self.env = env
        self.params = params
        # init actor and critic
        self.policy_net = get_model(params)
        self.value_function = MLPParam(params.state_dim, 1)

        self.log_step = 0
        # optimizers
        self.policy_optim = Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optim = Adam(self.value_function.parameters(), lr=3e-4)


    def save_state(self):
        raise NotImplementedError
    
    
    def decay_epsilon(self):
        self.steps_done += 1

        max_epsilon = 0.8
        min_epsilon = 0

        num_total_steps = self.total_eps*self.num_rollouts

        new_eps = max(min_epsilon, max_epsilon*self.steps_done / num_total_steps)
        self.epsilon = new_eps

    def get_actions(self, s):
        s = s.unsqueeze(0)
        pi = F.softmax(self.policy_net(s), dim=1)

        m = Categorical(pi)
        action = m.sample()
        log_prob_acts = m.log_prob(action)

        v = self.value_function(s)

        return action, log_prob_acts, v
    
    def reset_env(self):
        s = self.env.reset()
        s = tensorize(s)

        return s

    def step_env(self, action):
        """
        Action is a tensor
        """
        new_s, r, done, _  = self.env.step(action.item())

        # tensorize
        new_s = tensorize(new_s)
        r = tensorize(r).view(1)

        return new_s, r, done, _

    def do_rollout(self):
        with torch.no_grad():
            s = self.reset_env()
            traj = OnPolicyTrajectory()
            last_val = torch.zeros(1).to(self.params.device).view(1,1)
            total_rew = 0
            for i in range(self.params.rollout_len):

                action, log_prob_act, values = self.get_actions(s)

                new_s, r, done, _ = self.step_env(action)
                total_rew += r.item()
                if done:
                    last_val = self.value_function(new_s.unsqueeze(0))
                    break

                traj.add(s, new_s, r, action, log_prob_act, values)

                s = new_s

            on_policy_batch = traj.get_batch(last_val, self.params.parallel)
            self.log_step += 1
            if self.log_step %100 == 0:
                print(f'Total reward is {total_rew}')
            return on_policy_batch


    def compute_loss(self, on_policy_batch: OnPolicyBatch):
        states, actions, advantages, old_log_p_actions = on_policy_batch.states, on_policy_batch.actions, on_policy_batch.advantages, on_policy_batch.log_action_probs

        # Policy Loss
        # get action distribution
        pi = F.softmax(self.policy_net(states), dim=1)
        m = Categorical(pi)

        # get log probs of current actions
        new_log_probs = m.log_prob(actions)
        # print(new_log_probs, old_log_p_actions)
        ratio = torch.exp(new_log_probs - old_log_p_actions)
        clip_adv = torch.clamp(ratio, 1 - self.params.clip_ratio, 1 + self.params.clip_ratio) * advantages
        loss_pi = -(torch.min(ratio * advantages, clip_adv)).mean()


        # useful extra info
        approx_kl = (old_log_p_actions - new_log_probs).mean().item()
        ent = m.entropy().mean().item()
        clipped = ratio.gt(1+self.params.clip_ratio) | ratio.lt(1-self.params.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info


    def compute_value_loss(self, batch: OnPolicyBatch):
        obs = batch.states
        val = self.value_function(obs)
        tar = batch.returns_to_go.to(torch.float32)
        return F.mse_loss(val, tar)

    def update_policy(self, on_policy_batch: OnPolicyBatch, target_kl=0.01):

        # policy learning
        for i in range(self.params.policy_itrs):
            self.policy_optim.zero_grad()
            loss_pi, pi_info = self.compute_loss(on_policy_batch)
            if self.params.parallel:
                kl_mean, _ = get_mean_std_across_processes(pi_info['kl'])
            else:
                kl_mean = pi_info['kl']
            if  kl_mean > 1.5*target_kl:
                print('Early stopping..')
                break
            loss_pi.backward()
            if self.params.parallel:
                average_gradients(self.policy_net)
            self.policy_optim.step()
        
        # value function learning
        for i in range(self.params.value_func_itrs):
            loss = self.compute_value_loss(on_policy_batch)
            
            self.value_optim.zero_grad()
            loss.backward()
            if self.params.parallel:
                average_gradients(self.value_function)
            self.value_optim.step()
        


    def algorithm(self):

        for _ in range(self.params.total_eps):

            # sample rollout
            on_policy_batch = self.do_rollout()

            self.update_policy(on_policy_batch)


def train_ppo():
    env = gym.make('CartPole-v1')
    params = {
        'state_dim': 4,
        'action_dim': 2,
        'total_eps': 10000,
        'value_func_itrs': 10,
        'policy_itrs': 10,
        'clip_ratio': 0.2,
        'rollout_len': 500,
        'hidden_dim': 50,
        'network_type': "mlp",
        "device": "cpu",
        "parallel": False
    }
    params = Params(params)

    ppo = PPO(env, params)
    ppo.algorithm()

train_ppo()