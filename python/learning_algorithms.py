import copy
import pickle
import random
import gymnasium as gym
import torch
from collections import deque, namedtuple
from gymnasium.utils.save_video import save_video
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from utils import *


# Class for training an RL agent with Actor-Critic
class ACTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.agent = ACAgent(env=self.env, params=self.params)
        self.actor_net = ActorNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.critic_net = CriticNet(input_size=self.env.observation_space.shape[0], output_size=1, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.actor_optimizer = Adam(params=self.actor_net.parameters(), lr=self.params['actor_lr'])
        self.critic_optimizer = Adam(params=self.critic_net.parameters(), lr=self.params['critic_lr'])
        self.trajectory = None

    def run_training_loop(self):
        list_ro_reward = list()
        for ro_idx in range(self.params['n_rollout']):
            self.trajectory = self.agent.collect_trajectory(policy=self.actor_net)
            self.update_critic_net()
            self.estimate_advantage()
            self.update_actor_net()
            # TODO: Calculate avg reward for this rollout
            # HINT: Add all the rewards from each trajectory. There should be "ntr" trajectories within a single rollout.
            avg_ro_reward = 0
            for t in range(len(self.trajectory['reward'])):
                avg_ro_reward = avg_ro_reward + np.sum(r for r in self.trajectory['reward'][t])
            avg_ro_reward = avg_ro_reward / self.params['n_trajectory_per_rollout']
            print(f'End of rollout {ro_idx}: Average trajectory reward is {avg_ro_reward: 0.2f}')
            # Append average rollout reward into a list
            list_ro_reward.append(avg_ro_reward)
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ro_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def update_critic_net(self):
        for critic_iter_idx in range(self.params['n_critic_iter']):
            self.update_target_value()
            for critic_epoch_idx in range(self.params['n_critic_epoch']):
                critic_loss = self.estimate_critic_loss_function()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()

    def update_target_value(self, gamma=0.99):
        # TODO: Update target values
        # HINT: Use definition of target-estimate from equation 7 of teh assignment PDF
        target_value = list()
        state_value = list()

        for k in range(self.params['n_trajectory_per_rollout']):
            target_value_for_trajectory = list()
            state_value_for_trajectory = list()
            for t in range(len(self.trajectory['reward'][k]) - 1):
                reward_kt = self.trajectory['reward'][k][t]
                current_state_value = self.critic_net(self.trajectory['obs'][k][t])
                next_state_value = self.critic_net(self.trajectory['obs'][k][t+1])
                target_value_for_trajectory.append(reward_kt + (gamma * next_state_value.item()))
                state_value_for_trajectory.append(current_state_value.item())
            target_value.append(target_value_for_trajectory)
            state_value.append(state_value_for_trajectory)
        self.trajectory['target_value'] = target_value
        self.trajectory['state_value'] = state_value

    def estimate_advantage(self, gamma=0.99):
        # TODO: Estimate advantage
        # HINT: Use definition of advantage-estimate from equation 6 of teh assignment PDF

        advantage = list()
        for k in range(self.params['n_trajectory_per_rollout']):
            advantage_for_trajectory = list()
            for t in range(len(self.trajectory['reward'][k]) - 1):
                reward_kt = self.trajectory['reward'][k][t]
                state_value = self.critic_net(self.trajectory['obs'][k][t])
                next_state_value = self.critic_net(self.trajectory['obs'][k][t+1])
                advantage_for_trajectory.append((reward_kt + (gamma * next_state_value.item())) - state_value.item())
            advantage.append(advantage_for_trajectory)
        self.trajectory['advantage'] = advantage

    def update_actor_net(self):
        actor_loss = self.estimate_actor_loss_function()
        print(actor_loss)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

    def estimate_critic_loss_function(self):
        # TODO: Compute critic loss function
        # HINT: Use definition of critic-loss from equation 7 of teh assignment PDF. It is the MSE between target-values and state-values.
        critic_loss = torch.zeros(1, requires_grad=True)
        for k in range(self.params['n_trajectory_per_rollout']):
            trajectory_loss = torch.zeros(1)
            step = len(self.trajectory['target_value'][k])
            for t in range(step):
                trajectory_loss = trajectory_loss + (self.trajectory['target_value'][k][t] - self.trajectory['state_value'][k][t]) ** 2
            critic_loss = critic_loss + (trajectory_loss / step)
        return critic_loss / self.params['n_trajectory_per_rollout']
       

    def estimate_actor_loss_function(self):
        actor_loss = list()
        for t_idx in range(self.params['n_trajectory_per_rollout']):
            advantage = apply_discount(self.trajectory['advantage'][t_idx])
            # TODO: Compute actor loss function
            reward_for_trajectory = 0
            for t in range(len(self.trajectory['advantage'][t_idx])):
                reward_for_trajectory = reward_for_trajectory + (advantage[t] * self.trajectory['log_prob'][t_idx][t])
            actor_loss.append(-1 * reward_for_trajectory)
        actor_loss = torch.stack(actor_loss).mean()
        return actor_loss

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action_idx, log_prob = self.actor_net(torch.tensor(obs, dtype=torch.float32, device=get_device()))
            obs, reward, terminated, truncated, info = self.env.step(self.agent.action_space[action_idx.item()])
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)

# CLass for actor-net
class ActorNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(ActorNet, self).__init__()
        # TODO: Define the actor net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.ff_net = nn.Sequential(nn.Linear(input_size, hidden_dim),nn.Linear(hidden_dim, output_size))

    def forward(self, obs):
        # TODO: Forward pass of actor net
        # HINT: (use Categorical from torch.distributions to draw samples and log-prob from model output)
        action_probs = self.ff_net(obs)
        probability_distribution = Categorical(logits=action_probs)
        action_index = probability_distribution.sample()
        log_prob = probability_distribution.log_prob(action_index)
        return action_index, log_prob

# CLass for actor-net
class CriticNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(CriticNet, self).__init__()
        # TODO: Define the critic net
        # HINT: You can use nn.Sequential to set up a 2 layer feedforward neural network.
        self.ff_net = nn.Sequential(nn.Linear(input_size, hidden_dim),nn.Linear(hidden_dim, output_size))
        # self.ff_net = ???

    def forward(self, obs):
        # TODO: Forward pass of critic net
        # HINT: (get state value from the network using the current observation)
        state_value = torch.relu(self.ff_net(obs))
        return state_value

# Class for agent
class ACAgent:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params
        self.action_space = [action for action in range(self.env.action_space.n)]

    def collect_trajectory(self, policy):
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        rollout_buffer = list()
        for _ in range(self.params['n_trajectory_per_rollout']):
            trajectory_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
            while True:
                obs = torch.tensor(obs, dtype=torch.float32, device=get_device())
                # Save observation
                trajectory_buffer['obs'].append(obs)
                action_idx, log_prob = policy(obs)
                obs, reward, terminated, truncated, info = self.env.step(self.action_space[action_idx.item()])
                # Save log-prob and reward into the buffer
                trajectory_buffer['log_prob'].append(log_prob)
                trajectory_buffer['reward'].append(reward)
                # Check for termination criteria
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    rollout_buffer.append(trajectory_buffer)
                    break
        rollout_buffer = self.serialize_trajectory(rollout_buffer)
        return rollout_buffer

    # Converts a list-of-dictionary into dictionary-of-list
    @staticmethod
    def serialize_trajectory(rollout_buffer):
        serialized_buffer = {'obs': list(), 'log_prob': list(), 'reward': list()}
        for trajectory_buffer in rollout_buffer:
            serialized_buffer['obs'].append(torch.stack(trajectory_buffer['obs']))
            serialized_buffer['log_prob'].append(torch.stack(trajectory_buffer['log_prob']))
            serialized_buffer['reward'].append(trajectory_buffer['reward'])
        return serialized_buffer

class DQNTrainer:
    def __init__(self, params):
        self.params = params
        self.env = gym.make(self.params['env_name'])
        self.q_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net = QNet(input_size=self.env.observation_space.shape[0], output_size=self.env.action_space.n, hidden_dim=self.params['hidden_dim']).to(get_device())
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = self.params['init_epsilon']
        self.optimizer = Adam(params=self.q_net.parameters(), lr=self.params['lr'])
        self.replay_memory = ReplayMemory(capacity=self.params['rm_cap'])

    def run_training_loop(self):
        list_ep_reward = list()
        obs, _ = self.env.reset(seed=self.params['rng_seed'])
        for idx_episode in range(self.params['n_episode']):
            ep_len = 0
            while True:
                ep_len += 1
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    self.epsilon = max(self.epsilon*self.params['epsilon_decay'], self.params['min_epsilon'])
                    next_obs = None
                    self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                    list_ep_reward.append(ep_len)
                    print(f'End of episode {idx_episode} with epsilon = {self.epsilon: 0.2f} and reward = {ep_len}, memory = {len(self.replay_memory.buffer)}')
                    obs, _ = self.env.reset()
                    break
                self.replay_memory.push(obs, action, reward, next_obs, not (terminated or truncated))
                obs = copy.deepcopy(next_obs)
                self.update_q_net()
                self.update_target_net()
        # Save avg-rewards as pickle files
        pkl_file_name = self.params['exp_name'] + '.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(list_ep_reward, f)
        # Save a video of the trained agent playing
        self.generate_video()
        # Close environment
        self.env.close()

    def get_action(self, obs):
        # TODO: Implement the epsilon-greedy behavior
        # HINT: The agent will will choose action based on maximum Q-value with
        # '1-ε' probability, and a random action with 'ε' probability.
    
        if np.random.uniform() < self.epsilon:
            # Choose a random action with probability epsilon
            action = self.env.action_space.sample()
        else:
            # Choose the action with highest Q-value with probability 1 - epsilon
            with torch.no_grad():
                state_value = self.q_net(torch.FloatTensor(obs).to(get_device()))
                action = torch.argmax(state_value).item()
        return action
    
    def update_q_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        # TODO: Update Q-net
        # HINT: You should draw a batch of random samples from the replay buffer
        # and train your Q-net with that sampled batch.

        samples = self.replay_memory.sample(self.params['batch_size'])

        predicted_state_values = list()
        target_values = list()

        for idx in range(len(samples)):
            if (samples[idx].not_done_batch) :
                next_obs_reward = self.q_net(torch.tensor(samples[idx].next_obs_batch))
                next_obs_reward = samples[idx].reward_batch + self.params['gamma'] * torch.max(next_obs_reward)
            else:
                next_obs_reward = samples[idx].reward_batch
            
            current_obs= self.target_net(torch.tensor(samples[idx].obs_batch))
            current_action = current_obs[samples[idx].action_batch]

            predicted_state_values.append(next_obs_reward)
            target_values.append(current_action)
        
        predicted_state_value = torch.tensor(predicted_state_values, requires_grad= True)
        target_value = torch.tensor(target_values, requires_grad=True)

        criterion = nn.SmoothL1Loss()
        q_loss = criterion(predicted_state_value, target_value.unsqueeze(1))
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        if len(self.replay_memory.buffer) < self.params['batch_size']:
            return
        q_net_state_dict = self.q_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = self.params['tau']*q_net_state_dict[key] + (1 - self.params['tau'])*target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def generate_video(self, max_frame=1000):
        self.env = gym.make(self.params['env_name'], render_mode='rgb_array_list')
        self.epsilon = 0.0
        obs, _ = self.env.reset()
        for _ in range(max_frame):
            action = self.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        save_video(frames=self.env.render(), video_folder=self.params['env_name'][:-3], fps=self.env.metadata['render_fps'], step_starting_index=0, episode_index=0)


class ReplayMemory:
    # TODO: Implement replay buffer
    # HINT: You can use python data structure deque to construct a replay buffer
    # Reference: https://www.geeksforgeeks.org/namedtuple-in-python/#
    def __init__(self, capacity):
        self.capacity = capacity
        self.sampletuple = namedtuple('sample',['obs_batch', 'action_batch', 'reward_batch', 'next_obs_batch', 'not_done_batch'])
        self.buffer = deque(maxlen =  capacity)

    def push(self, *args):
        self.buffer.append(self.sampletuple(*args))

    def sample(self, n_samples):
        sample_list = list()
        for i in range(n_samples):
            sample_list.append(self.buffer[random.randint(0, len(self.buffer)-1)])
        return sample_list


class QNet(nn.Module):
    # TODO: Define Q-net
    # This is identical to policy network from HW1
    def __init__(self, input_size, output_size, hidden_dim):
        super(QNet, self).__init__()
        self.ff_net = nn.Sequential(nn.Linear(input_size, hidden_dim),nn.Linear(hidden_dim, output_size))

    def forward(self, obs):
        return self.ff_net(obs)

