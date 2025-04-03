import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ion()  # enable interactive mode for live plots

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import gym_race

from gymnasium.envs.registration import register
try:
    register(
        id='Pyrace-v1',
        entry_point='gym_race.envs:RaceEnv',
        max_episode_steps=2000,
    )
except:
    pass

VERSION_NAME = 'DQN_v03_extended'
REPORT_EPISODES = 10
DISPLAY_EPISODES = 100
NUM_EPISODES = 20000
MAX_T = 2000
BATCH_SIZE = 128
REPLAY_MEMORY_SIZE = 100000
TARGET_UPDATE_FREQ = 10
LEARNING_RATE = 0.0001

MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.99

env = gym.make("Pyrace-v1").unwrapped
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

DECAY_FACTOR = 10000.0

# --- DQN Neural Network ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # First layer
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256, track_running_stats=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second layer
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256, track_running_stats=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Third layer
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128, track_running_stats=True)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        # Training mode will use dropout and proper batch norm
        if self.training:
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.dropout3(x)
        # Eval mode will use running statistics for batch norm and no dropout
        else:
            with torch.no_grad():
                x = F.relu(self.bn1(self.fc1(x)))
                x = F.relu(self.bn2(self.fc2(x)))
                x = F.relu(self.bn3(self.fc3(x)))
        return self.out(x)

# --- Experience Replay Buffer ---
from collections import deque

class ReplayMemory:
    def __init__(self, capacity=REPLAY_MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- Decay Schedulers ---
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, 0.1 + 0.9 * math.exp(-t / 5000))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, LEARNING_RATE * math.exp(-t / DECAY_FACTOR))

# --- DQN Agent with Target Network ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self.memory = ReplayMemory()
        self.loss_fn = nn.MSELoss()
        self.steps_done = 0

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, explore_rate):
        self.steps_done += 1
        if random.random() < explore_rate:
            return random.randrange(action_dim)
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
        self.model.train()  # Set back to training mode
        return torch.argmax(q_values).item()

    def remember(self, *args):
        self.memory.push(args)

    def replay(self, batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(np.array(states))  # Convert to numpy array first
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))  # Convert to numpy array first
        dones = torch.FloatTensor(dones).unsqueeze(1)

        self.model.train()  # Ensure training mode
        current_q_values = self.model(states).gather(1, actions)

        self.target_model.eval()  # Set target network to eval mode
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + gamma * max_next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

# --- Training Loop ---
def simulate(learning=True, episode_start=0):
    agent = DQNAgent(state_dim, action_dim)
    total_rewards = []
    max_reward = -float("inf")
    episode_losses = []

    if not os.path.exists(f"models_{VERSION_NAME}"):
        os.makedirs(f"models_{VERSION_NAME}")

    checkpoint_path = f"models_{VERSION_NAME}/checkpoint.pth"
    start_episode = episode_start

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_rewards = checkpoint['total_rewards']
        max_reward = checkpoint['max_reward']
        start_episode = checkpoint['episode'] + 1
        print(f"✅ Resumed from checkpoint at episode {start_episode}")
    elif os.path.exists(f"models_{VERSION_NAME}/best_model.pth"):
        agent.model.load_state_dict(torch.load(f"models_{VERSION_NAME}/best_model.pth"))
        agent.update_target_network()
        print("✅ Loaded best model to start training.")

    env.set_view(True)

    for episode in range(start_episode, NUM_EPISODES + start_episode):
        total_reward = 0
        state, _ = env.reset()

        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = learning_rate

        if not learning:
            agent.model.eval()
            env.pyrace.mode = 2

        for t in range(MAX_T):
            action = agent.act(state, explore_rate if learning else 0)
            next_state, reward, done, _, info = env.step(action)
            
            if learning:
                agent.remember(state, action, reward, next_state, done)
                agent.replay()

            state = next_state
            total_reward += reward

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs([
                    'SIMULATE',
                    f'Episode: {episode}',
                    f'Time steps: {t}',
                    f'check: {info["check"]}',
                    f'dist: {info["dist"]}',
                    f'crash: {info["crash"]}',
                    f'Reward: {total_reward:.0f}',
                    f'Explore Rate: {explore_rate:.4f}'
                ])
                env.render()

            if done or t >= MAX_T - 1:
                if total_reward > max_reward and learning:
                    max_reward = total_reward
                    torch.save(agent.model.state_dict(), f"models_{VERSION_NAME}/best_model.pth")
                break

        if episode % TARGET_UPDATE_FREQ == 0 and learning:
            agent.update_target_network()

        total_rewards.append(total_reward)

        if episode % DISPLAY_EPISODES == 0:
            plt.clf()
            plt.plot(total_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Training Reward Progress")
            plt.savefig(f"models_{VERSION_NAME}/rewards_progress.png")
            plt.close()

        if learning and episode % REPORT_EPISODES == 0:
            # Save intermediate model
            torch.save(agent.model.state_dict(), f"models_{VERSION_NAME}/model_episode_{episode}.pth")
            
            # Save checkpoint with all training state
            torch.save({
                'episode': episode,
                'model_state_dict': agent.model.state_dict(),
                'target_model_state_dict': agent.target_model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'total_rewards': total_rewards,
                'max_reward': max_reward
            }, checkpoint_path)

        print(f"Episode {episode} — Total reward: {total_reward:.2f} — Explore Rate: {explore_rate:.4f} — Learning Rate: {learning_rate:.6f}")

if __name__ == "__main__":
    simulate()
