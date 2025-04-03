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

VERSION_NAME = 'DQN_v02_improved'
REPORT_EPISODES = 1
DISPLAY_EPISODES = 1
NUM_EPISODES = 10000
MAX_T = 2000

MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.99

env = gym.make("Pyrace-v1").unwrapped
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

DECAY_FACTOR = 10000.0

# --- DQN Neural Network ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# --- Experience Replay Buffer ---
from collections import deque

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- Decay Schedulers ---
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, 0.05 + 0.95 * math.exp(-t / 5000))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

# --- DQN Agent with Target Network ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = ReplayMemory()
        self.loss_fn = nn.MSELoss()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, explore_rate):
        if random.random() < explore_rate:
            return random.randrange(action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, *args):
        self.memory.push(args)

    def replay(self, batch_size=64, gamma=DISCOUNT_FACTOR):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_q = self.target_model(next_states).max(1)[0].detach().unsqueeze(1)
        target_q = rewards + gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# --- Training Loop ---
def simulate(learning=True, episode_start=0):
    agent = DQNAgent(state_dim, action_dim)
    total_rewards = []
    max_reward = -float("inf")

    if not os.path.exists(f"models_{VERSION_NAME}"):
        os.makedirs(f"models_{VERSION_NAME}")

    checkpoint_path = f"models_{VERSION_NAME}/checkpoint.pth"
    start_episode = episode_start

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.update_target_network()
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

        if not learning:
            env.pyrace.mode = 2

        for t in range(MAX_T):
            action = agent.act(state, explore_rate if learning else 0)
            next_state, reward, done, _, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            if learning:
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
                    f'Max Reward: {max_reward:.0f}',
                ])
                env.render()

            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                    torch.save(agent.model.state_dict(), f"models_{VERSION_NAME}/best_model.pth")
                break

        if episode % 10 == 0:
            agent.update_target_network()

        total_rewards.append(total_reward)

        plt.clf()
        plt.plot(total_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Live Training Reward Progress")
        plt.pause(0.01)

        if learning and episode % REPORT_EPISODES == 0:
            plt.savefig(f"models_{VERSION_NAME}/rewards_{episode}.png")
            torch.save(agent.model.state_dict(), f"models_{VERSION_NAME}/model_episode_{episode}.pth")
            torch.save({
                'episode': episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'total_rewards': total_rewards,
                'max_reward': max_reward
            }, checkpoint_path)
            plt.close()

        print(f"Episode {episode} — Total reward: {total_reward:.2f} — Explore Rate: {explore_rate:.4f}")

if __name__ == "__main__":
    simulate()
