import sys, os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import gym_race

# --- Hyperparameters ---
GAMMA = 0.99
LR = 1e-3
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
NUM_EPISODES = 500

# --- Environment setup ---
env = gym.make("Pyrace-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# --- DQN Model ---
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

# --- Replay Memory ---
from collections import deque

class ReplayMemory:
    def __init__(self, capacity=MEMORY_SIZE):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = ReplayMemory()
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def replay(self, batch_size=BATCH_SIZE):
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
        next_q = self.model(next_states).max(1)[0].detach().unsqueeze(1)
        target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# --- Simulation Function ---
def simulate():
    agent = DQNAgent(state_dim, action_dim)
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1} — Total reward: {total_reward}")

if __name__ == "__main__":
    simulate()
