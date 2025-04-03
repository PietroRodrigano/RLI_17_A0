import torch
import gymnasium as gym
import gym_race
from gymnasium.envs.registration import register
import numpy as np

# Register the custom environment
try:
    register(
        id='Pyrace-v1',
        entry_point='gym_race.envs:RaceEnv',
        max_episode_steps=2000,
    )
except:
    pass

# Load environment
env = gym.make("Pyrace-v1").unwrapped
env.set_view(True)
env.pyrace.mode = 2  # Evaluation/render mode

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# --- Define your DQN model architecture ---
import torch.nn as nn
import torch.nn.functional as F

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

# --- Load the model ---
model = DQN(state_dim, action_dim)
model_path = "models_DQN_v02_improved/checkpoint.pth"  # Using checkpoint since best_model.pth doesn't exist
model.load_state_dict(torch.load(model_path)['model_state_dict'])  # Load from checkpoint
model.eval()

# --- Run an episode ---
state, _ = env.reset()
total_reward = 0

for t in range(2000):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    next_state, reward, done, _, info = env.step(action)
    total_reward += reward
    state = next_state

    env.set_msgs([
        'BEST MODEL INFERENCE',
        f'Time step: {t}',
        f'check: {info["check"]}',
        f'dist: {info["dist"]}',
        f'crash: {info["crash"]}',
        f'Reward: {total_reward:.0f}'
    ])
    env.render()

    if done:
        print(f"Episode finished after {t+1} timesteps — Total Reward: {total_reward:.2f}")
        break
