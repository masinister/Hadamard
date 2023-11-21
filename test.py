import torch
import gymnasium as gym
import gym_hadamard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

env = gym.make('gym_hadamard/hadamard-v0', dim = 2, order = 3, render_mode = "human")

observation, info = env.reset(seed = 42)

for _ in range(1000):
    action = env.action_space.sample()
    print("action: \n", action)
    observation, reward, terminated, truncated, info = env.step(action)
    print("state: \n", observation)
    if terminated or truncated:
        observation, info = env.reset()

env.close()