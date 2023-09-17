# %%
import gymnasium as gym
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import random
import matplotlib.pyplot as plt
from collections import deque
import cv2
import math

from stable_baselines3.ddpg.policies import CnnPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import HumanRendering
from gymnasium.spaces import Box


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
def print_frames(frame_stack):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axes[i].imshow(frame_stack[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

# %%
class GameEnv(gym.Wrapper):
    def __init__(self, env):
        super(GameEnv, self).__init__(env)
        self.action_space = self.env.action_space
        self.frame_skip = 4
        self.frame_stack = 4
        self.observation_space = Box(0, 255, self.reset()[0].shape, dtype=np.uint8)
        self.last_stack = np.array([])

    def skip_frames(self, action, skip_count=100):
        total_reward = 0 # not sure
        for _ in range(skip_count):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward 
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, info

    def stack_frames(self, new_frame, reset=False):
        if reset:
            self.last_stack = np.tile(new_frame, (self.frame_stack, 1, 1))
        else:
            self.last_stack = np.concatenate((self.last_stack[1:], new_frame[np.newaxis]), axis=0)
        return self.last_stack
    
    def process_image(self, observation):
        observation = cv2.cvtColor(np.array(observation), cv2.COLOR_BGR2GRAY)/255.0 # normalize
        return observation[:84,6:90]

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation, _, _, _, info = self.skip_frames([0,0,0])
        observation = self.process_image(observation)
        observation = self.stack_frames(observation, True)

        return observation, info

    def step(self, action):
        observation, total_reward, terminated, truncated, info = self.skip_frames(action, self.frame_skip)
        observation = self.process_image(observation)
        observation = self.stack_frames(observation)

        return observation, total_reward, terminated, truncated, info

# %%

env = GameEnv(gym.make('CarRacing-v2', continuous=True, render_mode="rgb_array"))
wrapped = HumanRendering(env)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("CnnPolicy", env, verbose=1, buffer_size=100000, batch_size=32, learning_rate=10000, device=DEVICE)
vec_env = model.get_env()
model.learn(total_timesteps=500000, log_interval=10)
model.save("ddpg_carracing")

model = DDPG.load("ddpg_pendulum", env=wrapped)
vec_env = model.get_env()
obs = vec_env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = vec_env.step(action)
    


# %%
