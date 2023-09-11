
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
from IPython.display import clear_output
import pygame
import os
from torch.distributions import Categorical


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPISODES = 500
MAX_STEPS = 1000
GAMMA = 0.95
LR = 0.001
TERM_REWARD = -25


class GameEnv():
    def __init__(self, render=False):
        if render:
            self.env = gym.make('CarRacing-v2', continuous=False, render_mode='human')
        else:
            self.env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.frame_skip = 4
        self.frame_stack = 4
        self.last_stack = np.array([])

    def skip_frames(self, action, skip_count=50):
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

    def reset(self):
        observation, info = self.env.reset()
        observation, _, _, _, info = self.skip_frames(0)
        observation = self.process_image(observation)
        observation = self.stack_frames(observation, True)

        return observation, info

    def step(self, action):
        observation, total_reward, terminated, truncated, info = self.skip_frames(action, self.frame_skip)
        observation = self.process_image(observation)
        observation = self.stack_frames(observation)

        return observation, total_reward, terminated, truncated, info


class ConvNet(nn.Module):
    def __init__(self, input_channel, possible_actions):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Conv2d(input_channel, 16, kernel_size=8, padding=1, stride=2) # 40x40
        self.layernorm1 = nn.BatchNorm2d(16)
        self.layerpool1 = nn.MaxPool2d(kernel_size=2) # 20x20

        self.fc1 = nn.Linear(16*20*20, 256)
        self.fc2 = nn.Linear(256, possible_actions)

    def forward(self, state):
        out = self.layer1(state)
        out = self.layernorm1(out)
        out = F.relu(out)
        out = self.layerpool1(out)

        out = out.reshape(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return F.softmax(out, dim=1)


class PolicyGradient(nn.Module):
    def __init__(self, state_shape, actions, channel):
        super(PolicyGradient, self).__init__()
        self.policy = ConvNet(channel, actions).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)

        self.total_steps = 0
        self.log_probs = []
        self.rewards = []

    def select_action(self, state, training=True):

        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        action_probs = self.policy(state.unsqueeze(0))

        action_probs = Categorical(action_probs)
        action = action_probs.sample()

        self.log_probs.append(action_probs.log_prob(action))

        return action.item()

    def learn(self):
        R = 0
        policy_loss = []
        rewards = []
        for reward in self.rewards[::-1]:
            R = reward + GAMMA * R
            rewards.append(R)

        rewards = torch.tensor(rewards[::-1], dtype=torch.float32, device=DEVICE)
        reward = (rewards - rewards.mean()) / (rewards.std() + 0.00000001)

        for r, prob in zip(rewards, self.log_probs):
            policy_loss.append(-1*r*prob)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.log_probs[:]


def evaluate(agent, n_evals=5):
    eval_env = GameEnv()
    #agent.policy.load_state_dict(torch.load("./yello/dqn3.pt"))
    scores = 0
    for _ in range(n_evals):
        state, _ = eval_env.reset()
        ret = 0
        while True:
            action = agent.select_action(state, training=False)
            next_state, r, terminated, truncated, _ = eval_env.step(action)
            state = next_state
            ret += r
            done = terminated or truncated
            if done:
                break
        scores += ret
    pygame.display.quit()
    pygame.quit()
    return np.round(scores / n_evals, 4)


env = GameEnv()
state, _ = env.reset()
actions = env.action_space.n

history = {'Episode': [], 'AvgReturn': []}

agent = PolicyGradient(state.shape, actions, state.shape[0])

for episode in range(MAX_EPISODES):
    print(f"Episode {episode} starting ...")
    state, _ = env.reset()
    episode_rewards = []
    for _ in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        agent.rewards.append(reward)
        episode_rewards.append(reward)
        
        state = next_state

        if episode % 100 == 0:
            ret = evaluate(agent)
            history['Episode'].append(agent.total_steps)
            history['AvgReturn'].append(ret)
            clear_output()
            plt.figure(figsize=(8, 5))
            plt.plot(history['Episode'], history['AvgReturn'], 'r-')
            plt.xlabel('Step', fontsize=16)
            plt.ylabel('AvgReturn', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(axis='y')
            plt.savefig("rewardspolicy.png")

            torch.save(agent.policy.state_dict(), f'dqn-policy{episode}.pt')

        if terminated or truncated or np.sum(episode_rewards)<TERM_REWARD:
            break

    agent.learn()

    print(f"Episode {episode} ends ...")

evaluate(agent)