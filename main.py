
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SIZE = 10000
BATCH_SIZE = 32
EPS_MAX = 0.95
EPS_MIN = 0.1
EPS_DECAY = 1000
DROP = 0.2
GAMMA = 0.95
LR = 0.001


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
        


transition = namedtuple("Transition", ("state","action", "reward", "next_state"))


class ReplayMemory():
    def __init__(self, state_shape):
        self.state = torch.tensor(np.zeros((MAX_SIZE, *state_shape), dtype=float), dtype=torch.float32).to(DEVICE)
        self.action = torch.tensor(np.zeros((MAX_SIZE, 1), dtype=float), dtype=torch.float32).to(DEVICE)
        self.reward = torch.tensor(np.zeros((MAX_SIZE, 1), dtype=float), dtype=torch.float32).to(DEVICE)
        self.next_state = torch.tensor(np.zeros((MAX_SIZE, *state_shape), dtype=float), dtype=torch.float32).to(DEVICE)

        self.current_size = 0

    def push(self, transition):
        self.state[self.current_size] = torch.tensor(np.array(transition.state), dtype=torch.float32).to(DEVICE)
        self.action[self.current_size] = torch.tensor(np.array(transition.action), dtype=torch.float32).to(DEVICE)
        self.reward[self.current_size] = torch.tensor(np.array(transition.reward), dtype=torch.float32).to(DEVICE)
        self.next_state[self.current_size] = torch.tensor(np.array(transition.next_state), dtype=torch.float32).to(DEVICE)

        self.current_size = (self.current_size + 1) % MAX_SIZE

    def sample(self):
        sample_idx = np.random.randint(len(self.state), size=BATCH_SIZE) 
        return self.state[sample_idx], self.action[sample_idx], self.reward[sample_idx], self.next_state[sample_idx]


class ConvNet(nn.Module):
    def __init__(self, input_channel, possible_actions, input_size):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Conv2d(input_channel, 16, kernel_size=6, stride=2, padding=1) # 41x41
        self.layernorm1 = nn.BatchNorm2d(16)
        self.layerpool1 = nn.MaxPool2d(kernel_size=5, stride=2) # 42x42

        self.layer2 = nn.Conv2d(16, 32, kernel_size=4, stride=1) # 16x16
        self.layernorm2 = nn.BatchNorm2d(32)
        self.layerpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8x8

        #self.cnv = int((input_size-4)/4)
        self.fc1 = nn.Linear(32*8*8, 256)
        self.fc2 = nn.Linear(256, possible_actions)
    
    def forward(self, state):
        out = self.layer1(state)
        out = self.layernorm1(out)
        out = F.relu(out)
        out = self.layerpool1(out)

        out = self.layer2(out)
        out = self.layernorm2(out)
        out = F.relu(out)
        out = self.layerpool2(out)

        out = out.reshape(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out


class DQN(nn.Module):
    def __init__(self, state_shape, actions, channel):
        super(DQN, self).__init__()
        self.memory = ReplayMemory(state_shape)
        self.target_net = ConvNet(channel, actions, state_shape[1]).to(DEVICE)
        self.active_net = ConvNet(channel, actions, state_shape[1]).to(DEVICE)
        self.target_net.load_state_dict(self.active_net.state_dict())

        self.optimizer = torch.optim.Adam(self.active_net.parameters(), lr=LR)
        self.loss = nn.SmoothL1Loss()

        self.total_steps = 0
        self.target_update = 1000

    def select_action(self, state, training=True):
        self.target_net.train(training)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            greedy_action = self.active_net(state.unsqueeze(0)).max(1)[1].item()

        epsilon = EPS_MIN + (EPS_MAX - EPS_MIN) * math.exp(-1. * self.total_steps / EPS_DECAY)
        self.total_steps += 1

        if np.random.random(1) < epsilon and training:
            return np.random.randint(5)
        return greedy_action

    def learn(self):
        state, action, reward, next_state = self.memory.sample()
        reward = (reward - reward.mean()) / (reward.std() + 0.000001)

        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0]
            target_q = reward + GAMMA * next_q.unsqueeze(1)
        
        approx_q = self.active_net(state).gather(1, action.long())

        temp_diff = self.loss(approx_q, target_q)
        self.optimizer.zero_grad()
        temp_diff.backward()
        self.optimizer.step()

        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.active_net.state_dict()) 
            
        result = {
            'total_steps': self.total_steps,
            'value_loss': temp_diff.item()
        }
        return result


def evaluate(agent, n_evals=5):
    eval_env = GameEnv(render=True)
    agent.active_net.load_state_dict(torch.load("./yello/dqn.pt")) 
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

history = {'Step': [], 'AvgReturn': []}

agent = DQN(state.shape, actions, state.shape[0])
episode_count = 2000
action_taken = 0


for episode in range(episode_count):
    print(f"Episode {episode} starting ...")
    state, _ = env.reset()
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        action_taken += 1 
        agent.memory.push(transition(state, action, reward, next_state))
        if action_taken > 1000:
            result = agent.learn()

        state = next_state

        if agent.total_steps % 10000 == 0:
            ret = evaluate(agent)
            history['Step'].append(agent.total_steps)
            history['AvgReturn'].append(ret)
            clear_output()
            plt.figure(figsize=(8, 5))
            plt.plot(history['Step'], history['AvgReturn'], 'r-')
            plt.xlabel('Step', fontsize=16)
            plt.ylabel('AvgReturn', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(axis='y')
            plt.savefig("rewards.png")
            
            torch.save(agent.target_net.state_dict(), 'dqn.pt')

        if terminated or truncated:
            break
    print(f"Episode {episode} ends in step {action_taken} ...")

evaluate(agent)