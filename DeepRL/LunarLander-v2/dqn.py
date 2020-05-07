import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import time
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, buffer_size):
        return random.sample(self.memory, buffer_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size * 2)
        self.hidden2 = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.001)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        actions = self.output_layer(x)

        return actions

class Brain(object):

    def __init__(self, input_size, hidden_size, output_size,
            buffer_size, gamma, EPS, EPS_START, EPS_END, DECAY_RATE):

        self.n_actions = output_size

        # Constants
        self.BUFFER_SIZE = buffer_size
        self.GAMMA = gamma
        self.EPS = {'EPS': EPS, 'START': EPS_START, 'END': EPS_END, 'DECAY': DECAY_RATE}

        # Network Initialization
        self.DQN = DQN(input_size, hidden_size, output_size)
        print("Model Summary\n", self.DQN)

        # Memory
        self.memory = ReplayMemory(1000000)

    def store_transitions(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def update_eps(self, episode):
        if self.EPS['END'] >= episode >= self.EPS['START']:
                self.EPS['EPS'] -= self.EPS['DECAY']

    def select_action(self, state):
        state = torch.tensor(state)
        sample = random.random()

        if sample > self.EPS['EPS']:
            with torch.no_grad():
                return torch.argmax(self.DQN(state)).item()
        else:
            return random.randrange(self.n_actions)

    def optimize_model(self):
        if len(self.memory) < self.BUFFER_SIZE:
            return
        
        self.DQN.optimizer.zero_grad()

        transitions = self.memory.sample(self.BUFFER_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.int64)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)

        non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_terminal_state = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32)

        # Q - Learning : Q*(s, a) = E [ r + GAMMA * max(Q(s', a'))]

        # Q*(s, a)
        state_action_values = self.DQN(state_batch)
        target_state_action_values = state_action_values.clone()

        next_state_action_values = torch.zeros(self.BUFFER_SIZE)
        next_state_action_values[non_terminal_mask] = self.DQN(non_terminal_state).max(1)[0]
        
        batch_index = torch.arange(0, self.BUFFER_SIZE, dtype=torch.int64)
        target_state_action_values[batch_index, action_batch] = reward_batch + next_state_action_values * self.GAMMA

        loss = self.DQN.loss(target_state_action_values, state_action_values)
        
        loss.backward()
        # print("Loss:", loss.data)
        self.DQN.optimizer.step()





