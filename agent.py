import torch
import random
import numpy as np
import copy
import torch.nn as nn

from config import *

class Agent:
    def __init__(self, model, optimizer, memory, device=None):

        self.device = device if device is not None else torch.device("cpu")
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)

        self.optimizer = optimizer
        self.memory = memory
        self.epsilon = EPSILON
        self.device = device

        self.loss_fn = nn.MSELoss()

        
        self.update_target_every = 50
        self.step_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)

        return torch.argmax(q_values).item()

    def train(self):
        if self.memory.size() < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        # 🔥 Convert batch to tensors
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

       
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            targets = rewards + GAMMA * max_next_q * (1 - dones)

       
        loss = self.loss_fn(q_values, targets)

       
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

       
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)