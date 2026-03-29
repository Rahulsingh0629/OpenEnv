import numpy as np
import torch
import torch.nn as nn

# ================= ENV =================
class StrategyGameEnv:
    def __init__(self, grid_size=10, num_agents=2):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))

        self.agent_positions = {
            i: [np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)]
            for i in range(self.num_agents)
        }

        self.resources = [
            [np.random.randint(0, self.grid_size),
             np.random.randint(0, self.grid_size)]
            for _ in range(5)
        ]

        self.agent_health = {i: 100 for i in range(self.num_agents)}
        return self.get_observations()

    def get_observations(self):
        obs = {}
        for i, pos in self.agent_positions.items():
            x, y = pos

            # FIX: Always return 5x5 (pad if needed)
            view = np.zeros((5, 5))

            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        view[dx + 2, dy + 2] = self.grid[nx, ny]

            obs[i] = view

        return obs

    def step(self, actions):
        rewards = {i: 0 for i in range(self.num_agents)}

        for agent_id, action in actions.items():
            x, y = self.agent_positions[agent_id]

            # Move
            if action == 0: x -= 1
            elif action == 1: x += 1
            elif action == 2: y -= 1
            elif action == 3: y += 1

            # FIX: correct clipping
            x = np.clip(x, 0, self.grid_size - 1)
            y = np.clip(y, 0, self.grid_size - 1)

            self.agent_positions[agent_id] = [x, y]

            # Resource collection
            for r in self.resources:
                if [x, y] == r:
                    rewards[agent_id] += 10

            # Combat
            for other_id, other_pos in self.agent_positions.items():
                if other_id != agent_id and other_pos == [x, y]:
                    self.agent_health[other_id] -= 10
                    rewards[agent_id] += 5

        done = any(h <= 0 for h in self.agent_health.values())
        return self.get_observations(), rewards, done, {}

# ================= MODEL =================
class MultiAgentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.net(x)

# ================= ACTION =================
def select_actions(model, observations):
    actions = {}
    for agent_id, obs in observations.items():
        obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32)
        logits = model(obs_tensor)
        action = torch.argmax(logits).item()
        actions[agent_id] = action
    return actions

# ================= TRAIN =================
env = StrategyGameEnv()
model = MultiAgentNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for episode in range(50):
    obs = env.reset()

    for step in range(30):
        actions = select_actions(model, obs)
        next_obs, rewards, done, _ = env.step(actions)

        loss = 0

        for agent_id in rewards:
            reward = torch.tensor([rewards[agent_id]], dtype=torch.float32)

            pred = model(
                torch.tensor(obs[agent_id].flatten(), dtype=torch.float32)
            )[0]

            loss = loss + loss_fn(pred, reward)

        optimizer.zero_grad()
        loss.backward() # type: ignore
        optimizer.step()

        obs = next_obs

        if done:
            break

    print(f"Episode {episode} completed")