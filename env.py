import numpy as np
from config import GRID_SIZE, NUM_AGENTS

WIN_SCORE = 50

class StrategyGameEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.num_agents = NUM_AGENTS
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

        self.agent_scores = {i: 0 for i in range(self.num_agents)}

        return self.get_observations()

    def get_observations(self):
        obs = {}

        for i, pos in self.agent_positions.items():
            x, y = pos
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

            if action == 0: x -= 1
            elif action == 1: x += 1
            elif action == 2: y -= 1
            elif action == 3: y += 1

            x = np.clip(x, 0, self.grid_size - 1)
            y = np.clip(y, 0, self.grid_size - 1)

            self.agent_positions[agent_id] = [x, y]

        new_resources = []
        for r in self.resources:
            collected = False

            for agent_id, pos in self.agent_positions.items():
                if pos == r:
                    rewards[agent_id] += 10
                    self.agent_scores[agent_id] += 10
                    collected = True

            if not collected:
                new_resources.append(r)

        self.resources = new_resources

        for i, pos_i in self.agent_positions.items():
            for j, pos_j in self.agent_positions.items():
                if i != j and pos_i == pos_j:
                    self.agent_health[j] -= 10
                    rewards[i] += 5

        done, winner = self.check_game_over()

        return self.get_observations(), rewards, done, {"winner": winner}

    def check_game_over(self):
        for agent_id, score in self.agent_scores.items():
            if score >= WIN_SCORE:
                return True, agent_id

        for agent_id, health in self.agent_health.items():
            if health <= 0:
                return True, 1 - agent_id

        return False, None