import torch
from env import StrategyGameEnv
from model import DQN
from memory import ReplayBuffer
from agent import Agent
from config import *
from utils import flatten_obs
from render import Renderer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = StrategyGameEnv()
model = DQN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

agent = Agent(model, optimizer, memory, device=device)
renderer = Renderer(GRID_SIZE)

episode_rewards = []

for episode in range(EPISODES):
    obs = flatten_obs(env.reset())
    total_reward = 0

    for step in range(MAX_STEPS):
        renderer.handle_events()

        actions = {}

        for agent_id in obs:
            if agent_id == 0:
               
                actions[agent_id] = renderer.get_human_action()
            else:
                
                actions[agent_id] = agent.select_action(obs[agent_id])

        next_obs, rewards, done, _ = env.step(actions)
        next_obs = flatten_obs(next_obs)

       
        for agent_id in obs:
            memory.store((
                obs[agent_id],
                actions[agent_id],
                rewards[agent_id],
                next_obs[agent_id],
                done
            ))

        
        agent.train()

        obs = next_obs
        total_reward += sum(rewards.values())

       
        renderer.draw(env, episode=episode, rewards=rewards)

        done, winner = env.check_game_over()

        if done:
            renderer.draw(env, episode, rewards)
            renderer.show_winner(winner)
            break

    episode_rewards.append(total_reward)

    print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")