# 🧠 Multi-Agent RTS AI Environment (OpenEnv Project)

## 🚀 Overview

This project is a **multi-agent reinforcement learning RTS environment** where agents compete for resources and survival.

It combines:

* 🎮 Game simulation
* 🤖 AI learning (DQN)
* 🧠 Multi-agent interaction
* 📊 Real-time visualization

---

## 🎯 Features

* Multi-agent environment
* Human vs AI gameplay
* Deep Q-Network (DQN)
* Target Network stabilization
* Real-time Pygame UI
* Reward tracking system
* Win/Lose conditions

---

## 🏗️ Architecture

* `env.py` → Game logic
* `agent.py` → AI agent
* `model.py` → Neural network
* `render.py` → UI engine
* `main.py` → Training loop

*         ┌──────────────────────┐
        │      Pygame UI       │
        │ (render.py)          │
        └─────────┬────────────┘
                  │
        ┌─────────▼────────────┐
        │   Environment        │
        │ (env.py)             │
        └─────────┬────────────┘
                  │
        ┌─────────▼────────────┐
        │      Agent           │
        │ (DQN / PPO)          │
        └─────────┬────────────┘
                  │
        ┌─────────▼────────────┐
        │   Training Loop      │
        │ (main.py)            │
        └──────────────────────┘

---

## 🎮 Controls

* Arrow Keys → Move player
* AI → autonomous learning

---

## ▶️ Run Project

```bash
pip install pygame torch
python main.py
```

---

## 🧠 Use Cases

* Reinforcement learning research
* Multi-agent systems
* Game AI development
* Open environment simulations

---

## 🔥 Future Work

* PPO implementation
* Web-based UI
* Multi-player mode
* Advanced RTS mechanics

---

## 🏆 Hackathon Ready

This project demonstrates a full-stack AI system combining simulation, learning, and visualization.
