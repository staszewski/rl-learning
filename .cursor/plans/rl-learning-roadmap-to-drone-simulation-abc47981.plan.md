<!-- abc47981-18bf-4533-bd40-49acc75a07e5 e56b806c-3175-43c0-993f-53487e0eb7cb -->
# RL Learning Roadmap to Drone Simulation (Mentorship Mode)

This plan outlines a progressive learning path where you will build RL algorithms from scratch. I will act as a mentor, providing concepts, boilerplate, and guidance, while you implement the core logic.

## Philosophy

- **Ground Up**: You write the algorithm logic (e.g., the Q-learning update equation).
- **Interactive**: I explain the concept -> You implement it -> We debug and refine together.
- **No Magic**: We avoid high-level libraries like Stable Baselines initially to ensure deep understanding.

## Phase 1: Project Setup & Foundations

- **Goal**: Get a stable environment running with `uv`.
- **Action**: I will provide the setup commands. You will run them and verify the environment works.

## Phase 2: Tabular Q-Learning (Discrete States & Actions)

- **Environment**: `FrozenLake-v1` (Simple grid world).
- **Concept**: "The Cheat Sheet" (Q-Table).
- **Your Task**: Implement the Q-Table update rule: \( Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max Q(s',a') - Q(s,a)] \).
- **Outcome**: You solve the frozen lake yourself.

## Phase 3: Deep Q-Learning (DQN)

- **Environment**: `CartPole-v1`.
- **Concept**: Replacing the Q-Table with a Neural Network.
- **Your Task**: Build a simple neural network in PyTorch and implement the training loop with Experience Replay.

## Phase 4: Continuous Control (Policy Gradients)

- **Environment**: `Pendulum-v1`.
- **Concept**: Outputting continuous values (motor torque) instead of choices.
- **Your Task**: Implement a Policy Network (Actor) that outputs actions directly.

## Phase 5: Drone Simulation

- **Environment**: 2D Drone Lander or `gym-pybullet-drones`.
- **Your Task**: Apply your continuous control agent to a drone model.

## Technical Stack

- **Language**: Python 3.10+
- **Manager**: `uv`
- **Libraries**: `gymnasium`, `numpy`, `torch`, `matplotlib`.

### To-dos

- [ ] Initialize uv project and install dependencies (gymnasium, numpy, notebook)
- [ ] Create a 'Hello World' script to run a random agent in Gymnasium
- [ ] Implement Q-Learning for FrozenLake-v1
- [ ] Visualize training results (Reward vs Episodes)