import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # Define a simple network:
        # Input -> Hidden Layer (64 neurons) -> ReLU -> Hidden Layer (64 neurons) -> ReLU -> Output
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def main():
    # 1. Environment
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 2. Hyperparameters
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.0001
    target_update = 20

    # 3. Components
    policy_net = QNetwork(state_size, action_size)
    target_net = QNetwork(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target net is just for reference, not training

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer(10000)

    print("Training DQN...")

    episodes = 700
    rewards_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)  # Convert to tensor
        total_reward = 0
        done = False

        while not done:
            # 1. Action Selection
            if random.random() > epsilon:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()
            else:
                action = env.action_space.sample()

            # 2. Step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_tensor = torch.FloatTensor(next_state)

            # 3. Store Experience
            memory.push(state.numpy(), action, reward, next_state, done)

            state = next_state_tensor
            total_reward += reward

            # 4. Train
            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                batch_state = torch.FloatTensor(np.array(batch_state))
                batch_action = torch.LongTensor(batch_action).unsqueeze(1)  # Shape (64, 1)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
                batch_next_state = torch.FloatTensor(np.array(batch_next_state))
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1)

                # Current Q: Q(s, a)
                current_q = policy_net(batch_state).gather(1, batch_action)

                # Max Next Q: max(Q(s', a'))
                next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)

                # Target: R + gamma * max_Q * (1 - done)
                expected_q = batch_reward + (gamma * next_q * (1 - batch_done))

                criterion = nn.MSELoss()
                loss = criterion(current_q, expected_q)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(total_reward)

        if len(rewards_history) >= 10:
            avg_reward = np.mean(rewards_history[-10:])
            if avg_reward >= 475:
                print(f"Solved at episode {episode}. Avg reward: {avg_reward}")
                break

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    print("Training done")
    # 5. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Training progress")
    plt.show()
    torch.save(policy_net.state_dict(), "dqn_cartpole.pth")
    print("Model saved")


def evaluate():
    env = gym.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = QNetwork(state_size, action_size)
    try:
        model.load_state_dict(torch.load("dqn_cartpole.pth"))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: dqn_cartpole.pth not found. Train the model first.")
        return

    model.eval()

    episodes = 5
    for i in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0
        done = False

        while not done:
            env.render()

            with torch.no_grad():
                # Select the best action (argmax)
                action = model(state).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = torch.FloatTensor(next_state)
            total_reward += reward

        print(f"Episode {i + 1}: Total Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    # main()
    evaluate()

