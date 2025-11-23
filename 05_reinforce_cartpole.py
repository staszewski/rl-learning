import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

GYM_ENV = "CartPole-v1"


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        return torch.softmax(self.fc2(x), dim=1)


def main():
    env = gym.make(GYM_ENV)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    episodes = 500
    epsilon = 1e-9
    gamma = 0.98
    lr = 0.01

    policy_net = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        rewards_history = []
        log_probs = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            output = policy_net(state_tensor)
            out_distribution = Categorical(output)

            action = out_distribution.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            state = next_state

            log_probs.append(out_distribution.log_prob(action))
            rewards_history.append(reward)

            done = terminated or truncated

        R = 0
        returns = []
        for r in reversed(rewards_history):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + epsilon)

        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards_history)
        total_rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}, total reward {total_reward}")

        if total_reward >= 500:
            print(f"Solved at episode {episode}")
            break

    print("Training done")

    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Training progress")
    plt.show()


if __name__ == "__main__":
    main()
