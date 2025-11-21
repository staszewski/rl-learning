import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random


def main():
    # 1. Create the FrozenLake environment
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode=None)
    # 2. Initialize Q-Table
    state_size = env.observation_space.n  # 4x4
    action_size = env.action_space.n  # 4 up, down, left, right
    Q = np.zeros((state_size, action_size))
    # 3. Hyperparameters: total_episodes, lr (alpha), gamma, epsilon, max/min epsilon, decay_rate
    total_episodes = 1000
    lr = 0.5
    gamma = 0.95
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.005

    rewards = []
    print("Training started...")
    # 4. Training
    for episode in range(total_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # greedy policy
            rnd = random.uniform(0, 1)

            if rnd < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(Q[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # Q-learning Formula: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s,a))
            target = reward + gamma * np.max(Q[new_state, :])
            td_error = target - Q[state, action]
            Q[state, action] += lr * td_error

            state = new_state
            total_reward += reward

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_reward)

    print("Training finished.")

    # 5. Plot
    plt.plot(np.convolve(rewards, np.ones(100) / 100, mode="valid"))
    plt.xlabel("Episode")
    plt.ylabel("Average rewards - running mean")
    plt.title("Agent training progress")
    plt.show()
    # 6. Test the agent
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
    state = env.reset()
    env.render()

    done = False

    while not done:
        action = np.argmax(Q[state, :])
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    main()
