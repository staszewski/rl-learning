import gymnasium as gym

def main():
    # 1. Create the environment
    # "render_mode='human'" allows us to see the simulation in a window
    env = gym.make("CartPole-v1", render_mode="human")

    # 2. Reset the environment to start
    observation, info = env.reset()

    print("Starting random agent...")
    
    for _ in range(100):
        # 3. Sample a random action (0 or 1)
        action = env.action_space.sample()

        # 4. Step the environment
        # observation: The new state of the world
        # reward: The reward for the action
        # terminated: True if the episode ended (e.g., pole fell)
        # truncated: True if the episode was cut short (e.g., time limit)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished, resetting...")
            observation, info = env.reset()

    env.close()
    print("Done!")

if __name__ == "__main__":
    main()

