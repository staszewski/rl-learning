import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import matplotlib.pyplot as plt
import random
from typing import List, Tuple


GYM_ENV = "Pendulum-v1"
STATE_DICT = "a2c_pendulum.pth"
EPS = 1e-5


def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.mu_layer = nn.Linear(128, out_dim)
        self.log_std_layer = nn.Linear(128, out_dim)

        initialize_uniformly(self.mu_layer)
        initialize_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.hidden1(state))

        mu = torch.tanh(self.mu_layer(x)) * 2
        log_std = F.softplus(self.log_std_layer(x))
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.out = nn.Linear(128, 1)

        initialize_uniformly(self.out)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.hidden1(state))
        return self.out(x)


class A2CAgent:
    def __init__(self, env: gym.Env, gamma: float, entropy_weight: float, seed: int = 777):
        self.env = env
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device", self.device)

        # nn's
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.transition: list = list()
        self.total_step = 0
        self.is_test = False

    def select_action(self, state: np.ndarray):
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend(([next_state, reward, done]))

        return next_state, reward, done

    def update_model(self):
        state, log_prob, next_state, reward, done = self.transition

        mask = 1 - done
        next_state = torch.FloatTensor(next_state).to(self.device)

        predicted_value = self.critic(state)
        target_value = reward + self.gamma * self.critic(next_state) * mask

        value_loss = F.smooth_l1_loss(predicted_value, target_value.detach())

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        advantage = (target_value - predicted_value).detach()
        policy_loss = -advantage * log_prob
        policy_loss += self.entropy_weight * -log_prob

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False

        actor_losses, critic_losses, scores = [], [], []
        state, _ = self.env.reset(seed=self.seed)
        score = 0

        for self.total_step in range(1, num_frames + 1):
            print(self.total_step)
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            actor_loss, critic_loss = self.update_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            state = next_state
            score += reward

            if done:
                state, _ = self.env.reset()
                print(score)
                scores.append(score)
                score = 0

        self._plot(self.total_step, scores, actor_losses, critic_losses)
        self.save_model(STATE_DICT)
        self.env.close()

    def save_model(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)
        print(f"Model saved {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        print(f"Model loaded from {path}")

    def _plot(
        self, frame_idx: int, scores: List[float], actor_losses: List[float], critic_losses: List[float]
    ):
        subplot_params = [
            (131, f"frame {frame_idx}, score: {np.mean(scores[-10:])}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        plt.show()


def evaluate():
    env = gym.make(GYM_ENV, render_mode="human")

    gamma = 0.9
    entropy_weight = 1e-2
    agent = A2CAgent(env, gamma, entropy_weight)

    try:
        agent.load_model(STATE_DICT)
    except FileNotFoundError:
        print(f"Error, file not found {STATE_DICT}")
        return

    agent.is_test = True
    agent.actor.eval()
    agent.critic.eval()

    episodes = 5
    for i in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            env.render()

            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            step_count += 1

        print(f"Episode {i + 1}, total reward = {total_reward:.2f}, Steps = {step_count}")


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main():
    seed = 777
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)

    num_frames = 100000
    gamma = 0.9
    entropy_weight = 1e-2

    env = gym.make(GYM_ENV, render_mode="rgb_array")
    agent = A2CAgent(env, gamma, entropy_weight, seed)

    agent.train(num_frames)


if __name__ == "__main__":
    main()
    # evaluate()
