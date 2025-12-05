import random
from utils.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, List


class DuelingNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.feature_layer = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU())

        self.advantage_layer = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim))

        self.value_layer = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(x)

        # both layers take feature as an input
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q


class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        seed: int,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.98,
    ) -> None:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # nn's
        self.dqn = DuelingNetwork(obs_dim, action_dim).to(self.device)
        self.dqn_target = DuelingNetwork(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters())

        self.transition = list()

        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(self.dqn.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def train(self, num_frames: int):
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        update_cnt, score = 0, 0
        epsilons, losses, scores = [], [], []

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            if done:
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                self.epsilon = max(
                    self.min_epsilon,
                    self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                )
                epsilons.append(self.epsilon)

                if update_cnt % self.target_update == 0:
                    # target update
                    self.dqn_target.load_state_dict(self.dqn.state_dict())

        self._plot(frame_idx, scores, losses, epsilons)
        self.env.close()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        current_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done

        target = (reward + self.gamma * next_q_value * mask).to(device)

        return F.smooth_l1_loss(current_q_value, target)

    def _plot(
        self,
        frame_idx: int,
        scores: List[float],
        losses: List[float],
        epsilons: List[float],
    ):
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title("frame %s. score: %s" % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title("loss")
        plt.plot(losses)
        plt.subplot(133)
        plt.title("epsilons")
        plt.plot(epsilons)
        plt.show()


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

    env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")
    num_frames = 10000
    memory_size = 1000
    batch_size = 32
    target_update = 100
    epsilon_decay = 1 / 2000

    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, seed)

    agent.train(num_frames)


if __name__ == "__main__":
    main()
