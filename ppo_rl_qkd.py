import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from simulator import rl_simulator

# 환경 정의
class QKDRLEnv:
    def __init__(self):
        # [mu, nu, vac, p_mu, p_nu, p_vac, p_X, q_X]
        self.low = np.array([0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4])
        self.high = np.array([0.9, 0.9, 0.1, 1.0, 1.0, 1.0, 0.6, 0.6])
        self.param_dim = 8
        self.reset()

    def reset(self):
        # vac(2번째 인덱스)는 0으로 고정
        state = np.random.uniform(self.low, self.high)
        #state[2] = 0.0
        # 확률(p_mu, p_nu, p_vac) 정규화
        #probs = state[3:6]
        #probs = probs / np.sum(probs)
        #state[3:6] = probs
        self.state = state
        return self.state.copy()

    def step(self, action):
        # vac(2번째 인덱스)는 항상 0으로 유지
        next_state = np.clip(self.state + action, self.low, self.high)
        #next_state[2] = 0.0
        # 확률 정규화
        #probs = next_state[3:6]
        #probs = probs / np.sum(probs)
        #next_state[3:6] = probs
        self.state = next_state
        reward = rl_simulator(None, self.state, 0)
        done = False
        return self.state.copy(), reward, done, {}

# PPO 네트워크 정의
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(x)

def train_ppo():
    env = QKDRLEnv()
    state_dim = env.param_dim
    action_dim = env.param_dim

    policy_net = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=3e-4)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)

    gamma = 0.99
    clip_param = 0.2
    epochs = 5000
    batch_size = 32

    for episode in range(epochs):
        state = env.reset()
        log_probs, states, actions, rewards = [], [], [], []

        for t in range(batch_size):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_mean = policy_net(state_tensor).detach().numpy()[0]
            action = np.random.normal(action_mean, 0.05)
            next_state, reward, done, _ = env.step(action)

            # 로그 확률 계산 (정규분포 가정)
            log_prob = -0.5 * np.sum(((action - action_mean) / 0.05) ** 2)
            log_probs.append(log_prob)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # Advantage 계산
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs)

        # PPO 업데이트
        for _ in range(4):
            action_means = policy_net(states)
            dist = torch.distributions.Normal(action_means, 0.05)
            log_probs_new = dist.log_prob(actions).sum(dim=1)
            ratios = torch.exp(log_probs_new - log_probs_old)
            advantages = returns - value_net(states).squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            value_loss = (returns - value_net(states).squeeze()).pow(2).mean()
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Best Reward (SKR): {np.max(rewards):.10f}")

if __name__ == "__main__":
    train_ppo()