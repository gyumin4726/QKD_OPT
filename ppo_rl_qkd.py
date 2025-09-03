import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from simulator import rl_simulator

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

if torch.cuda.is_available():
    print(f"GPU 정보: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("GPU 모드로 실행합니다.")
else:
    print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")
    print("CPU 모드로 실행합니다.")

# 환경 정의
class QKDRLEnv:
    def __init__(self):
        # 기준이 될 초기값 (ga로 찾은 최적 파라미터)
        # vaccum이 0이 아님.
        # mu, nu, vac, p_mu, p_nu, p_vac, p_X, q_X
        initial_state = np.array([0.49998366, 0.22605707, 0.07603441, 0.19593278, 0.76712322, 0.0755698, 0.45442425, 0.14845363])
        
        self.initial_state = initial_state.copy() 
        
        # 탐색 범위를 결정할 작은 값
        delta = 0.02
        
        # initial_state를 기준으로 새로운 low와 high 범위 설정
        self.low = initial_state - delta
        self.high = initial_state + delta
        #self.low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #self.high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        self.param_dim = 8
        self.state = None # 초기 상태는 reset()에서 설정
        
        self.reset() # reset() 호출하여 초기 상태 설정

    def reset(self):
        # 고정된 초기값에서 시작
        self.state = self.initial_state.copy()
        # 무작위 초기화는 아래 주석 처리
        # self.state = np.random.uniform(self.low, self.high)
        
        return self.state.copy()

    def step(self, action):
        # action을 적용하여 상태 업데이트
        next_state = np.clip(self.state + action, self.low, self.high)
        self.state = next_state
        
        # 보상 계산
        reward = 10e5*rl_simulator(None, self.state, 0)
        done = False # 에피소드 종료 조건 없음
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

# 가치 네트워크 정의
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(x)

# PPO 훈련 루프
def train_ppo():
    env = QKDRLEnv()
    state_dim = env.param_dim
    action_dim = env.param_dim

    policy_net = PolicyNet(state_dim, action_dim).to(device)
    value_net = ValueNet(state_dim).to(device)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=3e-4)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)

    gamma = 0.99        # 할인 인자
    clip_param = 0.3    # PPO 클리핑 파라미터
    epochs = 1000          # episodes 수
    batch_size = 256   # 각 episode 당 step 수
    action_scale = 0.01 # action의 스케일 조정

    best_reward = -np.inf 
    best_state = None

    for episode in range(epochs):
        state = env.reset()
        log_probs, states, actions, rewards = [], [], [], []

        for t in range(batch_size):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_mean = policy_net(state_tensor).detach().cpu().numpy()[0]
            action = action_mean * action_scale + np.random.normal(0, 0.05) * action_scale
            next_state, reward, done, _ = env.step(action)

            # 최고 보상 갱신은 루프 안에서 직접 처리
            if reward > best_reward:
                best_reward = reward
                best_state = next_state.copy()
                print(f"Episode {episode}, New Best Reward: {best_reward:.10f}")

            # 로그 확률 계산 (정규분포 가정)
            log_prob = -0.5 * np.sum(((action - action_mean * action_scale) / (0.05 * action_scale)) ** 2)
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
        returns = torch.FloatTensor(returns).to(device)
        
        # 경고를 없애기 위해 리스트를 먼저 NumPy 배열로 변환
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        log_probs_old = torch.FloatTensor(np.array(log_probs)).to(device)

        # PPO 업데이트
        for _ in range(4):
            action_means_scaled = policy_net(states) * action_scale
            dist = torch.distributions.Normal(action_means_scaled, 0.05 * action_scale)
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

    # 훈련이 끝난 후 최종 결과 출력
    print("\n--- Training Finished ---")
    print(f"Highest Reward Found: {best_reward:.10f}")
    print(f"Parameters for Highest Reward: {best_state}")
    
if __name__ == "__main__":
    train_ppo()