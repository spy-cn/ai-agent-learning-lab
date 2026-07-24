# CartPole 平衡控制

CartPole 是强化学习的 "Hello World"——用一根杆子保持平衡，入门 RL 的最佳实战项目。

## 环境介绍

```
CartPole-v1
├── 目标: 通过左右移动小车，使杆子尽可能长时间不倒
├── 状态 (4维): [小车位置, 小车速度, 杆子角度, 角速度]
├── 动作 (2个): [向左推, 向右推]
├── 奖励: 每步 +1（杆子未倒）
├── 终止条件:
│   ├── 杆子角度 > ±12°
│   ├── 小车位置 > ±2.4
│   └── 步数达到 500（v1版本）
└── 满分: 500分
```

```
            │ 角度θ
            │
         ┌──┴──┐
         │     │  ← 杆子
         │     │
    ─────┴─────┴─────
         └──┬──┘
            │  ← 小车
    ═══════════════════  ← 轨道
         ←  →           动作: 左/右
```

## 方法对比

| 方法 | 预期分数 | 训练时间 | 难度 |
|------|----------|----------|------|
| Q-Learning (离散化) | ~150 | 快 | ★★☆ |
| DQN | ~500 | 中 | ★★★ |
| PPO | ~500 | 中 | ★★★ |
| SAC (连续动作版) | ~500 | 慢 | ★★★★ |

## 1. 基线：随机策略

```python
import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v1')

def random_policy():
    """随机策略基线"""
    scores = []
    for episode in range(10):
        obs, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        scores.append(total_reward)
    
    print(f"随机策略平均分: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    return scores

random_policy()
# 随机策略平均分: 22.5 ± 8.3
```

## 2. DQN 解决 CartPole

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=10000)
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = 10
        self.step_count = 0
    
    def select_action(self, state):
        """ε-greedy 策略"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax().item()
    
    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 当前 Q 值
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 目标 Q 值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ε 衰减
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # 目标网络更新
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()

def train_dqn(episodes=500):
    env = gym.make('CartPole-v1')
    agent = DQNAgent()
    scores = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.buffer.push(obs, action, reward, next_obs, done)
            agent.update()
            
            obs = next_obs
            total_reward += reward
        
        scores.append(total_reward)
        
        if (ep + 1) % 50 == 0:
            avg = np.mean(scores[-50:])
            print(f"Episode {ep+1:4d} | Score: {total_reward:7.1f} | "
                  f"Avg(50): {avg:7.1f} | ε: {agent.epsilon:.3f}")
    
    env.close()
    return agent, scores

# 训练
agent, scores = train_dqn(episodes=300)
```

## 3. 使用 Stable-Baselines3 快速解决

```python
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# 创建环境
env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

# 创建 PPO 模型
model = PPO(
    "MlpPolicy", 
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./logs/cartpole/"
)

# 训练
model.learn(total_timesteps=50000)

# 评估
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"\n评估结果: {mean_reward:.1f} ± {std_reward:.1f}")

# 保存模型
model.save("ppo_cartpole")

env.close()
```

## 4. 可视化训练过程

```python
import matplotlib.pyplot as plt

def plot_training(scores, window=50):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 原始分数
    axes[0].plot(scores, alpha=0.3, color='blue', label='Score')
    
    # 滑动平均
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(scores)), moving_avg, 
                     color='red', linewidth=2, label=f'Moving Avg ({window})')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ε 衰减曲线（示意）
    epsilons = [max(0.01, 1.0 * (0.995 ** i)) for i in range(len(scores))]
    axes[1].plot(epsilons, color='green')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Epsilon')
    axes[1].set_title('Epsilon Decay')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cartpole_training.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_training(scores)
```

## 5. 录制训练视频

```python
from gymnasium.wrappers import RecordVideo

def record_evaluation(model_path="ppo_cartpole"):
    """录制评估视频"""
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = RecordVideo(env, video_folder='./videos/', 
                      episode_trigger=lambda x: x == 0)
    env = DummyVecEnv([lambda: env])
    
    model = PPO.load(model_path, env=env)
    
    obs = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if done:
            break
    
    env.close()
    print("视频已保存到 ./videos/")

record_evaluation()
```

## 超参数调优建议

```
CartPole DQN 调优清单:
┌──────────────────────────────────────────────────┐
│ 学习率 (lr):        1e-4 ~ 1e-3                 │
│ 折扣因子 (γ):       0.95 ~ 0.99                 │
│ Buffer 大小:        10000 ~ 50000               │
│ Batch Size:         32 ~ 128                    │
│ 目标网络更新频率:    5 ~ 20 步                   │
│ ε 衰减:             0.99 ~ 0.999                │
│ 隐藏层:             [64, 64] ~ [256, 256]       │
│ 激活函数:           ReLU                        │
│                                                  │
│ 收敛判据: 连续 100 episode 平均分 > 475          │
└──────────────────────────────────────────────────┘
```

## 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 分数不增长 | 学习率过大/过小 | 尝试 1e-4、3e-4、1e-3 |
| 分数达到后骤降 | Q值过估计 | 使用 Double DQN |
| 训练不稳定 | buffer 太小 | 增大 replay buffer |
| 收敛后分数波动 | ε 太小 | 保持 ε_min ≥ 0.01 |
| GPU 不工作 | tensor 未移到 GPU | 检查 .to(device) |

## 小结

CartPole 是验证 RL 算法正确性的标准测试环境：

- **快速验证**: DQN/PPO 可在几分钟内达到满分
- **调试工具**: SB3 一行代码对比基线
- **过渡跳板**: 从 CartPole → Atari → MuJoCo

---

| [← 回到目录](../README.md) | [上一章：自博弈与种群方法](../08-多智能体强化学习篇/05-自博弈与种群方法.md) | [下一章：Atari游戏通关](02-Atari游戏通关.md) |
|---|---|---|
