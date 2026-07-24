# DQN及其变体

DQN（Deep Q-Network）将深度学习引入Q-Learning，是深度强化学习的里程碑。

## DQN核心思想

```
传统Q-Learning: Q(s,a)用表格存储
                 ↓ 状态空间太大时不可行
DQN:           Q(s,a;θ)用神经网络近似
                 ↓ 可以处理连续/高维状态空间

关键创新:
1. Experience Replay（经验回放）
2. Target Network（目标网络）
3. Frame Stacking（帧堆叠）
```

## 网络结构

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    """Q网络：状态→各动作的Q值"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
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


# 对于图像输入（如Atari）
class ConvQNetwork(nn.Module):
    """CNN Q网络：用于图像状态（Atari）"""
    
    def __init__(self, input_channels, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def forward(self, x):
        x = x / 255.0  # 归一化
        conv_out = self.conv(x)
        return self.fc(conv_out)
```

## 经验回放（Experience Replay）

```python
from collections import deque
import random

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
```

## 完整DQN实现

```python
class DQN:
    """完整的DQN算法"""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 buffer_size=100000, batch_size=64, 
                 target_update_freq=10, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # ε-greedy参数
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # 两个网络：在线网络 + 目标网络
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # 目标网络只用于推理
        
        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.train_step = 0
    
    def act(self, state, training=True):
        """ε-greedy选择动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            action = q_values.argmax().item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def learn(self):
        """DQN核心学习步骤"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 1. 从经验回放中采样
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 2. 计算当前Q值: Q(s,a)
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 3. 计算目标Q值: r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 4. MSE损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 5. 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（稳定训练）
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        # 6. 衰减epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # 7. 定期更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()
    
    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.epsilon = checkpoint['epsilon']


# ========== 训练DQN ==========
def train_dqn(env_name="CartPole-v1", n_episodes=500):
    import gymnasium as gym
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQN(state_dim, action_dim, lr=1e-3, gamma=0.99)
    
    scores = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 1. 选择动作
            action = agent.act(state)
            
            # 2. 执行动作
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            # 3. 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 4. 学习
            agent.learn()
            
            state = next_state
            episode_reward += reward
        
        scores.append(episode_reward)
        avg_score = np.mean(scores[-100:])
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{n_episodes}, "
                  f"Score: {episode_reward:.0f}, "
                  f"Avg: {avg_score:.1f}, "
                  f"ε: {agent.epsilon:.3f}")
        
        # 提前终止
        if avg_score >= 475:
            print(f"环境解决于第{episode+1}回合！")
            break
    
    agent.save("dqn_cartpole.pth")
    env.close()
    return agent, scores

# agent, scores = train_dqn()
```

## DQN的问题与改进

### 问题1：过估计偏差（Overestimation Bias）

Q-Learning的max操作会导致Q值被系统性高估：

```
max(E[X]) ≤ E[max(X)]  ← Jensen不等式

如果Q值有噪声:
  E[max(Q)] > max(E[Q])
  → 目标值被高估 → 学到的Q值偏高
```

## Double DQN

分离动作选择和动作评估：

```
DQN:          Y = r + γ max_a' Q_target(s', a')
              └── 选择和评估都用target_net ──┘

Double DQN:   Y = r + γ Q_target(s', argmax_a' Q_online(s', a'))
              └── online选动作 ──┘  └── target评估 ──┘
```

```python
class DoubleDQN(DQN):
    """Double DQN: 减少过估计"""
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # === Double DQN的核心改进 ===
        with torch.no_grad():
            # 用online网络选择动作
            next_actions = self.q_net(next_states).argmax(dim=1)
            # 用target网络评估Q值
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        # === 改进结束 ===
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()
```

## Dueling DQN

将Q值分解为状态值和优势：

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')$$

```python
class DuelingQNetwork(nn.Module):
    """Dueling架构: V(s) + A(s,a)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 状态值分支 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势分支 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        
        value = self.value_stream(features)          # [batch, 1]
        advantage = self.advantage_stream(features)   # [batch, action_dim]
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q
```

## Prioritized Experience Replay (PER)

优先回放重要的经验：

```python
class PrioritizedReplayBuffer:
    """优先经验回放"""
    
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级程度
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
    
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """按优先级采样"""
        if len(self.buffer) == self.capacity:
            probs = self.priorities
        else:
            probs = self.priorities[:self.pos]
        
        probs = probs ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        
        # 重要采样权重（修正偏差）
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """更新优先级（用TD误差）"""
        for i, p in zip(indices, priorities):
            self.priorities[i] = p
```

## Rainbow DQN

Rainbow集成了6种DQN改进：

| 改进 | 效果 |
|------|------|
| **Double DQN** | 减少过估计 |
| **Dueling DQN** | 更好的值函数分解 |
| **Prioritized Replay** | 高效利用重要经验 |
| **N-step Returns** | 加速传播奖励 |
| **Distributional RL (C51)** | 学习回报分布 |
| **Noisy Nets** | 智能探索替代ε-greedy |

## Noisy Networks

替代ε-greedy的智能探索：

```python
class NoisyLinear(nn.Module):
    """带参数噪声的线性层"""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 可学习的均值和标准差
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_eps', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_eps', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(epsilon_out.ger(epsilon_in))
        self.bias_eps.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)
```

## DQN变体总结

| 变体 | 核心改进 | 效果 |
|------|----------|------|
| **Double DQN** | 分离选择和评估 | 减少过估计 |
| **Dueling DQN** | V+A分解 | 更高效学习 |
| **PER** | 优先回放 | 样本效率高 |
| **N-step DQN** | 多步回报 | 更快传播奖励 |
| **C51** | 分布式RL | 学习回报分布 |
| **Noisy DQN** | 参数噪声探索 | 智能探索 |
| **Rainbow** | 全部集成 | 最佳性能 |

---

[← 上一篇：时序差分学习](03-时序差分学习.md) | [下一篇：值函数近似 →](05-值函数近似.md)
