# TD3：双延迟DDPG

TD3（Twin Delayed Deep Deterministic Policy Gradient）是DDPG的改进版，通过三个关键技巧解决了DDPG的Q值过估计和训练不稳定问题。

## DDPG的问题

```
DDPG的Q值过估计:
  target = r + γ · max_a' Q_target(s', a')
                        ↑ max操作导致系统性高估

  当Q值有噪声时:
  E[max(Q)] > max(E[Q])  ← Jensen不等式

  → Q值被高估 → Actor追逐错误目标 → 性能崩溃
```

## TD3的三个改进

```
┌──────────────────────────────────────────────────┐
│              TD3 的三大改进                       │
├──────────────────────────────────────────────────┤
│                                                  │
│  1. Clipped Double-Q (Twin Q)                    │
│     用两个Q网络取最小值 → 减少过估计              │
│                                                  │
│  2. Delayed Policy Updates                        │
│     Actor更新频率低于Critic → 更稳定             │
│                                                  │
│  3. Target Policy Smoothing                       │
│     给目标动作加噪声 → 防止利用Q函数尖峰         │
│                                                  │
└──────────────────────────────────────────────────┘
```

## 改进详解

### 1. Clipped Double-Q

```python
# DDPG: target = r + γ Q_target(s', μ(s'))
# TD3:  target = r + γ min(Q1_target, Q2_target)(s', μ(s'))

# 用两个独立训练的Q网络取最小值
# 减缓max操作带来的过估计

def compute_target(reward, gamma, q1_next, q2_next, done):
    """Clipped Double Q-learning"""
    min_q = torch.min(q1_next, q2_next)  # 取较小值
    target = reward + gamma * (1 - done) * min_q
    return target
```

### 2. Delayed Policy Updates

```python
# Critic每步更新
# Actor每d步更新（通常d=2）

# 为什么？因为Q值还不准时，更新Actor没意义
for step in range(n_steps):
    update_critic()
    if step % policy_update_freq == 0:  # 每2步
        update_actor()
        soft_update_targets()
```

### 3. Target Policy Smoothing

```python
# 给目标策略的动作加噪声
# 防止Actor利用Q函数的错误尖峰

# DDPG: a' = μ(s')
# TD3:  a' = μ(s') + clip(noise, -c, c)

def smooth_target_action(actor_target, next_state, noise_std=0.2, noise_clip=0.5):
    """目标策略平滑"""
    action = actor_target(next_state)
    noise = (torch.randn_like(action) * noise_std).clamp(-noise_clip, noise_clip)
    action = (action + noise).clamp(-max_action, max_action)
    return action
```

```
Q值景观（有噪声尖峰）:
     Q
     │        ↑ 尖峰（错误的高Q值）
     │       ╱│╲
     │      ╱ │ ╲
     │─────╱  │  ╲─────
     │   ←平滑后→
     │  目标动作加噪声
     └──────────────→ a

不加噪声: Actor可能学到尖峰处的动作（错误）
加噪声: 平均化Q值，更鲁棒
```

## 完整TD3实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class TD3:
    """完整的TD3算法"""
    
    def __init__(self, state_dim, action_dim, max_action,
                 lr=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, 
                 policy_freq=2, actor_hidden=256, critic_hidden=256):
        
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        
        # === Actor ===
        self.actor = Actor(state_dim, action_dim, max_action, actor_hidden)
        self.actor_target = Actor(state_dim, action_dim, max_action, actor_hidden)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # === Twin Critics ===
        self.critic = TwinCritic(state_dim, action_dim, critic_hidden)
        self.critic_target = TwinCritic(state_dim, action_dim, critic_hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state)
        return action.numpy()[0]
    
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        
        # 采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # === 1. Critic更新 ===
        with torch.no_grad():
            # 改进3: Target Policy Smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (
                self.actor_target(next_states) + noise
            ).clamp(-self.max_action, self.max_action)
            
            # 改进1: Clipped Double-Q
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            target_q = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        # 当前Q值
        current_q1, current_q2 = self.critic(states, actions)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # === 2. Actor更新（延迟） ===
        if self.total_it % self.policy_freq == 0:
            # 改进2: Delayed Update
            actor_actions = self.actor(states)
            q1, _ = self.critic(states, actor_actions)
            actor_loss = -q1.mean()  # 最大化Q值
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item()


class Actor(nn.Module):
    """确定性Actor"""
    
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.max_action * self.net(state)


class TwinCritic(nn.Module):
    """双Q网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2（独立参数）
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
```

## TD3的探索策略

TD3使用简单的高斯噪声（而非DDPG的OU噪声）：

```python
class TD3Exploration:
    """TD3探索策略"""
    
    def __init__(self, action_dim, expl_noise=0.1):
        self.action_dim = action_dim
        self.expl_noise = expl_noise
    
    def get_action(self, agent, state, max_action):
        """动作 + 高斯噪声"""
        action = agent.select_action(state)
        if self.expl_noise > 0:
            noise = np.random.normal(0, self.expl_noise, size=self.action_dim)
            action = (action + noise).clip(-max_action, max_action)
        return action
```

## DDPG vs TD3 vs SAC

| 特性 | DDPG | TD3 | SAC |
|------|------|-----|-----|
| **Q网络** | 单Q | Twin Q | Twin Q |
| **策略** | 确定性 | 确定性 | 随机性 |
| **探索** | OU噪声 | 高斯噪声 | 最大熵 |
| **Actor更新** | 每步 | 每2步 | 每步 |
| **目标平滑** | 无 | 有 | 内置 |
| **过估计** | 严重 | 缓解 | 缓解 |
| **稳定性** | 差 | 好 | 最好 |
| **性能** | 基准 | 好 | 最佳 |

## TD3的超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `lr` | 3e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | 软更新速率 |
| `policy_noise` | 0.2 | 目标平滑噪声 |
| `noise_clip` | 0.5 | 噪声裁剪范围 |
| `policy_freq` | 2 | Actor更新频率 |
| `expl_noise` | 0.1 | 探索噪声标准差 |
| `batch_size` | 256 | 批大小 |

## 小结

TD3的三个改进解决了DDPG的核心问题：
- **Twin Q**: 缓解Q值过估计
- **延迟更新**: 让Actor等Critic学好再更新
- **目标平滑**: 防止利用Q函数缺陷

| 技巧 | 效果 |
|------|------|
| **Clipped Double-Q** | min(Q1, Q2) 减少过估计 |
| **Delayed Update** | 每d步更新Actor |
| **Target Smoothing** | 目标动作加噪声 |

---

[← 上一篇：SAC柔性Actor-Critic](03-SAC柔性Actor-Critic.md) | [下一篇：连续控制算法对比 →](05-连续控制算法对比.md)
