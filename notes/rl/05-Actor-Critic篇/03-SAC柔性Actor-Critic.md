# SAC：柔性Actor-Critic

SAC（Soft Actor-Critic）是最先进的连续控制RL算法，基于**最大熵强化学习**框架。

## 核心思想

```
标准RL目标: 最大化期望回报
  J = E[Σ γᵗ rₜ]

最大熵RL目标: 最大化回报 + 策略熵
  J = E[Σ γᵗ (rₜ + α · H(π(·|sₜ)))]
                    └───┬───┘
                  熵奖励（鼓励探索）

α: 温度参数，控制探索程度
  α → 0: 退化为标准RL（纯利用）
  α → ∞: 均匀随机策略（纯探索）
```

## 为什么最大熵？

1. **鼓励探索**：高熵策略保持随机性，避免过早收敛
2. **多模态奖励**：如果有多个好策略，保留全部而非选一个
3. **鲁棒性**：不依赖单一策略，适应环境变化
4. **训练稳定**：熵正则化平滑优化景观

```
标准RL:        最大熵RL:
   收获              收获
    │ ╱              │ ╱╲
    │/               │/  ╲
    │    →          │    ╲    → 保留两个模式
    │                │      ╲
    └──→             └──────→
   选择一个峰值      两个策略模式都保留
```

## Soft Bellman方程

$$Q^\pi_{soft}(s,a) = r + \gamma E_{s' \sim P}\left[V^\pi_{soft}(s')\right]$$

$$V^\pi_{soft}(s) = E_{a \sim \pi}\left[Q^\pi_{soft}(s,a) - \alpha \log \pi(a|s)\right]$$

$$V^\pi_{soft}(s) = \alpha \log \sum_a \exp\left(\frac{Q^\pi_{soft}(s,a)}{\alpha}\right)$$

## 完整SAC实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class GaussianPolicy(nn.Module):
    """高斯策略网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 action_space=None):
        super().__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 动作缩放
        if action_space is not None:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            )
        else:
            self.action_scale = torch.ones(action_dim)
            self.action_bias = torch.zeros(action_dim)
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # 限制范围
        std = log_std.exp()
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        
        # 重参数化技巧
        x = dist.rsample()  # 可导的采样
        action = torch.tanh(x) * self.action_scale + self.action_bias
        
        # log概率（需要tanh修正）
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(self.action_scale * (1 - torch.tanh(x)**2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action


class QNetwork(nn.Module):
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
        
        # Q2
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


class SAC:
    """完整的SAC算法"""
    
    def __init__(self, state_dim, action_dim, action_space=None,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 auto_alpha=True, target_entropy=None):
        
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        
        # Actor
        self.actor = GaussianPolicy(state_dim, action_dim, action_space=action_space)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critics (Twin Q)
        self.critic = QNetwork(state_dim, action_dim)
        self.critic_target = QNetwork(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 自动温度调节
        if auto_alpha:
            self.target_entropy = target_entropy or -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha
    
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.numpy()[0]
    
    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return
        
        # 采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # === 1. Critic更新 ===
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * q_next
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # === 2. Actor更新 ===
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.alpha * log_probs - q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # === 3. 温度参数α更新 ===
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # === 4. 软更新目标网络 ===
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        return critic_loss.item(), actor_loss.item()
```

## 重参数化技巧（Reparameterization Trick）

```python
# 不能直接对采样过程求导
# action = dist.sample()  ← 不可导！

# 重参数化: 把随机性移到外部
# ε ~ N(0,1)
# action = mean + std * ε  ← 对mean, std可导
#         = dist.rsample()

"""
这允许梯度从Q值传回到策略参数:
  ∇_θ Q(s, a) = ∇_θ Q(s, μ(s) + σ(s)·ε)
              = ∇_a Q · ∇_θ μ + ∇_a Q · ∇_θ σ
"""
```

## 自动温度调节

```python
"""
α自动调节的目标:
  使策略熵 ≈ target_entropy

target_entropy = -|A|（经验值）

如果熵 > target: α减小 → 更少探索
如果熵 < target: α增大 → 更多探索
"""

# 调节过程
alpha_loss = -log_alpha * (log_prob + target_entropy).detach()
# 当 log_prob + target_entropy > 0 (熵太高)
#   → alpha_loss < 0 → log_alpha增大 → alpha增大？不对
# 
# 重新理解:
# H = -log_prob（策略熵）
# alpha_loss = -log_alpha * (log_prob + target_entropy)
#            = -log_alpha * (-H + target_entropy)
#            = log_alpha * (H - target_entropy)
#
# 当H > target: alpha_loss > 0 → log_alpha减小 → alpha减小 → 减少探索 ✓
# 当H < target: alpha_loss < 0 → log_alpha增大 → alpha增大 → 增加探索 ✓
```

## SAC的优势

| 特性 | 说明 |
|------|------|
| **最大熵** | 内在探索，不需要ε-greedy或噪声 |
| **Twin Q** | 减少Q值过估计（与TD3类似） |
| **Off-policy** | 样本效率高 |
| **自动α** | 不需要手动调探索参数 |
| **稳定** | 多种技术保证收敛 |
| **SOTA** | 连续控制的state-of-the-art |

## 与其他算法对比

| 特性 | DDPG | TD3 | SAC |
|------|------|-----|-----|
| **探索** | OU噪声 | 高斯噪声 | 最大熵 |
| **Q网络** | 单Q | Twin Q | Twin Q |
| **策略类型** | 确定性 | 确定性 | 随机性 |
| **温度参数** | 无 | 无 | 自动调节 |
| **性能** | 一般 | 好 | 最好 |
| **稳定性** | 差 | 好 | 最好 |

## 小结

SAC是连续控制领域的最佳选择：
- **最大熵框架**：自动探索
- **Twin Q**：减少过估计
- **自动温度**：免调参
- **Off-policy**：高样本效率

| 组件 | 作用 |
|------|------|
| **Actor** | 高斯策略 + tanh |
| **Twin Critic** | 双Q取min |
| **α** | 自动调节探索强度 |
| **重参数化** | 允许梯度回传 |

---

[← 上一篇：A2C与A3C](02-A2C与A3C.md) | [下一篇：TD3双延迟DDPG →](04-TD3双延迟DDPG.md)
