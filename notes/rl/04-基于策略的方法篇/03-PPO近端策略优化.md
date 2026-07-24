# PPO：近端策略优化

PPO（Proximal Policy Optimization）是当前工业界最常用的RL算法，用简单的一阶方法近似TRPO的效果。

## 核心思想

```
TRPO: 硬约束 D_KL(π_old || π_new) ≤ δ  → 二阶方法，复杂
PPO:  软约束 clip(ratio, 1-ε, 1+ε)     → 一阶方法，简单

ratio = π_new(a|s) / π_old(a|s)  重要采样比

如果 ratio > 1+ε 或 < 1-ε，就裁剪，防止更新太大
```

## PPO-Clip 目标函数

$$L^{CLIP}(\theta) = E\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

### Clip的直觉

```
情况1: A > 0（好动作）
  想要增大ratio → 但clip在1+ε停住 → 不会过度增大概率

情况2: A < 0（坏动作）  
  想要减小ratio → 但clip在1-ε停住 → 不会过度减小概率

结果: 每次更新有上限，保证稳定
```

```
     L^CLIP
       ↑
       │     ╱  ← min(ratio·A, (1+ε)·A)
  (1+ε)│ ───╱────────
       │   ╱
       │  ╱
       │ ╱  ← ratio·A
       │╱
───────┼───────────────→ ratio
       ││
       │└───  ← ratio·A
  (1-ε)│ ────╱────────
       │    ╱ ← min(ratio·A, (1-ε)·A)
       │
```

## 完整PPO实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal

class PPO:
    """完整的PPO算法"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 lr=3e-4, gamma=0.99, lambda_gae=0.95,
                 clip_eps=0.2, n_epochs=10, batch_size=64,
                 entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, continuous=False):
        
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        
        # 策略网络
        if continuous:
            self.actor = GaussianActor(state_dim, action_dim, hidden_dim)
        else:
            self.actor = CategoricalActor(state_dim, action_dim, hidden_dim)
        
        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优化器（共享或分离）
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
    
    def act(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, entropy = self.actor.get_action(state_t)
        return action, log_prob, entropy
    
    def compute_gae(self, rewards, values, dones, last_value):
        """广义优势估计 (GAE)"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lambda_gae * next_non_terminal * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns, values):
        """PPO核心更新"""
        states = torch.FloatTensor(states)
        if self.continuous:
            actions = torch.FloatTensor(actions)
        else:
            actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        n_samples = len(states)
        
        for epoch in range(self.n_epochs):
            # 随机打乱并分batch
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                # 取batch
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # 计算新的log_prob和entropy
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # 计算ratio: π_new / π_old
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # === PPO Clip 目标 ===
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # === 价值损失 ===
                current_values = self.critic(batch_states).squeeze()
                value_loss = nn.MSELoss()(current_values, batch_returns)
                
                # === 熵奖励 ===
                entropy_loss = -entropy.mean()
                
                # === 总损失 ===
                total_loss = (
                    policy_loss 
                    + self.value_coef * value_loss 
                    + self.entropy_coef * entropy_loss
                )
                
                # 梯度更新
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()


# ========== Actor网络 ==========
class CategoricalActor(nn.Module):
    """离散动作的Actor"""
    
    def __init__(self, state_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, state):
        logits = self.net(state)
        dist = Categorical(logits=logits)
        return dist
    
    def get_action(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob.item(), entropy.item()
    
    def evaluate_actions(self, states, actions):
        dist = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class GaussianActor(nn.Module):
    """连续动作的Actor"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        mean = self.mean_net(state)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        return dist
    
    def get_action(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action.numpy()[0], log_prob.item(), entropy.item()
    
    def evaluate_actions(self, states, actions):
        dist = self.forward(states)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy


# ========== 训练循环 ==========
def train_ppo(env_name="CartPole-v1", n_episodes=500):
    import gymnasium as gym
    env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    continuous = isinstance(env.action_space, gym.spaces.Box)
    action_dim = env.action_space.shape[0] if continuous else env.action_space.n
    
    agent = PPO(state_dim, action_dim, continuous=continuous)
    
    n_steps = 2048  # 每次收集的步数
    state, _ = env.reset()
    
    for update in range(100):
        # 1. 收集经验
        states, actions, rewards, dones = [], [], [], []
        log_probs = []
        values = []
        
        for step in range(n_steps):
            action, log_prob, _ = agent.act(state)
            value = agent.critic(torch.FloatTensor(state).unsqueeze(0)).item()
            
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob)
            values.append(value)
            
            state = next_state
            if done:
                state, _ = env.reset()
        
        # 计算最后的value
        last_value = agent.critic(torch.FloatTensor(state).unsqueeze(0)).item()
        
        # 2. 计算GAE
        advantages, returns = agent.compute_gae(rewards, values, dones, last_value)
        
        # 3. PPO更新
        agent.update(states, actions, log_probs, advantages, returns, values)
        
        # 打印进度
        avg_reward = np.mean(rewards) * n_steps / sum(dones) if sum(dones) > 0 else 0
        print(f"Update {update+1}, 平均回合奖励: {avg_reward:.1f}")
    
    env.close()
    return agent

# agent = train_ppo()
```

## PPO的变体

### PPO-Penalty（KL惩罚版）

替代clip，用自适应KL惩罚：

$$L^{KPEN} = E[r_t A_t] - \beta \cdot D_{KL}(\pi_{old} || \pi_\theta)$$

```python
class PPOPenalty(PPO):
    """PPO with adaptive KL penalty"""
    
    def __init__(self, *args, target_kl=0.01, beta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_kl = target_kl
        self.beta = beta
    
    def update(self, states, actions, old_log_probs, advantages, returns, values):
        # 计算KL散度
        new_log_probs, _ = self.actor.evaluate_actions(states, actions)
        kl = (old_log_probs - new_log_probs).mean().item()
        
        # 自适应调整beta
        if kl < self.target_kl / 1.5:
            self.beta /= 2
        elif kl > self.target_kl * 1.5:
            self.beta *= 2
        
        # ...其余与标准PPO类似，但用KL惩罚替代clip
```

## PPO超参数指南

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `lr` | 3e-4 | 学习率 |
| `clip_eps` | 0.2 | clip范围 |
| `gamma` | 0.99 | 折扣因子 |
| `lambda_gae` | 0.95 | GAE参数 |
| `n_epochs` | 10 | 每批数据训练轮数 |
| `batch_size` | 64 | 小批量大小 |
| `entropy_coef` | 0.01 | 熵奖励系数 |
| `value_coef` | 0.5 | 价值损失系数 |
| `max_grad_norm` | 0.5 | 梯度裁剪 |
| `n_steps` | 2048 | 每次收集步数 |

## PPO为什么有效？

1. **Clip防止破坏性更新**：限制策略变化幅度
2. **多次重复使用数据**：n_epochs次更新提高样本效率
3. **GAE平衡偏差和方差**：λ参数控制TD和MC的混合
4. **熵奖励鼓励探索**：防止过早收敛
5. **实现简单**：一阶方法，不需要二阶信息

## 小结

PPO是当前最广泛使用的RL算法：
- **简单**：一阶梯度下降
- **稳定**：clip约束
- **高效**：数据多次复用
- **通用**：离散/连续动作都可

| 组件 | 作用 |
|------|------|
| **Clip目标** | 限制策略更新 |
| **GAE** | 降低方差的优势估计 |
| **熵奖励** | 鼓励探索 |
| **多epoch训练** | 提高样本效率 |
| **梯度裁剪** | 防止梯度爆炸 |

---

[← 上一篇：TRPO信任域方法](02-TRPO信任域方法.md) | [下一篇：确定性策略梯度 →](04-确定性策略梯度.md)
