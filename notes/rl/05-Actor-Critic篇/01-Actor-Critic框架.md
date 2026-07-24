# Actor-Critic框架

Actor-Critic结合了基于价值和基于策略方法的优点，是现代深度RL的主流范式。

## 核心思想

```
Actor-Critic = Actor（策略） + Critic（价值函数）

Actor:  π(a|s;θ)     策略网络，负责选动作
Critic: Q(s,a;w)     价值网络，负责评估动作好坏

分工协作:
  Actor负责行动 → "做什么"
  Critic负责评价 → "做得怎么样"
```

```
┌─────────────────────────────────────────────────┐
│              Actor-Critic 架构                   │
│                                                 │
│    状态 s ──→ ┌─────────┐                      │
│               │  Actor  │ ──→ 动作 a            │
│               │  π(θ)   │                      │
│               └─────────┘                      │
│                    ↑                            │
│                 梯度更新                         │
│                    │                            │
│               ┌─────────┐                      │
│    状态 s ──→ │  Critic │ ──→ Q值/优势 A       │
│               │  Q(w)   │                      │
│               └─────────┘                      │
│                    ↑                            │
│                 TD误差更新                       │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 与纯策略梯度的区别

| 特性 | REINFORCE | Actor-Critic |
|------|-----------|--------------|
| **Critic** | 无（用MC回报） | 有（V或Q网络） |
| **方差** | 高 | 低（用TD误差） |
| **偏差** | 无偏 | 有偏（Bootstrap） |
| **更新时机** | 回合结束 | 每步在线 |
| **收敛速度** | 慢 | 快 |

## 基��的Actor-Critic算法

### A2C（Advantage Actor-Critic）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    """Actor-Critic网络（共享特征层）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # 共享特征提取
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor头
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic头
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.feature(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def act(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.item(), dist.entropy()


class A2CAgent:
    """A2C: Advantage Actor-Critic"""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def update(self, trajectory, last_value=0):
        """
        trajectory: [(log_prob, value, reward, entropy), ...]
        """
        # 1. 计算回报
        returns = []
        R = last_value
        for _, _, reward, _ in reversed(trajectory):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # 2. 提取数据
        log_probs = torch.stack([t[0] for t in trajectory])
        values = torch.FloatTensor([t[1] for t in trajectory])
        entropies = torch.stack([t[3] for t in trajectory])
        
        # 3. 计算优势 A = R - V(s)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 4. 计算损失
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = nn.MSELoss()(values, returns)
        entropy_loss = -entropies.mean()
        
        total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        
        # 5. 更新
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item()
```

## 训练循环

```python
def train_a2c(env_name="CartPole-v1", n_episodes=1000):
    import gymnasium as gym
    env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = A2CAgent(state_dim, action_dim)
    
    scores = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        trajectory = []
        ep_reward = 0
        done = False
        
        while not done:
            # 1. 选动作
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value, entropy = agent.model.act(state_t)
            
            # 2. 执行
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            trajectory.append((log_prob, value, reward, entropy))
            state = next_state
            ep_reward += reward
        
        # 3. 回合更新
        agent.update(trajectory)
        
        scores.append(ep_reward)
        if (episode + 1) % 50 == 0:
            avg = np.mean(scores[-100:])
            print(f"Episode {episode+1}, Avg Score: {avg:.1f}")
    
    return agent

# agent = train_a2c()
```

## 优势函数的作用

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

```
为什么用优势函数？

原始策略梯度: ∇J = E[∇log π(a|s) · G_t]
                                ↑ 高方差（回报波动大）

Actor-Critic: ∇J = E[∇log π(a|s) · A(s,a)]
                                ↑ 低方差（优势已中心化）

A > 0: 动作比平均好 → 增大概率
A < 0: 动作比平均差 → 减小概率
```

## 优势函数的估计

### TD误差作为优势

$$A_t \approx \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

```python
def td_advantage(reward, next_value, value, gamma=0.99, done=False):
    """用TD误差近似优势"""
    if done:
        return reward - value
    return reward + gamma * next_value - value
```

### GAE（广义优势估计）

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

```python
def compute_gae(rewards, values, last_value, gamma=0.99, lam=0.95):
    """GAE: 平衡偏差和方差"""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return advantages
```

```
λ=0:  A = δ_t         → TD(0)，偏差大，方差小
λ=1:  A = G_t - V(s)  → MC，偏差小，方差大
中间λ: 平衡两者

实践中 λ=0.95 是经典选择
```

## 共享网络 vs 分离网络

### 共享特征层

```python
class SharedActorCritic(nn.Module):
    """共享特征层的Actor-Critic"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # 共享CNN/MLP特征
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor头
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic头
        self.critic_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        feat = self.features(x)
        return self.actor_head(feat), self.critic_head(feat)
```

**优点**：参数共享，训练快
**缺点**：Actor和Critic可能互相干扰

### 分离网络

```python
class SeparateActorCritic:
    """分离的Actor和Critic"""
    
    def __init__(self, state_dim, action_dim):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
```

**优点**：互不干扰
**缺点**：参数多，训练慢

## 小结

| 组件 | 作用 |
|------|------|
| **Actor** | 策略π(a\|s;θ)，选动作 |
| **Critic** | 价值V(s;w)，评估状态 |
| **优势A** | 指导Actor改进 |
| **GAE** | 平衡偏差与方差 |

Actor-Critic框架是所有现代RL算法（PPO、SAC、TD3）的基础。

---

[← 上一篇：离线策略评估](../04-基于策略的方法篇/05-离线策略评估.md) | [下一篇：A2C与A3C →](02-A2C与A3C.md)
