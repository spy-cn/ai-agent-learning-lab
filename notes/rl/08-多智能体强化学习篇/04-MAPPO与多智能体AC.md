# MAPPO：多智能体PPO

MAPPO（Multi-Agent PPO）将PPO扩展到多智能体场景，采用CTDE范式，是当前最成功的MARL算法之一。

## MAPPO概述

```
MADDPG: CTDE + 连续动作 + Off-policy
MAPPO:  CTDE + 离散/连续 + On-policy

MAPPO的特点:
1. 继承PPO的稳定性
2. 集中Critic用全局状态
3. 分散Actor用局部观测
4. 参数共享（同类智能体）
```

## 架构

```
┌──────────────────────────────────────────────────┐
│                MAPPO 架构                         │
│                                                  │
│  共享Actor（参数共享）:                            │
│                                                  │
│  obs₁ → ┐                                        │
│  obs₂ → ├──→ SharedActor(θ) → a₁, a₂, ..., aₙ   │
│  obsₙ → ┘                                        │
│                                                  │
│  集中Critic:                                      │
│                                                  │
│  全局状态s → CentralizedCritic(w) → V(s)        │
│                                                  │
│  训练: PPO clip + GAE                             │
│  执行: 只用Actor + 局部观测                       │
│                                                  │
└──────────────────────────────────────────────────┘
```

## 完整MAPPO实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal

class MAPPO:
    """Multi-Agent PPO with CTDE"""
    
    def __init__(self, n_agents, obs_dim, state_dim, action_dim,
                 continuous=False, lr=3e-4, gamma=0.99, 
                 lambda_gae=0.95, clip_eps=0.2, n_epochs=10,
                 batch_size=64, entropy_coef=0.01, 
                 value_coef=0.5, max_grad_norm=0.5,
                 share_params=True):
        
        self.n_agents = n_agents
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        self.share_params = share_params
        
        # Actor（分散，可能参数共享）
        if share_params:
            self.actor = SharedActor(obs_dim, action_dim, continuous)
        else:
            self.actors = [SharedActor(obs_dim, action_dim, continuous) 
                          for _ in range(n_agents)]
        
        # Critic（集中，用全局状态）
        self.critic = CentralizedCritic(state_dim)
        
        # 优化器
        actor_params = self.actor.parameters() if share_params else \
            [p for actor in self.actors for p in actor.parameters()]
        
        self.actor_optimizer = optim.Adam(actor_params, lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
    
    def get_actions(self, observations, deterministic=False):
        """所有智能体选动作（分散）"""
        all_actions = []
        all_log_probs = []
        all_entropies = []
        
        for i in range(self.n_agents):
            obs = torch.FloatTensor(observations[i])
            
            if self.share_params:
                actor = self.actor
            else:
                actor = self.actors[i]
            
            action, log_prob, entropy = actor.get_action(obs.unsqueeze(0), 
                                                          deterministic)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_entropies.append(entropy)
        
        return all_actions, all_log_probs, all_entropies
    
    def get_values(self, global_states):
        """集中Critic评估全局状态"""
        return self.critic(global_states)
    
    def compute_gae_multi(self, rewards, values, dones, last_value):
        """为每个智能体计算GAE"""
        all_advantages = []
        all_returns = []
        
        for i in range(self.n_agents):
            advantages = []
            gae = 0
            
            for t in reversed(range(len(rewards[i]))):
                if t == len(rewards[i]) - 1:
                    next_value = last_value
                    next_non_terminal = 1 - dones[t]
                else:
                    next_value = values[t + 1]
                    next_non_terminal = 1 - dones[t]
                
                delta = rewards[i][t] + self.gamma * next_value * next_non_terminal - values[t]
                gae = delta + self.gamma * self.lambda_gae * next_non_terminal * gae
                advantages.insert(0, gae)
            
            advantages = torch.FloatTensor(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + torch.FloatTensor(values[:-1] if len(values) > len(advantages) else values)
            
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        return all_advantages, all_returns
    
    def update(self, trajectories):
        """PPO更新（所有智能体共享batch）"""
        # 收集所有智能体的数据
        all_obs = []
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        
        for agent_traj in trajectories:
            for step_data in agent_traj:
                all_obs.append(step_data['obs'])
                all_states.append(step_data['state'])
                all_actions.append(step_data['action'])
                all_old_log_probs.append(step_data['log_prob'])
                all_advantages.append(step_data['advantage'])
                all_returns.append(step_data['return'])
        
        # 转tensor
        all_obs = torch.stack(all_obs)
        all_states = torch.stack(all_states)
        all_actions = torch.stack(all_actions)
        all_old_log_probs = torch.stack(all_old_log_probs)
        all_advantages = torch.stack(all_advantages)
        all_returns = torch.stack(all_returns)
        
        n_samples = len(all_obs)
        
        # PPO多轮epoch
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                idx = indices[start:start + self.batch_size]
                
                batch_obs = all_obs[idx]
                batch_states = all_states[idx]
                batch_actions = all_actions[idx]
                batch_old_log_probs = all_old_log_probs[idx]
                batch_advantages = all_advantages[idx]
                batch_returns = all_returns[idx]
                
                # Actor前向
                if self.continuous:
                    new_log_probs, entropy = self.actor.evaluate(batch_obs, batch_actions)
                else:
                    new_log_probs, entropy = self.actor.evaluate(batch_obs, batch_actions)
                
                # PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clip目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic损失
                values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                # 熵奖励
                entropy_loss = -entropy.mean()
                
                total_loss = (actor_loss + self.value_coef * critic_loss 
                              + self.entropy_coef * entropy_loss)
                
                # 更新Actor
                self.actor_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # 更新Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()


class SharedActor(nn.Module):
    """参数共享的Actor"""
    
    def __init__(self, obs_dim, action_dim, continuous=False, hidden_dim=64):
        super().__init__()
        self.continuous = continuous
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        if continuous:
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs):
        feat = self.net(obs)
        if self.continuous:
            mean = self.mean_head(feat)
            std = self.log_std.exp()
            return mean, std
        else:
            return self.action_head(feat)
    
    def get_action(self, obs, deterministic=False):
        if self.continuous:
            mean, std = self.forward(obs)
            if deterministic:
                return mean.squeeze(0).detach(), torch.tensor(0), torch.tensor(0)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            logits = self.forward(obs)
            if deterministic:
                action = logits.argmax(-1)
                return action.squeeze(0).detach(), torch.tensor(0), torch.tensor(0)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        return action.squeeze(0), log_prob.squeeze(0), entropy.squeeze(0)
    
    def evaluate(self, obs, actions):
        if self.continuous:
            mean, std = self.forward(obs)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            logits = self.forward(obs)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        return log_probs, entropy


class CentralizedCritic(nn.Module):
    """集中Critic: 全局状态 → V(s)"""
    
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)
```

## 训练流程

```python
def train_mappco(env_name="simple_spread", n_episodes=10000):
    from pettingzoo.mpe import simple_spread_v3
    
    env = simple_spread_v3.env(N=3, max_cycles=25)
    env.reset()
    
    n_agents = 3
    obs_dim = 18
    state_dim = obs_dim * n_agents  # 全局状态 = 所有观测拼接
    action_dim = 5
    
    agent = MAPPO(n_agents, obs_dim, state_dim, action_dim, continuous=False)
    
    for episode in range(n_episodes):
        env.reset()
        
        # 收集数据
        trajectory = {i: [] for i in range(n_agents)}
        global_states = []
        
        for agent_id in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue
            
            # 分散选动作
            actions, log_probs, entropies = agent.get_actions([obs] * n_agents)
            
            # 集中评估
            global_state = torch.FloatTensor([obs] * n_agents).flatten()
            value = agent.get_values(global_state.unsqueeze(0))
            
            env.step(actions[0])  # 简化
        
        # PPO更新
        agent.update([trajectory[i] for i in range(n_agents)])
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}")
```

## MAPPO vs MADDPG

| 特性 | MAPPO | MADDPG |
|------|-------|--------|
| **算法类型** | On-policy | Off-policy |
| **策略类型** | 随机 | 确定性 |
| **参数共享** | 支持 | 支持 |
| **稳定性** | 好 | 中 |
| **样本效率** | 低 | 高 |
| **超参敏感** | 低 | 中 |
| **大规模** | 好 | 中 |
| **主流场景** | 合作 | 竞争/合作 |

## 大规模MARL技巧

### 1. 参数共享

```python
# 同类智能体共享网络
# n_agents=100时:
#   不共享: 100个独立网络
#   共享: 1个网络，输入obs+agent_id

class ParameterSharedActor(nn.Module):
    def __init__(self, obs_dim, action_dim, n_agents):
        super().__init__()
        # agent_id嵌入
        self.agent_embed = nn.Embedding(n_agents, 16)
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 16, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs, agent_id):
        embed = self.agent_embed(agent_id)
        x = torch.cat([obs, embed], dim=-1)
        return self.net(x)
```

### 2. 均值场近似

```python
class MeanFieldMAPPO:
    """用邻居的均值动作代替精确动作"""
    
    def __init__(self, n_agents, neighbor_radius=2):
        self.neighbor_radius = neighbor_radius
    
    def get_mean_action(self, actions, adjacency):
        """计算邻居的均值动作"""
        mean_actions = []
        for i in range(self.n_agents):
            neighbors = self.get_neighbors(i, adjacency)
            mean_action = np.mean([actions[j] for j in neighbors], axis=0)
            mean_actions.append(mean_action)
        return mean_actions
```

## 小结

MAPPO是当前合作MARL的最佳选择：
- **稳定**：继承PPO的稳定性
- **简单**：On-policy，无需经验回放
- **扩展性好**：参数共享处理大规模
- **性能强**：在SMAC等基准上SOTA

| 特点 | 说明 |
|------|------|
| **CTDE** | 集中训练分散执行 |
| **参数共享** | 同类智能体共享网络 |
| **集中Critic** | 用全局状态评估 |
| **分散Actor** | 用局部观测决策 |

---

[← 上一篇：MADDPG](03-MADDPG.md) | [下一篇：自博弈与种群方法 →](05-自博弈与种群方法.md)
