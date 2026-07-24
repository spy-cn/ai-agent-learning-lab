# MADDPG：多智能体深度确定性策略梯度

MADDPG（Multi-Agent Deep Deterministic Policy Gradient）通过**集中训练分散执行**（CTDE）解决了多智能体环境中的非平稳性问题。

## 核心思想：CTDE

```
集中训练分散执行 (Centralized Training, Decentralized Execution):

训练时（集中）:
  Critic可以看到所有智能体的观测和动作
  → 环境对Critic是平稳的（知道所有信息）

执行时（分散）:
  Actor只能看到自己的观测
  → 部署时不需要通信

训练: Critic(s₁,a₁,s₂,a₂,...,sₙ,aₙ) ← 有全局信息
执行: Actorᵢ(sᵢ) → aᵢ              ← 只有局部信息
```

```
┌──────────────────────────────────────────────────┐
│                MADDPG 架构                        │
│                                                  │
│  训练阶段:                                        │
│                                                  │
│  全局状态 ──→ ┌─────────────────────┐            │
│  所有动作 ──→ │  集中Critic_i        │ → Q_i     │
│               └─────────────────────┘            │
│                                                  │
│  局部观测 ──→ ┌─────────────────────┐            │
│               │  分散Actor_i         │ → a_i     │
│               └─────────────────────┘            │
│                                                  │
│  执行阶段:                                        │
│                                                  │
│  局部观测 ──→ Actor_i ──→ 动作a_i                 │
│  （不需要其他智能体信息）                          │
│                                                  │
└──────────────────────────────────────────────────┘
```

## MADDPG算法

每个智能体i有：
- Actor网络 μᵢ(oᵢ|θᵢ)：只用自己的观测
- Critic网络 Qᵢ(o₁,a₁,...,oₙ,aₙ|wᵢ)：用所有信息

### Critic更新

$$L(w_i) = E\left[(Q_i^{\mu}(o_1, a_1, ..., o_N, a_N) - y_i)^2\right]$$

$$y_i = r_i + \gamma Q_i^{\mu'}(o'_1, a'_1, ..., o'_N, a'_N)\big|_{a'_j = \mu'_j(o'_j)}$$

### Actor更新

$$\nabla_{\theta_i} J = E\left[\nabla_{\theta_i} \mu_i(o_i) \cdot \nabla_{a_i} Q_i(o_1, a_1, ..., o_N, a_N)\big|_{a_i = \mu_i(o_i)}\right]$$

## 完整实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MADDPG:
    """完整的多智能体DDPG"""
    
    def __init__(self, n_agents, obs_dims, action_dims, hidden_dim=64,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.01,
                 buffer_size=100000, batch_size=1024):
        self.n_agents = n_agents
        self.obs_dims = obs_dims  # [obs_dim_1, ..., obs_dim_N]
        self.action_dims = action_dims
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)
        
        # 每个智能体的Actor和Critic
        self.actors = []
        self.actors_target = []
        self.critics = []
        self.critics_target = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(n_agents):
            # Actor: 只用自己的观测
            actor = MADDPGActor(obs_dims[i], action_dims[i], hidden_dim)
            actor_target = MADDPGActor(obs_dims[i], action_dims[i], hidden_dim)
            actor_target.load_state_dict(actor.state_dict())
            
            # Critic: 用所有智能体的观测和动作
            critic = MADDPGCritic(total_obs_dim, total_action_dim, hidden_dim)
            critic_target = MADDPGCritic(total_obs_dim, total_action_dim, hidden_dim)
            critic_target.load_state_dict(critic.state_dict())
            
            self.actors.append(actor)
            self.actors_target.append(actor_target)
            self.critics.append(critic)
            self.critics_target.append(critic_target)
            
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr_actor))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr_critic))
        
        # 集中经验回放（存储全局信息）
        self.buffer = MultiAgentReplayBuffer(buffer_size)
    
    def act(self, observations, explore=True, noise_scale=0.1):
        """分散执行: 每个智能体只用自己的观测"""
        actions = []
        for i in range(self.n_agents):
            obs = torch.FloatTensor(observations[i]).unsqueeze(0)
            action = self.actors[i](obs).detach().numpy()[0]
            
            if explore:
                action += np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action, -1, 1)
            
            actions.append(action)
        return actions
    
    def update(self):
        """集中训练"""
        if len(self.buffer) < self.batch_size:
            return
        
        # 采样全局经验
        batch = self.buffer.sample(self.batch_size)
        
        # 解包: 每个智能体的数据
        all_obs = [batch['obs'][i] for i in range(self.n_agents)]  # [N, batch, obs_dim_i]
        all_actions = [batch['actions'][i] for i in range(self.n_agents)]
        all_rewards = [batch['rewards'][i] for i in range(self.n_agents)]
        all_next_obs = [batch['next_obs'][i] for i in range(self.n_agents)]
        dones = batch['dones']
        
        # 拼接全局观测和动作
        global_obs = torch.cat(all_obs, dim=-1)          # [batch, sum(obs_dim)]
        global_actions = torch.cat(all_actions, dim=-1)   # [batch, sum(action_dim)]
        global_next_obs = torch.cat(all_next_obs, dim=-1)
        
        for i in range(self.n_agents):
            # === Critic更新 ===
            with torch.no_grad():
                # 目标策略只用各自观测（分散执行）
                target_next_actions = torch.cat([
                    self.actors_target[j](all_next_obs[j]) 
                    for j in range(self.n_agents)
                ], dim=-1)
                
                target_q = self.critics_target[i](
                    global_next_obs, target_next_actions
                ).squeeze()
                
                target = all_rewards[i] + self.gamma * (1 - dones) * target_q
            
            current_q = self.critics[i](global_obs, global_actions).squeeze()
            critic_loss = nn.MSELoss()(current_q, target)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            # === Actor更新 ===
            # 当前策略的输出
            current_actions = torch.cat([
                self.actors[j](all_obs[j]) if j != i 
                else self.actors[j](all_obs[j])  # 需要梯度
                for j in range(self.n_agents)
            ], dim=-1)
            
            actor_loss = -self.critics[i](global_obs, current_actions).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
            # === 软更新目标网络 ===
            self.soft_update(self.actors[i], self.actors_target[i])
            self.soft_update(self.critics[i], self.critics_target[i])
    
    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class MADDPGActor(nn.Module):
    """分散Actor: 只用自己的观测"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 连续动作
        )
    
    def forward(self, obs):
        return self.net(obs)


class MADDPGCritic(nn.Module):
    """集中Critic: 用所有智能体的观测和动作"""
    
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, global_obs, global_actions):
        x = torch.cat([global_obs, global_actions], dim=-1)
        return self.net(x)


class MultiAgentReplayBuffer:
    """多智能体经验回放"""
    
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, obs_list, action_list, reward_list, next_obs_list, done):
        """存储一条全局经验"""
        self.buffer.append({
            'obs': obs_list,
            'actions': action_list,
            'rewards': reward_list,
            'next_obs': next_obs_list,
            'dones': done,
        })
        
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # 整理为tensor
        n_agents = len(batch[0]['obs'])
        
        result = {
            'obs': [[] for _ in range(n_agents)],
            'actions': [[] for _ in range(n_agents)],
            'rewards': [[] for _ in range(n_agents)],
            'next_obs': [[] for _ in range(n_agents)],
            'dones': [],
        }
        
        for exp in batch:
            for i in range(n_agents):
                result['obs'][i].append(exp['obs'][i])
                result['actions'][i].append(exp['actions'][i])
                result['rewards'][i].append(exp['rewards'][i])
                result['next_obs'][i].append(exp['next_obs'][i])
            result['dones'].append(exp['dones'])
        
        # 转tensor
        for i in range(n_agents):
            result['obs'][i] = torch.FloatTensor(np.array(result['obs'][i]))
            result['actions'][i] = torch.FloatTensor(np.array(result['actions'][i]))
            result['rewards'][i] = torch.FloatTensor(result['rewards'][i])
            result['next_obs'][i] = torch.FloatTensor(np.array(result['next_obs'][i]))
        
        result['dones'] = torch.FloatTensor(result['dones'])
        
        return result
    
    def __len__(self):
        return len(self.buffer)
```

## MADDPG的优势

| 优势 | 说明 |
|------|------|
| **解决非平稳性** | Critic有全局信息，环境平稳 |
| **分散执行** | 部署时不需要通信 |
| **灵活** | 合作/竞争/混合都适用 |
| **稳定** | 比IQL训练更稳定 |

## 对抗版本

对于竞争环境，MADDPG可以自然处理：

```python
class CompetitiveMADDPG(MADDPG):
    """竞争环境的MADDPG"""
    
    def update(self):
        # 对于零和博弈:
        # 智能体A的奖励 = -智能体B的奖励
        # MADDPG自动处理，因为每个智能体有自己的Critic
        
        # 如果想让对手更难对付:
        # 可以对对手的Actor做梯度上升
        pass
```

## 训练循环

```python
def train_maddpg(env_name="simple_adversary", n_episodes=5000):
    from pettingzoo.mpe import simple_adversary_v3
    
    env = simple_adversary_v3.env(N=2, max_cycles=25)
    env.reset()
    
    n_agents = len(env.agents)
    obs_dims = [env.observation_space(agent).shape[0] for agent in env.agents]
    action_dims = [env.action_space(agent).shape[0] for agent in env.agents]
    
    agent = MADDPG(n_agents, obs_dims, action_dims)
    
    for episode in range(n_episodes):
        env.reset()
        
        observations = {agent_id: None for agent_id in env.agents}
        actions = {agent_id: None for agent_id in env.agents}
        rewards = {agent_id: 0 for agent_id in env.agents}
        
        obs_list = []
        action_list = []
        reward_list = []
        next_obs_list = []
        
        for agent_id in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                action = agent.act([obs])[0]  # 简化: 单个智能体
                observations[agent_id] = obs
                actions[agent_id] = action
            
            rewards[agent_id] = reward
            env.step(action)
        
        # 存储经验
        agent.buffer.push(
            list(observations.values()),
            list(actions.values()),
            list(rewards.values()),
            list(observations.values()),  # 简化
            False
        )
        
        # 训练
        agent.update()
        
        if (episode + 1) % 100 == 0:
            total_reward = sum(rewards.values())
            print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")
    
    env.close()
```

## 小结

| 特点 | MADDPG |
|------|--------|
| **范式** | CTDE（集中训练分散执行） |
| **Critic** | 集中（看全局信息） |
| **Actor** | 分散（只用局部信息） |
| **适用** | 合作/竞争/混合 |
| **改进** | 解决非平稳性 |

---

[← 上一篇：独立Q学习](02-独立Q学习.md) | [下一篇：MAPPO与多智能体AC →](04-MAPPO与多智能体AC.md)
