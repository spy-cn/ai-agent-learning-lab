# 多智能体强化学习基础概念

多智能体强化学习（MARL）研究多个智能体在同一环境中交互学习的问题。

## 单智能体 vs 多智能体

```
单智能体RL:
  环境 ← → 1个智能体
  
  环境: 静态的（不随策略变化）

多智能体RL:
  环境 ← → 智能体1, 智能体2, ..., 智能体N

  环境: 非静态的（其他智能体也在学习，环境对每个智能体来说在变化）
  → 从单个智能体的视角，面临"移动目标"问题
```

## MARL问题分类

```
┌────────────────────────────────────────────────────┐
│                 MARL 分类矩阵                       │
├──────────┬─────────────────────────────────────────┤
│          │           关系类型                       │
│  设置    ├────────────┬────────────┬───────────────┤
│          │  完全合作   │  完全竞争   │   混合        │
├──────────┼────────────┼────────────┼───────────────┤
│ 完全可观测│ 团队协作   │ 零和博弈   │ 一般和博弈    │
│ (完全信息)│ 共同奖励   │ 对抗奖励   │ 不同奖励      │
│          │ 例: 团队游戏│ 例: 棋类   │ 例: 资源分配  │
├──────────┼────────────┼────────────┼───────────────┤
│ 部分可观测│ Dec-POMDP  │ 零和POSG   │ 一般和POSG   │
│ (POMDP)  │ 例: 多机器人│ 例: 扑克   │ 例: 交通控制  │
└──────────┴────────────┴────────────┴───────────────┘
```

## 博弈论基础

### 标准形式博弈

```python
import numpy as np

# 经典博弈的收益矩阵

# 囚徒困境
#         沉默    告密
# 沉默   [(-1,-1) (-3, 0)]
# 告密   [( 0,-3) (-2,-2)]

prisoners_dilemma = {
    ('silent', 'silent'): (-1, -1),
    ('silent', 'confess'): (-3, 0),
    ('confess', 'silent'): (0, -3),
    ('confess', 'confess'): (-2, -2),
}

# 纳什均衡: 两人都告密（虽然合作更好，但单方面告密是优势策略）
```

### 纳什均衡

**纳什均衡（Nash Equilibrium）**：没有任何智能体能通过单方面改变策略来获得更好结果的状态。

```python
def find_nash_equilibrium(payoff_matrix_1, payoff_matrix_2):
    """寻找纳什均衡（简化版）"""
    n_rows, n_cols = payoff_matrix_1.shape
    nash_equilibria = []
    
    for i in range(n_rows):
        for j in range(n_cols):
            # 检查玩家1是否有激励偏离
            best_for_1 = (payoff_matrix_1[i, j] == payoff_matrix_1[:, j].max())
            # 检查玩家2是否有激励偏离
            best_for_2 = (payoff_matrix_2[i, j] == payoff_matrix_2[i, :].max())
            
            if best_for_1 and best_for_2:
                nash_equilibria.append((i, j))
    
    return nash_equilibria

# 示例
p1 = np.array([[3, 0], [5, 1]])  # 玩家1的收益
p2 = np.array([[3, 5], [0, 1]])  # 玩家2的收益
nash = find_nash_equilibrium(p1, p2)
print(f"纳什均衡: {nash}")  # [(1,1)]
```

### 博弈类型

| 博弈类型 | 特点 | 玩家收益和 | 示例 |
|----------|------|-----------|------|
| **合作** | 玩家目标一致 | 最大 | 团队任务 |
| **零和** | 一方赢=另一方输 | = 0 | 棋类 |
| **一般和** | 任意关系 | 任意 | 经济博弈 |
| **混合** | 部分合作部分竞争 | 任意 | 交通 |

## 马尔可夫博弈（Markov Game）

MDP的多智能体推广：

$$\langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}, \mathcal{P}, \{R_i\}, \gamma \rangle$$

- $\mathcal{N}$: N个智能体的集合
- $\mathcal{S}$: 全局状态空间
- $\mathcal{A}_i$: 智能体i的动作空间
- $\mathcal{P}$: 联合转移概率 $P(s'|s, a_1, ..., a_N)$
- $R_i$: 智能体i的奖励函数
- $\gamma$: 折扣因子

```python
class MarkovGame:
    """多智能体环境基类"""
    
    def __init__(self, n_agents):
        self.n_agents = n_agents
    
    def reset(self):
        """返回全局状态"""
        pass
    
    def step(self, actions):
        """
        actions: [agent1_action, agent2_action, ...]
        
        返回:
            next_state: 全局状态
            rewards: [r1, r2, ...] 每个智能体的奖励
            done: 是否结束
            infos: 额外信息
        """
        pass


# 示例: 多智能体粒子环境
class ParticleEnvironment(MarkovGame):
    """多智能体粒子物理环境"""
    
    def __init__(self, n_agents, n_landmarks):
        super().__init__(n_agents)
        self.n_landmarks = n_landmarks
    
    def step(self, actions):
        # 物理模拟
        for i, agent in enumerate(self.agents):
            agent.apply_force(actions[i])
        
        self.physics_step()
        
        # 计算每个智能体的奖励
        rewards = []
        for i in range(self.n_agents):
            # 合作: 覆盖所有地标
            reward = self.compute_coverage_reward(i)
            rewards.append(reward)
        
        return self.get_state(), rewards, self.done, {}
```

## 联合动作空间爆炸

```
问题: 动作空间随智能体数量指数增长

单智能体: |A| = 5
2个智能体: |A₁ × A₂| = 25
5个智能体: |A₁ × ... × A₅| = 3125
10个智能体: |A₁ × ... × A₁₀| = 9,765,625

→ 联合Q值表 Q(s, a₁, a₂, ..., aₙ) 不可行
→ 需要函数近似或分解
```

## MARL的核心挑战

### 1. 非平稳性

```python
# 从单个智能体的视角:
# 其他智能体也在学习和改变策略
# → 最优策略也在变化

# 这打破了标准RL的收敛假设
# 经验回放缓冲区中的数据可能过时
```

### 2. 信用分配

```python
# 团队赢了，谁的功劳？
# - 团队奖励需要分配到每个智能体

# 方法:
# 1. 均分: 每人获得 team_reward / N
# 2. 差异化: 根据个人贡献
# 3. 反事实: 如果没有这个智能体会怎样？
```

### 3. 可扩展性

```python
# 智能体数量增加 → 学习变难

# 解决方案:
# 1. 参数共享（同类智能体共享网络）
# 2. 均值场近似（Mean-Field）
# 3. 图神经网络（建模智能体关系）
```

## MARL环境接口

```python
import numpy as np

class MultiAgentEnv:
    """标准多智能体环境接口"""
    
    def __init__(self, n_agents, observation_dim, action_dim):
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
    
    def reset(self):
        """重置环境，返回每个智能体的观测"""
        observations = []
        for i in range(self.n_agents):
            obs = self._get_observation(i)
            observations.append(obs)
        return observations
    
    def step(self, actions):
        """执行所有智能体的动作"""
        # actions: list of each agent's action
        
        # 环境动力学更新
        self._update_physics(actions)
        
        # 获取观测
        observations = [self._get_observation(i) for i in range(self.n_agents)]
        
        # 计算奖励
        rewards = [self._get_reward(i) for i in range(self.n_agents)]
        
        # 终止条件
        dones = [self._check_done(i) for i in range(self.n_agents)]
        
        return observations, rewards, dones, {}


# PettingZoo: 标准多智能体环境库
"""
PettingZoo API:
  env.reset() → dict of observations
  env.step(action) → (observations, rewards, dones, infos)
  
所有输入输出都是字典: {agent_id: value}
"""
```

## 常用MARL环境

| 环境 | 类型 | 智能体数 | 特点 |
|------|------|----------|------|
| **Pistonball** | 合作 | 5-20 | 移动球 |
| **Cooperative Pong** | 合作 | 2 | 对打乒乓球 |
| **Multiwalker** | 合作 | 2-4 | 联合搬运 |
| **PettingZoo** | 多种 | 多种 | 标准接口 |
| **SMAC** | 合作 | 多 | 星际争霸微操 |
| **Google Football** | 合作 | 11 | 足球 |
| **Hanabi** | 合作 | 2-5 | 不完全信息卡牌 |

## 小结

| 概念 | 说明 |
|------|------|
| **马尔可夫博弈** | MDP的多智能体推广 |
| **纳什均衡** | 没有激励单方面偏离 |
| **非平稳性** | 其他智能体在学习 |
| **信用分配** | 团队奖励的个体归因 |
| **联合动作爆炸** | 指数增长的动作空间 |

---

[← 上一篇：分层强化学习](../07-高级强化学习篇/05-分层强化学习.md) | [下一篇：独立Q学习 →](02-独立Q学习.md)
