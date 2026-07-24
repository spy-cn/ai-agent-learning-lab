# TRPO：信任域策略优化

TRPO（Trust Region Policy Optimization）通过限制策略更新幅度，保证单调改进，解决了REINFORCE训练不稳定的问题。

## 核心思想

```
问题: REINFORCE的步长难以选择
  - 步长太大: 策略突变，性能崩溃
  - 步长太小: 收敛太慢

TRPO的解决方案:
  限制新旧策略的KL散度 ≤ δ
  保证每次更新都是改进

π_new = argmax L(θ)        ← 最大化目标
s.t.  D_KL(π_old || π_new) ≤ δ  ← 信任域约束
```

## 理论基础：策略改进界限

TRPO基于以下理论结果：

$$J(\pi_{new}) \geq J(\pi_{old}) + \frac{1}{1-\gamma} E_{s \sim d^{\pi_{old}}, a \sim \pi_{new}}\left[A^{\pi_{old}}(s,a)\right] - \frac{2\epsilon\gamma}{(1-\gamma)^2} D_{KL}^{max}(\pi_{old} || \pi_{new})$$

简化后：

$$J(\pi_{new}) \geq \mathcal{L}_{\pi_{old}}(\pi_{new}) - \frac{4\epsilon\gamma}{(1-\gamma)^2} D_{KL}^{max}$$

**结论**：只要KL散度足够小，就能保证 $J(\pi_{new}) \geq J(\pi_{old})$。

## TRPO算法

### 1. 代理目标函数

$$\mathcal{L}(\theta) = E\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{\theta_{old}}}(s,a)\right]$$

### 2. KL约束

$$D_{KL}(\pi_{\theta_{old}} || \pi_\theta) \leq \delta$$

### 3. 用自然梯度求解

TRPO用**自然梯度**（而非普通梯度）来处理约束：

```python
# 普通梯度下降: θ ← θ + α ∇L
# 自然梯度下降: θ ← θ + α F⁻¹ ∇L
#   F = Fisher信息矩阵
```

## 完整TRPO实现

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque

class TRPO:
    """TRPO算法实现"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 lr=1e-3, gamma=0.99, lambda_gae=0.95,
                 delta=0.01, max_kl=0.01, cg_iters=10, 
                 damping=0.1, beta=1.0):
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.delta = delta  # KL约束
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.damping = damping
        
        # 策略网络
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
    
    def compute_gae(self, rewards, values, dones, last_value):
        """计算GAE（广义优势估计）"""
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
        
        return advantages
    
    def surrogate_loss(self, states, actions, advantages, old_log_probs):
        """计算代理目标函数"""
        new_log_probs = self.policy.log_prob(states, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate = (ratio * advantages).mean()
        return surrogate
    
    def get_kl(self, states):
        """计算KL散度"""
        mean, std = self.policy(states)
        old_mean, old_std = self.policy.old_mean, self.policy.old_std
        
        kl = torch.log(std / old_std) + (old_std**2 + (old_mean - mean)**2) / (2 * std**2) - 0.5
        return kl.mean()
    
    def flat_grad(self, loss, params, **kwargs):
        """展平梯度"""
        grads = torch.autograd.grad(loss, params, **kwargs)
        return torch.cat([grad.view(-1) for grad in grads])
    
    def flat_params(self, model):
        """展平参数"""
        return torch.cat([p.data.view(-1) for p in model.parameters()])
    
    def set_params(self, model, flat_params):
        """设置展平的参数"""
        prev_idx = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_idx:prev_idx + flat_size].view(param.size())
            )
            prev_idx += flat_size
    
    def conjugate_gradient(self, Ax, b, cg_iters=10, residual_tol=1e-10):
        """共轭梯度法: 解 Fx = -∇L"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(cg_iters):
            Ap = Ax(p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        
        return x
    
    def fisher_vector_product(self, p, states):
        """Fisher信息矩阵 × 向量: Fp"""
        kl = self.get_kl(states)
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        
        grad_vp = torch.dot(flat_grad_kl, p)
        hessian_vp = torch.autograd.grad(grad_vp, self.policy.parameters())
        flat_hessian_vp = torch.cat([grad.contiguous().view(-1) for grad in hessian_vp])
        
        return flat_hessian_vp + self.damping * p  # 加阻尼项保证正定
    
    def line_search(self, step_direction, loss_before, states, actions, 
                    advantages, old_log_probs):
        """线性搜索满足KL约束的步长"""
        params = self.flat_params(self.policy)
        max_backtracks = 10
        
        for i in range(max_backtracks):
            alpha = 0.5 ** i
            new_params = params + alpha * step_direction
            self.set_params(self.policy, new_params)
            
            loss = self.surrogate_loss(states, actions, advantages, old_log_probs)
            kl = self.get_kl(states)
            
            if kl <= self.max_kl and loss >= loss_before:
                return True
        
        # 恢复参数
        self.set_params(self.policy, params)
        return False
    
    def update(self, trajectories):
        """TRPO更新"""
        # 收集数据
        states = torch.cat([t['states'] for t in trajectories])
        actions = torch.cat([t['actions'] for t in trajectories])
        rewards = torch.cat([t['rewards'] for t in trajectories])
        old_log_probs = torch.cat([t['log_probs'] for t in trajectories])
        
        # 计算GAE优势
        with torch.no_grad():
            values = self.value_net(states).squeeze()
        
        advantages = self.compute_gae(rewards.tolist(), values.tolist(), 
                                       [0]*len(rewards), values[-1].item())
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # === TRPO核心步骤 ===
        
        # 1. 计算代理目标的梯度
        loss = self.surrogate_loss(states, actions, advantages, old_log_probs)
        loss_before = loss.item()
        grads = self.flat_grad(loss, self.policy.parameters(), retain_graph=True)
        
        # 2. 用共轭梯度法求 x = F⁻¹g
        step_direction = self.conjugate_gradient(
            lambda p: self.fisher_vector_product(p, states),
            grads,
            cg_iters=self.cg_iters
        )
        
        # 3. 计算步长
        shs = 0.5 * torch.dot(step_direction, 
                              self.fisher_vector_product(step_direction, states))
        if shs < 0:
            return  # Fisher不正定，跳过
        
        lm = np.sqrt(self.delta / (shs + 1e-8))
        full_step = lm * step_direction
        
        # 4. 线性搜索
        success = self.line_search(full_step, loss_before, states, actions,
                                   advantages, old_log_probs)
        
        # 5. 更新价值网络
        returns = advantages + values.detach()
        for _ in range(80):
            value_pred = self.value_net(states).squeeze()
            value_loss = nn.MSELoss()(value_pred, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
```

## TRPO的优缺点

### 优点
- **单调改进保证**：理论上保证性能不下降
- **超参数鲁棒**：KL约束δ不敏感
- **稳定训练**：不会崩溃

### 缺点
- **计算复杂**：共轭梯度+Hessian计算
- **实现复杂**：代码量大，容易出bug
- **二阶信息**：计算Hessian-vector product开销大
- **速度慢**：比PPO慢得多

## TRPO vs PPO

| 特性 | TRPO | PPO |
|------|------|-----|
| **约束** | 硬KL约束 | 软clip |
| **计算** | 二阶（Fisher矩阵） | 一阶 |
| **实现** | 复杂 | 简单 |
| **性能** | 稳定但慢 | 稍不稳定但快 |
| **主流使用** | 学术 | 工业界标准 |

## 小结

TRPO是策略优化的理论里程碑，证明了策略改进的保证。但其复杂性导致了PPO的诞生——用简单的方法近似TRPO的效果。

| 关键概念 | 说明 |
|----------|------|
| **信任域** | 限制策略更新幅度 |
| **代理目标** | 用重要采样比 × 优势 |
| **自然梯度** | 用Fisher矩阵的逆 |
| **共轭梯度** | 不显式求F⁻¹ |
| **线性搜索** | 找满足KL约束的步长 |

---

[← 上一篇：策略梯度定理](01-策略梯度定理.md) | [下一篇：PPO近端策略优化 →](03-PPO近端策略优化.md)
