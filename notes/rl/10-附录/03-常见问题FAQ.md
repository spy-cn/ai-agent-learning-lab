# RL 常见问题 FAQ

强化学习实践中的常见问题、疑难解答和调试技巧。

## 基础概念

### Q1: RL 和监督学习有什么本质区别？

```
监督学习:                      强化学习:
┌──────────┐                  ┌──────────┐
│  数据     │                  │  环境     │
│ (输入,标签)│                  │ (状态)    │
└────┬─────┘                  └────┬─────┘
     ↓                             ↓
┌──────────┐                  ┌──────────┐
│  模型     │                  │  智能体    │
└────┬─────┘                  └────┬─────┘
     ↓                             ↓
┌──────────┐                  ┌──────────┐
│  预测     │                  │  动作     │
└──────────┘                  └────┬─────┘
                                   ↓
                              ┌──────────┐
                              │  奖励/新状态│
                              └──────────┘

区别:
1. 数据来源: SL有固定数据集, RL通过交互产生数据
2. 目标信号: SL有标签, RL只有延迟奖励
3. 数据分布: SL独立同分布, RL数据高度相关
4. 决策影响: SL不影响数据, RL动作影响未来
```

### Q2: On-policy 和 Off-policy 的区别？

| 属性 | On-policy | Off-policy |
|------|-----------|------------|
| 数据来源 | 当前策略产生 | 任何策略产生 |
| 能否复用旧数据 | ❌ | ✅ |
| 样本效率 | 低 | 高 |
| 代表算法 | PPO, A2C, REINFORCE | DQN, SAC, TD3 |
| 稳定性 | 高 | 中等 |

```
比喻:
  On-policy: 只能从自己的经验中学习（实时学习）
  Off-policy: 可以从别人的经验中学习（看书学习）
```

### Q3: 什么是探索-利用困境？

```
利用 (Exploitation): 选择已知最好的动作 → 短期收益最大化
探索 (Exploration): 尝试新动作 → 发现可能更好的策略

例子: 去餐厅吃饭
  利用: 去你最喜欢的餐厅 (安全但可能错过更好的)
  探索: 尝试新餐厅 (可能发现惊喜，也可能踩雷)

解决方法:
  ε-greedy: 90%利用 + 10%探索
  UCB: 基于不确定性选择
  SAC: 熵正则化鼓励探索
```

## 训练调试

### Q4: 模型完全不学习怎么办？

**排查步骤：**

```
Step 1: 检查环境
  □ 随机策略能正常返回不同奖励吗？
  □ reward 的范围是否合理？
  □ done 条件是否正确？

Step 2: 检查基线
  □ 随机策略的平均分是多少？
  □ 固定策略（如全选动作0）的平均分？

Step 3: 检查网络
  □ 输入是否归一化？
  □ 输出范围是否正确？
  □ loss 是否在下降？

Step 4: 检查超参数
  □ 学习率是否太大/太小？
  □ gamma 是否合理？
  □ batch_size 是否合适？

Step 5: 简化问题
  □ 能否用表格型 Q-Learning 解决简化版？
  □ 能否用更简单的环境验证代码？
```

```python
# 调试代码模板
def debug_agent(env, agent, n_episodes=5):
    """调试智能体"""
    for ep in range(n_episodes):
        obs, _ = env.reset()
        print(f"\n=== Episode {ep+1} ===")
        print(f"Initial obs: {obs}")
        
        total_reward = 0
        for step in range(100):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            if step < 5 or step % 20 == 0:  # 打印关键步
                print(f"  Step {step}: obs={obs}, action={action}, "
                      f"reward={reward:.3f}, done={terminated}")
            
            total_reward += reward
            obs = next_obs
            if terminated or truncated:
                break
        
        print(f"Total reward: {total_reward:.2f}")
```

### Q5: 训练不稳定（分数忽高忽低）？

```
可能原因及解决方案:

┌─────────────────────┬─────────────────────────────┐
│ 原因                │ 解决方案                     │
├─────────────────────┼─────────────────────────────┤
│ 学习率太大          │ 降低 lr (1e-4 ~ 3e-4)       │
│ batch_size 太小     │ 增大 batch_size (64~256)    │
│ 没有目标网络        │ 添加 target network          │
│ 奖励不归一化        │ 归一化 reward                │
│ 观测不归一化        │ 使用 VecNormalize            │
│ buffer 太小         │ 增大 replay buffer           │
│ 探索不充分          │ 增大初始 ε 或减缓衰减        │
│ 网络太深/太宽       │ 减小网络: [64, 64]          │
└─────────────────────┴─────────────────────────────┘
```

### Q6: DQN 的 Q 值发散怎么办？

```python
# 监控 Q 值
def monitor_q_values(model, env, n_steps=1000):
    """监控 Q 值是否发散"""
    q_values = []
    obs = env.reset()
    
    for step in range(n_steps):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            q = model.q_net(obs_t)
            q_values.append(q.max().item())
        
        action = q.argmax().item()
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(q_values)
    plt.title('Max Q-values over time')
    plt.xlabel('Step')
    plt.ylabel('Max Q-value')
    plt.grid(True)
    plt.show()

# 如果 Q 值持续增长 → 使用 Double DQN
```

### Q7: PPO 表现不好怎么调？

```
PPO 调参清单:

1. 学习率
   默认: 3e-4
   范围: 1e-5 ~ 1e-3
   
2. n_steps (每次更新的步数)
   默认: 2048
   范围: 128 ~ 8192
   注: 太小→batch噪声大, 太大→on-policy偏移
   
3. batch_size
   必须: 能整除 n_steps × n_envs
   
4. n_epochs (每次更新的轮数)
   默认: 10
   范围: 3 ~ 30
   注: 太大→过拟合
   
5. clip_range
   默认: 0.2
   范围: 0.1 ~ 0.3
   
6. gae_lambda
   默认: 0.95
   范围: 0.9 ~ 1.0
   
7. ent_coef (熵系数)
   默认: 0.0 (SB3) 或 0.01
   增大→更多探索
   
8. vf_coef (价值函数系数)
   默认: 0.5
   增大→更好学习价值函数
   
9. max_grad_norm
   默认: 0.5
   作用: 梯度裁剪，稳定训练
   
10. normalize_advantage
    默认: True
    必须开启！
```

## 环境相关

### Q8: 如何设计好的奖励函数？

```
奖励设计原则:

1. 稀疏 vs 稠密
   稀疏: 只在终态给奖励 (如赢/输)
   稠密: 每步都有奖励信号
   → 稠密奖励更容易学习，但可能引入偏差

2. 奖励塑造 (Reward Shaping)
   添加中间奖励引导学习:
   r'(s,a,s') = r(s,a,s') + γ·F(s') - F(s)
   其中 F 是势函数 (potential function)
   → 理论保证不改变最优策略

3. 常见陷阱
   ✗ 奖励黑客: agent找到漏洞获取高奖励但不解决任务
   例子: 赛车游戏中来回撞墙弹跳得分
   
   ✗ 短视行为: 只看眼前奖励
   解决: 增大 γ 或增加长期奖励成分
   
4. 多目标奖励
   r_total = w1·r1 + w2·r2 + w3·r3
   注意权重平衡
```

### Q9: 如何处理稀疏奖励？

```
方法1: 奖励塑造
  → 添加稠密的中间奖励

方法2: 课程学习 (Curriculum Learning)
  → 从简单任务开始，逐步增加难度

方法3: HER (Hindsight Experience Replay)
  → 将失败轨迹视为新目标下的成功轨迹

方法4: 内在奖励 (Intrinsic Motivation)
  → ICM, RND 等好奇心驱动探索

方法5: 模仿学习预训练
  → 先用 BC 初始化，再用 RL 微调

方法6: 分层 RL
  → 分解为子任务，降低难度
```

### Q10: 如何做 Sim-to-Real 迁移？

```
Sim-to-Real 技术栈:

1. 域随机化 (Domain Randomization)
   → 在仿真中随机化物理参数
   → 让策略对参数变化鲁棒
   env_params = {
       'friction': uniform(0.5, 2.0),
       'mass': uniform(0.8, 1.2),
       'gravity': uniform(9.5, 10.0),
   }

2. 系统辨识 (System Identification)
   → 用真实数据校准仿真器

3. 域适应 (Domain Adaptation)
   → 学习仿真和现实的映射

4. 渐进式迁移
   → 先仿真 → 再 sim+real 混合 → 最后纯 real

5. 安全 RL
   → Constrained RL 保证安全约束
```

## 实现问题

### Q11: 如何正确实现经验回放？

```python
# 常见错误和正确实现

# ✗ 错误: 存储了旧引用
class BadReplayBuffer:
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        # 问题: state 和 next_state 可能是同一个数组的引用!

# ✓ 正确: 存储副本
class GoodReplayBuffer:
    def push(self, state, action, reward, next_state):
        self.buffer.append((
            state.copy(),    # 关键: 复制
            action,
            reward,
            next_state.copy()
        ))
```

### Q12: PyTorch 中 detach() 和 no_grad() 的区别？

```python
# torch.no_grad(): 上下文管理器，内部不追踪梯度
with torch.no_grad():
    q_values = model(states)  # 不计算梯度，节省内存

# .detach(): 分离 tensor，使其不需要梯度
target = (rewards + gamma * next_q.max()).detach()
# target 不参与梯度计算，但 model(states) 会

# RL 中的典型用法:
# 计算 Q target 时用 detach (不需要梯度流过 target)
target_q = rewards + gamma * next_q.detach().max()
loss = mse_loss(current_q, target_q)  # 只对 current_q 求梯度
```

### Q13: GPU 训练但不加速？

```python
# 检查清单:
# 1. 模型在 GPU 上
model = model.to('cuda')

# 2. 输入在 GPU 上
states = states.to('cuda')

# 3. 检查是否真的用了 GPU
print(next(model.parameters()).device)  # 应该是 cuda:0

# 4. batch_size 太小 → GPU 利用率低
# 建议至少 batch_size = 64

# 5. 数据搬运太频繁
# → 减少 .to('cuda') 调用
# → 批量搬运数据

# 6. CPU 瓶颈
# → 环境步进通常在 CPU
# → 用向量化环境 (VecEnv)
```

## 评估相关

### Q14: 如何正确评估 RL 智能体？

```python
# 正确评估方法
def evaluate_properly(model, env, n_episodes=20, seeds=None):
    """
    正确的评估流程
    """
    results = {'reward': [], 'length': [], 'success': []}
    
    for ep in range(n_episodes):
        # 使用不同的种子确保多样性
        seed = seeds[ep] if seeds else None
        obs, info = env.reset(seed=seed)
        
        total_reward = 0
        length = 0
        success = False
        
        while True:
            # 确定性策略评估 (关闭探索)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1
            success = info.get('is_success', False)
            
            if terminated or truncated:
                break
        
        results['reward'].append(total_reward)
        results['length'].append(length)
        results['success'].append(success)
    
    # 报告统计
    print(f"Reward: {np.mean(results['reward']):.1f} ± "
          f"{np.std(results['reward']):.1f}")
    print(f"Success Rate: {np.mean(results['success']):.1%}")
    
    return results

# 重要: 评估时用 deterministic=True!
```

### Q15: 不同种子结果差异很大？

```
原因: RL 训练本质上随机性强
  - 网络初始化随机
  - 环境初始化随机
  - 探索动作随机
  - mini-batch 采样随机

解决方案:
1. 多种子实验 (至少 3~5 个)
   seeds = [0, 42, 123, 456, 789]
   
2. 报告均值 ± 标准差
   mean_reward = np.mean(all_seed_results)
   std_reward = np.std(all_seed_results)
   
3. 绘制带置信区间的曲线
   plt.fill_between(steps, mean-std, mean+std, alpha=0.3)

4. 如果方差太大:
   → 增大 batch_size
   → 使用更稳定的算法 (PPO > DQN)
   → 使用目标网络
```

## 进阶问题

### Q16: 如何选择网络大小？

```
经验法则:

低维观测 (<10维):
  → [64, 64] 或 [128, 128]
  
中等维度 (10~100维):
  → [256, 256]
  
高维观测 (>100维):
  → [512, 512] 或更深的网络
  
图像输入:
  → CNN (NatureCNN 或 ResNet)
  
注意:
  - 太大: 过拟合、训练慢
  - 太小: 欠拟合
  - 先从 [64, 64] 开始，逐步增大
```

### Q17: 什么时候需要优先经验回放 (PER)？

```
PER 适合场景:
  ✓ 稀疏奖励环境
  ✓ 大量冗余经验的场景
  ✓ Atari 游戏
  ✓ 大规模 replay buffer

PER 不适合场景:
  ✗ 小规模 buffer (TD-error差别不大)
  ✗ on-policy 方法 (PPO 不需要)
  ✗ 对速度要求高的场景 (PER增加开销)

注意: PER 会引入 bias (重要采样)
  需要用 beta 系数修正
```

### Q18: 多智能体 RL 为什么比单智能体难？

```
三大挑战:

1. 非平稳性 (Non-stationarity)
   → 所有智能体同时在学习
   → 对手的策略在变化
   → 从固定策略的角度看环境是非平稳的

2. 联合动作空间爆炸
   → n个智能体 × m个动作 = m^n 种联合动作
   → 指数增长

3. 信用分配 (Credit Assignment)
   → 团队成功了，谁的功劳？
   → COMA, Difference Rewards 等方法解决

解决: CTDE 范式 (集中训练分散执行)
  → 训练时看到所有信息
  → 执行时只看自己的观测
```

## 工具相关

### Q19: TensorBoard 和 W&B 用哪个？

```
TensorBoard:
  ✓ 免费、本地运行
  ✓ PyTorch 原生支持
  ✓ 适合个人实验
  ✗ 不方便分享
  ✗ 对比实验不方便

Weights & Biases (W&B):
  ✓ 云端、自动同步
  ✓ 实验对比方便
  ✓ 团队协作
  ✓ 超参搜索集成
  ✗ 免费版有限制

建议: 开发阶段用 TensorBoard，正式实验用 W&B
```

### Q20: 如何分享我的 RL 项目？

```
分享清单:

1. 代码仓库 (GitHub)
   → 清晰的 README
   → requirements.txt
   → 可复现的命令

2. 训练日志
   → 分享 TensorBoard 或 W&B 链接

3. 模型权重
   → Hugging Face Hub
   → model.save() + 上传

4. 演示视频
   → 录制 GIF 或 MP4
   → 展示训练前后对比

5. 技术博客
   → 写清方法、结果、经验教训
```

---

| [← 回到目录](../README.md) | [上一章：算法选择指南](02-算法选择指南.md) | [下一章：推荐资源](04-推荐资源.md) |
|---|---|---|
