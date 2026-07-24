# Atari 游戏通关

Atari 游戏是深度强化学习的标准基准——DQN 论文在 49 款 Atari 游戏上展示了通用 RL 智能体的可能性。

## Atari 环境概览

```
Atari 游戏环境 (ALE - Arcade Learning Environment)
├── 状态: 原始游戏画面 (210×160×3 RGB 图像)
├── 动作: 离散 (4~18 种，取决于游戏)
├── 奖励: 游戏内得分
├── 特点:
│   ├── 高维像素输入 → 需要CNN
│   ├── 稀疏奖励 → 需要探索
│   ├── 部分可观测 → 需要帧堆叠
│   └── 多样性 → 测试算法泛化能力
└── 经典游戏:
    ├── Breakout (打砖块)
    ├── Pong (乒乓球)
    ├── SpaceInvaders (太空入侵者)
    ├── Enduro (赛车)
    └── Seaquest (海底探险)
```

```
经典 Atari 游戏画面:

Breakout          Pong              SpaceInvaders
┌─────────┐      ┌─────────┐       ┌─────────┐
│ ▓▓▓▓▓▓▓ │      │         │       │ ▓ ▓ ▓ ▓ │
│ ▓▓▓▓▓▓▓ │      │    ●    │       │ ▓ ▓ ▓ ▓ │
│         │      │   ●     │       │         │
│    ▄    │      │  ●      │       │    ▄    │
│ ────────│      │ ──  ── │       │ ────────│
└─────────┘      └─────────┘       └─────────┘
```

## 环境预处理

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import cv2

# === 方式一: 使用 SB3 内置包装器（推荐）===
from stable_baselines3.common.atari_wrappers import (
    AtariWrapper, NoopResetEnv, MaxAndSkipEnv,
    EpisodicLifeEnv, FireResetEnv, WarpFrame,
    ClipRewardEnv, FrameStack
)

env = gym.make('BreakoutNoFrameskip-v4')
env = AtariWrapper(env)  # 一行搞定所有预处理

# === 方式二: 手动实现预处理 ===
class PreprocessAtari(gym.ObservationWrapper):
    """Atari 图像预处理管道"""
    
    def __init__(self, env, frame_stack=4, frame_size=(84, 84)):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frames = deque(maxlen=frame_stack)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(frame_stack, *frame_size),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        # 1. RGB → 灰度
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # 2. 调整大小
        resized = cv2.resize(gray, self.frame_size, 
                              interpolation=cv2.INTER_AREA)
        # 3. 加入帧堆叠
        self.frames.append(resized)
        while len(self.frames) < self.frame_stack:
            self.frames.append(resized)
        
        return np.array(self.frames)

def make_atari_env(game='Breakout'):
    """创建预处理的 Atari 环境"""
    env = gym.make(f'{game}NoFrameskip-v4')
    
    # 1. No-op 重置: 随机执行 1~30 个无操作
    env = NoopResetEnv(env, noop_max=30)
    # 2. 跳帧: 每 4 帧执行一次动作
    env = MaxAndSkipEnv(env, skip=4)
    # 3. episodic life: 将损失生命视为回合结束
    env = EpisodicLifeEnv(env)
    # 4. Fire 重置: 某些游戏需要按 FIRE 开始
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # 5. 图形变换: 缩放到 84×84 灰度图
    env = WarpFrame(env, width=84, height=84)
    # 6. 奖励裁剪: {-1, 0, 1}
    env = ClipRewardEnv(env)
    # 7. 帧堆叠: 堆叠 4 帧
    env = FrameStack(env, 4)
    
    return env

env = make_atari_env('Breakout')
print(f"观测空间: {env.observation_space}")
print(f"动作空间: {env.action_space.n}")
```

## DQN 网络架构

```python
class NatureDQN(nn.Module):
    """DeepMind 2015 Nature 论文的 CNN 架构"""
    
    def __init__(self, input_channels=4, n_actions=4):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        
        self.fc = nn.Linear(512, n_actions)
    
    def forward(self, x):
        # 输入归一化: [0, 255] → [0, 1]
        x = x.float() / 255.0
        features = self.features(x)
        return self.fc(features)

# 测试网络
net = NatureDQN(input_channels=4, n_actions=4)
dummy_input = torch.zeros(1, 4, 84, 84)
output = net(dummy_input)
print(f"输入: {dummy_input.shape} → 输出: {output.shape}")
print(f"参数量: {sum(p.numel() for p in net.parameters()):,}")
# 参数量: ~1.7M
```

## 使用 SB3 训练 Atari

```python
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env(env_id, seed=0):
    def _init():
        env = gym.make(env_id)
        env = AtariWrapper(env)
        env.seed(seed)
        return env
    return _init

# === 训练 Pong ===
env_id = "PongNoFrameskip-v4"
n_envs = 8

vec_env = DummyVecEnv([make_env(env_id, seed=i) for i in range(n_envs)])

model = DQN(
    "CnnPolicy",
    vec_env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=32,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/atari_pong/",
    device="auto",
)

# 保存检查点
checkpoint = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/pong/",
    name_prefix="dqn_pong"
)

# 训练
model.learn(total_timesteps=1_000_000, callback=checkpoint)
model.save("dqn_pong")

vec_env.close()
```

## 使用 Rainbow DQN

```python
from stable_baselines3 import DQN

# SB3 目前支持 PER (Prioritized Experience Replay)
# 完整 Rainbow 需要自定义实现

class RainbowConfig:
    """Rainbow DQN 超参数"""
    # 基础 DQN
    learning_rate = 6.25e-5
    buffer_size = 1_000_000
    batch_size = 32
    gamma = 0.99
    
    # Double DQN (SB3 默认开启)
    # target_update_interval = 32000  # SB3 DQN 默认
    
    # PER (需自定义)
    alpha = 0.5  # 优先级指数
    beta_start = 0.4  # 重要采样初始值
    beta_frames = 1_000_000
    
    # Dueling (需自定义网络)
    # NoisyNet (需自定义网络)
    # Distributional (需自定义网络)
    # N-step (需自定义)
    
    # Atari 特有
    learning_starts = 20000
    train_freq = 4
    exploration_final_eps = 0.01  # NoisyNet 可设为 0
```

## 评估与可视化

```python
def evaluate_atari(model_path, env_id, n_episodes=10):
    """评估 Atari 智能体"""
    env = make_atari_env(env_id.replace('NoFrameskip', '').replace('-v4', ''))
    env = DummyVecEnv([lambda: env])
    
    model = DQN.load(model_path, env=env)
    
    scores = []
    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        scores.append(total_reward)
        print(f"Episode {ep+1}: Score = {total_reward}")
    
    print(f"\n平均分: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    env.close()
    return scores

evaluate_atari("dqn_pong", "PongNoFrameskip-v4")
```

## 各游戏性能参考

| 游戏 | 随机基线 | 人类水平 | DQN | Rainbow |
|------|----------|----------|-----|---------|
| Pong | -20.7 | -3.0 | 18.9 | 21.0 |
| Breakout | 1.2 | 30.5 | 168.0 | 417.5 |
| SpaceInvaders | 148.0 | 1635.0 | 581.0 | 1898.0 |
| Enduro | 0.0 | 129.6 | 301.0 | 311.0 |
| Seaquest | 68.4 | 2905.0 | 5286.0 | 16264.0 |

## 训练技巧

```
Atari 训练最佳实践:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. 预处理至关重要                                   │
│     → 帧堆叠(4帧) + 灰度(84×84) + 奖励裁剪(-1,0,1) │
│                                                     │
│  2. 足够的训练步数                                   │
│     → DQN: ~10M steps | Rainbow: ~10M steps        │
│                                                     │
│  3. 探索要充分                                       │
│     → ε 从 1.0 衰减到 0.01 (前 10%)                │
│                                                     │
│  4. 大的 Replay Buffer                               │
│     → 1M transitions                               │
│                                                     │
│  5. GPU 是必需的                                     │
│     → CNN 计算量大，CPU 不可行                       │
│                                                     │
│  6. 监控训练                                        │
│     → TensorBoard 观察reward、loss、Q值             │
│                                                     │
│  7. 从简单游戏开始                                   │
│     Pong → Breakout → SpaceInvaders                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 小结

- Atari 是 **RL 通用性** 的黄金基准
- DQN 论文证明：**同一套超参数**可以在 49 种游戏上超越人类
- 关键技术：**CNN 特征提取 + 帧堆叠 + 经验回放**
- 实战建议：先在 Pong 上验证代码，再扩展到其他游戏

---

| [← 回到目录](../README.md) | [上一章：CartPole平衡控制](01-CartPole平衡控制.md) | [下一章：机器人连续控制](03-机器人连续控制.md) |
|---|---|---|
