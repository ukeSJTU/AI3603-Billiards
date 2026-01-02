# PPO 智能体详细指南

> AI3603 台球游戏 - 基于深度强化学习的 PPO 智能体完整文档

---

## 目录

1. [什么是 PPO](#1-什么是ppo)
2. [PPO 智能体架构详解](#2-ppo智能体架构详解)
3. [配置参数详解](#3-配置参数详解)
4. [训练监控指标](#4-训练监控指标)
5. [超参数调优指南](#5-超参数调优指南)
6. [常见问题与解决方案](#6-常见问题与解决方案)
7. [使用示例](#7-使用示例)

---

## 1. 什么是 PPO

### 1.1 PPO 简介

**PPO (Proximal Policy Optimization, 近端策略优化)** 是由 OpenAI 在 2017 年提出的一种深度强化学习算法，也是目前最流行和实用的强化学习算法之一。

**核心思想**：

-   PPO 试图在**策略更新的稳定性**和**学习效率**之间找到平衡
-   通过限制策略更新的步长，避免更新过大导致性能突然下降
-   使用**裁剪(Clipping)**机制确保新策略不会偏离旧策略太远

### 1.2 为什么选择 PPO

相比其他强化学习算法，PPO 有以下优势：

| 特性             | PPO           | DQN        | A3C           | DDPG       |
| ---------------- | ------------- | ---------- | ------------- | ---------- |
| **适用场景**     | 连续/离散动作 | 仅离散动作 | 连续/离散动作 | 仅连续动作 |
| **训练稳定性**   | ⭐⭐⭐⭐⭐    | ⭐⭐⭐     | ⭐⭐          | ⭐⭐⭐     |
| **样本效率**     | ⭐⭐⭐⭐      | ⭐⭐⭐⭐⭐ | ⭐⭐          | ⭐⭐⭐     |
| **实现复杂度**   | ⭐⭐⭐        | ⭐⭐       | ⭐⭐⭐⭐      | ⭐⭐⭐     |
| **超参数敏感度** | ⭐⭐          | ⭐⭐⭐     | ⭐⭐⭐⭐      | ⭐⭐⭐⭐   |

**对于台球游戏**：

-   ✅ 台球有**连续动作空间**（力度、角度、旋转）—— PPO 完美适配
-   ✅ 需要**长期规划**（一杆定乾坤 vs 步步为营）—— PPO 的 GAE 机制擅长这一点
-   ✅ 状态空间复杂（15 个球的位置组合）—— PPO 的神经网络能很好建模
-   ✅ 需要**探索与利用的平衡**—— PPO 的熵奖励机制保证探索

### 1.3 PPO 核心机制

#### 1.3.1 目标函数

PPO 的目标函数由三部分组成：

$$
L = L_{\mathrm{CLIP}}(\theta) + c_1 L_{\mathrm{VF}}(\theta) - c_2 H(\pi_{\theta})
$$

**第一部分：裁剪策略目标 (L_CLIP)**

$$
L_{\mathrm{CLIP}} = \mathbb{E}\left[\min\left(r_t(\theta) A_t,\; \operatorname{clip}\big(r_t(\theta), 1-\varepsilon, 1+\varepsilon\big) A_t\right)\right]
$$

其中：

-   $r_t(\theta) = \dfrac{\pi_{\theta}(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}$ （重要性采样比率）
-   $A_t$ = 优势函数（该动作比平均水平好多少）
-   $\varepsilon$ = 裁剪范围（通常为 $0.2$）

**直观理解**：

-   如果某个动作比预期好（$A_t > 0$），我们希望增加它的概率
-   但增加的幅度不能超过 $1+\varepsilon$（防止过度优化）
-   如果某个动作比预期差（$A_t < 0$），我们希望降低它的概率
-   但降低的幅度不能超过 $1-\varepsilon$（防止过度惩罚）

**第二部分：价值函数损失 (L_VF)**

$$
L_{\mathrm{VF}} = \mathbb{E}\left[\big(V_{\theta}(s_t) - V_{\mathrm{target}}\big)^2\right]
$$

其中：

-   $V_{\theta}(s_t)$ = 当前价值网络对状态 $s_t$ 的估计
-   $V_{\mathrm{target}}$ = 实际回报（通过 GAE 计算）

**第三部分：熵奖励 (H)**

$$
H(\pi_{\theta}) = \mathbb{E}\left[-\sum_a \pi_{\theta}(a\mid s) \log \pi_{\theta}(a\mid s)\right]
$$

**作用**：鼓励探索，防止策略过早收敛到次优解。

#### 1.3.2 GAE (Generalized Advantage Estimation)

**优势函数**衡量"某个动作比平均水平好多少"。GAE 通过加权多步 TD 误差来估计优势：

$$
A_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k \, \delta_{t+k},
\quad \text{其中} \quad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

-   $\gamma$ = 折扣因子（例如 $0.99$）
-   $\lambda$ = GAE 参数（例如 $0.95$）

**λ 的作用**（偏差-方差权衡）：

-   λ=0：只看一步（高偏差，低方差）—— 短视
-   λ=1：看完整轨迹（低偏差，高方差）—— 远视但不稳定
-   λ=0.95：平衡点 ✅

---

## 2. PPO 智能体架构详解

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      PPO训练系统                             │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
          ┌─────────▼─────────┐  ┌─────▼─────────┐
          │  PolicyNetwork    │  │ ValueNetwork  │
          │   (Actor)         │  │  (Critic)     │
          │  状态 → 动作分布   │  │  状态 → 价值   │
          └─────────┬─────────┘  └───────┬───────┘
                    │                    │
                    └──────────┬─────────┘
                               │
                    ┌──────────▼──────────┐
                    │  SelfPlayTrainer    │
                    │  - 自对弈数据收集    │
                    │  - PPO更新          │
                    │  - 对手池管理        │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
         │ExperienceBuffer│ HybridReward│ OpponentPool│
         │  轨迹存储     │ │ 奖励调度   │ │  对手快照   │
         └──────────────┘ └──────────┘ └─────────────┘
```

### 2.2 神经网络结构

#### 2.2.1 策略网络 (PolicyNetwork)

**输入**：76 维状态向量

**架构**：

```
输入层(76维)
    ↓
Linear(256) → LayerNorm → ReLU → Dropout(0.1)
    ↓
Linear(256) → LayerNorm → ReLU → Dropout(0.1)
    ↓
Linear(128) → LayerNorm → ReLU → Dropout(0.1)
    ↓
    ├─→ mean_head: Linear(5)      # 动作均值
    └─→ log_std_head: Linear(5)   # 动作标准差（对数）
```

**输出**：

-   `mean`: 5 个连续动作的均值 (V0, phi, theta, a, b)
-   `std`: 5 个连续动作的标准差（用于探索）

**动作采样**：

```python
# 训练时：从正态分布采样（探索）
action ~ Normal(mean, std)

# 评估时：直接使用均值（利用）
action = mean
```

**为什么选择这个架构**：

-   **3 层隐藏层**（256-256-128）：足够深以学习复杂模式，但不会过拟合
-   **LayerNorm**：比 BatchNorm 更适合强化学习（小批量训练）
-   **Dropout(0.1)**：轻度正则化，防止过拟合
-   **正交初始化**：加速收敛，提高训练稳定性

#### 2.2.2 价值网络 (ValueNetwork)

**输入**：76 维状态向量

**架构**：

```
输入层(76维)
    ↓
Linear(256) → LayerNorm → ReLU → Dropout(0.1)
    ↓
Linear(256) → LayerNorm → ReLU → Dropout(0.1)
    ↓
Linear(128) → LayerNorm → ReLU → Dropout(0.1)
    ↓
Linear(1) → V(s)  # 状态价值估计
```

**输出**：

-   单个标量值：当前状态的预期累积奖励

**与策略网络的区别**：

-   策略网络：告诉智能体"该做什么"（动作分布）
-   价值网络：告诉智能体"当前局面有多好"（状态评估）

### 2.3 状态编码详解

#### 2.3.1 76 维状态向量构成

```python
# 总维度：76 = 2 + 45 + 12 + 15 + 2

[0:2]    白球位置 (x, y)                      # 2维
[2:47]   15个彩球 × (x, y, is_pocketed)      # 45维
[47:59]  6个球袋 × (x, y)                    # 12维
[59:74]  目标球掩码 (15个二进制指示器)         # 15维
[74:76]  全局特征 (剩余球比例, 是否黑8阶段)    # 2维
```

#### 2.3.2 详细分解

**1. 白球位置 (2D)**

```python
cue_x = balls['cue'].state.rvw[0][0] / table.l  # 归一化到[0, 1]
cue_y = balls['cue'].state.rvw[0][1] / table.w
```

**2. 彩球状态 (45D)**

```python
for i in range(1, 16):  # 球号1-15
    ball = balls[str(i)]
    if ball.state.s == 4:  # 已进袋
        features.extend([0.0, 0.0, 1.0])
    else:  # 未进袋
        x = ball.state.rvw[0][0] / table.l
        y = ball.state.rvw[0][1] / table.w
        features.extend([x, y, 0.0])
```

**为什么不删除已进袋的球**？

-   神经网络需要**固定大小的输入**
-   使用标志位 `is_pocketed` 来标记，保持维度不变
-   已进袋的球位置设为(0, 0)，网络会学会忽略它们

**3. 球袋位置 (12D)**

```python
for pocket_id in ['lb', 'lc', 'lt', 'rb', 'rc', 'rt']:
    pocket_x = table.pockets[pocket_id].center[0] / table.l
    pocket_y = table.pockets[pocket_id].center[1] / table.w
    features.extend([pocket_x, pocket_y])
```

球袋位置是固定的，但提供给网络有助于：

-   几何计算（球到袋的距离、角度）
-   策略规划（选择最优目标袋）

**4. 目标球掩码 (15D)**

```python
my_targets_set = set(my_targets) - {'8'}  # 不包括黑8
for i in range(1, 16):
    is_my_target = 1.0 if str(i) in my_targets_set else 0.0
    features.append(is_my_target)
```

例如，如果我的目标是实心球(1-7)：

```
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
```

**5. 全局特征 (2D)**

```python
remaining_ratio = my_remaining_balls / 7.0  # 剩余球占比
is_black8_phase = 1.0 if my_targets == ['8'] else 0.0
```

-   `remaining_ratio`: 0.0（全进）到 1.0（全在）
-   `is_black8_phase`: 提示智能体现在是决胜阶段

#### 2.3.3 归一化的重要性

**为什么要归一化**？

-   神经网络对输入范围敏感
-   未归一化的特征会主导梯度更新
-   归一化后训练更稳定、收敛更快

**归一化方式**：

-   **位置坐标**：除以桌面尺寸 → [0, 1]
-   **二进制标志**：0 或 1
-   **比例特征**：已经在[0, 1]范围内

### 2.4 动作空间映射

#### 2.4.1 神经网络输出 → 游戏动作

网络输出是**无界的实数**，需要映射到有效范围：

```python
# 网络原始输出
action_raw = [raw_V0, raw_phi, raw_theta, raw_a, raw_b]

# 映射到游戏动作
V0    = sigmoid(raw_V0)    * 7.5 + 0.5   # [0.5, 8.0] m/s
phi   = sigmoid(raw_phi)   * 360         # [0, 360] 度
theta = sigmoid(raw_theta) * 90          # [0, 90] 度
a     = tanh(raw_a)        * 0.5         # [-0.5, 0.5] 无量纲
b     = tanh(raw_b)        * 0.5         # [-0.5, 0.5] 无量纲
```

数学表示为：

$$
V_0 = \mathrm{sigmoid}(\mathrm{raw\_V0}) \cdot 7.5 + 0.5,\quad
\phi = \mathrm{sigmoid}(\mathrm{raw\_\phi}) \cdot 360,\quad
	heta = \mathrm{sigmoid}(\mathrm{raw\_\theta}) \cdot 90,
$$

$$
a = \tanh(\mathrm{raw\_a}) \cdot 0.5,\quad
b = \tanh(\mathrm{raw\_b}) \cdot 0.5.
$$

**激活函数选择**：

-   **Sigmoid**：输出[0, 1]，适合非负范围（速度、角度）
-   **Tanh**：输出[-1, 1]，适合对称范围（旋转）

**物理意义**：

-   `V0`：初速度，控制击球力度
-   `phi`：水平角度，控制击球方向（0°=正右，90°=正上）
-   `theta`：垂直角度，控制高杆/低杆（0°=平击，90°=垂直向下）
-   `a`：横向偏转（侧旋）
-   `b`：纵向偏转（上/下旋）

### 2.5 奖励机制详解

#### 2.5.1 混合奖励策略

我们的 PPO 智能体使用**混合奖励**，在训练过程中从密集奖励平滑过渡到稀疏奖励：

```
最终奖励 = w_dense × 密集奖励 + w_sparse × 稀疏奖励

其中：
- w_dense = exp(-5 × iter / 1000)  # 指数衰减
- w_sparse = 1 - w_dense
```

**衰减曲线**：

```
权重
1.0 ┤
    │ 密集奖励
0.8 ┤╲
    │ ╲
0.6 ┤  ╲              稀疏奖励
    │   ╲            ╱
0.4 ┤    ╲          ╱
    │     ╲        ╱
0.2 ┤      ╲______╱
    │
0.0 ┤━━━━━━━━━━━━━━━━━━━━
    0   200  400  600  800  1000  迭代
```

#### 2.5.2 密集奖励 (Dense Reward)

基于每一杆的即时反馈：

```python
奖励 = 0

# 正奖励
+ 进自己的球数量 × 50
+ 合法未进球 × 10

# 负奖励
- 进对手的球数量 × 20
- 白球进袋 × 100
- 违规打错球 × 30
- 无进球且无碰库 × 30
- 误打黑8 × 150
- 白球+黑8同时进袋 × 150

# 归一化到[-1, 1]
公式表示为：
$$
	ext{最终奖励} = \operatorname{clip}\left(\frac{\text{奖励}}{150},\; -1,\; 1\right)
$$
```

**优点**：

-   ✅ 提供**频繁的学习信号**
-   ✅ 帮助智能体快速学会**基本技巧**（如何进球、如何避免犯规）

**缺点**：

-   ❌ 可能导致**局部最优**（只学会简单进球，不学长期策略）
-   ❌ 手工设计的奖励可能**不完美**

#### 2.5.3 稀疏奖励 (Sparse Reward)

只在游戏结束时给予奖励：

```python
if 赢了:
    奖励 = +1.0
elif 平局:
    奖励 = 0.0
else:  # 输了
    奖励 = -1.0
```

**优点**：

-   ✅ **目标明确**：就是要赢
-   ✅ 避免局部最优，鼓励**全局最优策略**

**缺点**：

-   ❌ 学习信号**稀疏**，早期训练困难
-   ❌ 需要大量探索才能发现有效策略

#### 2.5.4 为什么使用混合策略

**训练早期**（迭代 0-200）：

-   密集奖励占主导（100% → 37%）
-   智能体学会：如何打球、如何进球、如何避免犯规
-   像教小孩：一步步纠正，及时反馈

**训练中期**（迭代 200-500）：

-   混合奖励（37% → 7%）
-   智能体开始关注：如何赢得比赛
-   从"会打球"到"会比赛"

**训练后期**（迭代 500-1000）：

-   稀疏奖励占主导（7% → 1%）
-   智能体优化：最优获胜策略
-   纯粹追求胜利

### 2.6 自对弈机制

#### 2.6.1 为什么需要自对弈

**问题**：如果只和固定对手训练会怎样？

-   智能体会**过拟合**到该对手的弱点
-   无法泛化到其他对手
-   可能陷入"剪刀石头布"循环

**解决方案**：对手池 (Opponent Pool)

```
当前策略 ──────→ 保存快照 ──→ 对手池
    │                          ↑
    │                          │
    └──→ 训练 ←── 随机采样对手 ──┘
```

#### 2.6.2 对手池管理

**策略**：

-   每 50 次迭代保存一个对手快照
-   保留最近的 5 个快照
-   训练时随机选择一个对手

**示例时间线**：

```
迭代0-49:   自己 vs 自己（池为空）
迭代50:     保存快照1，池=[快照1]
迭代51-99:  自己 vs 快照1
迭代100:    保存快照2，池=[快照1, 快照2]
迭代101-149: 自己 vs 随机(快照1或快照2)
...
迭代250:    保存快照6，池=[快照2, 快照3, 快照4, 快照5, 快照6]
            （快照1被删除，保持池大小=5）
```

**为什么是 5 个快照**？

-   太少（<3）：多样性不足
-   太多（>10）：旧快照太弱，训练效率低
-   5 个：平衡点 ✅

#### 2.6.3 训练流程

单次迭代的完整流程：

```python
for iteration in range(1000):
    # 1. 清空经验缓冲区
    buffer.clear()

    # 2. 加载随机对手（或早期使用自己）
    opponent = load_random_opponent()

    # 3. 收集10局自对弈数据
    for game in range(10):
        # 自己(A) vs 对手(B)
        play_one_game()
        # 只保存自己的轨迹到buffer

    # 4. 计算优势函数 (GAE)
    advantages, returns = compute_gae(buffer)

    # 5. PPO更新（10个epoch）
    for epoch in range(10):
        for batch in shuffle(buffer):
            # 计算PPO损失
            loss = ppo_loss(batch)
            # 反向传播
            optimizer.step()

    # 6. 评估（每50次迭代）
    if iteration % 50 == 0:
        evaluate_vs_baselines()

    # 7. 保存对手快照（每50次迭代）
    if iteration % 50 == 0:
        save_opponent_snapshot()

    # 8. 保存检查点（每100次迭代）
    if iteration % 100 == 0:
        save_checkpoint()
```

---

## 3. 配置参数详解

### 3.1 训练参数 (training)

#### `n_iterations` (默认: 1000)

**含义**：总训练迭代次数

**一次迭代包含**：

1. 收集 `games_per_iteration` 局游戏数据
2. 进行 `n_epochs` 轮 PPO 更新
3. （可选）评估、保存快照

**如何选择**：

-   **快速测试**：10-50 次迭代（~30 分钟）
-   **初步训练**：200-500 次迭代（~10-20 小时）
-   **完整训练**：1000 次迭代（~40-50 小时，GPU）
-   **深度训练**：2000+次迭代（如果 1000 次后仍在提升）

**示例**：

```yaml
n_iterations: 1000 # 完整训练，预期达到85%+ vs BasicAgent
```

---

#### `games_per_iteration` (默认: 10)

**含义**：每次迭代收集多少局游戏数据

**影响**：

-   **样本多样性**：更多游戏 = 更多样的经验
-   **训练时间**：10 局 ≈ 2-5 分钟（取决于游戏长度）

**如何选择**：

-   **太少**（<5 局）：数据不够多样，可能过拟合
-   **太多**（>20 局）：收集数据耗时长，迭代慢
-   **推荐**：10 局 ✅

**计算**：

```
每次迭代的总转换数 ≈ games_per_iteration × 平均步数/局
                  ≈ 10 × 30 = 300 转换
```

---

#### `gamma` (默认: 0.99)

**含义**：折扣因子，衡量对未来奖励的重视程度

**公式**：

$$
G = r_0 + \gamma r_1 + \gamma^2 r_2 + \gamma^3 r_3 + \dots
$$

**直观理解**：

-   **γ = 0.9**：只看近 10 步（短视）

    -   第 1 步：权重 = 1.0
    -   第 10 步：权重 = 0.9^10 ≈ 0.35
    -   第 20 步：权重 ≈ 0.12

-   **γ = 0.99**（推荐）：看远 100 步（平衡）

    -   第 1 步：权重 = 1.0
    -   第 10 步：权重 ≈ 0.90
    -   第 50 步：权重 ≈ 0.61
    -   第 100 步：权重 ≈ 0.37

-   **γ = 0.999**：看远 1000 步（极度远视）
    -   可能导致训练不稳定

**台球游戏的选择**：

-   平均每局 30-40 杆
-   γ=0.99 意味着智能体会考虑未来 40 杆左右
-   适合学习长期策略（如连击、防守）

---

#### `gae_lambda` (默认: 0.95)

**含义**：GAE 参数，控制优势估计的偏差-方差权衡

**公式**：

$$
A_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k \, \delta_{t+k}
$$

**直观理解**：

    **λ = 0**：$A_t = \delta_t$（只看一步 TD 误差）

    -   低方差，但高偏差
    -   适合环境噪声大的情况

    **λ = 1**：$A_t$ = 完整 MC 回报

    -   无偏差，但高方差
    -   适合确定性环境

-   **λ = 0.95**（推荐）：平衡点
    -   中等偏差，中等方差
    -   适合大多数情况

**如何选择**：

-   环境**噪声大**（如台球物理有随机性）：降低 λ → 0.9
-   环境**确定性强**：提高 λ → 0.98
-   **默认**：保持 0.95 ✅

---

#### `clip_epsilon` (默认: 0.2)

**含义**：PPO 裁剪范围，限制策略更新幅度

**公式**：

$$
\mathrm{ratio} = \frac{\pi_{\text{new}}(a\mid s)}{\pi_{\text{old}}(a\mid s)},\qquad
\mathrm{clipped\_ratio} = \operatorname{clip}(\mathrm{ratio},\; 1-\varepsilon,\; 1+\varepsilon)
$$

**直观理解**：

| epsilon | ratio 范围 | 策略变化 | 稳定性    | 学习速度 |
| ------- | ---------- | -------- | --------- | -------- |
| 0.1     | [0.9, 1.1] | 小       | 高 ⭐⭐⭐ | 慢       |
| 0.2 ✅  | [0.8, 1.2] | 中       | 中 ⭐⭐   | 中       |
| 0.3     | [0.7, 1.3] | 大       | 低 ⭐     | 快       |

**如何选择**：

    训练**不稳定**（loss 震荡）：降低 $\varepsilon \to 0.1$
    训练**太慢**：提高 $\varepsilon \to 0.3$

-   **默认**：保持 0.2 ✅

---

#### `learning_rate_policy` (默认: 3e-4)

**含义**：策略网络的学习率

**推荐范围**：

-   **太小**（<1e-4）：学习太慢
-   **太大**（>1e-3）：不稳定，可能发散
-   **最佳**：3e-4（Adam 优化器的经典值）

**学习率调度**（可选）：

```python
# 线性衰减
lr = 3e-4 * (1 - iteration / n_iterations)

# 余弦衰减
lr = 3e-4 * 0.5 * (1 + cos(π * iteration / n_iterations))
```

---

#### `learning_rate_value` (默认: 1e-3)

**含义**：价值网络的学习率

**为什么比策略网络高**？

-   价值网络是**回归任务**（预测标量）
-   策略网络是**分布学习**（更复杂，需谨慎）
-   价值网络可以更快收敛

**比例关系**：

```
lr_value / lr_policy ≈ 3-5倍
```

---

#### `n_epochs` (默认: 10)

**含义**：每批数据重复训练多少轮

**单次迭代的更新次数**：

```
总更新次数 = n_epochs × (buffer_size / batch_size)
           = 10 × (2048 / 64)
           = 10 × 32
           = 320次梯度更新
```

**如何选择**：

-   **太少**（<5）：数据利用不充分
-   **太多**（>15）：可能过拟合到旧数据
-   **推荐**：10 ✅

---

#### `batch_size` (默认: 64)

**含义**：每次梯度更新使用的样本数

**影响**：

-   **小批量**（16-32）：

    -   ✅ 更新频繁，收敛可能更快
    -   ❌ 梯度估计噪声大，不稳定

-   **大批量**（128-256）：
    -   ✅ 梯度估计准确，训练稳定
    -   ❌ 更新慢，可能陷入局部最优

**推荐**：64（平衡点）

---

#### `buffer_size` (默认: 2048)

**含义**：经验缓冲区容量（存储多少转换）

**计算**：

```
10局游戏 × 平均30步/局 ≈ 300转换

每次迭代收集300转换，缓冲区可存2048转换
→ 缓冲区可容纳约6-7次迭代的数据
```

**如何选择**：

-   必须 **> games_per_iteration × 平均步数**
-   推荐：2048-4096

---

#### `max_grad_norm` (默认: 0.5)

**含义**：梯度裁剪阈值

**作用**：防止梯度爆炸

**公式**：

```
if ||grad|| > max_grad_norm:
    grad = grad * (max_grad_norm / ||grad||)
```

**如何选择**：

-   训练**发散**（loss 突然变 NaN）：降低到 0.3
-   训练**稳定**：可保持 0.5

---

#### `value_coef` (默认: 0.5)

**含义**：价值损失在总损失中的权重

**总损失**：

```
Loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

**如何选择**：

-   价值网络学得**太慢**：提高到 1.0
-   价值网络学得**太快**（过拟合）：降低到 0.3
-   **默认**：0.5 ✅

---

#### `entropy_coef` (默认: 0.01)

**含义**：熵奖励系数，鼓励探索

**影响**：

-   **太小**（<0.001）：策略过早收敛，探索不足
-   **太大**（>0.1）：策略太随机，难以利用已学知识
-   **推荐**：0.01 ✅

**监控指标**：

-   熵应该缓慢下降（从 2-3 降到 0.5-1）
-   如果熵快速降到 0：增大 entropy_coef
-   如果熵一直很高：减小 entropy_coef

---

### 3.2 网络参数 (network)

#### `state_dim` (固定: 76)

**含义**：状态向量维度

**构成**（见 2.3 节）：

```
76 = 2 (白球) + 45 (彩球) + 12 (球袋) + 15 (目标掩码) + 2 (全局特征)
```

**不建议修改**，除非改变状态编码方式。

---

#### `action_dim` (固定: 5)

**含义**：动作向量维度

**构成**：

```
5 = V0, phi, theta, a, b
```

---

#### `hidden_sizes` (默认: [256, 256, 128])

**含义**：MLP 隐藏层尺寸

**当前架构**：

```
76 → 256 → 256 → 128 → 输出
     ↑     ↑     ↑
   第1层 第2层 第3层
```

**如何选择**：

| 配置               | 参数量 | 适用场景             | 训练速度  |
| ------------------ | ------ | -------------------- | --------- |
| [128, 128]         | 少     | 简单环境，快速原型   | 快 ⭐⭐⭐ |
| [256, 256, 128] ✅ | 中     | 默认，适合大多数情况 | 中 ⭐⭐   |
| [512, 512, 256]    | 多     | 极复杂环境           | 慢 ⭐     |

**参数量计算**（当前配置）：

```
策略网络：
  76×256 + 256×256 + 256×128 + 128×5×2 ≈ 120K参数
价值网络：
  76×256 + 256×256 + 256×128 + 128×1 ≈ 118K参数
```

---

### 3.3 奖励参数 (reward)

#### `decay_type` (默认: "exponential")

**含义**：密集 → 稀疏奖励的衰减方式

**选项**：

**1. exponential（指数衰减）✅**

```python
w_dense = exp(-5 * t/T)
```

-   前期快速减少，后期缓慢
-   适合大多数情况

**2. linear（线性衰减）**

```python
w_dense = 1 - t/T
```

-   匀速减少
-   更平滑，但可能转换太慢

**3. step（阶梯衰减）**

```python
if t < 0.3T: w_dense = 1.0
elif t < 0.7T: w_dense = 0.5
else: w_dense = 0.0
```

-   突变式转换
-   可能导致训练不稳定

**可视化对比**：

```
w_dense
1.0 ┤███exponential
    │   ╲
0.8 ┤    ╲___linear
    │        ╲___
0.6 ┤            ╲___step
    │                ╲━━━
0.4 ┤
    │
0.2 ┤
0.0 ┤
    0   200  400  600  800  1000
```

---

#### `total_iterations` (默认: 1000)

**含义**：完全过渡到稀疏奖励的迭代数

**建议**：与 `n_iterations` 保持一致

---

### 3.4 自对弈参数 (self_play)

#### `max_opponent_pool_size` (默认: 5)

**含义**：对手池最大容量

**如何选择**：

-   **太小**（<3）：对手多样性不足
-   **太大**（>10）：旧对手太弱，降低训练质量
-   **推荐**：5 ✅

---

#### `snapshot_interval` (默认: 50)

**含义**：每隔多少次迭代保存一个对手快照

**如何选择**：

-   **太频繁**（<20）：对手差异小，池变化慢
-   **太稀疏**（>100）：对手跨度大，可能有断层
-   **推荐**：50 ✅

---

### 3.5 评估参数 (evaluation)

#### `eval_interval` (默认: 50)

**含义**：每隔多少次迭代评估一次

**作用**：监控训练进度

**如何选择**：

-   **频繁评估**（20）：密切监控，但耗时
-   **稀疏评估**（100）：节省时间，但可能错过问题
-   **推荐**：50 ✅

---

#### `eval_games` (默认: 20)

**含义**：每次评估打多少局

**统计显著性**：

-   20 局：标准误差 ≈ 11%（如胜率 50%）
-   50 局：标准误差 ≈ 7%
-   100 局：标准误差 ≈ 5%

**推荐**：20 局（平衡速度和准确性）

---

#### `opponents` (默认: [RandomAgent, BasicAgent, GeometryAgent])

**含义**：评估时的对手列表

**建议配置**：

```yaml
opponents:
    - RandomAgent # 底线（应该95%+胜率）
    - BasicAgent # 中等难度（目标85%+）
    - GeometryAgent # 强敌（目标65%+）
```

---

### 3.6 检查点参数 (checkpointing)

#### `checkpoint_interval` (默认: 100)

**含义**：每隔多少次迭代保存完整检查点

**检查点内容**：

-   策略网络权重
-   价值网络权重
-   优化器状态
-   训练元数据

---

#### `checkpoint_dir` (默认: "checkpoints/ppo")

**含义**：检查点保存目录

---

## 4. 训练监控指标

### 4.1 TensorBoard 概览

启动 TensorBoard：

```bash
tensorboard --logdir runs/ppo_training
```

访问：http://localhost:6006

### 4.2 损失指标 (Loss)

#### `Loss/policy` - 策略损失

**含义**：策略网络的优化目标（越小越好）

**健康曲线**：

```
Loss
0.5 ┤╲
    │ ╲
0.3 ┤  ╲___
    │      ╲___
0.1 ┤          ╲______
    │                 ━━━━━
    0   200  400  600  800  1000
```

**异常情况**：

**1. 不下降**

```
Loss
0.5 ┤━━━━━━━━━━━━━━━━━━━━
    0   200  400  600  800
```

**原因**：学习率太小，或策略已收敛
**解决**：增大 learning_rate_policy 到 5e-4

**2. 剧烈震荡**

```
Loss
0.5 ┤  ╱╲  ╱╲  ╱╲
    │ ╱  ╲╱  ╲╱  ╲
0.3 ┤╱
    0   200  400
```

**原因**：学习率太大，或 clip_epsilon 太大
**解决**：降低 learning_rate 或 clip_epsilon

---

#### `Loss/value` - 价值损失

**含义**：价值网络预测误差（越小越好）

**健康曲线**：

```
Loss
0.3 ┤╲
    │ ╲
0.2 ┤  ╲
    │   ╲___
0.1 ┤       ╲______
    │              ━━━━
    0   200  400  600  800
```

**目标**：最终稳定在 0.05-0.1 之间

**异常：持续很高**（>0.3）

-   价值网络容量不足：增大 hidden_sizes
-   学习率太小：增大 learning_rate_value

---

#### `Loss/entropy` - 策略熵

**含义**：策略的随机性（越大越探索）

**健康曲线**：

```
Entropy
3.0 ┤━━━╲
    │    ╲
2.0 ┤     ╲___
    │         ╲
1.0 ┤          ╲___
    │              ╲______
0.5 ┤                     ━━━━
    0   200  400  600  800  1000
```

**关键阶段**：

-   **0-200 迭代**：保持在 2-3（充分探索）
-   **200-600 迭代**：逐渐降到 1-2（探索转利用）
-   **600+迭代**：稳定在 0.5-1（主要利用，少量探索）

**异常：快速降到 0**

```
Entropy
3.0 ┤╲
    │ ╲
    │  ╲
0.0 ┤   ━━━━━━━━━━━━
    0  50 100 150
```

**问题**：策略过早收敛，失去探索能力
**解决**：增大 entropy_coef 到 0.05

**异常：一直很高**

```
Entropy
3.0 ┤━━━━━━━━━━━━━━━━━
    0   200  400  600
```

**问题**：策略太随机，无法利用
**解决**：减小 entropy_coef 到 0.005

---

### 4.3 PPO 诊断指标

#### `PPO/approx_kl` - 近似 KL 散度

**含义**：新旧策略的差异程度

**健康范围**：0.001 - 0.02

**曲线示例**：

```
KL
0.03┤
    │  ╱╲  ╱╲
0.02┤ ╱  ╲╱  ╲  ╱╲
    │╱         ╲╱  ╲___
0.01┤               ━━━━━━
    0   200  400  600  800
```

**异常：持续>0.02**

```
KL
0.05┤━━━━━━━━━━━━━
    0   200  400
```

**问题**：策略更新太激进
**解决**：

-   降低 learning_rate_policy 到 1e-4
-   减小 clip_epsilon 到 0.1
-   减少 n_epochs 到 5

**异常：接近 0**

```
KL
0.002┤━━━━━━━━━━━━
     0   200  400
```

**问题**：策略几乎不更新
**解决**：增大 learning_rate 或增加 n_epochs

---

#### `PPO/clipfrac` - 裁剪比例

**含义**：被裁剪的样本比例

**健康范围**：0.1 - 0.3

**直观理解**：

-   clipfrac = 0.2 → 20%的样本被裁剪
-   意味着这 20%的样本更新幅度超过了 clip_epsilon 限制

**曲线示例**：

```
Clipfrac
0.3 ┤  ╱╲    ╱╲
    │ ╱  ╲  ╱  ╲
0.2 ┤╱    ╲╱    ╲___
    │              ╲______
0.1 ┤                     ━━━━
    0   200  400  600  800
```

**异常：持续>0.5**

```
Clipfrac
0.6 ┤━━━━━━━━━━━━
    0   200  400
```

**问题**：太多样本被裁剪，PPO 约束过强
**解决**：增大 clip_epsilon 到 0.3

**异常：接近 0**

```
Clipfrac
0.05┤━━━━━━━━━━━━
    0   200  400
```

**问题**：PPO 约束不起作用
**解决**：可能已收敛，或需要增大 learning_rate

---

### 4.4 奖励指标

#### `Reward/mean` - 平均奖励

**含义**：每局游戏的平均奖励

**健康曲线**：

```
Reward
1.0 ┤               ╱━━━━
    │              ╱
0.5 ┤         ╱━━━╱
    │    ╱━━━╱
0.0 ┤━━━━╱
    │
-0.5┤
    0   200  400  600  800  1000
```

**阶段分析**：

-   **0-200 迭代**：密集奖励主导，逐步从负转正
-   **200-600 迭代**：混合奖励，快速提升
-   **600+迭代**：稀疏奖励，趋向+1（赢）或-1（输）

**异常：持续为负**

```
Reward
0.0 ┤
    │
-0.5┤━━━━━━━━━━━━━━
    0   200  400  600
```

**问题**：智能体学不会或对手太强
**解决**：

-   检查奖励函数设计
-   降低对手强度（暂时移除 GeometryAgent）
-   增大密集奖励权重（调整 decay_type 为 linear）

---

#### `Reward/std` - 奖励标准差

**含义**：奖励的波动程度

**健康曲线**：

```
Std
1.0 ┤━━━╲
    │    ╲
0.8 ┤     ╲___
    │         ╲
0.6 ┤          ╲___
    │              ╲______
0.4 ┤                     ━━━━
    0   200  400  600  800  1000
```

**含义**：

-   高标准差（早期）：策略不稳定，表现波动大
-   低标准差（后期）：策略稳定，表现可预测

---

#### `Reward/dense_weight` - 密集奖励权重

**含义**：当前密集奖励的权重

**曲线**（指数衰减）：

```
Weight
1.0 ┤╲
    │ ╲
0.8 ┤  ╲
    │   ╲
0.6 ┤    ╲
    │     ╲
0.4 ┤      ╲
    │       ╲
0.2 ┤        ╲___
    │            ╲______
0.0 ┤                   ━━━━
    0   200  400  600  800  1000
```

**用途**：确认奖励衰减按预期进行

---

### 4.5 评估指标

#### `Eval/win_rate_vs_randomagent`

**目标**：>95%

**曲线示例**：

```
Win Rate
100%┤              ╱━━━━━━━
    │             ╱
 80%┤        ╱━━━━╱
    │   ╱━━━╱
 50%┤━━━╱
    0   100  200  300  400  500
```

**里程碑**：

-   迭代 50：>80%
-   迭代 100：>90%
-   迭代 200：>95%

**异常：长期<80%**
→ 基本功能有问题，检查代码

---

#### `Eval/win_rate_vs_basicagent`

**目标**：>85%

**曲线示例**：

```
Win Rate
100%┤                    ╱━━━
    │                 ╱━━╱
 80%┤            ╱━━━━╱
    │       ╱━━━━╱
 50%┤━━━━━━━╱
    │
 20%┤
    0   200  400  600  800  1000
```

**里程碑**：

-   迭代 200：>50%
-   迭代 500：>70%
-   迭代 800：>85%

---

#### `Eval/win_rate_vs_geometryagent`

**目标**：>65%

**曲线示例**：

```
Win Rate
 80%┤                      ╱━━
    │                  ╱━━━╱
 60%┤             ╱━━━━╱
    │        ╱━━━━╱
 40%┤━━━━━━━━╱
    │
 20%┤
    0   200  400  600  800  1000
```

**里程碑**：

-   迭代 300：>40%
-   迭代 600：>55%
-   迭代 1000：>65%

**备注**：GeometryAgent 很强，达到 65%已经很优秀

---

#### `Eval/avg_shots_per_game`

**含义**：平均每局游戏的杆数

**健康趋势**：下降（更高效）

```
Shots
50 ┤━━━╲
   │    ╲
40 ┤     ╲___
   │         ╲
30 ┤          ╲___
   │              ╲______
25 ┤                     ━━━━
   0   200  400  600  800  1000
```

**含义**：

-   40-50 杆：新手水平（乱打）
-   30-35 杆：中等水平（有策略）
-   25-30 杆：高手水平（连击、精准）
-   <25 杆：大师水平（极致效率）

---

### 4.6 动作分布

#### `Action/V0` - 速度分布

**健康分布**（直方图）：

```
Frequency
    ╱╲
   ╱  ╲        早期：广泛探索
  ╱    ╲
━╱      ╲━━━━━━
0  2  4  6  8  V0

Frequency
      ╱╲
     ╱  ╲     后期：集中在有效范围
    ╱    ╲
━━━╱      ╲━━━
0  2  4  6  8  V0
```

**异常：极端分布**

```
Frequency
╱╲              全是最小速度
  ╲━━━━━━━━━━━━
0  2  4  6  8  V0
```

→ 策略退化，检查奖励函数

---

#### `Action/phi` - 水平角度分布

**健康分布**：应覆盖 0-360°

```
Frequency
━━━━━━━━━━━━    均匀分布（早期）
0  90 180 270 360  phi

Frequency
  ╱╲  ╱╲  ╱╲     多峰分布（后期，针对不同球袋）
━╱  ╲╱  ╲╱  ╲━━
0  90 180 270 360  phi
```

---

### 4.7 监控检查清单

**每日检查**（如果长期训练）：

-   [ ] **Loss/policy**：是否平稳下降？
-   [ ] **Loss/entropy**：是否在合理范围（0.5-2）？
-   [ ] **PPO/approx_kl**：是否<0.02？
-   [ ] **Reward/mean**：是否上升？
-   [ ] **Eval/win_rate_vs_basic**：是否提升？

**异常检查**：

-   [ ] 是否有 Loss 突然跳变（NaN/Inf）？
-   [ ] 是否有指标停滞不动？
-   [ ] 是否有评估胜率突然下降？

---

## 5. 超参数调优指南

### 5.1 训练不稳定

#### 症状 1：Loss 震荡

```
Loss
0.5 ┤╱╲╱╲╱╲╱╲╱╲
    0  50 100 150
```

**诊断**：

-   检查`PPO/approx_kl`：如果>0.03，策略更新太激进
-   检查`PPO/clipfrac`：如果>0.5，裁剪过多

**解决方案**（按优先级）：

1. **降低学习率**

```yaml
learning_rate_policy: 0.0001 # 从0.0003降到0.0001
learning_rate_value: 0.0005 # 从0.001降到0.0005
```

2. **收紧裁剪**

```yaml
clip_epsilon: 0.1 # 从0.2降到0.1
```

3. **减少 epoch**

```yaml
n_epochs: 5 # 从10降到5
```

4. **梯度裁剪**

```yaml
max_grad_norm: 0.3 # 从0.5降到0.3
```

---

#### 症状 2：Loss 突然变 NaN

```
Loss
0.3 ┤━━━━━╲
    │      ╲
    │       ╲
NaN ┤        ✕✕✕✕
```

**可能原因**：

-   梯度爆炸
-   数值不稳定（exp 溢出）
-   学习率太大

**紧急解决**：

1. **重启训练，降低学习率**

```yaml
learning_rate_policy: 0.0001
learning_rate_value: 0.0003
```

2. **加强梯度裁剪**

```yaml
max_grad_norm: 0.3
```

3. **检查数据**

```python
# 在代码中添加检查
assert not torch.isnan(loss).any(), "Loss is NaN!"
assert not torch.isinf(loss).any(), "Loss is Inf!"
```

---

### 5.2 学习太慢

#### 症状：Loss 下降缓慢

```
Loss
0.5 ┤━━━━━━━━━━━━━━━━━━╲
    0   200  400  600  800  ╲
```

**诊断**：

-   检查`Reward/mean`：是否在提升？
-   检查`Eval/win_rate`：是否在增长？

**如果奖励在提升但很慢**：

1. **增大学习率**

```yaml
learning_rate_policy: 0.0005 # 从0.0003增到0.0005
```

2. **增加更新频率**

```yaml
n_epochs: 15 # 从10增到15
```

3. **增大 batch size**

```yaml
batch_size: 128 # 从64增到128
```

**如果奖励不提升**：

1. **增大探索**

```yaml
entropy_coef: 0.03 # 从0.01增到0.03
```

2. **检查奖励函数**

-   确认奖励有区分度
-   检查是否所有动作都得到相似奖励

3. **增大网络容量**

```yaml
hidden_sizes: [512, 512, 256] # 从[256, 256, 128]增大
```

---

### 5.3 策略过早收敛

#### 症状：熵快速下降到 0

```
Entropy
3.0 ┤╲
    │ ╲
    │  ╲
0.0 ┤   ━━━━━━━━━
    0  50 100 150
```

同时`Eval/win_rate`不高（<70% vs BasicAgent）

**问题**：策略陷入局部最优，失去探索能力

**解决方案**：

1. **增大熵奖励**

```yaml
entropy_coef: 0.05 # 从0.01增到0.05
```

2. **温度调节**（修改代码）

```python
# 在PolicyNetwork中添加温度参数
std = torch.exp(log_std) * temperature  # temperature=2.0增加探索
```

3. **重启训练**（如果已经僵化）

-   从较早的 checkpoint 恢复
-   调高 entropy_coef 后重新训练

---

### 5.4 无法打败 GeometryAgent

#### 症状：胜率停滞在 40-50%

```
Win Rate vs Geometry
60% ┤
50% ┤━━━━━━━━━━━━━━━━
40% ┤
    0   400  800  1200
```

**可能原因**：

1. **训练不够**

-   解决：继续训练到 1500-2000 迭代
-   GeometryAgent 很强，需要更长时间

2. **奖励偏向短期**

-   问题：密集奖励鼓励进球，但不鼓励战略
-   解决：加速稀疏奖励过渡

```yaml
reward:
    decay_type: linear # 改为线性衰减，更快转向胜负
    total_iterations: 500 # 提前完成过渡
```

3. **探索不足**

```yaml
self_play:
    max_opponent_pool_size: 8 # 增大对手池
    snapshot_interval: 30 # 更频繁保存
```

4. **网络容量不足**

```yaml
hidden_sizes: [512, 512, 256] # 增大网络
```

---

### 5.5 样本效率低

#### 症状：需要很多局才能学会

**优化数据收集**：

1. **增大 buffer**

```yaml
buffer_size: 4096 # 从2048增到4096
```

2. **增加每次迭代的游戏数**

```yaml
games_per_iteration: 20 # 从10增到20
```

3. **更多 epoch 重用数据**

```yaml
n_epochs: 15 # 从10增到15
```

**注意**：过度重用数据会导致过拟合，需监控`PPO/approx_kl`

---

### 5.6 调优流程图

```
开始训练
    ↓
检查是否稳定？
    ├─ 否（震荡/NaN）→ 降低学习率、收紧裁剪
    └─ 是 ↓

检查是否学习？
    ├─ 否（奖励不升）→ 增大探索、检查奖励函数
    └─ 是 ↓

检查探索是否充分？
    ├─ 否（熵过低）→ 增大entropy_coef
    └─ 是 ↓

检查是否打败目标对手？
    ├─ 否 → 继续训练/调整奖励/增大网络
    └─ 是 ↓

完成训练！
```

---

### 5.7 推荐配置组合

#### 配置 1：快速测试（适合调试）

```yaml
training:
    n_iterations: 50
    games_per_iteration: 5
    learning_rate_policy: 0.001 # 高学习率快速收敛
    n_epochs: 5

evaluation:
    eval_interval: 10
    eval_games: 10
```

**预期**：30 分钟，达到>80% vs RandomAgent

---

#### 配置 2：标准训练（推荐）

```yaml
training:
    n_iterations: 1000
    games_per_iteration: 10
    learning_rate_policy: 0.0003
    n_epochs: 10

evaluation:
    eval_interval: 50
    eval_games: 20
```

**预期**：40-50 小时，达到>85% vs BasicAgent

---

#### 配置 3：深度训练（追求极致）

```yaml
training:
    n_iterations: 2000
    games_per_iteration: 15
    learning_rate_policy: 0.0002 # 后期降低学习率
    n_epochs: 12
    hidden_sizes: [512, 512, 256] # 更大网络

self_play:
    max_opponent_pool_size: 8

evaluation:
    eval_interval: 50
    eval_games: 50
```

**预期**：100+小时，达到>70% vs GeometryAgent

---

## 6. 常见问题与解决方案

### Q1: 训练速度太慢怎么办？

**A1: GPU 加速**

确认使用 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出`False`，安装 GPU 版本 PyTorch：

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**A2: 减少评估频率**

```yaml
evaluation:
    eval_interval: 100 # 从50改到100
    eval_games: 10 # 从20改到10
```

**A3: 使用更小的网络**

```yaml
hidden_sizes: [128, 128] # 从[256, 256, 128]减小
```

---

### Q2: 显存不足（CUDA out of memory）

**A1: 减小 batch size**

```yaml
batch_size: 32 # 从64减到32
```

**A2: 减小网络**

```yaml
hidden_sizes: [128, 128]
```

**A3: 清理 GPU 缓存**

```python
# 在训练代码中添加
import torch
torch.cuda.empty_cache()
```

---

### Q3: 训练到一半崩溃，如何恢复？

**A: 使用 checkpoint 恢复**

```bash
# 找到最近的checkpoint
ls checkpoints/ppo/

# 恢复训练
python src/train/scripts/train_ppo.py \
    --config configs/ppo_training.yaml \
    --resume checkpoints/ppo/checkpoint_800.pth
```

---

### Q4: 如何判断是否需要继续训练？

**检查以下指标**：

-   [ ] `Eval/win_rate_vs_basicagent` < 85%？→ 继续训练
-   [ ] `Loss/policy` 仍在下降？→ 继续训练
-   [ ] `Reward/mean` 仍在提升？→ 继续训练
-   [ ] 最近 100 次迭代无明显提升？→ 可以停止

---

### Q5: PPO vs GeometryAgent vs BasicAgent，该如何选择？

**场景选择**：

| 场景         | 推荐智能体      | 原因                     |
| ------------ | --------------- | ------------------------ |
| 快速原型     | GeometryAgent   | 无需训练，立即可用       |
| 研究 RL 算法 | PPO             | 学习曲线清晰，可调参数多 |
| 追求最强性能 | PPO（深度训练） | 理论上限最高             |
| 比赛/演示    | 根据对手选择    | 选择胜率最高的           |

---

### Q6: 如何可视化智能体的决策过程？

**A: 添加注意力可视化**（需要修改代码）

```python
# 在decision时记录状态和动作
def decision(self, balls, my_targets, table):
    state = self._encode_state(balls, my_targets, table)
    action = self.policy_net(state)

    # 可视化
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))

    # 左图：球桌布局
    plt.subplot(1, 2, 1)
    for bid, ball in balls.items():
        if ball.state.s != 4:  # 未进袋
            x, y = ball.state.rvw[0][:2]
            color = 'white' if bid == 'cue' else 'red'
            plt.scatter(x, y, c=color, s=100)

    # 右图：动作分布
    plt.subplot(1, 2, 2)
    plt.bar(['V0', 'phi', 'theta', 'a', 'b'], action)

    plt.show()
```

---

### Q7: 训练数据如何保存和分析？

**A: 导出 TensorBoard 数据**

```bash
# 导出为CSV
tensorboard --logdir runs/ppo_training --export_csv results.csv

# 使用pandas分析
import pandas as pd
df = pd.read_csv('results.csv')
df[df['tag'] == 'Eval/win_rate_vs_basicagent'].plot(x='step', y='value')
```

---

## 7. 使用示例

### 7.1 标准训练流程

```bash
# 1. 安装依赖
uv pip install torch tensorboard

# 2. 启动训练
python src/train/scripts/train_ppo.py \
    --config configs/ppo_training.yaml

# 3. 在新终端启动TensorBoard
tensorboard --logdir runs/ppo_training

# 4. 浏览器打开 http://localhost:6006 监控训练

# 5. 等待训练完成（~40-50小时）

# 6. 评估模型
python src/train/scripts/evaluate_ppo.py \
    --model_path checkpoints/ppo/best_model.pth \
    --n_games 120
```

---

### 7.2 快速测试（10 次迭代）

创建测试配置 `configs/ppo_test.yaml`：

```yaml
experiment_name: ppo_test
random_seed_enabled: true
random_seed: 42

training:
    n_iterations: 10
    games_per_iteration: 5
    gamma: 0.99
    gae_lambda: 0.95
    clip_epsilon: 0.2
    learning_rate_policy: 0.001 # 较高学习率
    learning_rate_value: 0.003
    n_epochs: 5
    batch_size: 32
    buffer_size: 1024

network:
    state_dim: 76
    action_dim: 5
    hidden_sizes: [128, 128] # 小网络快速测试

reward:
    decay_type: exponential
    total_iterations: 10

self_play:
    max_opponent_pool_size: 2

evaluation:
    eval_interval: 5
    eval_games: 5
    opponents:
        - RandomAgent

checkpointing:
    checkpoint_interval: 10
    checkpoint_dir: checkpoints/ppo_test
```

运行：

```bash
python src/train/scripts/train_ppo.py --config configs/ppo_test.yaml
```

**预期时间**：~15 分钟
**预期结果**：>60% vs RandomAgent

---

### 7.3 恢复中断的训练

```bash
# 从checkpoint_800恢复
python src/train/scripts/train_ppo.py \
    --config configs/ppo_training.yaml \
    --resume checkpoints/ppo/checkpoint_800.pth

# 训练会从第801次迭代继续
```

---

### 7.4 对比不同配置

**实验 A：标准配置**

```bash
python src/train/scripts/train_ppo.py \
    --config configs/ppo_training.yaml
```

**实验 B：大网络**

创建 `configs/ppo_large.yaml`（修改 hidden_sizes 为[512, 512, 256]）

```bash
python src/train/scripts/train_ppo.py \
    --config configs/ppo_large.yaml
```

**对比结果**：

```bash
tensorboard --logdir runs \
    --port 6006
```

在 TensorBoard 中同时加载两个实验，对比曲线。

---

### 7.5 使用训练好的模型

```bash
# 方式1：使用evaluate.py（已注册PPOAgent）
python src/train/evaluate.py \
    --config configs/ppo_vs_geometry.yaml \
    --n_games 120

# 方式2：使用专用评估脚本
python src/train/scripts/evaluate_ppo.py \
    --model_path checkpoints/ppo/best_model.pth \
    --opponents RandomAgent BasicAgent GeometryAgent BasicAgentPro \
    --n_games 120 \
    --output ppo_results.json

# 查看结果
cat ppo_results.json
```

---

### 7.6 自定义奖励函数

如果想修改奖励函数，编辑 `src/train/agents/ppo_trainer.py`：

```python
def _dense_reward(self, shot_result: Dict) -> float:
    """自定义密集奖励"""
    reward = 0.0

    # 你的自定义奖励逻辑
    reward += len(shot_result.get('ME_INTO_POCKET', [])) * 100  # 提高进球奖励
    reward -= shot_result.get('WHITE_BALL_INTO_POCKET', False) * 200  # 加重白球犯规惩罚

    # 添加新的奖励项：连击奖励
    if len(shot_result.get('ME_INTO_POCKET', [])) >= 2:
        reward += 50  # 一杆多球奖励

    return np.clip(reward / 200.0, -1.0, 1.0)
```

---

### 7.7 导出模型用于部署

```python
# 导出为单文件（只包含策略网络）
import torch
from src.train.agents.ppo import PPOAgent

agent = PPOAgent(model_path='checkpoints/ppo/best_model.pth')

# 保存为ONNX格式（可移植）
dummy_input = torch.randn(1, 76)
torch.onnx.export(
    agent.policy_net,
    dummy_input,
    'ppo_policy.onnx',
    export_params=True
)
```

---

## 8. 附录

### 8.1 文件结构总览

```
AI3603-Billiards/
├── src/train/agents/
│   ├── ppo_networks.py       # 神经网络定义
│   ├── ppo_trainer.py        # 训练器和工具函数
│   └── ppo.py                # PPOAgent类（推理用）
├── src/train/scripts/
│   ├── train_ppo.py          # 训练脚本
│   └── evaluate_ppo.py       # 评估脚本
├── configs/
│   ├── ppo_training.yaml     # 训练配置
│   └── ppo_vs_geometry.yaml  # 评估配置
├── checkpoints/ppo/          # 模型检查点
│   ├── opponent_50.pth
│   ├── checkpoint_100.pth
│   └── best_model.pth
└── runs/ppo_training/        # TensorBoard日志
    └── events.out.tfevents...
```

---

### 8.2 术语表

| 术语     | 英文                             | 解释                           |
| -------- | -------------------------------- | ------------------------------ |
| 策略     | Policy                           | 从状态到动作的映射（神经网络） |
| 价值函数 | Value Function                   | 估计状态的好坏                 |
| 优势函数 | Advantage Function               | 衡量某动作比平均好多少         |
| GAE      | Generalized Advantage Estimation | 优势估计方法                   |
| 裁剪     | Clipping                         | 限制策略更新幅度               |
| 熵       | Entropy                          | 策略的随机性（探索程度）       |
| 对手池   | Opponent Pool                    | 过去版本的策略集合             |
| 检查点   | Checkpoint                       | 训练快照，可恢复               |

---

### 8.3 相关论文

1. **PPO 原论文**
   Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
   [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

2. **GAE**
   Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
   [https://arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)

3. **自对弈**
   Silver et al., "Mastering the game of Go with deep neural networks and tree search" (2016)
   [https://www.nature.com/articles/nature16961](https://www.nature.com/articles/nature16961)

---

### 8.4 推荐资源

**教程**：

-   [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)
-   [强化学习圣经 - Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)

**实现参考**：

-   [Stable-Baselines3 PPO](https://github.com/DLR-RM/stable-baselines3)
-   [CleanRL PPO](https://github.com/vwxyzjn/cleanrl)

---

### 8.5 联系与支持

如有问题，请：

-   查看 TensorBoard 日志分析问题
-   参考本文档第 6 节"常见问题"
-   提交 GitHub Issue（如果是代码 bug）

---

**祝训练顺利，早日打败 GeometryAgent！** 🎱🚀

---

_文档版本：v1.0_
_最后更新：2026-01-01_
