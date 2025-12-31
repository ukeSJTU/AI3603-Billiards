# AI3603-Billiards

AI3603 课程台球大作业

---

# evaluate.py 轮换机制详解

## 核心概念区分

-   **agent_a / agent_b**: 配置文件中定义的两个智能体（要对比的双方)
-   **player A / player B**: 环境中的双方玩家（固定的游戏角色）

## 轮换机制设计思想

代码通过 **4 局一个循环** 来保证公平性：

```python
players = [agent_a, agent_b]  # 索引: 0=agent_a, 1=agent_b
target_ball_choice = ["solid", "solid", "stripe", "stripe"]  # 4局循环
```

## 逐局分析（以 120 局为例）

### 第 0 局 (i=0)

```python
# 球型分配
target_ball_choice[0 % 4] = "solid"
# → player A 打实心球(1-7)，player B 打条纹球(9-15)

# 击球决策
if player == "A":
    action = players[0 % 2].decision(...)  # players[0] = agent_a
else:
    action = players[1 % 2].decision(...)  # players[1] = agent_b

# 胜负统计
if winner == "A":
    results[["AGENT_A_WIN", "AGENT_B_WIN"][0 % 2]] += 1
    # → results["AGENT_A_WIN"] += 1
```

**结果**: agent_a 扮演 player A（打实心球），agent_b 扮演 player B（打条纹球）

---

### 第 1 局 (i=1)

```python
# 球型分配
target_ball_choice[1 % 4] = "solid"
# → player A 打实心球，player B 打条纹球（同上）

# 击球决策（关键变化！）
if player == "A":
    action = players[1 % 2].decision(...)  # players[1] = agent_b
else:
    action = players[2 % 2].decision(...)  # players[0] = agent_a

# 胜负统计
if winner == "A":
    results[["AGENT_A_WIN", "AGENT_B_WIN"][1 % 2]] += 1
    # → results["AGENT_B_WIN"] += 1
```

**结果**: agent_b 扮演 player A（打实心球），agent_a 扮演 player B（打条纹球）  
**效果**: **交换了先手**

---

### 第 2 局 (i=2)

```python
# 球型分配（关键变化！）
target_ball_choice[2 % 4] = "stripe"
# → player A 打条纹球(9-15)，player B 打实心球(1-7)

# 击球决策
if player == "A":
    action = players[2 % 2].decision(...)  # players[0] = agent_a
else:
    action = players[3 % 2].decision(...)  # players[1] = agent_b

# 胜负统计
if winner == "A":
    results[["AGENT_A_WIN", "AGENT_B_WIN"][2 % 2]] += 1
    # → results["AGENT_A_WIN"] += 1
```

**结果**: agent_a 扮演 player A（打条纹球），agent_b 扮演 player B（打实心球）  
**效果**: **交换了球型**

---

### 第 3 局 (i=3)

```python
# 球型分配
target_ball_choice[3 % 4] = "stripe"
# → player A 打条纹球，player B 打实心球（同上）

# 击球决策
if player == "A":
    action = players[3 % 2].decision(...)  # players[1] = agent_b
else:
    action = players[4 % 2].decision(...)  # players[0] = agent_a

# 胜负统计
if winner == "A":
    results[["AGENT_A_WIN", "AGENT_B_WIN"][3 % 2]] += 1
    # → results["AGENT_B_WIN"] += 1
```

**结果**: agent_b 扮演 player A（打条纹球），agent_a 扮演 player B（打实心球）  
**效果**: **同时交换了先手和球型**

---

## 4 局循环总结

**循环模式表:**

-   **局 0, 4, 8...**: agent_a 先手打实心球，agent_b 后手打条纹球（基准）
-   **局 1, 5, 9...**: agent_b 先手打实心球，agent_a 后手打条纹球（交换先手）
-   **局 2, 6, 10...**: agent_a 先手打条纹球，agent_b 后手打实心球（交换球型）
-   **局 3, 7, 11...**: agent_b 先手打条纹球，agent_a 后手打实心球（先手+球型都交换）

**公平性保证:**

1. **消除先手优势**: 每个 agent 各打 60 局先手，60 局后手
2. **消除球型优势**: 每个 agent 各打 60 局实心球，60 局条纹球
3. **组合公平性**: 4 种组合各打 30 局，完全对称

## 关键代码逻辑

**角色分配:**

```python
# 决定谁控制 player A/B
players[i % 2]      # i 为偶数 → agent_a，i 为奇数 → agent_b
players[(i+1) % 2]  # i 为偶数 → agent_b，i 为奇数 → agent_a
```

**胜负归属转换:**

```python
if winner == "A":
    # 如果 player A 赢了，功劳归谁?
    results[["AGENT_A_WIN", "AGENT_B_WIN"][i % 2]] += 1
    # i 偶数时 → AGENT_A_WIN（因为 agent_a 扮演 player A）
    # i 奇数时 → AGENT_B_WIN（因为 agent_b 扮演 player A）
```

这样设计确保了评测的**完全公平性**，任何一方都无法从先手或球型中获得统计学优势。

---

loguru logging 功能：

-   CRITICAL：用来记录当前程序运行到这里，无论如何都无法恢复的情况
-   ERROR：程序出错，但是我们提供了兜底的机制，例如 try 或者重试等等，功能失败但程序继续
-   WARNING：出现了不应该出现的情况，但是仍然继续执行
-   INFO：程序正常运行时，记录一些重要的事件（直接在 console 输出的信息）
-   DEBUG：调试信息，最全的日志信息，等待程序运行完成后再查看 log 文件

---

项目使用了`uv`来管理环境，在安装[`pooltool`]()这个依赖的过程中，遇到了比较麻烦的问题，解决办法是：

```bash
uv add pooltool-billiards --index https://archive.panda3d.org/ --index-strategy unsafe-best-match --prerelease allow
```

我个人认为问题在于`pooltool`这个包依赖的`panda3d`版本问题，可以看到[](https://pooltool.readthedocs.io/en/latest/getting_started/install.html)中推荐的安装方式是：

```bash
pip install pooltool-billiards --extra-index-url https://archive.panda3d.org/
```

> (Providing the Panda3D archive is required until Panda3D v1.11 is released)

TODO: 这里需要进一步补充信息

总之配置完成后，运行：

```bash
python -c "import pooltool; print(pooltool.__version__)"
0.5.0
```

---

Run code below to install pre-commit hooks:

```bash
uv run pre-commit install
```
