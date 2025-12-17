"""
gym_wrapper.py - 将 PoolEnv 包装为 Gymnasium 兼容环境

设计要点：
- 状态空间：将复杂的 balls/table 对象转为固定维度的数值向量
- 动作空间：5维连续动作 (V0, phi, theta, a, b)
- 奖励函数：使用改进的 analyze_shot_for_reward + 局面评估
- 单人训练模式：PPO Agent 对战固定的 BasicAgent
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent import BasicAgent, analyze_shot_for_reward
from poolenv import PoolEnv

# ============ Logger 导入（带降级） ============
try:
    from logger import get_logger

    log = get_logger(__name__)
except ImportError:

    class _FakeLogger:
        """简单的 print-based logger（降级模式）"""

        def info(self, msg):
            print(f"[INFO] {msg}")

        def debug(self, msg):
            pass

        def warning(self, msg):
            print(f"[WARNING] {msg}")

        def error(self, msg):
            print(f"[ERROR] {msg}")

    log = _FakeLogger()


class PoolGymEnv(gym.Env):
    """
    Gymnasium 兼容的台球环境

    观测空间 (74维):
        - 白球位置 (2D)
        - 每个球 [x, y, is_pocketed] × 15球 = 45D
        - 我的目标球标记 (15D, one-hot)
        - 6个袋口位置 (12D)

    动作空间 (5维):
        - V0: [0.5, 8.0]
        - phi: [0, 360]
        - theta: [0, 90]
        - a: [-0.5, 0.5]
        - b: [-0.5, 0.5]

    奖励设计：
        - 基础奖励：analyze_shot_for_reward 的得分
        - 局面奖励：双方剩余球数差 × 权重
        - 胜利奖励：赢 +500, 输 -500
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent_agent=None):
        super().__init__()

        # 环境
        self.env = PoolEnv()
        self.opponent = opponent_agent or BasicAgent()

        # 定义动作空间
        self.action_space = spaces.Box(
            low=np.array([0.5, 0.0, 0.0, -0.5, -0.5], dtype=np.float32),
            high=np.array([8.0, 360.0, 90.0, 0.5, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        # 定义观测空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(74,), dtype=np.float32
        )

        # 统计信息
        self.episode_reward = 0
        self.step_count = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        # 重置环境
        target_ball = options.get("target_ball", "solid") if options else "solid"
        self.env.reset(target_ball=target_ball)

        # 统计归零
        self.episode_reward = 0
        self.step_count = 0

        # 获取初始观测
        obs = self._get_observation()
        info = {}

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        self.step_count += 1

        # 转换动作格式
        action_dict = {
            "V0": float(action[0]),
            "phi": float(action[1]),
            "theta": float(action[2]),
            "a": float(action[3]),
            "b": float(action[4]),
        }

        # 保存击球前状态（用于奖励计算）
        balls_before = {
            bid: copy.deepcopy(ball) for bid, ball in self.env.balls.items()
        }
        my_targets_before = list(self.env.player_targets["A"])

        # 执行 PPO Agent 的动作
        step_info = self.env.take_shot(action_dict)
        done, game_info = self.env.get_done()

        # 计算奖励
        reward = self._compute_reward(
            step_info, balls_before, my_targets_before, done, game_info
        )
        self.episode_reward += reward

        # 如果游戏未结束且轮到对手，执行对手动作
        if not done and self.env.get_curr_player() == "B":
            reward += self._opponent_turn()
            done, game_info = self.env.get_done()

        # 获取新观测
        obs = self._get_observation()

        # 构建 info
        info = {
            "episode_reward": self.episode_reward,
            "step_count": self.step_count,
        }
        if done:
            info.update(game_info)

        return obs, reward, done, False, info

    def _get_observation(self) -> np.ndarray:
        """
        将环境状态转为数值向量

        返回 74 维向量：
        - [0:2]   白球位置
        - [2:47]  15个球 × 3 (x, y, is_pocketed)
        - [47:62] 我的目标球标记 (15D one-hot)
        - [62:74] 6个袋口位置 (12D)
        """
        balls, my_targets, table = self.env.get_observation("A")

        obs = []

        # 1. 白球位置
        cue_pos = balls["cue"].state.rvw[0][:2]
        obs.extend(cue_pos)

        # 2. 所有球的状态
        ball_ids = [str(i) for i in range(1, 16)]
        for bid in ball_ids:
            pos = balls[bid].state.rvw[0][:2]
            is_pocketed = 1.0 if balls[bid].state.s == 4 else 0.0
            obs.extend([pos[0], pos[1], is_pocketed])

        # 3. 我的目标球标记
        target_mask = [1.0 if str(i) in my_targets else 0.0 for i in range(1, 16)]
        obs.extend(target_mask)

        # 4. 袋口位置
        pocket_ids = ["lb", "lc", "lt", "rb", "rc", "rt"]
        for pid in pocket_ids:
            pos = table.pockets[pid].center[:2]
            obs.extend(pos)

        return np.array(obs, dtype=np.float32)

    def _compute_reward(
        self,
        step_info: Dict,
        balls_before: Dict,
        my_targets: List[str],
        done: bool,
        game_info: Dict,
    ) -> float:
        """
        计算奖励

        奖励组成：
        1. 击球奖励：analyze_shot_for_reward 得分
        2. 局面奖励：双方剩余球数差
        3. 终局奖励：胜/负/平
        """
        # 1. 基础击球奖励
        # 创建一个临时 System 对象用于 analyze_shot_for_reward
        # 注意：analyze_shot_for_reward 需要的是 System 对象，不是 balls dict
        # 我们需要从当前环境状态构建
        import pooltool as pt

        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=self.env.table, balls=self.env.balls, cue=cue)

        shot_reward = analyze_shot_for_reward(shot, balls_before, my_targets)

        # 2. 局面评估
        my_remaining = sum(
            1
            for bid in self.env.player_targets["A"]
            if bid != "8" and self.env.balls[bid].state.s != 4
        )
        opp_remaining = sum(
            1
            for bid in self.env.player_targets["B"]
            if bid != "8" and self.env.balls[bid].state.s != 4
        )
        position_reward = (opp_remaining - my_remaining) * 5  # 每球差距 +5

        # 3. 终局奖励
        terminal_reward = 0
        if done:
            if game_info["winner"] == "A":
                terminal_reward = 500
            elif game_info["winner"] == "B":
                terminal_reward = -500
            else:
                terminal_reward = 0

        total_reward = shot_reward + position_reward * 0.1 + terminal_reward
        return total_reward

    def _opponent_turn(self) -> float:
        """执行对手回合（返回额外奖励）"""
        obs = self.env.get_observation("B")
        action = self.opponent.decision(*obs)

        balls_before = {
            bid: copy.deepcopy(ball) for bid, ball in self.env.balls.items()
        }
        opp_targets = list(self.env.player_targets["B"])

        self.env.take_shot(action)

        # 对手得分 = 我方负奖励
        import pooltool as pt

        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=self.env.table, balls=self.env.balls, cue=cue)

        opp_reward = analyze_shot_for_reward(shot, balls_before, opp_targets)
        return -opp_reward * 0.5  # 对手得分的负值（降权）
