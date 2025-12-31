"""
BasicAgentPro - MCTS-based Advanced Agent

This is an advanced agent using Monte Carlo Tree Search (MCTS) for decision making.
It generates heuristic candidate actions and evaluates them through simulation.
"""

import copy
import math
import random
from typing import Any, List, Optional, override

import numpy as np
import pooltool as pt

from src.utils.logger import get_logger

from .base import ActionDict, Agent, BallsDict, analyze_shot_for_reward

logger = get_logger()


class BasicAgentPro(Agent):
    """基于MCTS（蒙特卡洛树搜索）的进阶 Agent"""

    def __init__(
        self,
        n_simulations: int = 50,
        c_puct: float = 1.414,
    ):
        """初始化 BasicAgentPro

        参数：
            n_simulations: MCTS 仿真次数
            c_puct: UCB 探索系数
        """
        super().__init__()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.ball_radius = 0.028575

        # 定义噪声水平 (与 poolenv 保持一致或略大)
        self.sim_noise = {
            "V0": 0.1,
            "phi": 0.15,
            "theta": 0.1,
            "a": 0.005,
            "b": 0.005,
        }

        logger.info("BasicAgentPro (MCTS版) 已初始化。")

    def _calc_angle_degrees(self, v: np.ndarray) -> float:
        """计算向量的角度（度）"""
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(
        self,
        cue_pos: np.ndarray,
        obj_pos: np.ndarray,
        pocket_pos: np.ndarray,
    ) -> tuple:
        """计算幽灵球位置和击球角度

        参数：
            cue_pos: 白球位置
            obj_pos: 目标球位置
            pocket_pos: 袋口位置

        返回：
            (phi角度, 白球到幽灵球距离)
        """
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0:
            return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def generate_heuristic_actions(
        self,
        balls: BallsDict,
        my_targets: List[str],
        table: pt.Table,
    ) -> List[ActionDict]:
        """生成候选动作列表

        参数：
            balls: 球状态字典
            my_targets: 目标球ID列表
            table: 球桌对象

        返回：
            候选动作列表
        """
        actions = []

        cue_ball = balls.get("cue")
        if not cue_ball:
            return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        # 获取所有目标球的ID
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]

        # 如果没有目标球了（理论上外部会处理转为8号，这里兜底）
        if not target_ids:
            target_ids = ["8"]

        # 遍历每一个目标球
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]

            # 遍历每一个袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                # 1. 计算理论进球角度
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)

                # 2. 根据距离简单的估算力度 (距离越远力度越大，基础力度2.0)
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)

                # 3. 生成几个变种动作加入候选池
                # 变种1：精准一击
                actions.append({"V0": v_base, "phi": phi_ideal, "theta": 0, "a": 0, "b": 0})
                # 变种2：力度稍大
                actions.append(
                    {
                        "V0": min(v_base + 1.5, 7.5),
                        "phi": phi_ideal,
                        "theta": 0,
                        "a": 0,
                        "b": 0,
                    }
                )
                # 变种3：角度微调 (左右偏移 0.5 度，应对噪声)
                actions.append(
                    {
                        "V0": v_base,
                        "phi": (phi_ideal + 0.5) % 360,
                        "theta": 0,
                        "a": 0,
                        "b": 0,
                    }
                )
                actions.append(
                    {
                        "V0": v_base,
                        "phi": (phi_ideal - 0.5) % 360,
                        "theta": 0,
                        "a": 0,
                        "b": 0,
                    }
                )

        # 如果通过启发式没有生成任何动作（极罕见），补充随机动作
        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())

        # 随机打乱顺序
        random.shuffle(actions)
        return actions[:30]

    def simulate_action(
        self,
        balls: BallsDict,
        table: pt.Table,
        action: ActionDict,
    ) -> Optional[pt.System]:
        """执行带噪声的物理仿真

        让 Agent 意识到由于误差的存在，某些"极限球"是不可打的

        参数：
            balls: 球状态字典
            table: 球桌对象
            action: 击球动作

        返回：
            模拟后的 System 对象，失败返回 None
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        try:
            # 注入高斯噪声
            noisy_V0 = np.clip(action["V0"] + np.random.normal(0, self.sim_noise["V0"]), 0.5, 8.0)
            noisy_phi = (action["phi"] + np.random.normal(0, self.sim_noise["phi"])) % 360
            noisy_theta = np.clip(
                action["theta"] + np.random.normal(0, self.sim_noise["theta"]), 0, 90
            )
            noisy_a = np.clip(action["a"] + np.random.normal(0, self.sim_noise["a"]), -0.5, 0.5)
            noisy_b = np.clip(action["b"] + np.random.normal(0, self.sim_noise["b"]), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    @override
    def decision(
        self,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[Any] = None,
    ) -> ActionDict:
        """MCTS 决策方法

        Args:
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象

        Returns:
            ActionDict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        if balls is None:
            return self._random_action()

        # 预处理
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ["8"]
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # 生成候选动作
        candidate_actions = self.generate_heuristic_actions(balls, my_targets, table)
        n_candidates = len(candidate_actions)

        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)

        # MCTS 循环
        for i in range(self.n_simulations):
            # Selection (UCB)
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                # 使用归一化后的 Q 进行计算
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(
                    np.log(total_n + 1) / (N + 1e-6)
                )
                idx = np.argmax(ucb_values)

            # Simulation (带噪声)
            shot = self.simulate_action(balls, table, candidate_actions[idx])

            # Evaluation
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)

            # 映射公式: (val - min) / (max - min)
            normalized_reward = (raw_reward - (-500)) / 650.0
            # 截断一下防止越界
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            # Backpropagation
            N[idx] += 1
            Q[idx] += normalized_reward  # 累加归一化后的分数

        # Final Decision
        # 选平均分最高的 (Robust Child)
        avg_rewards = Q / (N + 1e-6)
        best_idx = np.argmax(avg_rewards)
        best_action = candidate_actions[best_idx]

        # 简单打印一下当前最好的预测胜率
        logger.info(
            f"[BasicAgentPro] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {self.n_simulations})"
        )

        return best_action
