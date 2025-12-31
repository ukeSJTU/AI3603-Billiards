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
        """计算向量的角度(度)

        将二维向量转换为角度表示(0-360度)
        使用 atan2 可以正确处理所有象限的角度

        参数:
            v: 二维向量 [x, y]

        返回:
            角度值(度),范围 [0, 360)
        """
        angle = math.degrees(math.atan2(v[1], v[0]))  # atan2返回 [-180, 180],转换为度
        return angle % 360  # 归一化到 [0, 360)

    def _get_ghost_ball_target(
        self,
        cue_pos: np.ndarray,
        obj_pos: np.ndarray,
        pocket_pos: np.ndarray,
    ) -> tuple:
        """计算幽灵球位置和击球角度

        幽灵球方法(Ghost Ball Method)是台球中的经典瞄准技巧:
        1. 想象一个"幽灵球"紧贴在目标球后方
        2. 幽灵球的位置使得目标球恰好能沿着"目标球-袋口"连线进袋
        3. 白球只需瞄准并击中这个幽灵球的位置即可

        几何原理:
        - 目标球进袋的方向: 从目标球指向袋口
        - 幽灵球位置: 沿着"袋口-目标球"方向,距离目标球中心 2倍球半径
        - 击球角度: 从白球指向幽灵球的方向

        参数：
            cue_pos: 白球位置 [x, y, z]
            obj_pos: 目标球位置 [x, y, z]
            pocket_pos: 袋口位置 [x, y, z]

        返回：
            (phi角度, 白球到幽灵球距离)
        """
        # 计算目标球到袋口的方向向量
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)

        # 边界情况: 目标球已经在袋口位置
        if dist_obj_to_pocket == 0:
            return 0, 0

        # 归一化方向向量
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket

        # 计算幽灵球位置: 沿着"袋口-目标球"的反方向,距离为2倍球半径
        # 这样白球击中幽灵球位置时,会将目标球推向袋口
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)

        # 计算白球到幽灵球的向量和距离
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)

        # 计算击球角度(水平面上的方位角)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)

        return phi, dist_cue_to_ghost

    def generate_heuristic_actions(
        self,
        balls: BallsDict,
        my_targets: List[str],
        table: pt.Table,
    ) -> List[ActionDict]:
        """生成候选动作列表(启发式方法)

        这是MCTS的第一步:生成动作空间

        策略:
        1. 遍历所有目标球和所有袋口的组合
        2. 对每个组合,使用幽灵球方法计算理论最优角度和力度
        3. 为每个理论解生成多个变种(不同力度、微调角度)
        4. 这样可以覆盖更多可能性,让MCTS有足够的候选动作进行评估

        为什么需要变种:
        - 理论解假设了完美执行,但实际有噪声
        - 生成多个变种可以让MCTS找到在噪声环境下更鲁棒的策略
        - 例如: 理论角度可能很完美,但稍大的力度反而更容易进球

        参数：
            balls: 球状态字典 {ball_id: Ball对象}
            my_targets: 目标球ID列表 ['1', '2', ...]
            table: 球桌对象,包含袋口信息

        返回：
            候选动作列表,每个动作包含 {V0, phi, theta, a, b}
        """
        actions = []

        # 获取白球位置
        cue_ball = balls.get("cue")
        if not cue_ball:
            return [self._random_action()]  # 白球不存在时返回随机动作
        cue_pos = cue_ball.state.rvw[0]  # rvw[0] 是位置向量

        # 获取所有还在桌面上的目标球(state.s != 4 表示未进袋)
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]

        # 如果没有目标球了,转为打8号球(黑球)
        if not target_ids:
            target_ids = ["8"]

        # 双重循环:遍历每一个目标球 × 每一个袋口
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]  # 目标球位置

            # 尝试将这个目标球打进每一个袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center  # 袋口中心位置

                # 步骤1: 计算理论最优击球角度(使用幽灵球方法)
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)

                # 步骤2: 根据距离估算力度
                # 经验公式: 基础力度1.5 + 距离相关项
                # 距离越远需要的力度越大
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)  # 限制在合理范围内

                # 步骤3: 生成多个变种动作
                # 这些变种探索了"理论解附近"的动作空间

                # 变种1: 精准一击(理论最优解)
                # theta=0表示水平击球,a=b=0表示击打球心
                actions.append({"V0": v_base, "phi": phi_ideal, "theta": 0, "a": 0, "b": 0})

                # 变种2: 力度稍大
                # 有时候更大的力度能让球更稳定地进袋(克服摩擦力等)
                actions.append(
                    {
                        "V0": min(v_base + 1.5, 7.5),  # 增加力度但不超过上限
                        "phi": phi_ideal,
                        "theta": 0,
                        "a": 0,
                        "b": 0,
                    }
                )

                # 变种3和4: 角度微调(左右偏移0.5度)
                # 应对噪声: 理论角度可能因为噪声而偏离,提前探索邻近角度
                actions.append(
                    {
                        "V0": v_base,
                        "phi": (phi_ideal + 0.5) % 360,  # 向右偏0.5度
                        "theta": 0,
                        "a": 0,
                        "b": 0,
                    }
                )
                actions.append(
                    {
                        "V0": v_base,
                        "phi": (phi_ideal - 0.5) % 360,  # 向左偏0.5度
                        "theta": 0,
                        "a": 0,
                        "b": 0,
                    }
                )

        # 边界情况: 如果启发式没有生成任何动作,补充随机动作
        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())

        # 随机打乱顺序,避免MCTS初期总是评估相同的动作
        random.shuffle(actions)
        return actions[:30]

    def simulate_action(
        self,
        balls: BallsDict,
        table: pt.Table,
        action: ActionDict,
    ) -> Optional[pt.System]:
        """执行带噪声的物理仿真

        这是MCTS中的Simulation(模拟)步骤

        关键设计: 注入高斯噪声
        - 模拟真实世界中击球的不确定性(手抖、球杆偏移等)
        - 让Agent在规划时就考虑到执行误差
        - 这样Agent会学会避免"理论完美但实际脆弱"的策略

        例如:
        - 某个角度需要极其精准才能进球 → 噪声会让它频繁失败 → 低分
        - 另一个角度容错性强,稍有偏差也能进 → 噪声影响小 → 高分

        让 Agent 意识到由于误差的存在,某些"极限球"是不可打的

        参数：
            balls: 球状态字典
            table: 球桌对象
            action: 击球动作 {V0, phi, theta, a, b}

        返回：
            模拟后的 System 对象(包含所有球的最终状态),失败返回 None
        """
        # 深拷贝,避免修改原始状态(模拟不应影响实际游戏)
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")  # 创建球杆对象
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)  # 创建模拟系统

        try:
            # 为每个动作参数注入高斯噪声(均值0,标准差由self.sim_noise定义)

            # 力度噪声: V0 ± 0.1
            noisy_V0 = np.clip(action["V0"] + np.random.normal(0, self.sim_noise["V0"]), 0.5, 8.0)

            # 水平角度噪声: phi ± 0.15度
            noisy_phi = (action["phi"] + np.random.normal(0, self.sim_noise["phi"])) % 360

            # 垂直角度噪声: theta ± 0.1度
            noisy_theta = np.clip(
                action["theta"] + np.random.normal(0, self.sim_noise["theta"]), 0, 90
            )

            # 击球点横向偏移噪声: a ± 0.005
            noisy_a = np.clip(action["a"] + np.random.normal(0, self.sim_noise["a"]), -0.5, 0.5)

            # 击球点纵向偏移噪声: b ± 0.005
            noisy_b = np.clip(action["b"] + np.random.normal(0, self.sim_noise["b"]), -0.5, 0.5)

            # 设置带噪声的击球参数并执行物理模拟
            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)  # inplace=True 直接修改shot对象
            return shot
        except Exception:
            # 模拟失败(例如参数异常导致物理引擎报错)
            logger.warning("[BasicAgentPro] 物理模拟失败，返回 None。")
            return None

    @override
    def decision(
        self,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[Any] = None,
    ) -> ActionDict:
        """MCTS 决策方法(核心算法)

        完整的MCTS流程包含四个步骤(循环执行):
        1. Selection(选择): 使用UCB公式选择一个候选动作进行评估
        2. Simulation(模拟): 执行带噪声的物理模拟
        3. Evaluation(评估): 计算模拟结果的奖励值
        4. Backpropagation(反向传播): 更新该动作的统计信息

        经过多次循环后,每个候选动作都会积累足够的统计数据,
        最终选择平均得分最高的动作作为最优决策。

        UCB (Upper Confidence Bound) 公式:
        UCB = 平均奖励 + c_puct * sqrt(ln(总访问次数) / 该动作访问次数)
              ^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              利用(Exploitation)         探索(Exploration)

        - 利用项: 倾向于选择已知效果好的动作
        - 探索项: 鼓励尝试访问次数少的动作(避免遗漏潜在最优解)
        - c_puct 控制探索和利用的平衡

        Args:
            balls: 球状态字典,{ball_id: Ball}
            my_targets: 目标球ID列表,['1', '2', ...]
            table: 球桌对象

        Returns:
            ActionDict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        # 边界情况: 没有球状态信息时返回随机动作
        if balls is None:
            return self._random_action()

        # 预处理: 确定当前的目标球
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]  # 筛选未进袋的球
        if len(remaining) == 0:
            my_targets = ["8"]  # 如果所有目标球都进了,转为打8号球

        # 保存当前状态快照,用于后续评估
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # === MCTS 初始化 ===
        # 生成候选动作池(使用启发式方法)
        candidate_actions = self.generate_heuristic_actions(balls, my_targets, table)
        n_candidates = len(candidate_actions)

        # 初始化统计数组
        N = np.zeros(n_candidates)  # N[i]: 第i个动作被访问(模拟)的次数
        Q = np.zeros(n_candidates)  # Q[i]: 第i个动作的累计奖励(未归一化的平均值 = Q/N)

        # === MCTS 主循环 ===
        for i in range(self.n_simulations):
            # === 步骤1: Selection(选择) ===
            # 使用UCB公式选择一个候选动作进行评估

            if i < n_candidates:
                # 前n_candidates次迭代: 确保每个动作至少被评估一次
                idx = i
            else:
                # 后续迭代: 使用UCB公式选择
                total_n = np.sum(N)  # 总访问次数

                # UCB公式: 平衡"利用"和"探索"
                # - Q / (N + 1e-6): 平均奖励(利用已知信息)
                # - c_puct * sqrt(...): 探索奖励(鼓励尝试访问少的动作)
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(
                    np.log(total_n + 1) / (N + 1e-6)
                )
                idx = np.argmax(ucb_values)  # 选择UCB值最大的动作

            # === 步骤2: Simulation(模拟) ===
            # 执行带噪声的物理模拟
            shot = self.simulate_action(balls, table, candidate_actions[idx])

            # === 步骤3: Evaluation(评估) ===
            # 根据模拟结果计算奖励值
            if shot is None:
                # 模拟失败(物理引擎报错等): 给予大幅负分
                raw_reward = -500.0
            else:
                # 调用奖励函数分析模拟结果
                # analyze_shot_for_reward 会考虑: 进球、犯规、球的最终位置等
                raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)

            # 将奖励归一化到 [0, 1] 区间
            # 假设原始奖励范围是 [-500, 150]
            # 归一化公式: (val - min) / (max - min)
            normalized_reward = (raw_reward - (-500)) / 650.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)  # 防止数值越界

            # === 步骤4: Backpropagation(反向传播) ===
            # 更新选中动作的统计信息
            N[idx] += 1  # 访问次数 +1
            Q[idx] += normalized_reward  # 累加奖励(注意: 这里是累加,不是平均)

        # === Final Decision(最终决策) ===
        # MCTS循环结束后,选择平均得分最高的动作

        # 计算每个动作的平均奖励: Q / N
        avg_rewards = Q / (N + 1e-6)  # 加1e-6防止除零

        # 选择平均得分最高的动作(Robust Child策略)
        # 注: 也可以选择访问次数最多的(Max Child策略),但平均分更稳定
        best_idx = np.argmax(avg_rewards)
        best_action = candidate_actions[best_idx]

        # 日志输出: 显示最佳动作的预期得分
        logger.info(
            f"[BasicAgentPro] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {self.n_simulations})"
        )

        return best_action
