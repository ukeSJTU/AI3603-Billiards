"""
GeometryAgent - Geometry-Guided Agent

An intelligent agent using geometry-based heuristics for shot selection:
1. Evaluates all target ball + pocket combinations
2. Checks for path obstructions
3. Selects the most feasible shot
4. Performs fine-grained sampling around the optimal angle
"""

import copy
from typing import List, Optional, Tuple, override

import numpy as np
import pooltool as pt

from src.utils.logger import get_logger

from .base import (
    ActionDict,
    Agent,
    BallsDict,
    analyze_shot_for_reward,
    simulate_with_timeout,
)

# Initialize logger
log = get_logger()


class GeometryAgent(Agent):
    """
    基于几何启发 + 多袋口评估 + 精细搜索的智能 Agent

    策略：
    1. 对每个目标球计算到所有6个袋口的可行性
    2. 检查路径上是否有其他球阻挡
    3. 选择最容易进袋的球-袋组合
    4. 在最优几何角度附近做密集采样+模拟验证
    """

    # 袋口ID列表
    POCKET_IDS = ["lb", "lc", "lt", "rb", "rc", "rt"]
    # 球的半径（米），标准台球
    BALL_RADIUS = 0.028575

    def __init__(
        self,
        n_candidates: int = 30,
        angle_spread: float = 5.0,
        v0_spread: float = 1.0,
    ):
        """初始化 GeometryAgent

        参数：
            n_candidates: 候选动作数量
            angle_spread: 角度搜索范围（度）
            v0_spread: 速度搜索范围
        """
        super().__init__()
        # 搜索参数 (now configurable via constructor)
        self.n_candidates = n_candidates
        self.angle_spread = angle_spread
        self.v0_spread = v0_spread
        log.info("GeometryAgent (Geometry-Guided) 已初始化。")

    # ============ 几何工具函数 ============
    @staticmethod
    def get_ball_pos(ball: pt.Ball) -> np.ndarray:
        """获取球的2D位置 (x, y)"""
        return ball.state.rvw[0][:2]

    @staticmethod
    def get_pocket_pos(table: pt.Table, pocket_id: str) -> np.ndarray:
        """获取袋口的2D位置 (x, y)"""
        return table.pockets[pocket_id].center[:2]

    @staticmethod
    def calc_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点间距离"""
        return float(np.linalg.norm(p2 - p1))

    @staticmethod
    def calc_angle(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
        """
        计算从 from_pos 到 to_pos 的角度（度）
        返回值范围：[0, 360)
        """
        delta = to_pos - from_pos
        angle_rad = np.arctan2(delta[1], delta[0])
        angle_deg = np.degrees(angle_rad)
        return angle_deg % 360

    def is_path_blocked(
        self,
        start: np.ndarray,
        end: np.ndarray,
        balls: BallsDict,
        exclude_ids: List[str],
    ) -> bool:
        """
        检查从 start 到 end 的路径是否被其他球阻挡

        Args:
            start: 起点位置 (x, y)
            end: 终点位置 (x, y)
            balls: 所有球
            exclude_ids: 排除的球ID（如白球、目标球本身）

        Returns:
            bool: True 表示路径被阻挡
        """
        direction = end - start
        path_length = np.linalg.norm(direction)
        if path_length < 1e-6:
            return False
        direction = direction / path_length  # 单位向量

        for bid, ball in balls.items():
            if bid in exclude_ids:
                continue
            if ball.state.s == 4:  # 已进袋，跳过
                continue

            ball_pos = self.get_ball_pos(ball)
            # 计算球心到路径直线的距离
            # 使用向量投影
            to_ball = ball_pos - start
            proj_length = np.dot(to_ball, direction)

            # 球必须在路径上（不在起点前或终点后）
            if proj_length < self.BALL_RADIUS or proj_length > path_length - self.BALL_RADIUS:
                continue

            # 计算垂直距离
            proj_point = start + proj_length * direction
            perp_dist = np.linalg.norm(ball_pos - proj_point)

            # 如果距离小于两个球的直径，则阻挡
            if perp_dist < 2 * self.BALL_RADIUS + 0.005:  # 加一点余量
                return True

        return False

    def evaluate_ball_pocket_pair(
        self,
        cue_pos: np.ndarray,
        target_ball_id: str,
        target_pos: np.ndarray,
        pocket_pos: np.ndarray,
        balls: BallsDict,
    ) -> Tuple[float, float, float]:
        """
        评估白球-目标球-袋口组合的可行性

        Returns:
            Tuple[float, float, float]: (分数, 理想phi角度, 理想V0)
            分数越高越好，-1表示不可行
        """
        # 1. 检查目标球到袋口的路径是否被阻挡
        if self.is_path_blocked(target_pos, pocket_pos, balls, ["cue", target_ball_id]):
            return -1, 0, 0

        # 2. 检查白球到目标球的路径是否被阻挡
        if self.is_path_blocked(cue_pos, target_pos, balls, ["cue", target_ball_id]):
            return -1, 0, 0

        # 3. 计算几何角度
        # 目标球到袋口的角度
        target_to_pocket_angle = self.calc_angle(target_pos, pocket_pos)
        # 理想击球点：目标球背面（相对于袋口）
        # 白球需要打到目标球上，使其朝袋口方向运动
        # 理想击球角度 = 从白球到目标球的角度，使得碰撞后目标球朝袋口
        # 简化：白球应该打向目标球的"背面"
        ideal_phi = target_to_pocket_angle  # 白球击向目标球，期望目标球朝袋口

        # 4. 计算实际需要的 phi（白球到目标球的角度）
        cue_to_target_angle = self.calc_angle(cue_pos, target_pos)

        # 5. 计算角度偏差（白球击球方向与理想方向的差异）
        # 理想情况：白球、目标球、袋口三点共线
        angle_diff = abs(cue_to_target_angle - ideal_phi)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # 角度偏差过大（>70度）基本不可能打进
        if angle_diff > 70:
            return -1, 0, 0

        # 6. 计算距离
        cue_to_target_dist = self.calc_distance(cue_pos, target_pos)
        target_to_pocket_dist = self.calc_distance(target_pos, pocket_pos)
        total_dist = cue_to_target_dist + target_to_pocket_dist

        # 7. 计算分数
        # 分数考虑：角度偏差（越小越好）、距离（越近越好）
        angle_score = max(0, 100 - angle_diff * 2)  # 角度偏差惩罚
        dist_score = max(0, 100 - total_dist * 30)  # 距离惩罚
        score = angle_score * 0.7 + dist_score * 0.3

        # 8. 估算理想速度
        # 简单估算：根据总距离
        ideal_v0 = np.clip(1.5 + total_dist * 2.0, 1.0, 6.0)

        return score, cue_to_target_angle, ideal_v0

    def find_best_shot(
        self,
        balls: BallsDict,
        my_targets: List[str],
        table: pt.Table,
    ) -> Tuple[Optional[str], Optional[str], float, float]:
        """
        找到最佳的球-袋组合

        Returns:
            Tuple[Optional[str], Optional[str], float, float]: (目标球ID, 袋口ID, 理想phi, 理想V0)
            如果没有可行方案，返回 (None, None, 0, 0)
        """
        cue_pos = self.get_ball_pos(balls["cue"])

        best_score = -1
        best_target = None
        best_pocket = None
        best_phi = 0
        best_v0 = 2.0

        # 遍历所有目标球
        for target_id in my_targets:
            if balls[target_id].state.s == 4:  # 已进袋
                continue

            target_pos = self.get_ball_pos(balls[target_id])

            # 遍历所有袋口
            for pocket_id in self.POCKET_IDS:
                pocket_pos = self.get_pocket_pos(table, pocket_id)

                score, phi, v0 = self.evaluate_ball_pocket_pair(
                    cue_pos, target_id, target_pos, pocket_pos, balls
                )

                if score > best_score:
                    best_score = score
                    best_target = target_id
                    best_pocket = pocket_id
                    best_phi = phi
                    best_v0 = v0

        if best_score < 0:
            return None, None, 0, 0

        log.debug(
            f"最佳方案: 球{best_target} -> 袋{best_pocket}, "
            f"分数={best_score:.1f}, phi={best_phi:.1f}, V0={best_v0:.1f}"
        )
        return best_target, best_pocket, best_phi, best_v0

    def generate_candidates(
        self,
        base_phi: float,
        base_v0: float,
    ) -> List[ActionDict]:
        """
        在基础参数附近生成候选动作
        """
        candidates = []

        for _ in range(self.n_candidates):
            phi = base_phi + np.random.uniform(-self.angle_spread, self.angle_spread)
            phi = phi % 360

            v0 = base_v0 + np.random.uniform(-self.v0_spread, self.v0_spread)
            v0 = np.clip(v0, 0.8, 7.0)

            # theta, a, b 使用小范围随机
            theta = np.random.uniform(0, 15)  # 小角度
            a = np.random.uniform(-0.2, 0.2)
            b = np.random.uniform(-0.2, 0.2)

            candidates.append(
                {
                    "V0": float(v0),
                    "phi": float(phi),
                    "theta": float(theta),
                    "a": float(a),
                    "b": float(b),
                }
            )

        # 添加基础动作（无扰动）
        candidates.append(
            {
                "V0": float(base_v0),
                "phi": float(base_phi),
                "theta": 2.0,
                "a": 0.0,
                "b": 0.0,
            }
        )

        return candidates

    def evaluate_candidate(
        self,
        action: ActionDict,
        balls: BallsDict,
        table: pt.Table,
        last_state: BallsDict,
        my_targets: List[str],
    ) -> float:
        """评估单个候选动作的得分"""
        # 创建沙盒环境
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        try:
            shot.cue.set_state(
                V0=action["V0"],
                phi=action["phi"],
                theta=action["theta"],
                a=action["a"],
                b=action["b"],
            )
            if not simulate_with_timeout(shot, timeout=3):
                return -100  # 超时
        except Exception:
            return -500

        return analyze_shot_for_reward(shot, last_state, my_targets)

    @override
    def decision(
        self,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[pt.Table] = None,
    ) -> ActionDict:
        """
        决策方法：几何启发 + 多袋口评估 + 精细搜索

        Args:
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象

        Returns:
            ActionDict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        # Step 0: 输入验证
        if balls is None or my_targets is None or table is None:
            log.warning("GeometryAgent: 输入不完整，使用随机动作")
            return self._random_action()

        try:
            # Step 1: 状态快照 + 目标检查
            last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            my_targets, switched = self.check_remaining_targets(balls, my_targets)
            if switched:
                log.info("目标球已清空，切换到8号球")

            # Step 2: 几何分析找最佳球-袋组合
            best_target, best_pocket, base_phi, base_v0 = self.find_best_shot(
                balls, my_targets, table
            )

            # Store original n_candidates
            original_n_candidates = self.n_candidates

            if best_target is None:
                # 没有明显好的方案，扩大搜索
                log.info("无明显最佳方案，使用扩大搜索")
                base_phi = np.random.uniform(0, 360)
                base_v0 = np.random.uniform(2.0, 5.0)
                self.n_candidates = 50  # 临时增加候选数
            else:
                log.info(f"几何分析: 目标球{best_target} -> 袋口{best_pocket}")

            # Step 3: 生成候选动作
            candidates = self.generate_candidates(base_phi, base_v0)

            # Restore original n_candidates
            self.n_candidates = original_n_candidates

            # Step 4: 评估所有候选
            best_action = None
            best_score = float("-inf")

            for action in candidates:
                score = self.evaluate_candidate(action, balls, table, last_state, my_targets)
                if score > best_score:
                    best_score = score
                    best_action = action

            # Step 5: 结果处理
            if best_action is None or best_score < -50:
                log.info(f"未找到好方案 (最高分: {best_score:.1f})，使用随机动作")
                return self._random_action()

            log.info(
                f"决策 (得分: {best_score:.1f}): "
                f"V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}"
            )
            return best_action

        except Exception as e:
            log.error(f"决策出错: {e}")
            import traceback

            traceback.print_exc()
            return self._random_action()
