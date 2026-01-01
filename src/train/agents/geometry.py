"""
GeometryAgent - Geometry-Guided Agent

An intelligent agent using geometry-based heuristics for shot selection:
1. Evaluates all target ball + pocket combinations
2. Checks for path obstructions
3. Selects the most feasible shot
4. Performs fine-grained sampling around the optimal angle
5. V2: Adds defensive strategy (enable_defense)
6. V3: Adds position optimization (enable_position)
7. Adaptive: Dynamic parameter adjustment (enable_adaptive)
"""

from typing import Dict, List, Optional, Tuple, override

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
    统一的几何启发式 Agent，支持多版本功能组合

    版本对应：
    - V1: 默认参数（所有 enable_*=False）
    - V2: enable_defense=True（防守增强）
    - V3: enable_defense=True, enable_position=True（连击优化）
    - Adaptive: 所有 enable_*=True（自适应参数）

    策略：
    1. 对每个目标球计算到所有6个袋口的可行性
    2. 检查路径上是否有其他球阻挡
    3. 选择最容易进袋的球-袋组合
    4. 在最优几何角度附近做密集采样+模拟验证
    5. 可选：评估对手威胁并生成防守动作
    6. 可选：预测白球停位并优化连击
    7. 可选：根据局势动态调整参数
    """

    # 袋口ID列表
    POCKET_IDS = ["lb", "lc", "lt", "rb", "rc", "rt"]

    # 物理常量
    BALL_RADIUS = 0.028575  # 球的半径（米）
    FRICTION_COEFF = 0.01  # 摩擦系数
    GRAVITY = 9.8  # 重力加速度
    ENERGY_LOSS_FACTOR = 0.3  # 碰撞后能量保留比例

    # 评分阈值
    MIN_VIABLE_SCORE = -50  # 最低可接受分数

    def __init__(
        self,
        # 搜索参数
        n_candidates: int = 30,
        angle_spread: float = 5.0,
        v0_spread: float = 1.0,
        # 策略开关
        enable_defense: bool = False,
        enable_position: bool = False,
        enable_adaptive: bool = False,
        # 权重系数
        defense_weight: float = 0.3,
        position_weight: float = 0.2,
        defense_threshold: float = 0.6,
        attack_threshold: float = 50.0,
    ):
        """初始化 GeometryAgent

        参数：
            # 搜索参数
            n_candidates: 候选动作数量
            angle_spread: 角度搜索范围（度）
            v0_spread: 速度搜索范围

            # 策略开关
            enable_defense: 启用防守策略（V2）
            enable_position: 启用连击优化（V3）
            enable_adaptive: 启用自适应参数（Adaptive）

            # 权重系数
            defense_weight: 防守候选占比（0-1）
            position_weight: 停位分数权重
            defense_threshold: 对手威胁触发防守阈值（0-1）
            attack_threshold: 进攻质量阈值
        """
        super().__init__()

        # 保存搜索参数
        self.n_candidates = n_candidates
        self.angle_spread = angle_spread
        self.v0_spread = v0_spread

        # 保存策略开关
        self.enable_defense = enable_defense
        self.enable_position = enable_position
        self.enable_adaptive = enable_adaptive

        # 保存权重系数
        self.defense_weight = defense_weight
        self.position_weight = position_weight
        self.defense_threshold = defense_threshold
        self.attack_threshold = attack_threshold

        # 保存原始参数（用于自适应重置）
        self._original_params = {
            "n_candidates": n_candidates,
            "angle_spread": angle_spread,
            "defense_weight": defense_weight,
            "position_weight": position_weight,
        }

        log.info(
            f"GeometryAgent 初始化: n_candidates={n_candidates}, "
            f"defense={enable_defense}, position={enable_position}, "
            f"adaptive={enable_adaptive}"
        )

    # ============================================================================
    # 几何工具方法
    # ============================================================================

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

    # ============================================================================
    # 路径检测
    # ============================================================================

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

    # ============================================================================
    # V1: 基础几何分析
    # ============================================================================

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

    # ============================================================================
    # V2: 防守策略
    # ============================================================================

    def _infer_opponent_targets(self, balls: BallsDict, my_targets: List[str]) -> List[str]:
        """推断对手的目标球

        逻辑：全部球号(1-15) - 我的目标 - 黑8 - 已进袋的球

        Args:
            balls: 球状态字典
            my_targets: 我的目标球列表

        Returns:
            对手的目标球ID列表
        """
        all_balls = {str(i) for i in range(1, 16)}
        my_set = set(my_targets) - {"8"}
        opponent_set = all_balls - my_set - {"8"}

        return [bid for bid in opponent_set if balls[bid].state.s != 4]

    def evaluate_opponent_threats(
        self, balls: BallsDict, opponent_targets: List[str], table: pt.Table
    ) -> List[Tuple[str, str, float]]:
        """从对手视角评估最容易进的球

        Args:
            balls: 球状态字典
            opponent_targets: 对手的目标球列表
            table: 球桌对象

        Returns:
            [(球ID, 袋口ID, 威胁分数), ...] 按威胁降序排列
            威胁分数 = 几何评分 / 100.0 (0-1之间)

        Note:
            复用 evaluate_ball_pocket_pair，从对手角度评估
        """
        threats = []
        cue_pos = self.get_ball_pos(balls["cue"])

        for target_id in opponent_targets:
            target_pos = self.get_ball_pos(balls[target_id])

            for pocket_id in self.POCKET_IDS:
                pocket_pos = self.get_pocket_pos(table, pocket_id)

                # 复用现有评估函数
                score, _, _ = self.evaluate_ball_pocket_pair(
                    cue_pos, target_id, target_pos, pocket_pos, balls
                )

                if score > 0:
                    # 归一化为0-1之间的威胁分数
                    threat = score / 100.0
                    threats.append((target_id, pocket_id, threat))

        # 按威胁从高到低排序
        threats.sort(key=lambda x: x[2], reverse=True)
        return threats

    def generate_defensive_candidates(
        self,
        balls: BallsDict,
        threats: List[Tuple[str, str, float]],
        table: pt.Table,
        n_defense: int,
    ) -> List[ActionDict]:
        """生成防守候选动作

        策略：将白球停在威胁球和袋口连线的中点附近

        Args:
            balls: 球状态字典
            threats: 威胁列表 [(球ID, 袋口ID, 威胁分数), ...]
            table: 球桌对象
            n_defense: 生成的防守候选数量

        Returns:
            防守候选动作列表

        Note:
            - 取前3个最高威胁
            - 力量：1.5-3.0（小力量，精确控制）
            - 仰角：12-18（中等，帮助白球停稳）
            - 下塞：-0.15 到 -0.05（轻微，增加稳定性）
        """
        if not threats or n_defense <= 0:
            return []

        candidates = []
        top_threats = threats[: min(3, len(threats))]

        # 为每个威胁生成防守候选
        cands_per_threat = max(1, n_defense // len(top_threats))

        for threat_ball_id, threat_pocket_id, _ in top_threats:
            cue_pos = self.get_ball_pos(balls["cue"])
            threat_pos = self.get_ball_pos(balls[threat_ball_id])
            pocket_pos = self.get_pocket_pos(table, threat_pocket_id)

            # 防守位置：威胁球和袋口连线的中点
            defense_pos = (threat_pos + pocket_pos) / 2
            vec = defense_pos - cue_pos
            distance = np.linalg.norm(vec)

            if distance < 0.05:  # 白球已经在防守位置附近
                continue

            phi_base = self.calc_angle(cue_pos, defense_pos)

            # 生成多个防守候选（微调）
            for _ in range(cands_per_threat):
                candidates.append(
                    {
                        "V0": float(
                            np.clip(1.5 + distance * 0.8 + np.random.uniform(-0.3, 0.3), 0.8, 3.0)
                        ),
                        "phi": float((phi_base + np.random.uniform(-3, 3)) % 360),
                        "theta": float(np.random.uniform(12, 18)),
                        "a": 0.0,
                        "b": float(np.random.uniform(-0.15, -0.05)),
                    }
                )

        return candidates

    # ============================================================================
    # V3: 连击优化
    # ============================================================================

    def _find_target_by_angle(
        self, phi: float, balls: BallsDict, cue_pos: np.ndarray
    ) -> Optional[str]:
        """通过击球角度推断最可能的目标球

        Args:
            phi: 击球角度（度）
            balls: 球状态字典
            cue_pos: 白球位置

        Returns:
            最可能的目标球ID，如果没有找到返回None

        Note:
            找到与 phi 方向最接近的球（未进袋的）
        """
        min_angle_diff = float("inf")
        best_target = None

        for bid, ball in balls.items():
            if bid == "cue" or ball.state.s == 4:  # 跳过白球和已进袋的球
                continue

            ball_pos = self.get_ball_pos(ball)
            ball_angle = self.calc_angle(cue_pos, ball_pos)

            # 计算角度差
            angle_diff = abs(ball_angle - phi)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_target = bid

        # 如果角度差太大（>45度），认为没有命中目标
        if min_angle_diff > 45:
            return None

        return best_target

    def predict_cue_final_position(
        self, action: ActionDict, balls: BallsDict, table: pt.Table
    ) -> np.ndarray:
        """使用简化物理模型预测白球停位

        Args:
            action: 击球动作
            balls: 球状态字典
            table: 球桌对象

        Returns:
            预测的白球最终位置 (x, y)

        Note:
            简化假设：
            1. 通过 phi 找到最可能的目标球
            2. 碰撞点 = 白球向目标球方向 90% 处
            3. 碰撞后速度 v_after = ENERGY_LOSS_FACTOR * V0
            4. 反弹角度 = phi + (theta - 45) * 2
            5. 滚动距离 = v² / (2 * μ * g)
        """
        cue_pos = self.get_ball_pos(balls["cue"])
        phi = action["phi"]
        theta = action["theta"]
        V0 = action["V0"]

        # 1. 找到目标球
        target_id = self._find_target_by_angle(phi, balls, cue_pos)
        if not target_id:
            # 预测失败，返回原位
            return cue_pos

        target_pos = self.get_ball_pos(balls[target_id])

        # 2. 碰撞点（90%距离处）
        direction = target_pos - cue_pos
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return cue_pos

        direction = direction / distance
        collision_point = cue_pos + 0.9 * distance * direction

        # 3. 反弹方向（受 theta 影响）
        bounce_angle = (phi + (theta - 45) * 2) % 360

        # 4. 碰撞后速度（能量损失）
        v_after = V0 * self.ENERGY_LOSS_FACTOR

        # 5. 滚动距离（摩擦减速）
        rolling_distance = (v_after**2) / (2 * self.FRICTION_COEFF * self.GRAVITY)
        rolling_distance = np.clip(rolling_distance, 0, 0.8)  # 限制最大距离

        # 6. 最终位置
        bounce_vec = np.array([np.cos(np.radians(bounce_angle)), np.sin(np.radians(bounce_angle))])
        final_pos = collision_point + bounce_vec * rolling_distance

        # 7. 边界限制
        final_pos[0] = np.clip(final_pos[0], 0.05, table.l - 0.05)
        final_pos[1] = np.clip(final_pos[1], 0.05, table.w - 0.05)

        return final_pos

    def evaluate_cue_position(
        self, predicted_pos: np.ndarray, balls: BallsDict, my_targets: List[str], table: pt.Table
    ) -> float:
        """评估白球停位质量

        Args:
            predicted_pos: 预测的白球位置
            balls: 球状态字典
            my_targets: 我的目标球列表
            table: 球桌对象

        Returns:
            停位质量分数（-50 到 100）

        Note:
            评估维度：
            1. 到最近目标球距离（权重0.5）：0.5米内满分，1.5米外0分
            2. 到袋口距离（权重0.3，负向）：<12cm扣40分，<20cm扣15分
            3. 球桌中心区域（权重0.2）：中心0.4米内加分，最多30分
        """
        score = 0.0

        # 1. 到最近目标球的距离
        min_dist = float("inf")
        for tid in my_targets:
            if tid == "8" or balls[tid].state.s == 4:  # 跳过黑8和已进袋的球
                continue
            target_pos = self.get_ball_pos(balls[tid])
            dist = self.calc_distance(predicted_pos, target_pos)
            min_dist = min(min_dist, dist)

        if min_dist < float("inf"):
            # 0.5米内100分，1.5米外0分，线性插值
            dist_score = np.clip(100 - (min_dist - 0.5) * 100, 0, 100)
            score += dist_score * 0.5

        # 2. 到袋口距离（太近危险）
        min_pocket_dist = min(
            self.calc_distance(predicted_pos, self.get_pocket_pos(table, pid))
            for pid in self.POCKET_IDS
        )
        if min_pocket_dist < 0.12:
            score -= 40  # 离袋口太近，扣分
        elif min_pocket_dist < 0.20:
            score -= 15

        # 3. 中心区域加分（更灵活的位置）
        center_pos = np.array([table.l / 2, table.w / 2])
        dist_to_center = self.calc_distance(predicted_pos, center_pos)
        if dist_to_center < 0.4:
            center_score = (0.4 - dist_to_center) / 0.4 * 30
            score += center_score

        return np.clip(score, -50, 100)

    # ============================================================================
    # Adaptive: 自适应策略
    # ============================================================================

    def assess_game_state(self, balls: BallsDict, my_targets: List[str]) -> Dict:
        """评估当前局势

        Args:
            balls: 球状态字典
            my_targets: 我的目标球列表

        Returns:
            局势字典: {
                "my_remaining": int,       # 我方剩余球数
                "opp_remaining": int,      # 对手剩余球数
                "advantage": str,          # "leading"/"even"/"behind"
                "my_on_black8": bool,      # 我方是否打黑8
                "opp_on_black8": bool      # 对手是否打黑8
            }
        """
        # 统计我方剩余球数
        my_set = set(my_targets) - {"8"}
        my_remaining = sum(1 for bid in my_set if balls[bid].state.s != 4)

        # 推断并统计对手剩余球数
        all_balls = {str(i) for i in range(1, 16)}
        opp_targets = list(all_balls - my_set - {"8"})
        opp_remaining = sum(1 for bid in opp_targets if balls[bid].state.s != 4)

        # 判断优势（剩余球数少的领先）
        diff = my_remaining - opp_remaining
        if diff < -1:
            advantage = "leading"  # 我方球少，领先
        elif diff > 1:
            advantage = "behind"  # 我方球多，落后
        else:
            advantage = "even"  # 均势

        return {
            "my_remaining": my_remaining,
            "opp_remaining": opp_remaining,
            "advantage": advantage,
            "my_on_black8": my_targets == ["8"],
            "opp_on_black8": opp_remaining == 0,
        }

    def adjust_parameters(self, game_state: Dict) -> None:
        """根据局势动态调整参数

        Args:
            game_state: assess_game_state() 返回的局势字典

        Note:
            参数调整策略：
            | 局势   | n_cand | angle | defense_w | position_w | 说明 |
            |--------|--------|-------|-----------|------------|------|
            | 黑8    | 80     | 3.0   | 0.5       | 0.05       | 精确+高防守 |
            | 领先   | 50     | 5.0   | 0.4       | 0.25       | 稳健+重走位 |
            | 落后   | 40     | 6.0   | 0.15      | 0.1        | 激进+快速 |
            | 均势   | 默认   | 默认  | 默认      | 默认       | 平衡 |
        """
        # 先重置到原始值
        for key, value in self._original_params.items():
            setattr(self, key, value)

        # 黑8阶段（最高优先级）
        if game_state["my_on_black8"] or game_state["opp_on_black8"]:
            self.n_candidates = 80
            self.angle_spread = 3.0
            self.defense_weight = 0.5
            self.position_weight = 0.05
            log.info("[Adaptive] 黑8阶段：精确+高防守 (n=80, angle=3.0, def=0.5)")
            return

        # 根据优势调整
        advantage = game_state["advantage"]

        if advantage == "leading":
            # 领先：稳健策略
            self.n_candidates = 50
            self.angle_spread = 5.0
            self.defense_weight = 0.4
            self.position_weight = 0.25
            log.info("[Adaptive] 领先：稳健+重走位 (n=50, def=0.4, pos=0.25)")

        elif advantage == "behind":
            # 落后：激进策略
            self.n_candidates = 40
            self.angle_spread = 6.0
            self.defense_weight = 0.15
            self.position_weight = 0.1
            log.info("[Adaptive] 落后：激进+快速 (n=40, angle=6.0, def=0.15)")

        # 均势：使用原始参数（已在上面重置）

    # ============================================================================
    # 候选生成与评估
    # ============================================================================

    def generate_attack_candidates(
        self, base_phi: float, base_v0: float, n_attack: int
    ) -> List[ActionDict]:
        """
        生成进攻候选动作

        Args:
            base_phi: 基础角度（度）
            base_v0: 基础速度
            n_attack: 生成的进攻候选数量

        Returns:
            进攻候选动作列表

        Note:
            - 如果 enable_position=True，扩大 theta 和旋转范围以优化停位
        """
        candidates = []

        # 根据是否启用连击优化，调整参数范围
        if self.enable_position:
            theta_range = (0, 45)  # 扩大 theta 范围
            spin_range = (-0.3, 0.3)  # 扩大旋转范围
        else:
            theta_range = (0, 15)  # 小角度
            spin_range = (-0.2, 0.2)  # 小旋转

        for _ in range(n_attack):
            phi = base_phi + np.random.uniform(-self.angle_spread, self.angle_spread)
            phi = phi % 360

            v0 = base_v0 + np.random.uniform(-self.v0_spread, self.v0_spread)
            v0 = np.clip(v0, 0.8, 7.0)

            theta = np.random.uniform(*theta_range)
            a = np.random.uniform(*spin_range)
            b = np.random.uniform(*spin_range)

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

    def generate_candidates(
        self,
        base_phi: float,
        base_v0: float,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[pt.Table] = None,
        threats: Optional[List[Tuple[str, str, float]]] = None,
    ) -> List[ActionDict]:
        """
        统一候选生成入口（整合进攻、防守、停位评估）

        Args:
            base_phi: 基础角度
            base_v0: 基础速度
            balls: 球状态字典（用于停位预测）
            my_targets: 我的目标球列表（用于停位评估）
            table: 球桌对象（用于停位预测和防守）
            threats: 对手威胁列表（用于防守）

        Returns:
            候选动作列表

        Note:
            - V1: 只生成进攻候选
            - V2: 根据 enable_defense 混合防守候选
            - V3: 根据 enable_position 评估停位分数
        """
        candidates = []

        # 计算候选数量分配
        if self.enable_defense and threats and balls and table:
            n_defense = int(self.n_candidates * self.defense_weight)
            n_attack = self.n_candidates - n_defense
        else:
            n_attack = self.n_candidates
            n_defense = 0

        # 1. 生成进攻候选
        attack_cands = self.generate_attack_candidates(base_phi, base_v0, n_attack)
        candidates.extend(attack_cands)

        # 2. 生成防守候选（如果启用）
        if n_defense > 0:
            defense_cands = self.generate_defensive_candidates(balls, threats, table, n_defense)
            candidates.extend(defense_cands)

        # 3. 停位预测与评估（如果启用）
        if self.enable_position and balls and my_targets and table:
            for action in candidates:
                try:
                    predicted_pos = self.predict_cue_final_position(action, balls, table)
                    position_score = self.evaluate_cue_position(
                        predicted_pos, balls, my_targets, table
                    )
                    action["_position_score"] = position_score
                except Exception as e:
                    log.warning(f"停位预测失败: {e}")
                    action["_position_score"] = 0

        return candidates

    def evaluate_candidate(
        self,
        action: ActionDict,
        balls: BallsDict,
        table: pt.Table,
        last_state: BallsDict,
        my_targets: List[str],
    ) -> float:
        """评估单个候选动作的得分

        Args:
            action: 候选动作
            balls: 球状态字典
            table: 球桌对象
            last_state: 上一次状态（用于评分）
            my_targets: 我的目标球列表

        Returns:
            综合评分 = 基础分数 + 停位分数 * position_weight

        Note:
            - 如果 enable_position=True，会额外评估白球停位质量
            - 停位分数从 action 的 "_position_score" 字段获取（如果已预先计算）
        """
        # 创建沙盒环境
        sim_balls = {bid: ball.copy() for bid, ball in balls.items()}
        sim_table = table.copy()
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

        # 基础分数（物理模拟结果）
        base_score = analyze_shot_for_reward(shot, last_state, my_targets)

        # 如果启用连击优化，加入停位分数
        if self.enable_position:
            position_score = action.get("_position_score", 0)
            total_score = base_score + position_score * self.position_weight
            return total_score
        else:
            return base_score

    # ============================================================================
    # 主决策流程
    # ============================================================================

    @override
    def decision(
        self,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[pt.Table] = None,
    ) -> ActionDict:
        """
        统一决策方法（支持 V1/V2/V3/Adaptive 所有版本）

        Args:
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象

        Returns:
            ActionDict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}

        Note:
            决策流程：
            0. 输入验证
            1. 自适应参数调整（如果启用）
            2. 状态快照 + 目标检查
            3. 几何分析找最佳球-袋组合
            4. 防守威胁评估（如果启用）
            5. 生成候选动作（进攻 + 防守 + 停位评估）
            6. 评估所有候选
            7. 返回最佳动作
        """
        # Step 0: 输入验证
        if balls is None or my_targets is None or table is None:
            log.warning("GeometryAgent: 输入不完整，使用随机动作")
            return self._random_action()

        try:
            # Step 1: 自适应参数调整（如果启用）
            if self.enable_adaptive:
                game_state = self.assess_game_state(balls, my_targets)
                self.adjust_parameters(game_state)

            # Step 2: 状态快照 + 目标检查
            last_state = {bid: ball.copy() for bid, ball in balls.items()}
            my_targets, switched = self.check_remaining_targets(balls, my_targets)
            if switched:
                log.info("目标球已清空，切换到8号球")

            # Step 3: 几何分析找最佳球-袋组合
            best_target, best_pocket, base_phi, base_v0 = self.find_best_shot(
                balls, my_targets, table
            )

            # 保存原始 n_candidates（防止后续修改影响）
            original_n_candidates = self.n_candidates

            if best_target is None:
                # 没有明显好的方案，扩大搜索
                log.info("无明显最佳方案，使用扩大搜索")
                base_phi = np.random.uniform(0, 360)
                base_v0 = np.random.uniform(2.0, 5.0)
                self.n_candidates = 50  # 临时增加候选数
            else:
                log.info(f"几何分析: 目标球{best_target} -> 袋口{best_pocket}")

            # Step 4: 防守威胁评估（如果启用）
            threats = None
            if self.enable_defense:
                opponent_targets = self._infer_opponent_targets(balls, my_targets)
                threats = self.evaluate_opponent_threats(balls, opponent_targets, table)
                if threats:
                    log.debug(f"检测到{len(threats)}个对手威胁，最高威胁={threats[0][2]:.2f}")

            # Step 5: 生成候选动作（整合所有策略）
            candidates = self.generate_candidates(
                base_phi, base_v0, balls=balls, my_targets=my_targets, table=table, threats=threats
            )

            # 恢复原始 n_candidates
            self.n_candidates = original_n_candidates

            # Step 6: 评估所有候选
            best_action = None
            best_score = float("-inf")

            for action in candidates:
                score = self.evaluate_candidate(action, balls, table, last_state, my_targets)
                if score > best_score:
                    best_score = score
                    best_action = action

            # Step 7: 结果处理
            if best_action is None or best_score < self.MIN_VIABLE_SCORE:
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
